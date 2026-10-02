"""Existing joint label involution adapted to visited-only reading dictionaries.

Inventory and segmentation stay fixed. Whole source contexts are rescored.
The target-only candidate is not an action-policy draw or a chain density.
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from voynich.reading_regrowth import ReadingPath, SourceRegrowth, exact_bernoulli
from voynich.source_action_proposal import ReadingState


@dataclass(frozen=True)
class LabelPoint:
    actions: tuple
    state: ReadingState
    source_terms: tuple
    numerator: int
    denominator: int
    law_token: object

    def log_potential(self):
        return math.log(self.numerator)-math.log(self.denominator)


def _guard(sampler, path):
    if (not isinstance(sampler, SourceRegrowth) or not isinstance(path, ReadingPath)
            or path.law_token is not sampler.token or not path.complete
            or sampler.env.rows < 2 or path.grid_bits != sampler.config.grid_bits):
        raise ValueError('Complete supported reading and at least two source rows required')
    sampler.env.validate(path.state)
    if sampler.env.selected_record(path.state) is not None:
        raise ValueError('Reported complete state is not literal completion')


def pair_probability(path, a, b):
    """MARGINAL unordered pair law: first a uniform USED row, then any other row."""
    rows = len(path.state.key)
    if (not path.complete or rows < 2 or type(a) is not int or type(b) is not int
            or not 0 <= a < b < rows):
        raise ValueError('Ordered distinct pair in a complete reading required')
    used = sum(k >= 0 for k in path.state.key)
    if not used:
        raise ValueError('Nonempty complete readings must visit a source row')
    orientations = int(path.state.key[a] >= 0)+int(path.state.key[b] >= 0)
    return Fraction(orientations, used*(rows-1))


def score_label_transport(sampler, path, a, b, *, progress=lambda: None):
    """Rescore the complete deterministic candidate without a reference q replay."""
    _guard(sampler, path)
    rows = sampler.env.rows
    if type(a) is not int or type(b) is not int or not 0 <= a < b < rows:
        raise ValueError('Ordered distinct source rows required')

    def rename(row):
        return b if row == a else a if row == b else row

    key = list(path.state.key)
    key[a], key[b] = key[b], key[a]
    texts = tuple(tuple(rename(row) for row in text) for text in path.state.texts)
    actions = tuple(2*rename(action//2)+action%2 for action in path.actions)
    state = ReadingState(tuple(key), path.state.offsets, texts)
    sampler.env.validate(state)
    assert sum(k >= 0 for k in state.key) == sum(k >= 0 for k in path.state.key)
    assert tuple(map(len, texts)) == tuple(map(len, path.state.texts))
    # Keep terms in ACTION order, matching the retained reference path. Records
    # interleave under the normalized-consumption scheduler; concatenating
    # per-record terms has the same product but invalid reference bookkeeping.
    terms = []
    offsets, indices = [0]*len(texts), [0]*len(texts)
    contexts = [sampler.root]*len(texts)
    for action in actions:
        progress()
        unfinished = [i for i, record in enumerate(sampler.env.records) if offsets[i] < len(record)]
        if not unfinished:
            raise ValueError('Actions continue beyond literal completion')
        selected = unfinished[0]
        for i in unfinished[1:]:
            if offsets[i]*len(sampler.env.records[selected]) < offsets[selected]*len(sampler.env.records[i]):
                selected = i
        row, length = action//2, action%2+1
        unit = sampler.env.records[selected][offsets[selected]:offsets[selected]+length]
        if (indices[selected] >= len(texts[selected]) or texts[selected][indices[selected]] != row
                or len(unit) != length or sampler.env.pool[key[row]] != unit):
            raise ValueError('Actions disagree with the complete transformed reading')
        pairs, exponents, _ = sampler._row(contexts[selected])
        terms.append((pairs[row][0], exponents[row]))
        contexts[selected] = int(sampler.t[contexts[selected], row])
        offsets[selected] += length
        indices[selected] += 1
    if tuple(offsets) != state.offsets or tuple(indices) != tuple(map(len, texts)):
        raise ValueError('Actions truncate the transformed reading')
    n, records = len(actions), len(texts)
    assert n == len(terms)
    stop, continuation = sampler.config.stop, 1-sampler.config.stop
    numerator = math.prod(p for p, _ in terms)*continuation.numerator**n*stop.numerator**records
    denominator = (1 << sum(e for _, e in terms))*continuation.denominator**n*stop.denominator**records
    denominator *= len(sampler.env.pool)**sum(k >= 0 for k in state.key)
    return LabelPoint(actions, state, tuple(terms), numerator, denominator, sampler.token)


def reference_reading(sampler, point, *, progress=lambda: None):
    """Reconstruct policy counts ONLY for a retained candidate/future regrowth.

    This reference probability does not enter the symmetric label acceptance.
    """
    if not isinstance(point, LabelPoint) or point.law_token is not sampler.token:
        raise ValueError('Target candidate belongs to a different source/policy instance')
    result = sampler.path(forced_actions=point.actions, progress=progress)
    if (result.state != point.state or result.source_terms != point.source_terms
            or sampler.target_integers(result) != (point.numerator, point.denominator)):
        raise ArithmeticError('Transport score differs from full literal/reference replay')
    return result


def label_transport_step(sampler, path, rng, *, progress=lambda: None):
    _guard(sampler, path)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')
    used = [row for row, key in enumerate(path.state.key) if key >= 0]
    a = used[int(rng.integers(len(used)))]
    b = int(rng.integers(sampler.env.rows-1))
    b += b >= a
    a, b = sorted((a, b))
    candidate = score_label_transport(sampler, path, a, b, progress=progress)
    forward = pair_probability(path, a, b)
    # m and whether the pair has one/two used rows are invariant under this swap.
    reversed_used = int(candidate.state.key[a] >= 0)+int(candidate.state.key[b] >= 0)
    reverse = Fraction(reversed_used, len(used)*(sampler.env.rows-1))
    assert forward == reverse and forward > 0
    old_n, old_d = sampler.target_integers(path)
    numerator, denominator = candidate.numerator*old_d, candidate.denominator*old_n
    accepted, blocks = exact_bernoulli(numerator, denominator, rng,
                                     maximum_blocks=sampler.config.maximum_acceptance_blocks)
    retained = reference_reading(sampler, candidate, progress=progress) if accepted else path
    return retained, candidate, {'pair': [a, b], 'accepted': accepted, 'acceptance_blocks': blocks,
        'pair_probability': str(forward), 'reference_policy_probability_used_for_acceptance': False}
