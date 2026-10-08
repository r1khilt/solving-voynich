"""Global literal pair merge/split with a fixed, state-independent triplet law.

This discrete partial involution changes every occurrence across all records.
Acceptance uses the full visited-state target, including length and row prior.
Reference policy counts are bookkeeping, never a proposal ratio for this move.
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from voynich.reading_regrowth import ReadingPath, SourceRegrowth, exact_bernoulli
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


@dataclass(frozen=True)
class PairRewrite:
    state: ReadingState
    branch: str
    replacements: int


def triplet_from_rank(rows, rank):
    """Bijection from a uniform rank to ALL ordered distinct row triplets."""
    if (type(rows) is not int or not 3 <= rows <= 64 or type(rank) is not int
            or not 0 <= rank < rows*(rows-1)*(rows-2)):
        raise ValueError('Bounded source rows and a valid uniform triplet rank required')
    a, tail = divmod(rank, (rows-1)*(rows-2))
    b, c = divmod(tail, rows-2)
    b += b >= a
    for excluded in sorted((a, b)):
        c += c >= excluded
    return a, b, c


def rewrite_pair(environment, state, a, b, c):
    """Invalid branches are identity; invalid inputs raise before any rewrite."""
    if (not isinstance(environment, ReadingEnvironment) or environment.rows < 3
            or any(type(x) is not int or not 0 <= x < environment.rows for x in (a, b, c))
            or len({a, b, c}) != 3):
        raise ValueError('Literal environment and ordered distinct triplet required')
    environment.validate(state)
    if state.offsets != tuple(map(len, environment.records)):
        raise ValueError('Only complete visited-only readings can be rewritten')
    key, pool = list(state.key), environment.pool
    has_pair = any(any(x == b and y == c for x, y in zip(t, t[1:])) for t in state.texts)
    replacements, texts = 0, []
    if key[a] < 0 and key[b] >= 0 and key[c] >= 0:
        if len(pool[key[b]]) != 1 or len(pool[key[c]]) != 1 or not has_pair:
            return PairRewrite(state, 'identity', 0)
        key[a] = pool.index(pool[key[b]]+pool[key[c]])
        for text in state.texts:
            new, i = [], 0
            while i < len(text):
                if text[i:i+2] == (b, c):
                    new.append(a)
                    replacements += 1
                    i += 2
                else:
                    new.append(text[i])
                    i += 1
            texts.append(tuple(new))
        used = set(row for text in texts for row in text)
        for row in (b, c):
            if row not in used:
                key[row] = -1
        branch = 'merge'
    elif key[a] >= 0 and len(pool[key[a]]) == 2 and not has_pair:
        unit = pool[key[a]]
        singles = (pool.index(unit[:1]), pool.index(unit[1:]))
        if key[b] not in (-1, singles[0]) or key[c] not in (-1, singles[1]):
            return PairRewrite(state, 'identity', 0)
        for text in state.texts:
            new = []
            for row in text:
                if row == a:
                    new.extend((b, c))
                    replacements += 1
                else:
                    new.append(row)
            texts.append(tuple(new))
        key[a], key[b], key[c] = -1, singles[0], singles[1]
        branch = 'split'
    else:
        return PairRewrite(state, 'identity', 0)
    assert replacements > 0
    result = ReadingState(tuple(key), state.offsets, tuple(texts))
    environment.validate(result)
    return PairRewrite(result, branch, replacements)


def reading_actions(environment, state):
    """Rebuild the normalized-consumption schedule; no old action splice."""
    environment.validate(state)
    if state.offsets != tuple(map(len, environment.records)):
        raise ValueError('A complete literal reading required')
    offsets, indices, actions = [0]*len(state.texts), [0]*len(state.texts), []
    while unfinished := [i for i, r in enumerate(environment.records) if offsets[i] < len(r)]:
        record = unfinished[0]
        for i in unfinished[1:]:
            if offsets[i]*len(environment.records[record]) < offsets[record]*len(environment.records[i]):
                record = i
        row = state.texts[record][indices[record]]
        length = len(environment.pool[state.key[row]])
        actions.append(2*row+length-1)
        offsets[record] += length
        indices[record] += 1
    assert tuple(indices) == tuple(map(len, state.texts))
    return tuple(actions)


def _guard(sampler, path):
    if (not isinstance(sampler, SourceRegrowth) or not isinstance(path, ReadingPath)
            or path.law_token is not sampler.token or not path.complete or sampler.env.rows < 3
            or path.grid_bits != sampler.config.grid_bits
            or not len(path.actions) == len(path.counts) == len(path.source_terms)):
        raise ValueError('Complete supported reference reading and at least three rows required')
    if reading_actions(sampler.env, path.state) != path.actions:
        raise ValueError('Reference actions disagree with the literal reading')


def pair_rewrite_candidate(sampler, path, a, b, c, *, progress=lambda: None):
    """Initial qualification uses full forced replay, including future contexts."""
    _guard(sampler, path)
    change = rewrite_pair(sampler.env, path.state, a, b, c)
    if change.branch == 'identity':
        return path, change
    candidate = sampler.path(forced_actions=reading_actions(sampler.env, change.state), progress=progress)
    if not candidate.complete or candidate.state != change.state:
        raise ArithmeticError('Structural rewrite disagrees with full reference replay')
    return candidate, change


def pair_rewrite_ratio(sampler, old, candidate):
    """COMPLETE target ratio under fixed uniform triplet selection."""
    _guard(sampler, old)
    _guard(sampler, candidate)
    old_n, old_d = sampler.target_integers(old)
    new_n, new_d = sampler.target_integers(candidate)
    return new_n*old_d, new_d*old_n


def pair_rewrite_step(sampler, path, rng, *, progress=lambda: None):
    _guard(sampler, path)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')
    size = math.prod((sampler.env.rows, sampler.env.rows-1, sampler.env.rows-2))
    rank = int(rng.integers(size))
    triplet = triplet_from_rank(sampler.env.rows, rank)
    candidate, change = pair_rewrite_candidate(sampler, path, *triplet, progress=progress)
    accepted, blocks = False, 0
    if change.branch != 'identity':
        numerator, denominator = pair_rewrite_ratio(sampler, path, candidate)
        accepted, blocks = exact_bernoulli(numerator, denominator, rng,
                                         maximum_blocks=sampler.config.maximum_acceptance_blocks)
    return candidate if accepted else path, candidate, {
        'rank': rank, 'triplet': list(triplet), 'triplet_probability': str(Fraction(1, size)),
        'branch': change.branch, 'replacements': change.replacements,
        'accepted': accepted, 'acceptance_blocks': blocks,
        'reference_policy_probability_used_for_acceptance': False}


def eligible_triplets(environment, state):
    """Exactly all nonidentity triplets, without scanning every invalid rewrite.

    A selector uniform on this STATE-DEPENDENT list needs its reverse factor.
    Sorted order is deterministic and also fixes seeded allocation identity.
    """
    if not isinstance(environment, ReadingEnvironment) or environment.rows < 3:
        raise ValueError('Literal environment and at least three source rows required')
    environment.validate(state)
    if state.offsets != tuple(map(len, environment.records)):
        raise ValueError('Only a complete reading has structural eligibility')
    unbound, singles, doubles = [], {}, []
    for row, index in enumerate(state.key):
        if index < 0:
            unbound.append(row)
        else:
            unit = environment.pool[index]
            if len(unit) == 1:
                singles.setdefault(unit, []).append(row)
            else:
                doubles.append((row, unit))
    adjacent = set(pair for text in state.texts for pair in zip(text, text[1:]))
    result = []
    for b, c in adjacent:
        if (b != c and len(environment.pool[state.key[b]]) == 1
                and len(environment.pool[state.key[c]]) == 1):
            result.extend((a, b, c) for a in unbound)
    for a, unit in doubles:
        for b in unbound+singles.get(unit[:1], []):
            for c in unbound+singles.get(unit[1:], []):
                if b != c and (b, c) not in adjacent:
                    result.append((a, b, c))
    return tuple(sorted(result))


def valid_pair_rewrite_ratio(sampler, old, candidate, old_degree, new_degree):
    if any(type(d) is not int or d <= 0 for d in (old_degree, new_degree)):
        raise ValueError('Positive independently counted eligibility sizes required')
    numerator, denominator = pair_rewrite_ratio(sampler, old, candidate)
    return numerator*old_degree, denominator*new_degree


def valid_pair_rewrite_step(sampler, path, rng, *, progress=lambda: None):
    """Uniform eligible selection; exact T(new)/T(old) * V(old)/V(new)."""
    _guard(sampler, path)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')
    eligible = eligible_triplets(sampler.env, path.state)
    if not eligible:
        return path, path, {'branch': 'identity', 'old_degree': 0, 'new_degree': 0,
            'triplet': None, 'accepted': False, 'acceptance_blocks': 0,
            'reference_policy_probability_used_for_acceptance': False}
    triplet = eligible[int(rng.integers(len(eligible)))]
    candidate, change = pair_rewrite_candidate(sampler, path, *triplet, progress=progress)
    reverse = eligible_triplets(sampler.env, candidate.state)
    if triplet not in reverse or change.branch == 'identity':
        raise ArithmeticError('Eligible rewrite has no matching reverse triplet')
    numerator, denominator = valid_pair_rewrite_ratio(sampler, path, candidate, len(eligible), len(reverse))
    accepted, blocks = exact_bernoulli(numerator, denominator, rng,
                                     maximum_blocks=sampler.config.maximum_acceptance_blocks)
    return candidate if accepted else path, candidate, {
        'triplet': list(triplet), 'old_degree': len(eligible), 'new_degree': len(reverse),
        'forward_triplet_probability': str(Fraction(1, len(eligible))),
        'reverse_triplet_probability': str(Fraction(1, len(reverse))),
        'branch': change.branch, 'replacements': change.replacements,
        'accepted': accepted, 'acceptance_blocks': blocks,
        'reference_policy_probability_used_for_acceptance': False}
