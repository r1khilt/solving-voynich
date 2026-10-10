"""Preparatory literal single-occurrence repair; not in the frozen replica run.

Boundaries stay fixed. Source-row equality and the visited inventory may change.
Uniform occurrence/compatible-row selection is symmetric, including identity.
"""
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from voynich.exact_radical import exact_radical_bernoulli, tempered_ratio
from voynich.reading_label_transport import reference_reading
from voynich.reading_pair_rewrite import _guard
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.tempered_reading import path_target, point_target, score_complete_state


@dataclass(frozen=True)
class OccurrenceChange:
    state: ReadingState
    old_row: int
    new_row: int
    compatible_rows: tuple[int, ...]
    visited_delta: int


def occurrence_rows(environment, state, record, index):
    """Free rows and rows already bound to this occurrence's observed unit."""
    if not isinstance(environment, ReadingEnvironment):
        raise ValueError('Literal reading environment required')
    environment.validate(state)
    if (state.offsets != tuple(map(len, environment.records))
            or type(record) is not int or not 0 <= record < len(state.texts)
            or type(index) is not int or not 0 <= index < len(state.texts[record])):
        raise ValueError('Valid occurrence in a complete reading required')
    unit = state.key[state.texts[record][index]]
    return tuple(row for row, bound in enumerate(state.key) if bound in (-1, unit))


def replace_occurrence(environment, state, record, index, new_row):
    """Change ONE source position, bind its new row and release an unused old row."""
    compatible = occurrence_rows(environment, state, record, index)
    if type(new_row) is not int or new_row not in compatible:
        raise ValueError('Replacement must be a compatible source row')
    old_row = state.texts[record][index]
    if new_row == old_row:
        return OccurrenceChange(state, old_row, new_row, compatible, 0)
    text = state.texts[record]
    texts = state.texts[:record]+(text[:index]+(new_row,)+text[index+1:],)+state.texts[record+1:]
    key = list(state.key)
    key[new_row] = state.key[old_row]
    if not any(old_row in other for other in texts):
        key[old_row] = -1
    changed = ReadingState(tuple(key), state.offsets, texts)
    environment.validate(changed)
    if occurrence_rows(environment, changed, record, index) != compatible:
        raise ArithmeticError('Occurrence repair changed its reverse selector set')
    delta = sum(k >= 0 for k in changed.key)-sum(k >= 0 for k in state.key)
    return OccurrenceChange(changed, old_row, new_row, compatible, delta)


def occurrence_step(sampler, path, rng, degree=1, *, progress=lambda: None):
    """Exact warmed MH for a uniform literal occurrence repair.

    Accepted target points reconstruct the unchanged reference policy for later
    regrowth. Its action-path probability is NOT a proposal correction here.
    Integer/block caps abort rather than approximate the acceptance decision.
    """
    _guard(sampler, path)
    tempered_ratio(F(1), F(1), degree)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')
    total = sum(map(len, path.state.texts))
    rank = int(rng.integers(total))
    index, record = rank, 0
    while index >= len(path.state.texts[record]):
        index -= len(path.state.texts[record])
        record += 1
    compatible = occurrence_rows(sampler.env, path.state, record, index)
    new_row = compatible[int(rng.integers(len(compatible)))]
    change = replace_occurrence(sampler.env, path.state, record, index, new_row)
    info = {'rank': rank, 'record': record, 'index': index, 'old_row': change.old_row,
        'new_row': new_row, 'compatible_rows': compatible, 'visited_delta': change.visited_delta,
        'forward_component_probability': str(F(1, total*len(compatible))),
        'reverse_component_probability': str(F(1, total*len(compatible))),
        'degree': degree, 'accepted': False, 'acceptance_blocks': 0,
        'reference_policy_probability_used_for_acceptance': False}
    if new_row == change.old_row:
        return path, path, None, info
    candidate = score_complete_state(sampler, change.state, progress=progress)
    ratio = tempered_ratio(point_target(candidate)/path_target(sampler, path), F(1), degree)
    accepted, blocks = exact_radical_bernoulli(ratio, rng,
        maximum_blocks=sampler.config.maximum_acceptance_blocks)
    info.update({'accepted': accepted, 'acceptance_blocks': blocks})
    retained = reference_reading(sampler, candidate, progress=progress) if accepted else path
    return retained, candidate, ratio, info
