"""Preparatory one/two-glyph window repair, outside the frozen search controller.

Uniform OBSERVED windows include invalid-boundary self loops. At a valid window,
uniform compatible replacements have identical forward/reverse candidate sets.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import itertools

import numpy as np

from voynich.exact_radical import exact_radical_bernoulli, tempered_ratio
from voynich.reading_label_transport import reference_reading
from voynich.reading_pair_rewrite import _guard
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.tempered_reading import path_target, point_target, score_complete_state


@dataclass(frozen=True)
class WindowChoices:
    record: int
    start: int
    width: int
    first: int
    last: int
    outside_key: tuple[int, ...]
    replacements: tuple[tuple[int, ...], ...]


def observed_window(environment, rank):
    """Rank all nonempty width1/2 observed windows, without a reading-dependent filter."""
    if not isinstance(environment, ReadingEnvironment):
        raise ValueError('Literal environment required')
    if type(rank) is not int or not 0 <= rank < sum(2*len(r)-1 for r in environment.records):
        raise ValueError('Observed-window rank out of range')
    for record, observation in enumerate(environment.records):
        count = 2*len(observation)-1
        if rank < count:
            return record, rank//2, rank % 2+1
        rank -= count
    raise ArithmeticError('Window rank failed to resolve')


def window_choices(environment, state, record, start, width):
    """Conditional family after integrating bindings used only inside this window.

    None means an endpoint falls inside an existing two-glyph emission; the
    uniformly selected observed window then contributes an ordinary identity.
    """
    if not isinstance(environment, ReadingEnvironment):
        raise ValueError('Literal environment required')
    environment.validate(state)
    if (state.offsets != tuple(map(len, environment.records))
            or type(record) is not int or not 0 <= record < len(state.texts)
            or type(start) is not int or start < 0 or type(width) is not int or width not in (1, 2)
            or start+width > len(environment.records[record])):
        raise ValueError('Width1/2 observed window in a complete reading required')
    boundaries, offset = {0: 0}, 0
    for i, row in enumerate(state.texts[record]):
        offset += len(environment.pool[state.key[row]])
        boundaries[offset] = i+1
    if start not in boundaries or start+width not in boundaries:
        return None
    first, last = boundaries[start], boundaries[start+width]
    outside = set(itertools.chain.from_iterable(
        text if r != record else text[:first]+text[last:] for r, text in enumerate(state.texts)))
    key = tuple(k if row in outside else -1 for row, k in enumerate(state.key))
    unit = environment.records[record][start:start+width]

    def compatible(glyphs):
        code = environment.pool.index(glyphs)
        return tuple(row for row, bound in enumerate(key) if bound in (-1, code))

    replacements = [(row,) for row in compatible(unit)]
    if width == 2:
        replacements += [(a, b) for a in compatible(unit[:1]) for b in compatible(unit[1:])
                         if a != b or unit[0] == unit[1]]
    assert state.texts[record][first:last] in replacements
    assert len(replacements) == len(set(replacements)) and len(replacements) <= environment.rows*(environment.rows+1)
    return WindowChoices(record, start, width, first, last, key, tuple(replacements))


def replace_window(environment, state, record, start, width, replacement):
    """Re-derive the conditional family; reject forged or stale choices."""
    choices = window_choices(environment, state, record, start, width)
    if choices is None or type(replacement) is not tuple or replacement not in choices.replacements:
        raise ValueError('Literal conditional window replacement required')
    if any(type(row) is not int for row in replacement):
        raise ValueError('Replacement rows must be integers')
    old = state.texts[record]
    changed_text = old[:choices.first]+replacement+old[choices.last:]
    texts = state.texts[:record]+(changed_text,)+state.texts[record+1:]
    key, offset = list(choices.outside_key), start
    length = width if len(replacement) == 1 else 1
    for row in replacement:
        unit = environment.records[record][offset:offset+length]
        code = environment.pool.index(unit)
        if key[row] not in (-1, code):
            raise ArithmeticError('Window family contains incompatible repeated row')
        key[row] = code
        offset += length
    assert offset == start+width
    changed = ReadingState(tuple(key), state.offsets, texts)
    environment.validate(changed)
    reverse = window_choices(environment, changed, record, start, width)
    if (reverse.outside_key, reverse.replacements) != (choices.outside_key, choices.replacements):
        raise ArithmeticError('Window repair changed its reverse conditional family')
    return changed


def window_step(sampler, path, rng, degree=1, *, progress=lambda: None):
    """Uniform-window/conditional-family exact MH; no action-policy q factor."""
    _guard(sampler, path)
    tempered_ratio(F(1), F(1), degree)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')
    count = sum(2*len(r)-1 for r in sampler.env.records)
    rank = int(rng.integers(count))
    record, start, width = observed_window(sampler.env, rank)
    choices = window_choices(sampler.env, path.state, record, start, width)
    info = {'rank': rank, 'record': record, 'start': start, 'width': width, 'degree': degree,
        'valid_endpoints': choices is not None, 'replacement': None, 'family_size': 0,
        'accepted': False, 'acceptance_blocks': 0, 'visited_delta': 0, 'source_length_delta': 0,
        'reference_policy_probability_used_for_acceptance': False}
    if choices is None:
        return path, path, None, info
    replacement = choices.replacements[int(rng.integers(len(choices.replacements)))]
    state = replace_window(sampler.env, path.state, record, start, width, replacement)
    info.update({'replacement': replacement, 'family_size': len(choices.replacements),
        'forward_component_probability': str(F(1, count*len(choices.replacements))),
        'reverse_component_probability': str(F(1, count*len(choices.replacements))),
        'visited_delta': sum(k >= 0 for k in state.key)-sum(k >= 0 for k in path.state.key),
        'source_length_delta': sum(map(len, state.texts))-len(path.actions)})
    if state == path.state:
        return path, path, None, info
    candidate = score_complete_state(sampler, state, progress=progress)
    ratio = tempered_ratio(point_target(candidate)/path_target(sampler, path), F(1), degree)
    accepted, blocks = exact_radical_bernoulli(ratio, rng,
        maximum_blocks=sampler.config.maximum_acceptance_blocks)
    info.update({'accepted': accepted, 'acceptance_blocks': blocks})
    retained = reference_reading(sampler, candidate, progress=progress) if accepted else path
    return retained, candidate, ratio, info
