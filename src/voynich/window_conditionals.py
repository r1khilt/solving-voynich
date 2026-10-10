"""Preparatory exact local source ratios and cold conditional heat-bath repair.

The unchanged exterior source tail cancels ONLY after contexts match. This is
not an immediate-letter surrogate or an irrational-temperature heat bath.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import bisect
import math

import numpy as np

from voynich.reading_label_transport import reference_reading
from voynich.reading_pair_rewrite import _guard
from voynich.reading_window import observed_window, replace_window, window_choices
from voynich.tempered_reading import score_complete_state

MAX_BITS = 4_194_304
MAX_CHOICES = 4160
MAX_TOTAL_BITS = 64*1024**2  # Eight MiB of raw integer payload, before Python overhead.


@dataclass(frozen=True)
class WindowRatio:
    state: object
    ratio: F
    suffix_steps: int
    omitted_suffix_steps: int
    contexts_matched: bool
    source_length_delta: int
    visited_delta: int


def window_ratio(sampler, path, record, start, width, replacement):
    """Exact full-target ratio with cancellation of identical exterior factors."""
    _guard(sampler, path)
    family = window_choices(sampler.env, path.state, record, start, width)
    if family is None:
        raise ValueError('Exact ratio requires valid window endpoints')
    state = replace_window(sampler.env, path.state, record, start, width, replacement)
    text = path.state.texts[record]
    context = sampler.root
    for row in text[:family.first]:
        context = int(sampler.t[context, row])

    def sequence(rows, initial):
        value, current = F(1), initial
        for row in rows:
            pairs, _, _ = sampler._row(current)
            value *= F(*pairs[row])
            current = int(sampler.t[current, row])
        return value, current

    old_inner = text[family.first:family.last]
    old_value, old_context = sequence(old_inner, context)
    new_value, new_context = sequence(replacement, context)
    ratio = new_value/old_value
    suffix = text[family.last:]
    steps = 0
    for row in suffix:
        if old_context == new_context:
            break
        old_term, old_context = sequence((row,), old_context)
        new_term, new_context = sequence((row,), new_context)
        ratio *= new_term/old_term
        steps += 1
    length_delta = len(replacement)-len(old_inner)
    visited_delta = sum(k >= 0 for k in state.key)-sum(k >= 0 for k in path.state.key)
    ratio *= (1-sampler.config.stop)**length_delta*F(len(sampler.env.pool))**(-visited_delta)
    return WindowRatio(state, ratio, steps, len(suffix)-steps, old_context == new_context,
                       length_delta, visited_delta)


def integer_weights(weights, *, maximum_bits=MAX_BITS):
    """Convert a bounded positive rational family to exact integer weights."""
    if (type(weights) is not tuple or not 1 <= len(weights) <= MAX_CHOICES
            or any(not isinstance(w, F) or w <= 0 for w in weights)
            or type(maximum_bits) is not int or not 1 <= maximum_bits <= MAX_BITS):
        raise ValueError('Bounded positive rational weights and integer cap required')
    denominator = 1
    if sum(w.numerator.bit_length()+w.denominator.bit_length() for w in weights) > MAX_TOTAL_BITS:
        raise MemoryError('Categorical aggregate input integer payload exceeds cap')
    for w in weights:
        if max(w.numerator.bit_length(), w.denominator.bit_length()) > maximum_bits:
            raise MemoryError('Rational categorical input exceeds integer cap')
        factor = w.denominator//math.gcd(denominator, w.denominator)
        if denominator.bit_length()+factor.bit_length()-1 > maximum_bits:
            raise MemoryError('Categorical common denominator exceeds cap before multiplication')
        denominator *= factor
        if denominator.bit_length() > maximum_bits:
            raise MemoryError('Categorical common denominator exceeds actual integer cap')
    result, total, payload_bits = [], 0, 0
    for w in weights:
        factor = denominator//w.denominator
        if w.numerator.bit_length()+factor.bit_length()-1 > maximum_bits:
            raise MemoryError('Categorical aligned weight exceeds cap before multiplication')
        value = w.numerator*factor
        payload_bits += value.bit_length()
        if payload_bits > MAX_TOTAL_BITS:
            raise MemoryError('Categorical aggregate aligned integer payload exceeds cap')
        if max(value.bit_length(), (total+value).bit_length()) > maximum_bits:
            raise MemoryError('Categorical total exceeds integer cap')
        result.append(value)
        total += value
    divisor = math.gcd(*result)
    return tuple(value//divisor for value in result)


def rational_categorical(weights, rng, *, maximum_blocks=16, maximum_bits=MAX_BITS):
    """Draw exactly by locating a uniform raw64 interval inside an integer CDF.

    A boundary-straddling interval consumes another block; cap failure aborts,
    never rounds or renormalizes. Even a one-choice draw consumes one block.
    """
    if (not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64)
            or type(maximum_blocks) is not int or not 1 <= maximum_blocks <= 16):
        raise ValueError('PCG64 and bounded acceptance blocks required')
    values = integer_weights(weights, maximum_bits=maximum_bits)
    cumulative, running = [], 0
    for value in values:
        running += value
        cumulative.append(running)
    # Bound temporary interval products before consuming any RNG.
    if running.bit_length()+64*maximum_blocks+1 > maximum_bits:
        raise MemoryError('Categorical interval products exceed integer cap')
    prefix = 0
    for blocks in range(1, maximum_blocks+1):
        prefix = (prefix << 64)+int(rng.bit_generator.random_raw())
        scale = 1 << (64*blocks)
        index = bisect.bisect_right(cumulative, prefix*running//scale)
        if (prefix+1)*running <= cumulative[index]*scale:
            return index, blocks
    raise RuntimeError('Exact rational categorical undecided at fixed block cap')


def cold_window_heatbath(sampler, path, rng, *, progress=lambda: None):
    """Exact cold conditional repair. All family members, no MH rejection.

    Source-ratio evaluation is complete despite cancelling matched suffixes.
    Invalid-boundary windows are identity; warm conditional roots are unsupported.
    This method is NOT in the active/frozen replica-recovery controller.
    """
    _guard(sampler, path)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('PCG64 required')
    count = sum(2*len(r)-1 for r in sampler.env.records)
    rank = int(rng.integers(count))
    record, start, width = observed_window(sampler.env, rank)
    family = window_choices(sampler.env, path.state, record, start, width)
    info = {'rank': rank, 'record': record, 'start': start, 'width': width,
        'valid_endpoints': family is not None, 'family_size': 0, 'selected': None,
        'raw64_blocks': 0, 'changed': False, 'candidate_suffix_steps': 0,
        'candidate_omitted_suffix_steps': 0, 'degree': 1}
    if family is None:
        return path, info
    evaluated = []
    for replacement in family.replacements:
        progress()
        evaluated.append(window_ratio(sampler, path, record, start, width, replacement))
    weights = tuple(value.ratio for value in evaluated)
    index, blocks = rational_categorical(weights, rng,
        maximum_blocks=sampler.config.maximum_acceptance_blocks)
    chosen = evaluated[index]
    info.update({'family_size': len(evaluated), 'selected': family.replacements[index],
        'raw64_blocks': blocks, 'changed': chosen.state != path.state,
        'candidate_suffix_steps': sum(v.suffix_steps for v in evaluated),
        'candidate_omitted_suffix_steps': sum(v.omitted_suffix_steps for v in evaluated)})
    if chosen.state == path.state:
        return path, info
    point = score_complete_state(sampler, chosen.state, progress=progress)
    # Complete target replay is still required before returning a retained path.
    old_n, old_d = sampler.target_integers(path)
    assert F(point.numerator*old_d, point.denominator*old_n) == chosen.ratio
    return reference_reading(sampler, point, progress=progress), info
