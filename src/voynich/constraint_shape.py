"""Exact visited-key marginalization and shape-only reversible proposals.

This module has no neural model, corpus acquisition or recovery controller.
Warm states contain labels and spans only; the hard dictionary is uniquely forced.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
import math

from voynich import constraint_reading as explicit
from voynich.source_action_proposal import ReadingEnvironment, ReadingState

MAX_PROFILE_WORK = 1_000_000


@dataclass(frozen=True)
class ReadingShape:
    widths: tuple[tuple[int, ...], ...]
    texts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ShapeMove:
    state: ReadingShape
    forward: F
    reverse: F
    kind: str


def validate(env, state):
    if (not isinstance(env, ReadingEnvironment) or not isinstance(state, ReadingShape)
            or type(state.widths) is not tuple or type(state.texts) is not tuple
            or len(state.widths) != len(env.records) or len(state.texts) != len(env.records)):
        raise ValueError('Literal observation environment and bounded reading shape required')
    for observed, widths, text in zip(env.records, state.widths, state.texts, strict=True):
        if (type(widths) is not tuple or type(text) is not tuple or not text
                or len(widths) != len(text)
                or any(type(w) is not int or w not in (1, 2) for w in widths)
                or sum(widths) != len(observed)
                or any(type(a) is not int or not 0 <= a < env.rows for a in text)):
            raise ValueError('Complete one/two-glyph span partition and source rows required')


def from_hard(env, state):
    # Includes literal validation and complete-reading requirement, no inferred future key.
    extended = explicit.from_hard(env, state)
    return ReadingShape(extended.widths, extended.texts)


def to_hard(env, state):
    validate(env, state)
    key = [-1]*env.rows
    for observed, widths, text in zip(env.records, state.widths, state.texts, strict=True):
        pos = 0
        for width, row in zip(widths, text, strict=True):
            unit = env.pool.index(observed[pos:pos+width])
            if key[row] not in (-1, unit):
                raise ValueError('A row has incompatible observed units across its occurrences')
            key[row] = unit
            pos += width
    answer = ReadingState(tuple(key), tuple(map(len, env.records)), state.texts)
    env.validate(answer)
    return answer


def profiles(env, state):
    """Per-visited-row unit costs, across all records, without choosing any key."""
    validate(env, state)
    histogram = Counter()
    for observed, widths, text in zip(env.records, state.widths, state.texts, strict=True):
        pos = 0
        for width, row in zip(widths, text, strict=True):
            histogram[row, observed[pos:pos+width]] += 1
            pos += width
    if len(env.pool)*len(histogram) > MAX_PROFILE_WORK:
        raise MemoryError('Registered polynomial profile-work bound; no truncation')
    costs = {row: [0]*len(env.pool) for row, _ in histogram}
    for (row, observed), count in histogram.items():
        for code, unit in enumerate(env.pool):
            costs[row][code] += count*explicit._edits(unit, observed)
    return tuple((row, tuple(costs[row])) for row in sorted(costs))


def channel(env, state, epsilon):
    """U^-m product of per-row unit sums; no key sampling or extra alignment count."""
    explicit._epsilon(epsilon)
    validate(env, state)
    if epsilon == 0:
        try:
            hard = to_hard(env, state)
        except ValueError:
            return F(0)
        return F(1, len(env.pool)**sum(k >= 0 for k in hard.key))
    if epsilon == 1:
        return F(1)
    result, powers = F(1), {}
    for _, costs in profiles(env, state):
        polynomial = F(0)
        for cost in costs:
            if cost not in powers:
                powers[cost] = explicit._power(epsilon, cost)
            polynomial += powers[cost]
            if max(polynomial.numerator.bit_length(), polynomial.denominator.bit_length()) > explicit.MAX_BITS:
                raise MemoryError('Exact row polynomial integer-work cap; no rounding')
        result = explicit._product(result, polynomial/len(env.pool))
    return result


def source_mass(source, env, state, *, stop=F(1, 225)):
    validate(env, state)
    if not isinstance(stop, F) or not 0 < stop < 1:
        raise ValueError('Positive exact stop parameter required')
    result = explicit._product(explicit._power(stop, len(state.texts)),
        explicit._power(1-stop, sum(map(len, state.texts))))
    for text in state.texts:
        context = source.state('')
        for row in text:
            value = float(source.probabilities[context, row])
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('Positive finite source coefficients required')
            result = explicit._product(result, F.from_float(value))
            context = int(source.transitions[context, row])
    return result


def target(source, env, state, epsilon):
    coefficient = channel(env, state, epsilon)
    return explicit._product(source_mass(source, env, state), coefficient) if coefficient else F(0)


def _replace(env, state, record, first, last, widths, labels):
    validate(env, state)
    if (type(record) is not int or not 0 <= record < len(state.texts)
            or type(first) is not int or type(last) is not int
            or not 0 <= first < last <= len(state.texts[record]) or last-first not in (1, 2)
            or type(widths) is not tuple or type(labels) is not tuple
            or not 1 <= len(labels) <= 2 or len(widths) != len(labels)
            or any(type(w) is not int or w not in (1, 2) for w in widths)
            or any(type(a) is not int or not 0 <= a < env.rows for a in labels)
            or sum(widths) != sum(state.widths[record][first:last]) or sum(widths) > 2):
        raise ValueError('Complete bounded shape replacement required')
    spans, texts = list(state.widths), list(state.texts)
    spans[record] = spans[record][:first]+widths+spans[record][last:]
    texts[record] = texts[record][:first]+labels+texts[record][last:]
    result = ReadingShape(tuple(spans), tuple(texts))
    validate(env, result)
    return result


def relabel(env, state, record, index, row):
    validate(env, state)
    if (type(record) is not int or not 0 <= record < len(state.texts)
            or type(index) is not int or not 0 <= index < len(state.texts[record])):
        raise ValueError('Current-span relabel anchor required')
    result = _replace(env, state, record, index, index+1, (state.widths[record][index],), (row,))
    q = F(1, sum(map(len, state.texts))*env.rows)
    return ShapeMove(result, q, q, 'relabel')


def boundary(env, state, rank, labels):
    validate(env, state)
    anchors = [(r, i) for r, record in enumerate(env.records) for i in range(len(record)-1)]
    if not anchors:
        if type(rank) is not int or rank != 0 or labels != ():
            raise ValueError('Empty boundary allocation has one explicit identity')
        return ShapeMove(state, F(1), F(1), 'boundary_identity')
    if type(rank) is not int or not 0 <= rank < len(anchors):
        raise ValueError('Uniform ALL observed width2 anchors required')
    record, start = anchors[rank]
    offsets, offset = {0: 0}, 0
    for index, width in enumerate(state.widths[record]):
        offset += width
        offsets[offset] = index+1
    if start not in offsets or start+2 not in offsets:
        if labels != ():
            raise ValueError('Invalid boundary endpoint cannot carry row choices')
        return ShapeMove(state, F(1, len(anchors)), F(1, len(anchors)), 'boundary_identity')
    first, last = offsets[start], offsets[start+2]
    widths = (1, 1) if last-first == 1 else (2,)
    if type(labels) is not tuple or len(labels) != len(widths):
        raise ValueError('One row for merge or two separate rows for split required')
    result = _replace(env, state, record, first, last, widths, labels)
    return ShapeMove(result, F(1, len(anchors)*env.rows**len(labels)),
                     F(1, len(anchors)*env.rows**(last-first)), 'boundary')


def swap_ratio(env, left, right, epsilon_left, epsilon_right):
    current = explicit._product(channel(env, left, epsilon_left), channel(env, right, epsilon_right))
    if not current:
        raise ValueError('Both current shapes need positive channel mass in their layers')
    proposed = explicit._product(channel(env, right, epsilon_left), channel(env, left, epsilon_right))
    return explicit._product(proposed, 1/current)
