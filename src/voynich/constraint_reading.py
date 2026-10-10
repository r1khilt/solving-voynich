"""Preparation: finite relaxed channel states and explicit local proposal components.

There is no production sampler, neural policy or empirical qualification here.
At violation zero the state/target coincide with the original complete reading.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import itertools
import math

from voynich.source_action_proposal import ReadingEnvironment, ReadingState

MAX_BITS = 4_194_304


@dataclass(frozen=True)
class ConstraintReading:
    key: tuple[int, ...]
    widths: tuple[tuple[int, ...], ...]
    texts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ConstraintMove:
    state: ConstraintReading
    forward: F
    reverse: F
    kind: str


def validate(env, state):
    if not isinstance(env, ReadingEnvironment) or not isinstance(state, ConstraintReading):
        raise ValueError('Literal environment and relaxed reading required')
    if (type(state.key) is not tuple or len(state.key) != env.rows
            or any(type(k) is not int or not -1 <= k < len(env.pool) for k in state.key)
            or type(state.widths) is not tuple or type(state.texts) is not tuple
            or len(state.widths) != len(env.records) or len(state.texts) != len(env.records)):
        raise ValueError('Exact visited key and record allocation required')
    used = set()
    for record, widths, text in zip(env.records, state.widths, state.texts, strict=True):
        if (type(widths) is not tuple or type(text) is not tuple or not text or len(widths) != len(text)
                or any(type(w) is not int or w not in (1, 2) for w in widths) or sum(widths) != len(record)
                or any(type(a) is not int or not 0 <= a < env.rows for a in text)):
            raise ValueError('Bounded partition and source labels required')
        used.update(text)
    if used != {a for a, k in enumerate(state.key) if k >= 0}:
        raise ValueError('Only and all visited rows must have bindings')


def _edits(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        following = [i]
        for j, y in enumerate(b, 1):
            following.append(min(row[j]+1, following[-1]+1, row[j-1]+(x != y)))
        row = following
    return row[-1]


def violations(env, state):
    validate(env, state)
    result = 0
    for observed, widths, text in zip(env.records, state.widths, state.texts, strict=True):
        offset = 0
        for width, row in zip(widths, text, strict=True):
            result += _edits(env.pool[state.key[row]], observed[offset:offset+width])
            offset += width
    return result


def from_hard(env, state):
    env.validate(state)
    if state.offsets != tuple(map(len, env.records)):
        raise ValueError('Only complete hard readings embed into the relaxed space')
    result = ConstraintReading(state.key,
        tuple(tuple(len(env.pool[state.key[a]]) for a in text) for text in state.texts), state.texts)
    validate(env, result)
    assert violations(env, result) == 0
    return result


def to_hard(env, state):
    if violations(env, state):
        raise ValueError('An inconsistent channel state cannot become a hard reading')
    result = ReadingState(state.key, tuple(map(len, env.records)), state.texts)
    env.validate(result)
    return result


def _epsilon(epsilon):
    if not isinstance(epsilon, F) or not 0 <= epsilon <= 1:
        raise ValueError('Exact rational constraint parameter in [0,1] required')


def _product(left, right):
    if max(left.numerator.bit_length()+right.numerator.bit_length(),
           left.denominator.bit_length()+right.denominator.bit_length()) > MAX_BITS:
        raise MemoryError('Exact relaxed-target integer-work cap; no rounding')
    return left*right


def _power(value, exponent):
    if max(value.numerator.bit_length(), value.denominator.bit_length())*exponent > MAX_BITS:
        raise MemoryError('Exact relaxed-target power-work cap; no rounding')
    return value**exponent


def target(source, env, state, epsilon, *, stop=F(1, 225)):
    """Exact binary64 source-factor target; hard epsilon0 is the original target."""
    _epsilon(epsilon)
    d = violations(env, state)
    if not isinstance(stop, F) or not 0 < stop < 1:
        raise ValueError('Positive exact stop parameter required')
    if epsilon == 0 and d:
        return F(0)
    n = sum(map(len, state.texts))
    mass = _product(_power(stop, len(state.texts)), _power(1-stop, n))
    mass = _product(mass, F(1, len(env.pool)**sum(k >= 0 for k in state.key)))
    mass = _product(mass, _power(epsilon, d))
    for text in state.texts:
        context = source.state('')
        for a in text:
            coefficient = float(source.probabilities[context, a])
            if not math.isfinite(coefficient) or not 0 < coefficient <= 1:
                raise ValueError('Positive finite source coefficients required')
            mass = _product(mass, F.from_float(coefficient))
            context = int(source.transitions[context, a])
    return mass


def _replace(env, state, record, first, last, widths, rows, bindings):
    """Local block, releasing exactly the rows with no remaining exterior occurrence."""
    validate(env, state)
    if (type(record) is not int or not 0 <= record < len(env.records)
            or type(first) is not int or type(last) is not int
            or not 0 <= first < last <= len(state.texts[record]) or last-first not in (1, 2)
            or type(widths) is not tuple or type(rows) is not tuple or not 1 <= len(rows) <= 2
            or len(widths) != len(rows) or any(type(w) is not int or w not in (1, 2) for w in widths)
            or any(type(a) is not int or not 0 <= a < env.rows for a in rows)
            or sum(widths) != sum(state.widths[record][first:last]) or sum(widths) > 2):
        raise ValueError('One/two-glyph block and its complete replacement required')
    outside = set(itertools.chain.from_iterable(t if i != record else t[:first]+t[last:]
        for i, t in enumerate(state.texts)))
    fresh = set(rows)-outside
    if (type(bindings) is not dict or set(bindings) != fresh
            or any(type(a) is not int or type(k) is not int or not 0 <= k < len(env.pool)
                   for a, k in bindings.items())):
        raise ValueError('Exactly one unit draw for each newly visited row required')
    key = [k if a in outside else -1 for a, k in enumerate(state.key)]
    for a, k in bindings.items():
        key[a] = k
    texts = list(state.texts)
    spans = list(state.widths)
    texts[record] = texts[record][:first]+rows+texts[record][last:]
    spans[record] = spans[record][:first]+widths+spans[record][last:]
    result = ConstraintReading(tuple(key), tuple(spans), tuple(texts))
    validate(env, result)
    old_fresh = set(state.texts[record][first:last])-outside
    return result, len(old_fresh), len(fresh)


def rebind(env, state, row, unit):
    validate(env, state)
    if (type(row) is not int or not 0 <= row < env.rows
            or type(unit) is not int or not 0 <= unit < len(env.pool)):
        raise ValueError('Uniform all-row/all-unit component required')
    key = list(state.key)
    if key[row] >= 0:
        key[row] = unit
    result = ConstraintReading(tuple(key), state.widths, state.texts)
    validate(env, result)
    q = F(1, env.rows*len(env.pool))
    return ConstraintMove(result, q, q, 'rebind')


def relabel(env, state, record, index, row, bindings):
    validate(env, state)
    if (type(record) is not int or not 0 <= record < len(env.records)
            or type(index) is not int or not 0 <= index < len(state.texts[record])):
        raise ValueError('Uniform current-span anchor required')
    result, old, new = _replace(env, state, record, index, index+1,
        (state.widths[record][index],), (row,), bindings)
    anchors = sum(map(len, state.texts))
    return ConstraintMove(result, F(1, anchors*env.rows*len(env.pool)**new),
        F(1, anchors*env.rows*len(env.pool)**old), 'relabel')


def boundary(env, state, rank, rows, bindings):
    """Uniform ALL observed width2 anchors, toggling pair-span versus two singles."""
    validate(env, state)
    anchors = [(r, i) for r, c in enumerate(env.records) for i in range(len(c)-1)]
    if not anchors:
        if type(rank) is not int or rank != 0 or rows != () or bindings != {}:
            raise ValueError('Empty boundary allocation is an explicit identity')
        return ConstraintMove(state, F(1), F(1), 'boundary_identity')
    if type(rank) is not int or not 0 <= rank < len(anchors):
        raise ValueError('Nonempty observed width2 anchor allocation required')
    record, start = anchors[rank]
    offsets, offset = {0: 0}, 0
    for j, width in enumerate(state.widths[record]):
        offset += width
        offsets[offset] = j+1
    if start not in offsets or start+2 not in offsets:
        if rows != () or bindings != {}:
            raise ValueError('Invalid-boundary identity consumes no row/binding draws')
        q = F(1, len(anchors))
        return ConstraintMove(state, q, q, 'boundary_identity')
    first, last = offsets[start], offsets[start+2]
    widths = (1, 1) if last-first == 1 else (2,)
    if type(rows) is not tuple or len(rows) != len(widths):
        raise ValueError('Complete split/merge row tuple required')
    result, old, new = _replace(env, state, record, first, last, widths, rows, bindings)
    return ConstraintMove(result, F(1, len(anchors)*env.rows**len(rows)*len(env.pool)**new),
        F(1, len(anchors)*env.rows**(last-first)*len(env.pool)**old), 'boundary')


def swap_ratio(env, x, y, epsilon_x, epsilon_y):
    _epsilon(epsilon_x)
    _epsilon(epsilon_y)
    dx, dy = violations(env, x), violations(env, y)
    denominator = _product(_power(epsilon_x, dx), _power(epsilon_y, dy))
    if not denominator:
        raise ValueError('Current replica states must have positive target support')
    numerator = _product(_power(epsilon_x, dy), _power(epsilon_y, dx))
    return _product(numerator, 1/denominator)
