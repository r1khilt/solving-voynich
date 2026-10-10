"""Independent small-key marginal checks and direct shape proposal construction."""
from collections import defaultdict
from fractions import Fraction as F
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_constraint_reading import states as key_states
from tests.test_constraint_sampling import full_mass
from tests.test_window_support_obstruction import obstruction
from voynich import constraint_shape as shape
from voynich.source_action_proposal import ReadingEnvironment

EPSILONS = (F(0), F(1, 8), F(1, 2), F(1))


def source(rows, contextual, root=0):
    base = np.array([.5, .25, .25] if rows == 3 else [.5, .25, .125, .125])
    count = rows if contextual else 1
    return SimpleNamespace(probabilities=np.array([np.roll(base, i) for i in range(count)]),
        transitions=np.array([[(i+a+1) % count for a in range(rows)] for i in range(count)]), state=lambda _: root)


def direct_shapes(env):
    def partitions(n):
        if n == 0:
            yield ()
        for width in (1, 2):
            if width <= n:
                for rest in partitions(n-width):
                    yield (width,)+rest
    for widths in itertools.product(*(tuple(partitions(len(r))) for r in env.records)):
        for flat in itertools.product(range(env.rows), repeat=sum(map(len, widths))):
            offset, texts = 0, []
            for spans in widths:
                texts.append(flat[offset:offset+len(spans)])
                offset += len(spans)
            yield shape.ReadingShape(widths, tuple(texts))


def summed_keys(env, expert):
    maps = {epsilon: defaultdict(F) for epsilon in EPSILONS}
    for state in key_states(env):
        keyless = shape.ReadingShape(state.widths, state.texts)
        for epsilon in EPSILONS:
            maps[epsilon][keyless] += full_mass(expert, env, state, epsilon)
    return {e: dict(values) for e, values in maps.items()}


def independent_components(env, state, kernel):
    def replacement(record, first, last, widths, labels):
        texts, spans = list(state.texts), list(state.widths)
        texts[record] = texts[record][:first]+labels+texts[record][last:]
        spans[record] = spans[record][:first]+widths+spans[record][last:]
        return shape.ReadingShape(tuple(spans), tuple(texts))
    if kernel == 'relabel':
        n = sum(map(len, state.texts))
        for record, text in enumerate(state.texts):
            for index in range(len(text)):
                for row in range(env.rows):
                    candidate = replacement(record, index, index+1, (state.widths[record][index],), (row,))
                    yield candidate, F(1, n*env.rows), (record, index, row)
    else:
        anchors = [(r, i) for r, record in enumerate(env.records) for i in range(len(record)-1)]
        if not anchors:
            yield state, F(1), (0, ())
        for rank, (record, start) in enumerate(anchors):
            endpoints = tuple(itertools.accumulate((0,)+state.widths[record]))
            if start not in endpoints or start+2 not in endpoints:
                yield state, F(1, len(anchors)), (rank, ())
                continue
            first, last = endpoints.index(start), endpoints.index(start+2)
            widths = (1, 1) if last-first == 1 else (2,)
            for labels in itertools.product(range(env.rows), repeat=len(widths)):
                yield replacement(record, first, last, widths, labels), F(1, len(anchors)*env.rows**len(widths)), (rank, labels)


@pytest.mark.parametrize('records', (((0, 1),), ((0,), (1,)), ((0, 1, 0),)))
def test_complete_shape_marginals_proposal_flux_and_exchanges(records):
    # R3 contextual with nonzero root; no original source or stochastic controller.
    env, expert = ReadingEnvironment(records, rows=3, glyphs=2), source(3, True, root=2)
    ordered, weights = tuple(direct_shapes(env)), summed_keys(env, expert)
    assert set(ordered) == set(weights[F(1)])
    for epsilon in EPSILONS:
        assert {s: shape.target(expert, env, s, epsilon) for s in ordered} == weights[epsilon]
    proposal = {}
    for kernel in ('relabel', 'boundary'):
        proposal[kernel] = {}
        for state in ordered:
            law = defaultdict(F)
            for candidate, q, args in independent_components(env, state, kernel):
                actual = getattr(shape, kernel)(env, state, *args)
                assert actual.state == candidate and actual.forward == q
                law[candidate] += q
            assert sum(law.values()) == 1
            proposal[kernel][state] = dict(law)
        for state in ordered:
            for candidate, q, args in independent_components(env, state, kernel):
                actual = getattr(shape, kernel)(env, state, *args)
                if candidate != state:
                    assert actual.reverse == proposal[kernel][candidate][state]
        for epsilon in EPSILONS:
            incoming = dict.fromkeys(ordered, F(0))
            for state in ordered:
                p = weights[epsilon][state]
                if not p:
                    continue
                stay = F(1)
                for candidate, q in proposal[kernel][state].items():
                    if candidate == state:
                        continue
                    qr, other = proposal[kernel][candidate][state], weights[epsilon][candidate]
                    moved = q*min(F(1), other*qr/(p*q))
                    incoming[candidate] += p*moved
                    stay -= moved
                    if other:
                        assert p*moved == other*qr*min(F(1), p*q/(other*qr))
                incoming[state] += p*stay
            assert incoming == weights[epsilon]
    for left, right in zip(EPSILONS, EPSILONS[1:]):
        for x in ordered:
            if not weights[left][x]:
                continue
            for y in ordered:
                old = weights[left][x]*weights[right][y]
                new = weights[left][y]*weights[right][x]
                assert shape.swap_ratio(env, x, y, left, right) == new/old
    for state in ordered:
        if weights[F(0)][state]:
            assert shape.from_hard(env, shape.to_hard(env, state)) == state


def test_original_shape_obstruction_has_collapsed_warm_escape():
    env, initial, _ = obstruction()
    state = shape.from_hard(env, initial)
    changed = shape.relabel(env, state, 0, 0, 1).state
    assert shape.channel(env, changed, F(0)) == 0
    assert shape.channel(env, changed, F(1, 8)) > 0
    assert shape.channel(env, state, F(0)) == F(1, 42**23)
    with pytest.raises(ValueError):
        shape.to_hard(env, changed)


def test_shape_forged_inputs_invalid_boundary_and_work_guards(monkeypatch):
    env = ReadingEnvironment(((0, 1, 0),), rows=3, glyphs=2)
    state = shape.ReadingShape(((2, 1),), ((0, 1),))
    assert shape.boundary(env, state, 1, ()).state == state
    with pytest.raises(ValueError):
        shape.boundary(env, state, 1, (0,))
    for invalid in (shape.ReadingShape((('bad', 1),), ((0, 1),)),
                    shape.ReadingShape(((2, 1),), ((False, 1),))):
        with pytest.raises(ValueError):
            shape.validate(env, invalid)
    with pytest.raises(ValueError):
        shape.channel(env, state, .125)
    with pytest.raises(ValueError):
        shape.relabel(env, state, True, 0, 0)
    monkeypatch.setattr(shape, 'MAX_PROFILE_WORK', 1)
    with pytest.raises(MemoryError):
        shape.channel(env, state, F(1, 8))
