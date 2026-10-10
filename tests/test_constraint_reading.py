"""Artificial preparation: soft states, hard bijection, reverse probabilities and support."""
from dataclasses import replace
from fractions import Fraction as F
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_reading_label_landscape001 import independent_point
from tests.test_window_support_obstruction import obstruction
from voynich import constraint_reading as c
from voynich.source_action_proposal import ReadingEnvironment


def states(env):
    def partitions(length):
        if not length:
            yield ()
        for width in (1, 2):
            if width <= length:
                for suffix in partitions(length-width):
                    yield (width,)+suffix
    answer = set()
    for widths in itertools.product(*(tuple(partitions(len(r))) for r in env.records)):
        for labels in itertools.product(range(env.rows), repeat=sum(map(len, widths))):
            texts, offset = [], 0
            for spans in widths:
                texts.append(labels[offset:offset+len(spans)])
                offset += len(spans)
            used = sorted(set(labels))
            for codes in itertools.product(range(len(env.pool)), repeat=len(used)):
                key = [-1]*env.rows
                for row, unit in zip(used, codes, strict=True):
                    key[row] = unit
                value = c.ConstraintReading(tuple(key), widths, tuple(texts))
                c.validate(env, value)
                answer.add(value)
    return answer


def fresh_bindings(env, state, record, first, last, rows):
    outside = set(a for i, text in enumerate(state.texts) for j, a in enumerate(text)
                  if i != record or not first <= j < last)
    fresh = sorted(set(rows)-outside)
    for codes in itertools.product(range(len(env.pool)), repeat=len(fresh)):
        yield dict(zip(fresh, codes, strict=True))


def components(env, state, kernel):
    if kernel == 'rebind':
        for row, unit in itertools.product(range(env.rows), range(len(env.pool))):
            yield c.rebind(env, state, row, unit)
    elif kernel == 'relabel':
        for r, text in enumerate(state.texts):
            for j in range(len(text)):
                for row in range(env.rows):
                    for bindings in fresh_bindings(env, state, r, j, j+1, (row,)):
                        yield c.relabel(env, state, r, j, row, bindings)
    else:
        anchors = [(r, i) for r, obs in enumerate(env.records) for i in range(len(obs)-1)]
        if not anchors:
            yield c.boundary(env, state, 0, (), {})
        for rank, (r, start) in enumerate(anchors):
            boundaries, offset = {0: 0}, 0
            for j, width in enumerate(state.widths[r]):
                offset += width
                boundaries[offset] = j+1
            if start not in boundaries or start+2 not in boundaries:
                yield c.boundary(env, state, rank, (), {})
                continue
            first, last = boundaries[start], boundaries[start+2]
            new_size = 2 if last-first == 1 else 1
            for rows in itertools.product(range(env.rows), repeat=new_size):
                for bindings in fresh_bindings(env, state, r, first, last, rows):
                    yield c.boundary(env, state, rank, rows, bindings)


def source():
    p = np.array([[.25, .5, .25], [.5, .25, .25]], dtype=np.float64)
    t = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    return SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)


@pytest.mark.parametrize('records', [((0, 1),), ((0,), (1,)), ((0,), (0,))])
def test_every_small_state_hard_bijection_normalized_components_reverse_and_stationarity(records):
    env = ReadingEnvironment(records, rows=3, glyphs=2)
    all_states, expert = states(env), source()
    hard = [s for s in all_states if c.violations(env, s) == 0]
    assert len(all_states) in (234, 252) and hard
    for s in hard:
        original = c.to_hard(env, s)
        assert c.from_hard(env, original) == s
        assert c.target(expert, env, s, F(0)) == F(*independent_point(expert, env, original, lambda: None))
    proposals = {}
    for kernel in ('rebind', 'relabel', 'boundary'):
        proposals[kernel] = {}
        for s in all_states:
            moves = list(components(env, s, kernel))
            assert sum(m.forward for m in moves) == 1
            q = {}
            for move in moves:
                assert move.state in all_states
                q[move.state] = q.get(move.state, F(0))+move.forward
            proposals[kernel][s] = (q, moves)
        for s, (_, moves) in proposals[kernel].items():
            for move in moves:
                if move.state != s:
                    assert move.reverse == proposals[kernel][move.state][0][s]
    # Full mixture graph, not merely a positive score or a single path.
    reached, frontier = {next(iter(all_states))}, list(all_states)[:1]
    while frontier:
        s = frontier.pop()
        for kernel in proposals.values():
            for t in kernel[s][0]:
                if t not in reached:
                    reached.add(t)
                    frontier.append(t)
    assert reached == all_states
    for epsilon in (F(0), F(1, 8), F(1, 2), F(1)):
        targets = {s: c.target(expert, env, s, epsilon) for s in all_states}
        positive = {s for s, value in targets.items() if value > 0}
        for kernel in proposals.values():
            incoming = dict.fromkeys(positive, F(0))
            for s in positive:
                q = kernel[s][0]
                stay = F(1)
                for t, forward in q.items():
                    if t == s or targets[t] == 0:
                        continue
                    reverse = kernel[t][0][s]
                    moved = forward*min(F(1), targets[t]*reverse/(targets[s]*forward))
                    backward = reverse*min(F(1), targets[s]*forward/(targets[t]*reverse))
                    assert targets[s]*moved == targets[t]*backward
                    incoming[t] += targets[s]*moved
                    stay -= moved
                assert 0 <= stay <= 1
                incoming[s] += targets[s]*stay
            assert incoming == {s: targets[s] for s in positive}


def test_hard_soft_exchange_all_pairs_matches_full_joint_target():
    env = ReadingEnvironment(((0, 1),), rows=3, glyphs=2)
    all_states, expert = states(env), source()
    for ex, ey in ((F(0), F(1, 8)), (F(1, 8), F(1, 2))):
        tx = {s: c.target(expert, env, s, ex) for s in all_states}
        ty = {s: c.target(expert, env, s, ey) for s in all_states}
        for x, y in itertools.product(all_states, repeat=2):
            if tx[x] and ty[y]:
                actual = c.swap_ratio(env, x, y, ex, ey)
                assert actual == tx[y]*ty[x]/(tx[x]*ty[y])
                old, new = tx[x]*ty[y], tx[y]*ty[x]
                if new:
                    reverse = c.swap_ratio(env, y, x, ex, ey)
                    assert old*min(F(1), actual) == new*min(F(1), reverse)
                else:
                    assert actual == 0


def test_locked_original_shape_has_positive_soft_escape_and_strict_hard_rejection():
    env, locked, _ = obstruction()
    s = c.from_hard(env, locked)
    move = c.rebind(env, s, 0, 0)
    assert move.state != s and c.violations(env, move.state) > 0
    p = np.ones((1, 23), dtype=np.float64)/23
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    expert = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    assert c.target(expert, env, move.state, F(1, 8)) > 0
    assert c.target(expert, env, move.state, F(0)) == 0
    assert c.swap_ratio(env, s, move.state, F(0), F(1, 8)) == 0
    with pytest.raises(ValueError):
        c.to_hard(env, move.state)


def test_forged_bindings_parameters_and_work_caps_rejected(monkeypatch):
    env = ReadingEnvironment(((0, 1),), rows=3, glyphs=2)
    s = next(iter(states(env)))
    for bad in (replace(s, key=(True,)*3), replace(s, widths=((3,),)), replace(s, key=(-1,)*3)):
        if bad != s:
            with pytest.raises(ValueError):
                c.validate(env, bad)
    with pytest.raises(ValueError):
        c.target(source(), env, s, .5)
    with pytest.raises(ValueError):
        c.relabel(env, s, 0, 0, 0, {True: 0})
    monkeypatch.setattr(c, 'MAX_BITS', 8)
    with pytest.raises(MemoryError):
        c.target(source(), env, s, F(1, 8))
