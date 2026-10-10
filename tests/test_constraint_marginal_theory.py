"""Preparatory exact algebra; no production collapsed dispatcher or recovery."""
from collections import defaultdict
from fractions import Fraction as F
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_constraint_reading001 import independent_distance
from tests.test_constraint_reading import states, source
from tests.test_constraint_sampling import full_mass
from voynich import constraint_reading as law
from voynich.source_action_proposal import ReadingEnvironment


def row_polynomials(env, state, epsilon):
    """Key-free product of independently accumulated per-row unit factors."""
    observed = defaultdict(list)
    for record, widths, labels in zip(env.records, state.widths, state.texts, strict=True):
        offset = 0
        for width, row in zip(widths, labels, strict=True):
            observed[row].append(record[offset:offset+width])
            offset += width
    def edits(a, b):
        # Separate length1/2 case formula, not the production dynamic program.
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b, strict=True))
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        return 1 if shorter[0] in longer else 2
    return {row: sum(epsilon**sum(edits(unit, span) for span in spans)
                     for unit in env.pool) for row, spans in observed.items()}


@pytest.mark.parametrize('rows', (3, 4))
@pytest.mark.parametrize('contextual', (False, True))
@pytest.mark.parametrize('records', (((0, 1),), ((0, 0),), ((0,), (1,))))
def test_exact_marginal_and_hard_return_identity(rows, contextual, records):
    probabilities = np.array([.5, .25, .25] if rows == 3 else [.5, .25, .125, .125])
    count = rows if contextual else 1
    expert = SimpleNamespace(
        probabilities=np.array([np.roll(probabilities, i) for i in range(count)]),
        transitions=np.array([[(i+a+1) % count for a in range(rows)] for i in range(count)]),
        state=lambda _: 0)
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    ordered = states(env)
    hard = next(s for s in ordered if independent_distance(env, s) == 0)
    for epsilon in (F(0), F(1, 8), F(1, 2), F(1)):
        groups = defaultdict(F)
        representatives = {}
        for state in ordered:
            shape = (state.widths, state.texts)
            representatives[shape] = state
            groups[shape] += full_mass(expert, env, state, epsilon)
        for shape, state in representatives.items():
            # At epsilon1 each key has the same Q/U^m; exact sum separates by row.
            base = full_mass(expert, env, state, F(1))
            polynomial = row_polynomials(env, state, epsilon)
            assert groups[shape] == base*math.prod(polynomial.values())
            if epsilon == 1:
                assert groups[shape] == base*len(env.pool)**len(polynomial)
            if epsilon == 0:
                assert all(value in (0, 1) for value in polynomial.values())
        if epsilon:
            weights = {s: full_mass(expert, env, s, epsilon) for s in ordered}
            accepted = sum(w*min(F(1), law.swap_ratio(env, hard, s, F(0), epsilon))
                           for s, w in weights.items())
            z0 = sum(full_mass(expert, env, s, F(0)) for s in ordered)
            assert accepted == z0
            assert accepted/sum(weights.values()) == z0/sum(groups.values())


def test_fixed_singleton_sector_noise_is_not_structural_incompatibility():
    env = ReadingEnvironment(((0, 0, 0),), rows=3, glyphs=2)
    expert = source()
    ordered = [s for s in states(env) if s.widths == ((1, 1, 1),)
               and s.texts == ((0, 1, 2),)]
    assert len(ordered) == 6**3
    for epsilon in (F(1, 8), F(1, 2), F(1)):
        total = sum(full_mass(expert, env, s, epsilon) for s in ordered)
        valid = sum(full_mass(expert, env, s, F(0)) for s in ordered)
        assert valid/total == (1+4*epsilon+epsilon**2)**-3
    # These conditional-sector calculations do not describe the full Latin posterior.
    assert (1+4*F(1, 8)+F(1, 8)**2)**-23 < F(1, 10000)
