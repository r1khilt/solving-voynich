"""Artificial preparation: literal reverse support, exact cold flux and RNG paths."""
from collections import defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_reading_pair_rewrite001 import direct_target, enumerate_complete, source_fixture
from voynich.reading_occurrence import occurrence_rows, occurrence_step, replace_occurrence
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def direct_replace(env, state, record, index, row):
    """Rebuild visited bindings from old units at every resulting occurrence."""
    old = state.texts[record][index]
    units = [None if k < 0 else env.pool[k] for k in state.key]
    units[row] = units[old]
    texts = tuple(tuple(row if (r, i) == (record, index) else a for i, a in enumerate(text))
                  for r, text in enumerate(state.texts))
    key = [-1]*env.rows
    for text, observation in zip(texts, env.records, strict=True):
        offset = 0
        for a in text:
            unit = units[a]
            assert observation[offset:offset+len(unit)] == unit
            assert key[a] in (-1, env.pool.index(unit))
            key[a] = env.pool.index(unit)
            offset += len(unit)
        assert offset == len(observation)
    return ReadingState(tuple(key), state.offsets, texts)


@pytest.mark.parametrize('records', [((0, 1), (0, 1)), ((0, 0), (0, 0)), ((0, 0, 1),)])
@pytest.mark.parametrize('contextual', [False, True])
def test_every_artificial_state_selector_reverse_and_complete_cold_stationarity(records, contextual):
    env = ReadingEnvironment(records, rows=3, glyphs=2)
    source, stop = source_fixture(3, contextual), F(1, 3)
    states = enumerate_complete(env)
    masses = {state: direct_target(source, env, state, stop) for state in states}
    incoming = defaultdict(F)
    birth = death = equality_change = 0
    for state in states:
        total = sum(map(len, state.texts))
        outgoing = F(0)
        for r, text in enumerate(state.texts):
            for i, old in enumerate(text):
                candidates = tuple(a for a, k in enumerate(state.key) if k < 0 or k == state.key[old])
                assert occurrence_rows(env, state, r, i) == candidates
                q = F(1, total*len(candidates))
                for row in candidates:
                    change = replace_occurrence(env, state, r, i, row)
                    other = direct_replace(env, state, r, i, row)
                    assert change.state == other and other in masses
                    assert replace_occurrence(env, other, r, i, old).state == state
                    assert occurrence_rows(env, other, r, i) == candidates
                    assert tuple(map(len, other.texts)) == tuple(map(len, state.texts))
                    birth += change.visited_delta > 0
                    death += change.visited_delta < 0
                    equality_change += row != old and sum(t.count(old) for t in state.texts) > 1
                    acceptance = min(F(1), masses[other]/masses[state])
                    reverse = min(F(1), masses[state]/masses[other])
                    assert masses[state]*q*acceptance == masses[other]*q*reverse
                    outgoing += q
                    incoming[other] += masses[state]*q*acceptance
                    incoming[state] += masses[state]*q*(1-acceptance)
        assert outgoing == 1
    assert dict(incoming) == masses
    assert birth and death
    if sum(map(len, records)) >= 4:
        assert equality_change


def test_birth_death_cross_record_equality_and_identity_witnesses():
    env = ReadingEnvironment(((0, 0), (0,)), rows=3, glyphs=2)
    state = ReadingState((0, -1, -1), (2, 1), ((0, 0), (0,)))
    changed = replace_occurrence(env, state, 0, 1, 1)
    assert changed.state.texts == ((0, 1), (0,)) and changed.state.key == (0, 0, -1)
    assert changed.visited_delta == 1
    reverse = replace_occurrence(env, changed.state, 0, 1, 0)
    assert reverse.state == state and reverse.visited_delta == -1
    assert replace_occurrence(env, state, 0, 0, 0).state is state
    # Every row bound to a different unit: this move alone cannot change a key.
    full = ReadingEnvironment(((0, 1), (0, 0)), rows=3, glyphs=2)
    frozen = ReadingState((0, 1, 2), (2, 2), ((0, 1), (2,)))
    assert all(occurrence_rows(full, frozen, r, i) == (row,)
               for r, text in enumerate(frozen.texts) for i, row in enumerate(text))


def test_seeded_warm_paths_replay_reference_and_guards():
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 0), (0, 0)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    path = sampler.path(forced_actions=(0, 0, 0, 0))
    a, b = np.random.default_rng(96301), np.random.default_rng(96301)
    observed = set()
    for index in range(80):
        first = occurrence_step(sampler, path, a, (1, 2, 4, 16)[index % 4])
        second = occurrence_step(sampler, path, b, (1, 2, 4, 16)[index % 4])
        assert first == second and a.bit_generator.state == b.bit_generator.state
        path, candidate, ratio, info = first
        assert path == sampler.path(forced_actions=path.actions)
        assert info['forward_component_probability'] == info['reverse_component_probability']
        observed.add(path.state)
        if ratio is not None:
            target = direct_target(source, env, candidate.state, sampler.config.stop)
            assert F(candidate.numerator, candidate.denominator) == target
    assert len(observed) > 1
    for degree in (False, 0, 257):
        with pytest.raises(ValueError):
            occurrence_step(sampler, path, a, degree)
    for record, index, row in ((False, 0, 0), (0, -1, 0), (0, 0, False), (9, 0, 0)):
        with pytest.raises(ValueError):
            replace_occurrence(env, path.state, record, index, row)
