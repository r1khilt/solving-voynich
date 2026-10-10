"""Preparation on complete artificial spaces; no original-source or Gold data."""
from collections import defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_reading_pair_rewrite001 import direct_target, enumerate_complete, source_fixture
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.reading_window import observed_window, replace_window, window_choices, window_step
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def direct_window_family(env, state, record, start, width, states):
    """Filter an independently enumerated complete space by outside observed intervals.

    No production candidate construction or released-binding bookkeeping is used.
    """
    def outside(value):
        positions, endpoints = [], set()
        for r, text in enumerate(value.texts):
            offset = 0
            for row in text:
                end = offset+len(env.pool[value.key[row]])
                if r == record:
                    endpoints.update((offset, end))
                if r != record or end <= start or offset >= start+width:
                    positions.append((r, offset, end, row, value.key[row]))
                offset = end
        return tuple(positions), endpoints
    original, ends = outside(state)
    if start not in ends or start+width not in ends:
        return set()
    return {other for other in states if outside(other)[0] == original
            and start in outside(other)[1] and start+width in outside(other)[1]}


@pytest.mark.parametrize('records', [((0, 1), (0, 1)), ((0, 0), (0, 0)), ((0, 0, 1),)])
@pytest.mark.parametrize('contextual', [False, True])
def test_all_conditional_families_reverse_support_cold_flux_and_stationarity(records, contextual):
    env = ReadingEnvironment(records, rows=3, glyphs=2)
    source, stop = source_fixture(3, contextual), F(1, 3)
    states = enumerate_complete(env)
    targets = {state: direct_target(source, env, state, stop) for state in states}
    windows = [(r, start, width) for r, obs in enumerate(records) for start in range(len(obs))
               for width in (1, 2) if start+width <= len(obs)]
    assert [observed_window(env, i) for i in range(len(windows))] == windows
    incoming = defaultdict(F)
    split = merge = changed_inventory = failed_boundaries = 0
    for state in states:
        total = F(0)
        for r, start, width in windows:
            family = window_choices(env, state, r, start, width)
            oracle = direct_window_family(env, state, r, start, width, states)
            if family is None:
                assert not oracle
                incoming[state] += targets[state]/len(windows)
                total += F(1, len(windows))
                failed_boundaries += 1
                continue
            choices = {replace_window(env, state, r, start, width, x): x for x in family.replacements}
            assert set(choices) == oracle and len(choices) == len(family.replacements)
            q = F(1, len(windows)*len(choices))
            old = state.texts[r][family.first:family.last]
            for other, replacement in choices.items():
                reverse = window_choices(env, other, r, start, width)
                assert reverse.replacements == family.replacements and reverse.outside_key == family.outside_key
                assert replace_window(env, other, r, start, width, old) == state
                accept = min(F(1), targets[other]/targets[state])
                back = min(F(1), targets[state]/targets[other])
                assert targets[state]*q*accept == targets[other]*q*back
                incoming[other] += targets[state]*q*accept
                incoming[state] += targets[state]*q*(1-accept)
                total += q
                difference = sum(map(len, other.texts))-sum(map(len, state.texts))
                split += difference == 1
                merge += difference == -1
                changed_inventory += sum(k >= 0 for k in state.key) != sum(k >= 0 for k in other.key)
        assert total == 1
    assert dict(incoming) == targets
    assert split and merge and changed_inventory and failed_boundaries


def test_repeated_row_split_preexisting_pairs_and_one_glyph_degeneracy():
    env = ReadingEnvironment(((0, 0), (0, 0)), rows=3, glyphs=2)
    state = ReadingState((0, -1, -1), (2, 2), ((0, 0), (0, 0)))
    merged = replace_window(env, state, 0, 0, 2, (1,))
    assert merged == ReadingState((0, 2, -1), (2, 2), ((1,), (0, 0)))
    assert replace_window(env, merged, 0, 0, 2, (0, 0)) == state
    assert window_choices(env, merged, 0, 1, 1) is None
    # A row used only in the old window may receive a different unit there.
    replaced = replace_window(env, merged, 0, 0, 2, (1, 1))
    assert replaced.key == (0, 0, -1)
    short = ReadingEnvironment(((0,),), rows=3, glyphs=2)
    one = ReadingState((0, -1, -1), (1,), ((0,),))
    assert observed_window(short, 0) == (0, 0, 1)
    assert window_choices(short, one, 0, 0, 1).replacements == ((0,), (1,), (2,))


def test_seeded_warm_reference_replay_and_guards():
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 0), (0, 0)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    path = sampler.path(forced_actions=(0, 0, 0, 0))
    a, b = np.random.default_rng(96311), np.random.default_rng(96311)
    seen, lengths = set(), set()
    for index in range(160):
        first = window_step(sampler, path, a, (1, 2, 4, 16)[index % 4])
        second = window_step(sampler, path, b, (1, 2, 4, 16)[index % 4])
        assert first == second and a.bit_generator.state == b.bit_generator.state
        path, candidate, ratio, info = first
        assert path == sampler.path(forced_actions=path.actions)
        seen.add(path.state)
        lengths.add(len(path.actions))
        if ratio is not None:
            assert F(candidate.numerator, candidate.denominator) == direct_target(source, env, candidate.state, sampler.config.stop)
            assert info['forward_component_probability'] == info['reverse_component_probability']
    assert len(seen) > 1 and len(lengths) > 1
    for rank in (-1, False, 6):
        with pytest.raises(ValueError):
            observed_window(env, rank)
    for r, start, width, replacement in ((False, 0, 1, (0,)), (0, -1, 1, (0,)),
                                        (0, 0, False, (0,)), (0, 0, 2, (False,))):
        with pytest.raises(ValueError):
            replace_window(env, path.state, r, start, width, replacement)
