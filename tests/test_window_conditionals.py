"""Exact local full-target cancellation, categorical boundaries and cold law."""
from collections import defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_reading_pair_rewrite001 import direct_target, enumerate_complete, source_fixture, direct_actions
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.reading_window import observed_window, window_choices
from voynich.source_action_proposal import ReadingEnvironment
from voynich.window_conditionals import (cold_window_heatbath, integer_weights,
    rational_categorical, window_ratio)


def lattice_quantile(weights, rng, maximum_blocks=16):
    """Separate cumulative rational thresholds on the dyadic uniform lattice."""
    total, prefix = sum(weights), 0
    intervals, before = [], F(0)
    for w in weights:
        intervals.append((before/total, (before+w)/total))
        before += w
    for blocks in range(1, maximum_blocks+1):
        prefix = prefix*2**64+int(rng.bit_generator.random_raw())
        scale = 2**(64*blocks)
        for i, (low, high) in enumerate(intervals):
            lower = -(-low.numerator*scale//low.denominator)
            upper = high.numerator*scale//high.denominator
            if lower <= prefix and prefix+1 <= upper:
                return i, blocks
    raise RuntimeError('Independent rational threshold undecided')


def test_every_artificial_conditional_ratio_future_context_and_cold_heatbath_flux():
    env = ReadingEnvironment(((0, 0, 1), (0, 1)), rows=3, glyphs=2)
    source = source_fixture(3, True)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    states = enumerate_complete(env)
    target = {s: direct_target(source, env, s, sampler.config.stop) for s in states}
    windows = [observed_window(env, rank) for rank in range(sum(2*len(r)-1 for r in env.records))]
    incoming = defaultdict(F)
    touched_future = cancelled_future = 0
    for state in states:
        path = sampler.path(forced_actions=direct_actions(env, state))
        for window in windows:
            family = window_choices(env, state, *window)
            if family is None:
                incoming[state] += target[state]/len(windows)
                continue
            ratios = [window_ratio(sampler, path, *window, x) for x in family.replacements]
            total = sum(v.ratio for v in ratios)
            for value in ratios:
                assert value.ratio == target[value.state]/target[state]
                assert value.omitted_suffix_steps == 0 or value.contexts_matched
                touched_future += value.suffix_steps > 0
                cancelled_future += value.omitted_suffix_steps > 0
                incoming[value.state] += target[state]*value.ratio/total/len(windows)
    assert dict(incoming) == target and touched_future and cancelled_future


def test_rational_categorical_independent_rng_thresholds_and_multiblock_abort(monkeypatch):
    for weights in ((F(1),), (F(1, 3), F(2, 3)), (F(1, 2**160), F(1), F(23, 19)),
                    (F(3, 7), F(8, 9), F(1, 6), F(5, 11))):
        a, b = np.random.default_rng(96341), np.random.default_rng(96341)
        for _ in range(256):
            assert rational_categorical(weights, a) == lattice_quantile(weights, b)
            assert a.bit_generator.state == b.bit_generator.state
    v = int(np.random.default_rng(96343).bit_generator.random_raw())
    boundary = F(2*v+1, 2**65)
    weights = (boundary, 1-boundary)
    a, b = np.random.default_rng(96343), np.random.default_rng(96343)
    assert rational_categorical(weights, a) == lattice_quantile(weights, b)
    assert a.bit_generator.state == b.bit_generator.state
    assert rational_categorical(weights, np.random.default_rng(96343))[1] == 2
    with pytest.raises(RuntimeError):
        rational_categorical(weights, np.random.default_rng(96343), maximum_blocks=1)
    assert integer_weights((F(3, 14), F(9, 14))) == (1, 3)
    for weights in ((F(0),), (F(-1),), (1,), (), (F(1),)*4161):
        with pytest.raises(ValueError):
            integer_weights(weights)
    with pytest.raises(MemoryError):
        integer_weights((F(2**20),), maximum_bits=16)
    with pytest.raises(MemoryError):
        integer_weights((F(1, 251), F(1, 241)), maximum_bits=8)
    with pytest.raises(MemoryError):
        rational_categorical((F(1),), np.random.default_rng(1), maximum_bits=16)
    monkeypatch.setattr('voynich.window_conditionals.MAX_TOTAL_BITS', 48)
    with pytest.raises(MemoryError, match='aggregate aligned'):
        integer_weights((F(1, 251), F(1, 241), F(1, 239), F(1, 233)))
    with pytest.raises(MemoryError, match='aggregate input'):
        integer_weights((F(2**30), F(2**30)))


def test_seeded_heatbath_selects_exact_conditional_and_replays_reference():
    env = ReadingEnvironment(((0, 0, 1), (0, 1)), rows=3, glyphs=2)
    source = source_fixture(3, True)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    path = sampler.path(forced_actions=direct_actions(env, enumerate_complete(env)[0]))
    a, b = np.random.default_rng(96347), np.random.default_rng(96347)
    for _ in range(128):
        retained, info = cold_window_heatbath(sampler, path, a)
        rank = int(b.integers(sum(2*len(r)-1 for r in env.records)))
        window = observed_window(env, rank)
        assert info['rank'] == rank
        family = window_choices(env, path.state, *window)
        if family is not None:
            values = [window_ratio(sampler, path, *window, x) for x in family.replacements]
            # Use independent full per-record target values, not incremental ratios.
            weights = tuple(direct_target(source, env, v.state, sampler.config.stop) for v in values)
            index, blocks = lattice_quantile(weights, b)
            assert retained.state == values[index].state and info['raw64_blocks'] == blocks
        else:
            assert retained == path
        assert retained == sampler.path(forced_actions=retained.actions)
        assert a.bit_generator.state == b.bit_generator.state
        path = retained
