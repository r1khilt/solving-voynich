"""Artificial trajectory and exact score/proposal controls before scientific use."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_reading_pair_rewrite001 import source_fixture, enumerate_complete, direct_target
from voynich.reading_label_transport import reference_reading
from voynich.reading_pair_rewrite import reading_actions
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment
from voynich.tempered_reading import (exchange, local_step, path_target, point_target,
    replica_sweep, score_complete_state, suffix_correction)


def fixture():
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 1), (0, 1)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    return source, sampler


def test_fast_complete_score_matches_full_reference_at_every_short_state():
    source, sampler = fixture()
    for state in enumerate_complete(sampler.env):
        point = score_complete_state(sampler, state)
        path = sampler.path(forced_actions=reading_actions(sampler.env, state))
        assert reference_reading(sampler, point) == path
        assert point_target(point) == direct_target(source, sampler.env, state, sampler.config.stop)


def test_untempered_regrowth_bit_identical_and_suffix_law():
    _, sampler = fixture()
    path = sampler.path(forced_actions=(1, 1))
    a, b = np.random.default_rng(96061), np.random.default_rng(96061)
    for _ in range(50):
        old = path
        path, candidate, ratio, info = local_step(sampler, old, a, 1, kernel='regrowth')
        retained, other, legacy_info = sampler.step(old, b)
        assert retained == path and other == candidate
        assert {k: info[k] for k in ('cut', 'accepted', 'failed', 'acceptance_blocks')} == legacy_info
        assert a.bit_generator.state == b.bit_generator.state
        if candidate.complete:
            h = suffix_correction(sampler, old, candidate, info['cut'])
            expected = path_target(sampler, candidate)/path_target(sampler, old)*h
            assert F(ratio.numerator, ratio.denominator) == F(*sampler.ratio(old, candidate, info['cut'])) == expected


def test_replica_seed_identity_literal_and_postacceptance_reference():
    _, sampler = fixture()
    old = sampler.path(forced_actions=(1, 1))
    one = two = (old,)*4
    a, b = np.random.default_rng(96071), np.random.default_rng(96071)
    changed, kernels = set(), set()
    for sweep in range(80):
        one, locals_, swaps = replica_sweep(sampler, one, (1, 2, 4, 16), a, sweep)
        two, other, other_swaps = replica_sweep(sampler, two, (1, 2, 4, 16), b, sweep)
        assert one == two and locals_ == other and swaps == other_swaps
        assert a.bit_generator.state == b.bit_generator.state
        for path in one:
            assert path == sampler.path(forced_actions=path.actions)
            changed.add(path.state)
        for _, _, info in locals_:
            kernels.add(info['kernel'])
    assert len(changed) > 1 and kernels == {'regrowth', 'pair', 'label'}


def test_incomplete_foreign_ladder_and_kernel_guards():
    _, sampler = fixture()
    _, other = fixture()
    old = sampler.path(forced_actions=(1, 1))
    foreign = other.path(forced_actions=(1, 1))
    rng = np.random.default_rng(1)
    for path in (foreign, replace(old, complete=False)):
        with pytest.raises(ValueError):
            local_step(sampler, path, rng, 2, kernel='pair')
    for degree in (0, False, 257):
        with pytest.raises(ValueError):
            local_step(sampler, old, rng, degree, kernel='pair')
    with pytest.raises(ValueError):
        replica_sweep(sampler, (old, old), (2, 4), rng, 0)
    with pytest.raises(ValueError):
        local_step(sampler, old, rng, 2, kernel='unknown')
    with pytest.raises(ValueError):
        exchange(sampler, old, foreign, 1, 4, rng)
