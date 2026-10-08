"""Structural boundary controls; artificial data only, before scientific use."""
import hashlib
import itertools
import json
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from scripts import check_reading_pair_rewrite001 as run
from voynich.reading_pair_rewrite import (eligible_triplets, pair_rewrite_candidate, pair_rewrite_ratio,
    pair_rewrite_step, reading_actions, rewrite_pair, triplet_from_rank, valid_pair_rewrite_step)
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


@pytest.mark.parametrize('rows', (3, 4, 23, 64))
def test_uniform_triplet_bijection(rows):
    size = rows*(rows-1)*(rows-2)
    assert {triplet_from_rank(rows, r) for r in range(size)} == set(itertools.permutations(range(rows), 3))
    for rank in (-1, size, False):
        with pytest.raises(ValueError):
            triplet_from_rank(rows, rank)


def fixture():
    env = ReadingEnvironment(((0, 1, 0, 1), (0, 1)), rows=4, glyphs=2)
    sampler = SourceRegrowth(run.source_fixture(4, True), env)
    state = ReadingState((-1, env.pool.index((0,)), env.pool.index((1,)), -1),
                         (4, 2), ((1, 2, 1, 2), (1, 2)))
    return sampler, sampler.path(forced_actions=reading_actions(env, state))


def test_global_repeated_merge_release_and_full_reference_replay():
    sampler, old = fixture()
    candidate, change = pair_rewrite_candidate(sampler, old, 0, 1, 2)
    assert change.branch == 'merge' and change.replacements == 3
    assert candidate.state.texts == ((0, 0), (0,))
    assert candidate.state.key == (sampler.env.pool.index((0, 1)), -1, -1, -1)
    assert candidate.actions == (1, 1, 1)
    back, reverse = pair_rewrite_candidate(sampler, candidate, 0, 1, 2)
    assert reverse.branch == 'split' and back == old
    ratio = F(*pair_rewrite_ratio(sampler, old, candidate))
    source = run.source_fixture(4, True)
    assert ratio == run.direct_target(source, sampler.env, candidate.state, sampler.config.stop) / run.direct_target(
        source, sampler.env, old.state, sampler.config.stop)
    assert ratio != F(1)
    assert candidate.state.offsets == old.state.offsets


def test_merge_retains_single_rows_used_elsewhere_and_split_restores():
    env = ReadingEnvironment(((0, 1, 0), (1,)), rows=3, glyphs=2)
    state = ReadingState((-1, 0, 1), (3, 1), ((1, 2, 1), (2,)))
    change = rewrite_pair(env, state, 0, 1, 2)
    assert change.branch == 'merge' and change.state.key[1:] == (0, 1)
    assert rewrite_pair(env, change.state, 0, 1, 2).state == state


def test_existing_pair_rejection_prevents_noninvolution():
    env = ReadingEnvironment(((0, 1, 0, 1),), rows=3, glyphs=2)
    state = ReadingState((env.pool.index((0, 1)), 0, 1), (4,), ((0, 1, 2),))
    assert rewrite_pair(env, state, 0, 1, 2).branch == 'identity'
    unsafe, branch, _ = run.direct_rewrite(env, state, (0, 1, 2), unsafe_split=True)
    assert branch == 'split' and run.direct_rewrite(env, unsafe, (0, 1, 2))[0] != state


def test_record_boundary_is_not_an_adjacent_pair():
    env = ReadingEnvironment(((0,), (1,)), rows=3, glyphs=2)
    state = ReadingState((-1, 0, 1), (1, 1), ((1,), (2,)))
    assert rewrite_pair(env, state, 0, 1, 2).branch == 'identity'


def test_same_glyph_duplicate_codes_still_use_distinct_letters():
    env = ReadingEnvironment(((0, 0, 0, 0),), rows=3, glyphs=2)
    state = ReadingState((-1, 0, 0), (4,), ((1, 2, 1, 2),))
    merged = rewrite_pair(env, state, 0, 1, 2)
    assert merged.replacements == 2 and merged.state.texts == ((0, 0),)
    assert rewrite_pair(env, merged.state, 0, 1, 2).state == state


def test_guard_foreign_incomplete_action_and_rng():
    sampler, old = fixture()
    foreign, other = fixture()
    for bad in (other, replace(old, complete=False), replace(old, actions=old.actions[:-1])):
        with pytest.raises(ValueError):
            pair_rewrite_candidate(sampler, bad, 0, 1, 2)
    for triplet in ((0, 0, 1), (False, 1, 2), (0, 1, 4)):
        with pytest.raises(ValueError):
            rewrite_pair(sampler.env, old.state, *triplet)
    with pytest.raises(ValueError):
        reading_actions(sampler.env, sampler.env.initial)
    with pytest.raises(ValueError):
        pair_rewrite_step(sampler, old, np.random.Generator(np.random.MT19937(1)))
    assert foreign.token is not sampler.token


def test_step_seed_stability_and_no_reference_q_acceptance(monkeypatch):
    import voynich.reading_pair_rewrite as module
    sampler, old = fixture()
    monkeypatch.setattr(module, 'triplet_from_rank', lambda *_: (0, 1, 2))
    expected, _ = pair_rewrite_candidate(sampler, old, 0, 1, 2)
    captured = []

    def accept(n, d, rng, **kwargs):
        captured.append(F(n, d))
        assert F(n, d) == F(*sampler.target_integers(expected))/F(*sampler.target_integers(old))
        return True, 1

    monkeypatch.setattr(module, 'exact_bernoulli', accept)
    first = pair_rewrite_step(sampler, old, np.random.default_rng(95803))
    second = pair_rewrite_step(sampler, old, np.random.default_rng(95803))
    assert first == second and first[0] == expected
    assert not first[2]['reference_policy_probability_used_for_acceptance'] and len(captured) == 2


def test_small_independent_complete_space_and_negative_controls():
    panel = run.finite_panel(3, True, ((0, 1), (0, 1)))
    assert panel['triplet_checks'] == panel['states']*6
    assert panel['merge_edges'] == panel['split_edges'] > 0
    assert panel['omitted_prior_flux_failures'] > 0 and panel['wrong_policy_flux_failures'] > 0


def test_valid_selector_exact_reverse_factor_and_zero_degree(monkeypatch):
    import voynich.reading_pair_rewrite as module
    sampler, path = fixture()
    captured = []

    def accept(n, d, rng, **kwargs):
        captured.append(F(n, d))
        return True, 1

    monkeypatch.setattr(module, 'exact_bernoulli', accept)
    rng = np.random.default_rng(95809)
    for _ in range(30):
        retained, candidate, info = valid_pair_rewrite_step(sampler, path, rng)
        assert retained == candidate and info['branch'] != 'identity'
        dx, dy = len(eligible_triplets(sampler.env, path.state)), len(eligible_triplets(sampler.env, candidate.state))
        assert captured[-1] == F(*sampler.target_integers(candidate))/F(*sampler.target_integers(path))*F(dx, dy)
        assert info['old_degree'] == dx and info['new_degree'] == dy
        assert candidate == sampler.path(forced_actions=candidate.actions)
        path = retained
    env = ReadingEnvironment(((0,), (1,)), rows=3, glyphs=2)
    other = SourceRegrowth(run.source_fixture(3, True), env)
    no_move = other.path(forced_actions=(0, 2))
    assert eligible_triplets(env, no_move.state) == ()
    old_rng = json.dumps(rng.bit_generator.state, sort_keys=True)
    retained, candidate, info = valid_pair_rewrite_step(other, no_move, rng)
    assert retained == candidate == no_move and info['old_degree'] == 0
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == old_rng


def test_controller_receipt_and_exclusive_namespace(tmp_path, monkeypatch):
    from scripts.audit_reading_pair_rewrite001 import audit
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run, 'resource_report', lambda *_: {'wall_seconds': .1, 'cpu_seconds': .1,
                                                         'peak_rss_bytes': 1024, 'paid_spend_usd': 0})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_new(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream)
        return artifact(path)

    def artificial(rows, context, records):
        return dict(rows=rows, contextual=context, records=[list(r) for r in records],
                    states=1, triplets_per_state=rows*(rows-1)*(rows-2), stationarity_equations_per_selector=1,
                    triplet_checks=rows*(rows-1)*(rows-2), merge_edges=1, split_edges=1,
                    identity_edges=rows*(rows-1)*(rows-2)-2, corrected_valid_selection_edges=2)

    monkeypatch.setattr(run, 'artifact', artifact)
    monkeypatch.setattr(run, 'save_new', save_new)
    monkeypatch.setattr(run, 'finite_panel', artificial)
    totals = run.check('artificial-only')
    assert len(totals) == 7 and audit('artificial-only') == 'PASS_receipt_hash_and_arithmetic_closure'
    with pytest.raises(FileExistsError):
        run.check('artificial-only')
