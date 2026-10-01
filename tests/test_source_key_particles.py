"""Independent small-law enumeration, ancestry/prior replay and failure controls."""
import itertools
import math
from collections import defaultdict

import numpy as np
import pytest

from scripts.benchmark_source_prefix_systems001 import full_key_reference
from voynich.source_key_particles import (
    advance, categorical, coefficients, reconstruct, run_particles, select_records, terminal_groups,
)


def tiny_source():
    return np.array([[1/3, 2/3], [3/4, 1/4], [1/5, 4/5]], dtype=np.float64), np.tile([1, 2], (3, 1))


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
def test_exact_production_action_law_matches_full_key_reference(schedule):
    p, t = tiny_source()
    observations = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
    for cipher in itertools.product(observations, repeat=2):
        lengths = np.array(list(map(len, cipher)), dtype=np.int64)
        obs = np.full((2, 2), -1, dtype=np.int32)
        for i, c in enumerate(cipher):
            obs[i, :len(c)] = c
        found, proposal_total = defaultdict(float), [0.]

        def visit(keys, offsets, contexts, closed, texts, mass, proposal, importance, steps):
            assert steps <= sum(lengths)+2
            if closed.all():
                found[texts] += mass
                proposal_total[0] += proposal
                assert abs(mass-proposal*importance) < 1e-14
                return
            values, rec, singles, doubles = coefficients(keys, offsets, contexts, closed,
                obs, lengths, p, 2, .25, schedule)
            total = values.sum()
            if total == 0:
                proposal_total[0] += proposal
            for action in np.flatnonzero(values[0]):
                a = values[0, action]
                changed = advance(keys, offsets, contexts, closed, rec, singles, doubles,
                    np.array([0], dtype=np.int32), np.array([action], dtype=np.int32), t)
                new_texts = list(texts)
                if action < 4:
                    new_texts[int(rec[0])] += (int(action)//2,)
                visit(*changed[:4], tuple(new_texts), mass*a, proposal*a/total,
                    importance*total, steps+1)

        visit(np.full((1, 2), -1, dtype=np.int32), np.zeros((1, 2), dtype=np.int32),
            np.zeros((1, 2), dtype=np.uint32), np.zeros((1, 2), dtype=bool), ((), ()), 1., 1., 1., 0)
        reference = full_key_reference(cipher)
        assert found.keys() == reference.keys()
        assert all(abs(found[k]-float(reference[k])) < 1e-13 for k in found)
        assert abs(proposal_total[0]-1) < 1e-13


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
def test_single_possible_reading_exact_evidence_and_ancestry(schedule):
    p, t = np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint32)
    result = run_particles(((0,), (0,)), p, t, glyphs=1, rho=.25,
        particles=32, seed=3, schedule=schedule)
    assert result["summary"]["status"] == "complete_particles"
    assert abs(math.exp(result["summary"]["log_evidence_estimate"])-9/512) < 1e-14
    for i in range(32):
        assert reconstruct(result, i, 2) == ((0,), (0,))
        assert result["keys"][i].tolist() == [0]
        assert abs(result["scores"][i]-math.log(9/512)) < 1e-13


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
def test_two_records_requiring_different_key_extinct_without_refill(schedule):
    p, t = np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint32)
    result = run_particles(((0,), (1,)), p, t, glyphs=2, rho=.25,
        particles=32, seed=5, schedule=schedule)
    assert result["summary"]["status"] == "extinct"
    assert result["summary"]["log_evidence_estimate"] is None
    assert result["trace"][-1]["positive_parents"] == 0
    with pytest.raises(ValueError, match="Extinct"):
        reconstruct(result, 0, 2)


def test_early_completion_absorbs_and_literal_prior_replays():
    p, t = np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint32)
    result = run_particles(((0, 0), (0, 0)), p, t, glyphs=1, rho=.25,
        particles=256, seed=13)
    assert (result["labels"] == -2).any()
    for i in range(256):
        texts = reconstruct(result, i, 2)
        unit_length = int(result["keys"][i, 0])+1
        assert all(len(text)*unit_length == 2 for text in texts)
        score = math.log(.5)+2*math.log(.25)+sum(map(len, texts))*math.log(.75)
        assert abs(score-result["scores"][i]) < 1e-13
    groups, roots = terminal_groups(result)
    assert sum(count for _, count in groups) == 256 and 1 <= roots <= 256
    assert all(reconstruct(result, i, 2) in (((0,), (0,)), ((0, 0), (0, 0))) for i, _ in groups)


def test_repeat_seed_exact_all_outputs_no_source_mutation():
    p, t = tiny_source()
    before_p, before_t = p.copy(), t.copy()
    a = run_particles(((0, 1), (1, 0)), p, t, glyphs=2, rho=.25, particles=64, seed=31)
    b = run_particles(((0, 1), (1, 0)), p, t, glyphs=2, rho=.25, particles=64, seed=31)
    assert a["summary"] == b["summary"] and a["trace"] == b["trace"]
    for k in ("parents", "records", "labels", "keys", "scores", "offsets", "contexts", "closed"):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(t, before_t)


def test_balanced_schedule_exact_fraction_ties_closed_and_large_products():
    offsets = np.array([[1, 2], [0, 0], [2_000_000_000, 1_999_999_999]], dtype=np.int32)
    closed = np.array([[False, False], [True, True], [False, False]])
    lengths = np.array([2_000_000_001, 2_000_000_001], dtype=np.int64)
    assert select_records(offsets, closed, lengths, "balanced").tolist() == [0, -1, 1]


def test_zero_mass_categories_and_near_one_uniform():
    u = np.array([0., .5, np.nextafter(1., 0.)])
    assert categorical(np.array([0., 1., 0., 1., 0.]), u).tolist() == [1, 3, 3]
    weights = np.array([[0., 1., 0.], [1., 0., 1.], [0., 0., 1.]])
    assert categorical(weights, u).tolist() == [1, 2, 2]


@pytest.mark.parametrize("weights,u", [([-1, 2], [0]), ([0, 0], [.5]),
    ([1, float('nan')], [.5]), ([1, 2], [1.]), ([1, 2], [-.1])])
def test_bad_categorical_refused(weights, u):
    with pytest.raises(ValueError):
        categorical(weights, u)


@pytest.mark.parametrize("kwargs", [{"rho": 0}, {"rho": True}, {"seed": -1},
    {"particles": 0}, {"particles": True}, {"glyphs": True}, {"schedule": "random"}])
def test_invalid_config_refused(kwargs):
    p, t = tiny_source()
    with pytest.raises(ValueError):
        run_particles(((0,),), p, t, **kwargs)


def test_ancestry_resource_cap_before_allocation():
    p, t = tiny_source()
    with pytest.raises(MemoryError, match="ancestry"):
        run_particles(((0,)*100,), p, t, particles=10_000, max_ancestry_bytes=100)


def test_particle_work_envelope_before_allocation():
    p, t = tiny_source()
    with pytest.raises(MemoryError, match="work array"):
        run_particles(((0,),), p, t, particles=100, max_work_array_bytes=100)


@pytest.mark.parametrize("change", ["zero", "nan", "negative", "goto", "shape", "float32"])
def test_malformed_source_refused(change):
    p, t = tiny_source()
    if change == "zero":
        p[0] = 0
    elif change == "nan":
        p[0, 0] = np.nan
    elif change == "negative":
        p[0] = [-1, 2]
    elif change == "goto":
        t[0, 0] = 999
    elif change == "shape":
        t = t[:1]
    else:
        p = p.astype(np.float32)
    with pytest.raises(ValueError):
        run_particles(((0,),), p, t)


def test_actual_32_cell_cli_transport_and_audit_with_artificial_small_inputs(tmp_path, monkeypatch):
    import scripts.audit_source_particle_systems001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_particle_systems001 as runner

    class MockSource:
        probabilities = np.full((1, 23), 1/23, dtype=np.float64)
        transitions = np.zeros((1, 23), dtype=np.uint32)

        def row(self, state):
            return self.probabilities[state]

        def step(self, state, letter):
            return 0

    source = MockSource()
    def loader():
        return source, {"counts": {"path": "mock", "sha256": "mock", "bytes": 0}}
    counts, designs = (4, 8), (("repetitive", 3), ("repetitive", 5), ("markov", 3), ("markov", 5))
    out, bulk = tmp_path/"results"/runner.EXP, tmp_path/"outputs"/runner.EXP
    for module in (runner, auditor, storage, artifacts):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "COUNTS", counts)
        monkeypatch.setattr(module, "load_source", loader)
        monkeypatch.setattr(module, "require_frozen", lambda *args: None)
        monkeypatch.setattr(module, "limit_resources", lambda *args: None)
    monkeypatch.setattr(runner, "BULK", bulk)
    monkeypatch.setattr(runner, "DESIGNS", designs)
    runner.run("artificial-test-freeze")
    auditor.audit()
    import json
    assert json.loads((out/"audit.json").read_text())["workloads"] == 32
