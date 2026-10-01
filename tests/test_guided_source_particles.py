"""Independent surrogate enumeration and corrected proposal/target laws."""
import itertools
import math
from collections import defaultdict

import numpy as np
import pytest

from scripts.benchmark_source_prefix_systems001 import full_key_reference
from voynich.guided_source_particles import IidSuffixGuide, guided_actions, log_categorical, run_guided_particles
from voynich.source_key_particles import advance, reconstruct


def tiny_source():
    return np.array([[1/3, 2/3], [3/4, 1/4], [1/5, 4/5]], dtype=np.float64), np.tile([1, 2], (3, 1))


def test_surrogate_dp_matches_independent_string_and_fresh_unit_enumeration():
    cipher = ((0, 1, 0), (1, 0))
    guide = IidSuffixGuide(cipher, [1/3, 2/3], glyphs=2, rho=.25)
    pool = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
    for key in itertools.product(range(-1, 6), repeat=2):
        keys = np.array([key], dtype=np.int32)
        table = guide.build(keys)[0]
        for record, observed in enumerate(cipher):
            for pos in range(len(observed)+1):
                suffix, mass = observed[pos:], 0.
                for n in range(len(suffix)+1):
                    for text in itertools.product(range(2), repeat=n):
                        options = [pool if key[r] < 0 else (pool[key[r]],) for r in text]
                        for units in itertools.product(*options):
                            if tuple(g for unit in units for g in unit) != suffix:
                                continue
                            weight = .25*.75**n
                            for row in text:
                                weight *= guide.row[row]*(1/6 if key[row] < 0 else 1)
                            mass += weight
                assert abs(math.exp(table[record, pos])-mass) < 1e-14


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
def test_guided_full_target_telescopes_and_matches_independent_full_key_law(schedule):
    p, t = tiny_source()
    observations = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
    for cipher in itertools.product(observations, repeat=2):
        lengths = np.array(list(map(len, cipher)))
        obs = np.full((2, 2), -1, dtype=np.int32)
        for i, c in enumerate(cipher):
            obs[i, :len(c)] = c
        guide = IidSuffixGuide(cipher, p[0], glyphs=2, rho=.25, cache_entries=3)
        keys = np.full((1, 2), -1, dtype=np.int32)
        offsets = np.zeros((1, 2), dtype=np.int32)
        contexts = np.zeros((1, 2), dtype=np.uint32)
        closed = np.zeros((1, 2), dtype=bool)
        found, proposal_mass = defaultdict(float), [0.]
        h0 = math.exp(guide(keys, offsets, closed)[0])

        def visit(k, o, s, c, texts, mass, proposal, importance):
            if c.all():
                assert guide(k, o, c)[0] == 0
                found[texts] += mass
                proposal_mass[0] += proposal
                assert abs(proposal*importance-mass) < 1e-13
                return
            values, rec, singles, doubles, log_b = guided_actions(k, o, s, c, obs,
                lengths, p, t, 2, .25, schedule, guide)
            total = np.exp(log_b).sum()
            if total == 0:
                proposal_mass[0] += proposal
            for action in np.flatnonzero(values[0]):
                changed = advance(k, o, s, c, rec, singles, doubles,
                    np.array([0]), np.array([action]), t)
                next_texts = list(texts)
                if action < 4:
                    next_texts[int(rec[0])] += (int(action)//2,)
                visit(*changed[:4], tuple(next_texts), mass*values[0, action],
                    proposal*math.exp(log_b[0, action])/total, importance*total)

        visit(keys, offsets, contexts, closed, ((), ()), 1., 1., h0)
        reference = full_key_reference(cipher)
        assert found.keys() == reference.keys()
        assert all(abs(found[k]-float(reference[k])) < 1e-13 for k in found)
        assert abs(proposal_mass[0]-1) < 1e-13


def test_closed_eos_floor_cache_eviction_and_work_cap():
    guide = IidSuffixGuide(((0,), (1,)), [1.], glyphs=2, rho=.25, log_floor=-7., cache_entries=1)
    keys = np.array([[0], [1], [-1]], dtype=np.int32)
    offsets = np.zeros((3, 2), dtype=np.int32)
    closed = np.zeros((3, 2), dtype=bool)
    values = guide(keys, offsets, closed)
    assert values[0] == values[1] == -7
    np.testing.assert_array_equal(guide(keys, offsets, closed), values)
    assert len(guide.cache) == 1
    assert np.all(guide(keys, offsets, np.ones_like(closed)) == 0)
    with pytest.raises(RuntimeError, match="work cap"):
        capped = IidSuffixGuide(((0,),), [1.], glyphs=2, max_tables=1)
        capped(keys, offsets[:, :1], closed[:, :1])
    with pytest.raises(MemoryError, match="cache"):
        IidSuffixGuide(((0,)*100,), [1.], max_cache_bytes=100)


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
@pytest.mark.parametrize("guidance", ["none", "iid"])
def test_single_reading_guided_evidence_original_scores_and_replay(schedule, guidance):
    p, t = np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint32)
    a = run_guided_particles(((0,), (0,)), p, t, glyphs=1, rho=.25,
        particles=32, seed=3, schedule=schedule, guidance=guidance)
    b = run_guided_particles(((0,), (0,)), p, t, glyphs=1, rho=.25,
        particles=32, seed=3, schedule=schedule, guidance=guidance)
    assert a["summary"] == b["summary"] and a["trace"] == b["trace"]
    assert abs(math.exp(a["summary"]["log_evidence_estimate"])-9/512) < 1e-14
    for k in ("parents", "records", "labels", "keys", "scores", "offsets", "contexts", "closed"):
        np.testing.assert_array_equal(a[k], b[k])
    for i in range(32):
        assert reconstruct(a, i, 2) == ((0,), (0,))
        assert abs(a["scores"][i]-math.log(9/512)) < 1e-13


def test_original_support_extinction_without_refill():
    p, t = np.ones((1, 1)), np.zeros((1, 1), dtype=np.uint32)
    a = run_guided_particles(((0,), (1,)), p, t, glyphs=2, rho=.25, particles=32, seed=3)
    assert a["summary"]["status"] == "extinct"
    assert a["summary"]["log_evidence_estimate"] is None


def test_log_categorical_zeros_and_extreme_scales_never_need_exponentiation():
    rng = np.random.default_rng(123)
    assert np.all(log_categorical(np.array([-math.inf, -10000., -math.inf]), rng, 130) == 1)
    weights = np.array([[-10000., -math.inf], [-math.inf, -2000.]])
    assert log_categorical(weights, rng).tolist() == [0, 1]
    with pytest.raises(ValueError):
        log_categorical(np.array([-math.inf, -math.inf]), rng, 5)


def test_candidate_guidance_changes_initial_key_preferences():
    p, t = tiny_source()
    cipher = ((0, 0, 1), (1, 0, 1))
    obs = np.array(cipher)
    k = np.full((1, 2), -1, dtype=np.int32)
    o = np.zeros((1, 2), dtype=np.int32)
    s = np.zeros((1, 2), dtype=np.uint32)
    c = np.zeros((1, 2), dtype=bool)
    g = IidSuffixGuide(cipher, p[0], glyphs=2, rho=.25)
    a, _, _, _, log_b = guided_actions(k, o, s, c, obs, np.array([3, 3]), p, t, 2, .25, "balanced", g)
    assert not np.allclose(a/a.sum(), np.exp(log_b)/np.exp(log_b).sum())


@pytest.mark.parametrize("kwargs", [{"particles": 0}, {"seed": -1}, {"rho": True},
    {"glyphs": True}, {"schedule": "random"}, {"max_work_array_bytes": 1}, {"max_ancestry_bytes": 1}])
def test_config_and_memory_fail_before_run(kwargs):
    p, t = tiny_source()
    with pytest.raises((ValueError, MemoryError)):
        run_guided_particles(((0,),), p, t, **kwargs)


def test_actual_fixed_grid_transport_and_audit_with_mock_inputs(tmp_path, monkeypatch):
    import scripts.audit_source_guide001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_guide001 as runner

    class MockSource:
        probabilities = np.full((1, 23), 1/23, dtype=np.float64)
        transitions = np.zeros((1, 23), dtype=np.uint32)
        alphabet = tuple(range(23))

        def row(self, state):
            return self.probabilities[state]

        def step(self, state, row):
            return 0

    for module in (runner, auditor, storage, artifacts):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    out, bulk = tmp_path/"results"/runner.EXP, tmp_path/"outputs"/runner.EXP
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "PARTICLES", 4)
        monkeypatch.setattr(module, "load_source", lambda: (MockSource(), {"counts": {"mock": True}}))
    monkeypatch.setattr(runner, "BULK", bulk)
    monkeypatch.setattr(runner, "DESIGNS", (3, 3, 5, 5))
    runner.run("mock")
    auditor.audit()
    import json
    result = json.loads((out/"result.json").read_text())
    assert len(result["workloads"]) == 16
    assert json.loads((out/"audit.json").read_text())["workloads"] == 16
