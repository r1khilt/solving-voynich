from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import math
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from voynich.source_prefix_inverse import BoundedQueue, Node, search_source_prefix


def rational_row(prefix):
    if not prefix:
        return F(1, 3), F(2, 3)
    return (F(3, 4), F(1, 4)) if prefix[-1] == 0 else (F(1, 5), F(4, 5))


def provider(prefix):
    return tuple(map(float, rational_row(prefix)))


def source_weight(text, rho=F(1, 4)):
    weight = rho*(1-rho)**len(text)
    for i, row in enumerate(text):
        weight *= rational_row(text[:i])[row]
    return weight


def brute(cipher, glyphs=2):
    pool = [(g,) for g in range(glyphs)]+list(product(range(glyphs), repeat=2))
    texts = [tuple(x for n in range(1, len(c)+1) for x in product(range(2), repeat=n)) for c in cipher]
    result = defaultdict(F)
    for key in product(pool, repeat=2):
        choices = [tuple(x for x in options if tuple(g for r in x for g in key[r]) == c)
                   for c, options in zip(cipher, texts, strict=True)]
        for pair in product(*choices):
            weight = F(1, len(pool)**2)
            for x in pair:
                weight *= source_weight(x)
            result[pair] += weight
    return dict(result)


@pytest.mark.parametrize("records", [1, 2])
@pytest.mark.parametrize("bonus", [0., 4.])
def test_all_tiny_observations_against_full_key_and_source_rational_enumeration(records, bonus):
    observed = [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    for cipher in product(observed, repeat=records):
        expected = brute(cipher)
        result = search_source_prefix(cipher, provider, rows=2, glyphs=2, rho=.25,
                                      max_expanded=10000, max_frontier=10000, progress_bonus=bonus)
        assert result["complete_search"] and result["unresolved_log_mass_upper"] == -math.inf
        actual = {r["source_records"]: math.exp(r["log_mass"]) for r in result["readings"]}
        assert actual.keys() == expected.keys()
        for x, weight in expected.items():
            assert actual[x] == pytest.approx(float(weight), abs=1e-14, rel=1e-12)
        assert math.exp(result["found_log_mass"]) == pytest.approx(float(sum(expected.values())), abs=1e-14)
        for terminal in result["terminals"]:
            key = terminal["used_key"]
            assert tuple(tuple(g for r in x for g in key[r]) for x in terminal["source_records"]) == cipher
        assert len(result["terminals"]) == len({(r["source_records"], r["used_key"]) for r in result["terminals"]})


@pytest.mark.parametrize("budget", [1, 3, 8, 20, 60, 150])
@pytest.mark.parametrize("width", [1, 3, 50])
@pytest.mark.parametrize("bonus", [0., 4.])
def test_every_pruned_or_unfinished_mass_bound_covers_exact_evidence_and_decision(budget, width, bonus):
    cipher = ((0, 0), (0, 1))
    expected = brute(cipher)
    total = float(sum(expected.values()))
    result = search_source_prefix(cipher, provider, rows=2, glyphs=2, rho=.25,
                                  max_expanded=budget, max_frontier=width, progress_bonus=bonus)
    low, high = math.exp(result["found_log_mass"]), math.exp(result["evidence_log_upper"])
    assert low <= total+1e-13 <= high+2e-13
    for r in result["readings"]:
        assert math.exp(r["log_mass"]) <= float(expected[r["source_records"]])+1e-14
    if result["reading_bound_separated"]:
        winner = result["readings"][0]["source_records"]
        assert expected[winner] == max(expected.values())
    assert result["maximum_active_states"] <= width
    assert result["generated"] >= result["expanded"]+result["pruned_states"]


def test_terminal_cap_and_canonical_factor_do_not_claim_exhaustive_or_change_ranking():
    original = search_source_prefix(((0, 1),), provider, rows=2, glyphs=2, rho=.25)
    canonical = search_source_prefix(((0, 1),), provider, rows=2, glyphs=2, rho=.25, canonical=True)
    assert canonical["canonical_orbit_factor"] == 2
    assert canonical["found_log_mass"] == pytest.approx(original["found_log_mass"]+math.log(2))
    assert [r["source_records"] for r in canonical["readings"]] == [r["source_records"] for r in original["readings"]]
    partial = search_source_prefix(((0, 0), (0, 1)), provider, rows=2, glyphs=2, rho=.25, max_terminals=1)
    assert not partial["complete_search"] and partial["stop_reason"] == "terminal_cap"
    assert partial["unresolved_log_mass_upper"] > -math.inf
    with pytest.raises(ValueError, match="Canonical"):
        search_source_prefix(((1,),), provider, rows=2, glyphs=2, canonical=True)


@pytest.mark.parametrize("cipher,orbit", [(((0, 0),), 3), (((0, 1),), 6), (((0, 1, 2),), 6)])
def test_canonical_mass_with_one_two_and_three_seen_glyphs_matches_raw_key_reference(cipher, orbit):
    expected = brute(cipher, glyphs=3)
    result = search_source_prefix(cipher, provider, rows=2, glyphs=3, rho=.25,
                                  canonical=True, max_frontier=10000, max_expanded=10000)
    assert result["complete_search"] and result["canonical_orbit_factor"] == orbit
    for reading in result["readings"]:
        assert math.exp(reading["log_mass"]) == pytest.approx(float(expected[reading["source_records"]])*orbit, abs=1e-14)


def test_dual_queue_lazy_history_compacts_and_mass_of_evictions_is_preserved():
    queue = BoundedQueue(3)
    for i in range(100):
        node = Node((), (i,), 0, (), -float(i))
        queue.push(node, -float(i), -float(i))
    assert len(queue.live) == 3 and queue.pruned_count == 97
    assert len(queue.best) <= 4*queue.capacity+16 and len(queue.worst) <= 4*queue.capacity+16
    assert [queue.pop().prefix[0] for _ in range(3)] == [0, 1, 2]
    assert math.exp(queue.pruned_bound) == pytest.approx(sum(math.exp(-i) for i in range(3, 100)))


def test_unsupported_observation_is_exact_zero_and_source_histories_reset_and_share():
    unsupported = search_source_prefix(((0, 1, 0),), lambda _: (1.,), rows=1, glyphs=2)
    assert unsupported["complete_search"] and unsupported["found_log_mass"] == -math.inf
    assert unsupported["evidence_log_upper"] == -math.inf and not unsupported["reading_bound_separated"]
    seen = []

    def recorded(prefix):
        seen.append(prefix)
        return provider(prefix)

    result = search_source_prefix(((0, 0), (0, 1)), recorded, rows=2, glyphs=2, rho=.25)
    assert len(seen) == len(set(seen)) == result["source_calls"]
    assert result["source_requests"] > result["source_calls"] and seen.count(()) == 1


@pytest.mark.parametrize("probabilities", [(1., 1.), (-1., 2.), (math.nan, 1.), (1.,), (True, False)])
def test_invalid_source_callback_is_never_treated_as_a_zero_likelihood(probabilities):
    with pytest.raises(ValueError, match="callback"):
        search_source_prefix(((0,),), lambda _: probabilities, rows=2, glyphs=2)


@pytest.mark.parametrize("kwargs", [{"rho": 0.}, {"max_expanded": False}, {"progress_bonus": -1.}, {"numerical_margin": -1.}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        search_source_prefix(((0,),), provider, rows=2, glyphs=2, **kwargs)


def test_actual_artificial_benchmark_and_auditor_json_transport(tmp_path, monkeypatch):
    from scripts import audit_source_prefix_systems001 as checker
    from scripts import benchmark_source_prefix_systems001 as runner
    from scripts import run_blind_channel_dev004 as storage

    probabilities = np.full((1, 23), 1/23)
    transitions = np.zeros((1, 23), dtype=np.uint32)
    source = SimpleNamespace(alphabet=tuple("abcdefghiklmnopqrstuxyz"),
        probabilities=probabilities, transitions=transitions,
        row=lambda _: probabilities[0], state=lambda _: 0, step=lambda *_: 0,
        array_bytes=probabilities.nbytes+transitions.nbytes)
    assert len(source.alphabet) == 23
    selected = {"counts": {"artificial": True}}
    fixed = [{"cipher": ((0,), (0,)), "kind": "artificial-test-only", "length": 1}]*4

    def artifact(path):
        raw = path.read_bytes()
        return {"path": str(path.relative_to(tmp_path)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(storage, "ROOT", tmp_path)
    for module in (runner, checker):
        monkeypatch.setattr(module, "OUT", tmp_path/"result")
        monkeypatch.setattr(module, "BULK", tmp_path/"bulk")
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "load_source", lambda: (source, selected))
        monkeypatch.setattr(module, "fixtures", lambda _: fixed)
        monkeypatch.setattr(module, "artifact", artifact)
    runner.run("artificial-transport-test")
    checker.audit()
    result = json.loads((tmp_path/"result/result.json").read_text())
    audit = json.loads((tmp_path/"result/audit.json").read_text())
    assert result["tiny_calls"] == 288 and len(result["dense_results"]) == 8
    assert audit["workloads"] == 296
    text = (tmp_path/"bulk/workloads.jsonl").read_text()
    assert "Infinity" not in text and "NaN" not in text
    # The unsupported-mass null must not corrupt unused dictionary null rows.
    value = checker.restore_mass({"found_log_mass": None, "used_key": [None]})
    assert value == {"found_log_mass": -math.inf, "used_key": [None]}
