"""Exact finite faces, hard support, length controls and real16cell transport."""
import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from scripts.audit_source_coupling001 import validate_inventory
from scripts.run_source_coupling001 import (
    MODELS, POOL, decoy_targets, finite_difference, iid_control, probe, replace_rows,
)
from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.native_suffix_marginal import marginal_python


def fake_score(function):
    def score(key):
        value = function(key)
        return {m: {"log_key_mass": value} for m in MODELS}
    return score


def test_discrete_derivatives_match_multilinear_coefficients_exactly():
    coefficients = {(): Fraction(3), (0,): Fraction(1, 7), (1,): Fraction(2, 7),
        (2,): Fraction(-2, 5), (0, 1): Fraction(19, 3), (1, 2): Fraction(-4, 9),
        (0, 1, 2): Fraction(5, 11)}
    values = {s: sum(v for t, v in coefficients.items() if set(t) <= set(s))
        for k in range(4) for s in itertools.combinations(range(3), k)}
    for k in range(1, 4):
        for s in itertools.combinations(range(3), k):
            assert math.isclose(finite_difference(values, s), float(coefficients.get(s, 0)), abs_tol=1e-14)
    values[(0,)] = None
    assert finite_difference(values, (0, 1)) is None


def test_pair_support_bridge_is_separate_from_undefined_log_interaction():
    pool = ("A", "B", "C", "D")
    base, gold = ("A", "B"), ("B", "A")
    def objective(key):
        if key == base:
            return 0.
        if key == gold:
            return 2.
        return None
    payload = probe(base, gold, (0, 1), 3, fake_score(objective), pool=pool)
    summary = payload["summaries"]["context"]["gold"]
    pair = summary["by_size"]["2"]
    assert pair["joint_only_improving_faces"] == pair["support_bridges_with_all_proper_nonempty_subfaces_unsupported"] == 1
    assert pair["finite_derivatives"] == 0
    assert summary["single_local_optimum_within_point1_nats"]
    validate_inventory(json.loads(json.dumps(payload)), base, gold, (0, 1), pool)


def test_pure_triple_improvement_needs_every_proper_subface():
    base, gold, pool = ("A",)*3, ("B",)*3, ("A", "B", "C", "D")
    def objective(key):
        n = key.count("B")
        return 4. if n == 3 else -float(n)
    payload = probe(base, gold, (0, 1, 2), 7, fake_score(objective), pool=pool)
    s = payload["summaries"]["context"]["gold"]["by_size"]
    assert s["2"]["joint_only_improving_faces"] == 0
    assert s["3"]["joint_only_improving_faces"] == 1
    assert s["3"]["derivative_max"] == 7.
    validate_inventory(json.loads(json.dumps(payload)), base, gold, (0, 1, 2), pool)


def test_matched_decoys_change_rows_without_correcting_and_preserve_target_lengths():
    base, gold = ("A", "AB", "BC"), ("BC", "C", "BD")
    target = decoy_targets(base, gold, (0, 1, 2), 17)
    assert target == decoy_targets(base, gold, (0, 1, 2), 17)
    assert all(t not in (b, g) and len(t) == len(g) for t, b, g in zip(target, base, gold, strict=True))
    assert replace_rows(base, (0, 2), gold) == ("BC", "AB", "BD")


def test_iid_source_removes_context_and_preserves_original_arrays():
    compact = fit_compact(["abcaabca", "abbaacba"], "abc", order=2, minimum=1)
    source = DenseSuffixAdapter(compact, 1.)
    probabilities, transitions = source.probabilities.copy(), source.transitions.copy()
    iid = iid_control(source)
    assert iid.order == 0 and iid.probabilities.shape == (1, 3)
    assert np.array_equal(source.probabilities[0], iid.probabilities[0])
    assert not np.any(iid.transitions) and not iid.probabilities.flags.writeable
    assert np.array_equal(source.probabilities, probabilities) and np.array_equal(source.transitions, transitions)


@pytest.mark.parametrize("broken", ["nan", "support"])
def test_bad_score_or_inconsistent_support_refused(broken):
    def score(key):
        return {"context": {"log_key_mass": float("nan") if broken == "nan" else None},
            "iid": {"log_key_mass": 1.}}
    with pytest.raises((ArithmeticError, ValueError)):
        probe(("A", "B"), ("B", "A"), (0, 1), 3, score, pool=("A", "B", "C", "D"))


def test_actual16cell_runner_and_auditor_transport(tmp_path, monkeypatch):
    import scripts.audit_source_coupling001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_coupling001 as runner

    alphabet = "abcdefghiklmnopqrstuxyz"
    source = DenseSuffixAdapter(fit_compact([alphabet*2], alphabet, order=1, minimum=1), 1.)
    class Native:
        def __init__(self, s, _build):
            self.source = s
        def score(self, key, record, rho, **limits):
            return marginal_python(self.source, key, record, rho, **limits)
    for module in (runner, auditor, storage, artifacts):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    parent_out = tmp_path/"results/SOURCE-REVISE-001"
    out, bulk = tmp_path/"results/SOURCE-COUPLING-001", tmp_path/"outputs/SOURCE-COUPLING-001"
    monkeypatch.setattr(runner, "PARENT", parent_out)
    monkeypatch.setattr(runner, "BULK", bulk)
    base = POOL[:23]
    gold = ("C", "D", *base[2:])
    fixture = {"generation_key": [POOL.index(u) for u in gold], "generation_source": [[0, 1], [1, 0]], "cipher": [[2, 3], [3, 2]]}
    specs = []
    native = Native(source, {})
    for case, seed, guide in itertools.product(range(4), (75511, 75513), ("none", "iid")):
        name = f"case{case}-{seed}-{guide}"
        prediction = storage.save_new(parent_out/f"{name}-prediction.json", {"revised_key": base})
        before = runner.score_key(native, ("CD", "DC"), base)["log_key_mass"]
        gold_score = runner.score_key(native, ("CD", "DC"), gold)["log_key_mass"]
        specs.append(storage.save_new(parent_out/f"{name}.json", {"name": name, "case": case, "seed": seed,
            "guidance": guide, "prediction": prediction, "search_summary": {"best_score": before},
            "diagnostic": {"gold_full_key_log_mass": gold_score}}))
    parent = {"workloads": specs}
    storage.save_new(parent_out/"result.json", parent)
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "inputs", lambda *_: ({"build": {}}, parent, [fixture]*4))
        monkeypatch.setattr(module, "load_source", lambda: (source, {"counts": {"mock": True}}))
        monkeypatch.setattr(module, "NativeMarginal", Native)
    runner.run("mock")
    auditor.audit()
    assert json.loads((out/"audit.json").read_text())["workloads"] == 16
    assert json.loads((out/"result.json").read_text())["signals"]["not_recovery_qualification"]
