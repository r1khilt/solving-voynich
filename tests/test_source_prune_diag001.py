"""Observation-only controls, alternate arithmetic, and complete tiny transport."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_source_prune_diag001 import alternate_priorities, alternate_states
from scripts.run_source_prune_diag001 import config, summarize_trace, validate_trace
from scripts.run_source_state_systems001 import ARMS, config as original_config
from tests.test_source_state_lattice import source
from voynich.source_prune_observer import NativePruneObserver, build_prune_observer, golden_path


def test_only_timer_differs_from_closed_search():
    for arm in ARMS:
        assert config(arm) == {**original_config(arm), "max_seconds": 600.}
    with pytest.raises(ValueError):
        config("adaptive")


def test_independent_prefix_scheduler_contexts_priors_and_eos():
    p, goto = source()
    fixture = {"generation_source": [[0, 1, 0], [1, 0]], "generation_key": [0, 2], "cipher": [[0, 0, 0, 0], [0, 0, 0]]}
    for schedule in ("balanced", "sequential"):
        alt = alternate_states(fixture, p, goto, glyphs=2, rho=.25, schedule=schedule)
        targets, masses = golden_path(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], p, goto, glyphs=2, rho=.25, schedule=schedule)
        for layer, table, score, _ in alt:
            assert table == list(targets[layer])
            assert score == pytest.approx(masses[layer], abs=1e-12)


def test_summary_does_not_revive_path_or_confuse_key_with_reading():
    lost = {"alive_before": 1, "alive_after": 0, "pre_kept": 0, "processed": True, "aliases": 1}
    end = {"alive_before": 0, "alive_after": 0, "processed": True, "aliases": 1,
        "terminal_group_rank": 0, "terminal_used_mapping_best_rank": 7, "terminal_used_mapping_returned": 1}
    s = summarize_trace([lost, end])
    assert s["first_loss_stage"] == "geometric_prebeam" and not s["literal_path_survived"]
    assert s["terminal_used_mapping_best_rank"] == 7 and s["post_loss_structural_alias_layers"] == 1
    lost["pre_kept"] = 1
    assert summarize_trace([lost, end])["first_loss_stage"] == "final_beam"


def test_observer_arithmetic_full_tiny_and_invalid_trace(tmp_path):
    p, goto = source()
    fixture = {"generation_source": [[0, 1, 0], [1, 0]], "generation_key": [0, 2], "cipher": [[0, 0, 0, 0], [0, 0, 0]]}
    build = build_prune_observer(tmp_path/"build")
    cfg = {**config("guided"), "glyphs": 2, "rho": .25}
    result, trace = NativePruneObserver(p, goto, build).observe(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], **cfg)
    validate_trace(trace, fixture["cipher"], cfg)
    targets, _ = golden_path(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], p, goto, glyphs=2, rho=.25)
    checked, delta = alternate_priorities(fixture, p, goto, trace, [list(targets[d["target_glyph_layer"]]) for d in trace], cfg)
    assert checked >= len(trace) and delta < 1e-12
    broken = [dict(d) for d in trace]
    broken[0]["pre_rank"] = 0
    with pytest.raises(ValueError):
        validate_trace(broken, fixture["cipher"], cfg)
    assert result["stop_reason"] == "frontier_exhausted"


def test_actual16_native_diagnostic_cells_and_complete_audit(tmp_path, monkeypatch):
    import scripts.audit_source_prune_diag001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_prune_diag001 as runner
    from voynich.native_source_state import NativeStateLattice, build_source_state_native
    p, goto = np.full((1, 23), 1/23), np.zeros((1, 23), dtype=np.uint32)
    source_obj = SimpleNamespace(probabilities=p, transitions=goto)
    out, bulk, parent_dir = tmp_path/"results/SOURCE-PRUNE-DIAG-001", tmp_path/"outputs/SOURCE-PRUNE-DIAG-001", tmp_path/"results/SOURCE-STATE-SYSTEMS-001"
    for module in (runner, storage, artifacts):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BULK", bulk)
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "PARENT", parent_dir)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "load_source", lambda: (source_obj, {"counts": {"tiny": True}}))
    runner.prepare()
    build = json.loads((out/"native-build.json").read_text())
    base_build = build_source_state_native(tmp_path/"original")
    baseline = NativeStateLattice(p, goto, base_build)
    cases = [{"case": i, "cipher": [[0], [0]], "generation_source": [[0], [1]], "generation_key": [0]*23} for i in range(4)]
    fixtures = runner.save_new(bulk/"fixtures.json", {"fixtures": cases})
    cells = []
    for case in range(4):
        for arm in ARMS:
            name = f"case{case}-{arm}"
            fitted = runner.save_new(tmp_path/f"outputs/original/{name}.json.gz", runner.finite(baseline.search(cases[case]["cipher"], **original_config(arm))), compressed=True)
            cells.append(runner.save_new(parent_dir/f"{name}.json", {"name": name, "fitted": fitted}))
    closed = {"source": {"tiny": True}, "source_arrays": runner.array_identity(source_obj), "fixtures": fixtures, "workloads": cells}
    result_spec = runner.save_new(parent_dir/"result.json", closed)
    runner.save_new(parent_dir/"audit.json", {"status": "PASS_full_native_state_and_prediction_replay", "result": result_spec})
    for module in (runner, auditor):
        monkeypatch.setattr(module, "inputs", lambda _: ({}, {"fixtures": fixtures}, cases, {}, closed, build))
    runner.run("tiny-only")
    auditor.audit()
    result = json.loads((out/"result.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    assert result["ordinary_output_exact_match_cells"] == audit["workloads"] == 16
    assert audit["status"] == "PASS" and audit["maximum_alternate_delta"] < 1e-12
