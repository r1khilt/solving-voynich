"""Closed search/prediction boundaries and audit-prefix accounting."""
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from scripts.audit_source_state_systems001 import compare_replay, replay_config
from scripts.run_source_state_systems001 import ARMS, config, diagnose, finite, predict, summarize
from voynich.native_source_state import NativeStateLattice


def test_no_answer_arguments_and_fixed_width_controls():
    assert "fixture" not in inspect.signature(predict).parameters
    assert "generation_key" not in inspect.signature(NativeStateLattice.search).parameters
    a, b = config("unmerged"), config("merged")
    assert {**a, "merge": True} == b
    c = config("guided")
    assert {**b, "guidance": "iid"} == c
    assert config("wide_guided")["width"] == 32*c["width"]
    with pytest.raises(ValueError):
        config("adaptive")


def test_finite_mass_encoding_and_absent_prediction():
    assert finite({"x": (-math.inf, 0., [1])}) == {"x": [None, 0., [1]]}
    for invalid in (math.inf, math.nan):
        with pytest.raises(ArithmeticError):
            finite(invalid)
    assert predict({"terminals": []}, ((0,),), 0, None, None)["status"] == "no_complete_terminal"


def test_timed_layer_replay_fixes_work_not_new_time():
    stored = {"stop_reason": "time_cap", "expanded": 129, "trace": [{"x": 1}]}
    bounded = replay_config(stored, "guided")
    assert bounded["max_expanded"] == 129 and bounded["max_seconds"] == 600
    compare_replay(stored, {**stored, "stop_reason": "expansion_cap"})
    with pytest.raises(ValueError):
        compare_replay(stored, {**stored, "expanded": 130, "stop_reason": "expansion_cap"})
    with pytest.raises(ValueError):
        compare_replay(stored, stored)
    with pytest.raises(ValueError):
        replay_config({"stop_reason": "time_cap", "expanded": 0}, "guided")


def test_gold_diagnostics_distinguish_missing_rows_and_missing_readings():
    fixture = {"generation_source": [[0, 1]], "generation_key": [2, 3]}
    prediction = {"status": "no_complete_terminal"}
    result = {"terminals": [{"used_key": [2, -1]}, {"used_key": [2, 3]}, {"used_key": [1, 3]}]}
    row = diagnose(result, prediction, fixture, SimpleNamespace(alphabet="ab"))
    assert row["gold_used_mapping_in_returned_terminals"]
    assert row["returned_partial_keys_compatible_with_gold"] == 2
    assert row["edit_errors"] is None and row["matched_gold_used_rows"] is None
    assert row["true_source_letters"] == 2 and row["gold_used_rows"] == 2
    del result["terminals"][1]
    assert not diagnose(result, prediction, fixture, None)["gold_used_mapping_in_returned_terminals"]


def test_aggregate_never_counts_missing_reading_as_success():
    rows = [{"arm": arm, "summary": {"terminal_states": 0, "complete_search": False,
        "expanded": 10, "merged_arrivals": 0}, "diagnostic": {
        "gold_used_mapping_in_returned_terminals": False, "edit_errors": None, "exact_records": 0}}
        for arm in ARMS]
    totals = summarize(rows)
    assert all(t["read_cells"] == t["exact_records"] == t["cells_with_terminals"] == 0 for t in totals.values())


def test_preparation_converts_compiler_path_before_manifest(monkeypatch):
    import scripts.run_source_state_systems001 as runner
    observed = []
    monkeypatch.setattr(runner, "build_source_state_native", lambda _: {"library_path": "/tmp/library"})
    def check_path(path):
        assert isinstance(path, Path)
        return {"path": str(path)}
    monkeypatch.setattr(runner, "artifact", check_path)
    monkeypatch.setattr(runner, "save_new", lambda path, payload: observed.append((path, payload)))
    runner.prepare()
    assert observed[0][1]["no_empirical_search_or_training"]


def test_actual16_native_calls_close_predictions_before_gold_and_full_audit(tmp_path, monkeypatch):
    import scripts.audit_source_state_systems001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_state_systems001 as runner
    from voynich.native_suffix_marginal import marginal_python

    class Source:
        alphabet = tuple("abcdefghiklmnopqrstuxyz")
        probabilities = np.full((1, 23), 1/23, dtype=np.float64)
        transitions = np.zeros((1, 23), dtype=np.uint32)

        def state(self, _history):
            return 0

        def row(self, state):
            return self.probabilities[state]

        def step(self, _state, _row):
            return 0

    class Native:
        def __init__(self, source, _build):
            self.source = source

        def score(self, key, record, rho, **limits):
            return marginal_python(self.source, key, record, rho, **limits)

    out, bulk = tmp_path/"results/SOURCE-STATE-SYSTEMS-001", tmp_path/"outputs/SOURCE-STATE-SYSTEMS-001"
    for module in (runner, auditor, storage, artifacts):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BULK", bulk)
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "NativeMarginal", Native)
        monkeypatch.setattr(module, "load_source", lambda: (Source(), {"counts": {"mock": True}}))
    runner.prepare()
    build = json.loads((out/"native-build.json").read_text())
    cases = [{"case": i, "cipher": [[0], [0]], "generation_key": [0]*23,
        "generation_source": [[0], [1]]} for i in range(4)]
    fixtures = runner.save_new(bulk/"fixtures.json", {"fixtures": cases})
    parent = {"fixtures": fixtures}
    for module in (runner, auditor):
        monkeypatch.setattr(module, "inputs", lambda _: ({"build": {}}, parent, cases, build))
    original = runner.diagnose
    def closed_diagnostic(result, prediction, fixture, source):
        reads = list(bulk.glob("case*-prediction.json"))
        fits = list(bulk.glob("case*.json.gz"))
        assert reads and len(reads) == len(fits)
        return original(result, prediction, fixture, source)
    monkeypatch.setattr(runner, "diagnose", closed_diagnostic)
    runner.run("mock")
    auditor.audit()
    result = json.loads((out/"result.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    assert len(result["workloads"]) == audit["workloads"] == 16
    assert audit["alternate_fixed_key_record_scores"] == 32
    assert result["arm_totals"]["merged"]["merged_arrivals"] > 0
    assert all(t["read_cells"] == 4 for t in result["arm_totals"].values())
