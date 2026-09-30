import copy
from dataclasses import asdict
import json
import math

import pytest

from scripts import run_blind_channel_dev005 as runner
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import estimate_source
from voynich.higher_order_unit_refine import refine_units
from voynich.higher_order_unit_channel import decode_units
from voynich.unit_channel_search import channel_from_units
from voynich.finite_state_channel_fit import channel_code_bits


def fixture():
    source = estimate_source(["aaaabaaab"], ("a", "b"), 3, 1.)
    context = CodingContext(("a", "b"), ("X", "Y"), 8, 2, 2, 3, stop_probability=.2)
    trace = refine_units(source, ["XXX", "XY"], context, ("Y", "X"), max_seconds=10.)
    return trace, source.to_dict(), context


def test_full_trace_accounting_and_zero_sweep_are_well_defined():
    trace, source, context = fixture()
    result = runner.trace_accounting(trace, source, context)
    assert result["final_local_certificate"]
    assert result["neighbors_accounted"] == sum(len(row["neighbors"]) for row in trace["trace"])
    empty = refine_units(estimate_source(["aba"], ("a", "b"), 3, 1.), ["XXX"], context,
                         ("X", "Y"), max_sweeps=0)
    assert not runner.trace_accounting(empty, source, context)["final_local_certificate"]


@pytest.mark.parametrize("change", ["neighbor", "move", "code", "data", "selected", "count", "parent", "certificate"])
def test_adversarial_trace_mutations_fail(change):
    trace, source, context = fixture()
    trace = copy.deepcopy(trace)
    first = trace["trace"][0]
    if change == "neighbor":
        first["neighbors"][0]["units"].reverse()
    elif change == "move":
        first["neighbors"][0]["move"]["rows"] = [0, 0]
    elif change == "code":
        first["neighbors"][0]["score"]["model_bits"] += 1
    elif change == "data":
        first["neighbors"][0]["score"]["data_bits"] += .01
    elif change == "selected":
        first["selected_units"] = ["YY", "YY"]
    elif change == "count":
        trace["evaluated_neighbors"] += 1
    elif change == "parent":
        first["parent_units"].reverse()
    else:
        trace["best_is_certified_local_optimum"] = False
    with pytest.raises(ValueError):
        runner.trace_accounting(trace, source, context)


def test_campaign_refuses_existing_key_before_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "inputs", lambda _: None)
    path = runner.selected_path(runner.CASE_NAMES[0])
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("Must not launch a replacement"))
    with pytest.raises(FileExistsError):
        runner.campaign("unused")


def test_fit_rejects_unregistered_case_without_input_reads(monkeypatch):
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(runner, "inputs", lambda _: pytest.fail("Should not open inputs"))
    with pytest.raises(ValueError):
        runner.fit("unregistered", "unused")


def test_evaluation_preserves_unsupported_transfer_as_a_measured_failure(tmp_path, monkeypatch):
    source = estimate_source(["aaab"], ("a", "b"), 3, 1.)
    context = CodingContext(("a", "b"), ("X", "Y"), 8, 2, 2, 3, stop_probability=.2)
    learned = channel_from_units(source, context, ("X", "XX"))
    gold = channel_from_units(source, context, ("X", "Y"))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CASE_NAMES", ("B-key1",))
    selection = tmp_path / runner.SELECTION
    selection.parent.mkdir(parents=True)
    selection.write_text("{}")
    manifest = {"cases": {"B-key1": {"positive": True, "artifacts": {
        k: {"path": k, "sha256": "fixture"} for k in ("fit", "transfer", "answer")}}}}
    payloads = {"fit": {"records": ["XXX"], "context": asdict(context)},
                "transfer": {"records": ["Y"], "context": asdict(context)},
                "answer": {"gold_channel": gold.to_dict(), "plaintext": {"fit": ["aaa"], "transfer": ["b"]}}}
    fit_score = (channel_code_bits(learned, context)
                 - decode_units(source, ("X", "XX"), "XXX", .2).log_likelihood / math.log(2))
    chosen = runner.selected_path("B-key1")
    chosen.parent.mkdir(parents=True)
    chosen.write_text(json.dumps({"fit_sha256": "fixture", "source_selection_sha256": runner.digest(b"{}"),
                                  "channel": learned.to_dict(), "score": {"total_bits": fit_score}}))
    monkeypatch.setattr(runner, "inputs", lambda _: (manifest, source, source.to_dict()))
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    monkeypatch.setattr(runner, "checked_artifact", lambda spec: payloads[spec["path"]])
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(runner.signal, "alarm", lambda *_: None)
    saved = {}

    def save(path, payload, **_kwargs):
        saved[path.name] = payload
        return {"path": str(path), "sha256": "fixture", "bytes": 0}

    monkeypatch.setattr(runner, "save_new", save)
    runner.evaluate("fixture")
    observed = saved["evaluation.json"]["cases"]["B-key1"]["learned"]["transfer"]
    assert observed["supported_records"] == 0
    assert observed["edits"] == 1 and observed["exact_records"] == 0
    assert observed["log_likelihood"] is None and observed["total_bits"] is None
    assert observed["dictionary_floor"] == {"record_edits": [None], "edits": None}
