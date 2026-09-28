"""Synthetic API traces only; no experiment or source artifacts are opened."""
from __future__ import annotations

import copy
import gzip
import json
from dataclasses import asdict

import pytest

from scripts import summarize_blind_channel_dev003_search as audit
from voynich import unit_channel_search as search_module
from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich.unit_channel_search import UnitSearchConfig, channel_from_units, search_unit_channel


def fixture(config=None):
    source = SourceModel(("a", "b"), 1, {"": {"a": .75, "b": .25},
                         "a": {"a": .125, "b": .875}, "b": {"a": .625, "b": .375}})
    context = CodingContext(source.alphabet, ("x", "y"), 32, 2, 2, 3, stop_probability=.25)
    config = config or UnitSearchConfig(seed=19, restarts=3, max_sweeps=6, batch_size=3, max_units=6)
    full = search_unit_channel(source, ("x" * 12, "y", "", "xx"), context, config=config).to_dict()
    plain_context = json.loads(json.dumps(asdict(context)))
    return source, plain_context, full


def compact(full, name="B-key1"):
    frozen = {key: copy.deepcopy(value) for key, value in full.items() if key != "trace"}
    hashes = {path: "b" * 64 for path in audit.SOURCE_PATHS}
    hashes.update({audit.SOURCE_PATH: audit.SOURCE_SHA256, audit.MANIFEST_PATH: audit.MANIFEST_SHA256})
    frozen.update(experiment=audit.EXPERIMENT, case_id=name, source_freeze=audit.SOURCE_FREEZE,
                  source_sha256=audit.SOURCE_SHA256, input_sha256="c" * 64,
                  full_output={"path": f"outputs/{audit.EXPERIMENT}/{name}.json.gz", "sha256": "d" * 64, "bytes": 1},
                  cpu_seconds_before_final_write=.1, peak_rss_bytes=1024, source_hashes=hashes,
                  environment={"python": "synthetic", "numpy": "synthetic", "platform": "synthetic",
                               "numerical_thread_limits": {key: "1" for key in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    return frozen


def sweep(full):
    return next(event for event in full["trace"] if event["event"] == "sweep")


def test_complete_synthetic_trace_accounts_for_all_scored_work():
    _, context, full = fixture()
    result = audit.validate_case(compact(full), full, context)
    assert result["status"] == "verified" and result["returned_minimum_verified"]
    assert result["scored_initializations"] == 3
    assert result["completed_sweeps"] == full["completed_sweeps"]
    assert sum(result["move_counts"].values()) == full["evaluated_neighbors"]
    assert result["kernel_fallbacks"] == full["kernel_fallbacks"]
    assert len(result["restart_details"]) == 3


@pytest.mark.parametrize("corruption", [
    "pool", "unknown_payload", "seed", "method", "random_units", "coverage", "parent", "row_order",
    "move", "boolean_move", "status", "literal_cost", "log_arithmetic", "total_arithmetic", "selection",
    "acceptance", "cardinality", "batch_start", "batch_dimensions", "batch_fallback", "aggregate_fallback",
    "aggregate_neighbors", "certificate", "channel", "boolean_channel", "premature_restart", "final_score",
])
def test_adversarial_trace_corruption_is_rejected(corruption):
    _, context, full = fixture()
    event = sweep(full)
    neighbor = next(row for row in event["neighbors"] if row["score"] is not None)
    second = next(row for row in full["trace"] if row["event"] == "restart" and row["restart"] == 1)
    if corruption == "pool":
        full["unit_pool"].pop()
    elif corruption == "unknown_payload":
        full["unregistered"] = True
    elif corruption == "seed":
        second["seed"] += 1
    elif corruption == "method":
        second["method"] = "oracle"
    elif corruption == "random_units":
        second["units"].reverse()
    elif corruption == "coverage":
        full["trace"][0]["units"] = ["x", "x"]
    elif corruption == "parent":
        event["parent_units"].reverse()
    elif corruption == "row_order":
        event["neighbors"][0], event["neighbors"][1] = event["neighbors"][1], event["neighbors"][0]
    elif corruption == "move":
        event["neighbors"][0]["units"] = ["xx", "xx"]
    elif corruption == "boolean_move":
        event["neighbors"][0]["move"]["rows"] = [False, True]
    elif corruption == "status":
        neighbor["status"] = "unsupported"
    elif corruption == "literal_cost":
        neighbor["score"]["model_bits"] += 1
        neighbor["score"]["total_bits"] += 1
    elif corruption == "log_arithmetic":
        neighbor["score"]["log_likelihood"] -= 1
    elif corruption == "total_arithmetic":
        neighbor["score"]["total_bits"] += 1
    elif corruption == "selection":
        event["selected_units"] = ["xx", "xx"]
    elif corruption == "acceptance":
        event["accepted"] = not event["accepted"]
    elif corruption == "cardinality":
        event["expected_neighbors"] -= 1
    elif corruption == "batch_start":
        event["batches"][0]["first_neighbor"] = 1
    elif corruption == "batch_dimensions":
        event["batches"][0]["kernel_diagnostics"]["rolling_buffer_bytes"] += 8
    elif corruption == "batch_fallback":
        event["batches"][0]["kernel_diagnostics"]["log_domain_fallbacks"] = 1000
    elif corruption == "aggregate_fallback":
        full["kernel_fallbacks"] += 1
    elif corruption == "aggregate_neighbors":
        full["evaluated_neighbors"] += 1
    elif corruption == "certificate":
        full["best_is_certified_local_optimum"] = not full["best_is_certified_local_optimum"]
    elif corruption == "channel":
        full["channel"]["rows"][0]["emissions"][0]["glyphs"] = "yy"
    elif corruption == "boolean_channel":
        full["channel"]["initial"]["s0"] = True
    elif corruption == "premature_restart":
        full["trace"].remove(event)
    elif corruption == "final_score":
        full["score"]["log_likelihood"] -= 1
        full["score"]["data_bits"] = -full["score"]["log_likelihood"] / audit.math.log(2)
        full["score"]["total_bits"] = full["score"]["model_bits"] + full["score"]["data_bits"]
    with pytest.raises(ValueError):
        audit.validate_case(compact(full), full, context)


def test_every_payload_field_is_compared_between_compact_and_full():
    _, context, full = fixture()
    for key in full.keys() - {"trace"}:
        frozen = compact(full)
        frozen[key] = "inconsistent"
        with pytest.raises(ValueError, match="Compact/full payload mismatch"):
            audit.validate_case(frozen, full, context)


def test_subtolerance_minimum_has_no_borrowed_parent_certificate():
    config = UnitSearchConfig(restarts=1, max_sweeps=4, batch_size=3, max_units=6, improvement_tolerance_bits=1000.)
    _, context, full = fixture(config)
    result = audit.validate_case(compact(full), full, context)
    assert full["units"] != full["trace"][0]["units"]
    assert result["local_optima"] == 1 and not result["best_is_certified_local_optimum"]
    initial = full["trace"][0]
    full["units"], full["score"] = initial["units"], initial["score"]
    source, _, _ = fixture(config)
    full["channel"] = channel_from_units(source, CodingContext(**context), full["units"]).to_dict()
    with pytest.raises(ValueError, match="first best completed"):
        audit.validate_case(compact(full), full, context)


def test_partial_batch_trace_is_a_valid_prefix_but_cannot_claim_completion(monkeypatch):
    original = search_module.batch_log_likelihood
    now, calls = [0.], [0]

    def expire_after_finished_batch(*args, **kwargs):
        value = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            now[0] = 2.
        return value

    monkeypatch.setattr(search_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(search_module, "batch_log_likelihood", expire_after_finished_batch)
    config = UnitSearchConfig(restarts=3, max_sweeps=4, batch_size=3, max_units=6, max_seconds=1.)
    _, context, full = fixture(config)
    result = audit.validate_case(compact(full), full, context)
    assert result["evaluated_neighbors"] == 3 and result["completed_sweeps"] == 0
    event = sweep(full)
    event["complete"] = True
    with pytest.raises(ValueError, match="Completion/acceptance"):
        audit.validate_case(compact(full), full, context)


def write_panel(root, monkeypatch):
    def settings(index):
        return asdict(UnitSearchConfig(seed=100 + index, restarts=3, max_sweeps=6, batch_size=3, max_units=6))

    monkeypatch.setattr(audit, "registered_config", settings)
    for index, name in enumerate(audit.CASE_NAMES):
        _, context, full = fixture(UnitSearchConfig(**settings(index)))
        monkeypatch.setattr(audit, "CONTEXT", context)
        archive = gzip.compress(json.dumps(full, allow_nan=False).encode(), mtime=0)
        frozen = compact(full, name)
        frozen["full_output"].update(sha256=audit.digest(archive), bytes=len(archive))
        output = root / frozen["full_output"]["path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive)
        path = root / f"results/{audit.EXPERIMENT}/{name}_freeze.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frozen, allow_nan=False))
    campaign = {"experiment": audit.EXPERIMENT, "source_freeze": audit.SOURCE_FREEZE, "workers": 2,
                "status": "complete", "wall_seconds": 1., "cases": {
                    name: {"returncode": 0, "wall_seconds": .1, "log": f"outputs/{audit.EXPERIMENT}/{name}.log"}
                    for name in audit.CASE_NAMES}}
    (root / f"results/{audit.EXPERIMENT}/campaign.json").write_text(json.dumps(campaign))


def test_registered_panel_hashes_match_and_only_allowlisted_artifacts_are_read(tmp_path, monkeypatch):
    write_panel(tmp_path, monkeypatch)
    original = audit.Path.read_bytes
    reads = []

    def observed_read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(audit.Path, "read_bytes", observed_read)
    result = audit.summarize(tmp_path)
    assert result["status"] == "pass" and result["cases_verified"] == 4
    expected = {tmp_path / f"results/{audit.EXPERIMENT}/campaign.json", audit.Path(audit.__file__)}
    expected.update(tmp_path / f"results/{audit.EXPERIMENT}/{name}_freeze.json" for name in audit.CASE_NAMES)
    expected.update(tmp_path / f"outputs/{audit.EXPERIMENT}/{name}.json.gz" for name in audit.CASE_NAMES)
    assert set(reads) == expected
    assert not result["empirical_likelihood_replay_performed"]
    assert not result["source_hashes_authenticated_here"]
    for case in result["cases"].values():
        assert len(case["freeze_sha256"]) == len(case["full_output_sha256"]) == 64
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("corruption", ["hash", "size", "path", "gzip", "thread_limit", "source_hash", "config", "campaign_failure"])
def test_panel_corruption_fails_closed(tmp_path, monkeypatch, corruption):
    write_panel(tmp_path, monkeypatch)
    path = tmp_path / f"results/{audit.EXPERIMENT}/B-key1_freeze.json"
    frozen = json.loads(path.read_text())
    if corruption == "hash":
        frozen["full_output"]["sha256"] = "0" * 64
    elif corruption == "size":
        frozen["full_output"]["bytes"] += 1
    elif corruption == "path":
        frozen["full_output"]["path"] = "../source.json"
    elif corruption == "gzip":
        archive = b"not a gzip file"
        (tmp_path / frozen["full_output"]["path"]).write_bytes(archive)
        frozen["full_output"].update(sha256=audit.digest(archive), bytes=len(archive))
    elif corruption == "thread_limit":
        frozen["environment"]["numerical_thread_limits"]["OMP_NUM_THREADS"] = "8"
    elif corruption == "source_hash":
        frozen["source_hashes"][audit.SOURCE_PATH] = "0" * 64
    elif corruption == "config":
        frozen["config"]["seed"] += 1
    elif corruption == "campaign_failure":
        campaign_path = tmp_path / f"results/{audit.EXPERIMENT}/campaign.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["cases"]["B-key1"]["returncode"] = 1
        campaign_path.write_text(json.dumps(campaign))
    path.write_text(json.dumps(frozen))
    result = audit.summarize(tmp_path)
    assert result["status"] == "fail" and result["errors"]


def test_cli_guard_and_incomplete_panel_never_write_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    destination = tmp_path / "summary.json"
    with pytest.raises(SystemExit):
        audit.main(["--output", str(destination)])
    assert not destination.exists()
    with pytest.raises(SystemExit, match="incomplete"):
        audit.main(["--after-fitting", "--output", str(destination)])
    assert not destination.exists()
    write_panel(tmp_path, monkeypatch)
    (tmp_path / f"outputs/{audit.EXPERIMENT}/B-key2.json.gz").unlink()
    result = audit.summarize(tmp_path)
    assert result["status"] == "pending" and result["cases_verified"] == 0
    assert result["pending"] == ["B-key2"]


def test_cli_success_writes_only_requested_new_output(tmp_path, monkeypatch, capsys):
    write_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    audit.main(["--after-fitting", "--output", "summary.json"])
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["status"] == "pass"
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    with pytest.raises(FileExistsError):
        audit.main(["--after-fitting", "--output", "summary.json"])


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"score":NaN}', b'{"score":Infinity}'])
def test_ambiguous_or_nonfinite_json_is_rejected(raw):
    with pytest.raises(ValueError):
        audit.json_load(raw)
