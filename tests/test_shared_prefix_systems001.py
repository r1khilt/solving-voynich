import json
import math
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_shared_prefix_systems001 as runner
from scripts import audit_shared_prefix_systems001 as auditor
from voynich.recurrent_latin_source import ALPHABET, RecurrentSource
from voynich.recurrent_shared_prefix import decode_shared_prefix
from voynich.recurrent_unit_beam import RecurrentProvider


def test_complete_artificial_inventory_and_known_generated_support_are_reproducible():
    items = runner.workloads()
    assert items == runner.workloads()
    assert [(i["letters"], i["key_count"], i["beam"]) for i in items] == list(runner.SHAPES)
    for item in items:
        assert len(set(item["keys"])) == item["key_count"]
        assert math.fsum(math.exp(w) for w in item["log_weights"]) == pytest.approx(1)
        table = dict(zip(ALPHABET, item["keys"][0], strict=True))
        assert ["".join(table[c] for c in text) for text in item["plain"]] == item["records"]
        assert all(len(key) == 23 and all(1 <= len(unit) <= 2 for unit in key) for key in item["keys"])


def test_variable_instrumentation_keeps_native_shapes_scores_and_provider_lifetimes():
    import gc
    import weakref
    import numpy as np
    model = RecurrentSource(alphabet="ab", embedding=3, width=4, layers=1)
    trace = []
    measured = runner.MeasuredRecurrentProvider(model, "cpu", trace.append)
    native = RecurrentProvider(model)
    a, astates = measured.advance([2], [None])
    b, bstates = native.advance([2], [None])
    assert np.array_equal(a, b)
    a, _ = measured.advance([0, 1, 0], astates*3)
    b, _ = native.advance([0, 1, 0], bstates*3)
    assert np.array_equal(a, b)
    assert [row["shape"] for row in trace] == [[1, 1], [3, 1]]
    assert measured.calls == 2 and measured.real_rows == 4
    reference = weakref.ref(measured)
    del measured
    gc.collect()
    assert reference() is None


def test_whole_history_numeric_reference_and_global_key_reencoding_are_not_just_saved_score_checks(monkeypatch):
    monkeypatch.setattr(runner, "SHAPES", ((2, 1, 16),))
    item = runner.workloads()[0]
    torch.manual_seed(71901)
    model = RecurrentSource(embedding=5, width=7, layers=2)
    value = decode_shared_prefix(RecurrentProvider(model), item["keys"], item["records"], 1/225,
                                 log_weights=item["log_weights"], beam_width=16)
    assert runner.literal_checks(model, item, value)["maximum_delta"] < 2e-6
    with pytest.raises(ValueError, match="score"):
        runner.literal_checks(model, item, {**value, "joint_log_probability": value["joint_log_probability"]+1})
    with pytest.raises(ValueError, match="literal"):
        runner.literal_checks(model, item, {**value, "compatible_key_indices": ()})


def test_parent_keeps_every_failed_and_timed_out_arm_and_never_repeats(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        arm = command[command.index("--arm")+1]
        calls.append(arm)
        assert kwargs["timeout"] == 615 and kwargs["check"] is False
        if arm == "fixed8-host":
            raise subprocess.TimeoutExpired(command, 615)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.campaign("frozen")
    value = json.loads((runner.OUT/"campaign.json").read_text())
    assert calls == list(runner.ARMS)
    assert [p["returncode"] for p in value["processes"]] == [1, None]
    assert [p["outer_timeout"] for p in value["processes"]] == [False, True]
    with pytest.raises(FileExistsError):
        runner.campaign("frozen")
    assert calls == list(runner.ARMS)


def configure(tmp_path, monkeypatch):
    def save(path, value, compressed=False):
        import gzip
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode()
        with path.open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return artifact(path)
    def artifact(path):
        import hashlib
        raw = path.read_bytes()
        return {"path": str(path.relative_to(tmp_path)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    def archive(spec):
        import gzip
        assert artifact(tmp_path/spec["path"]) == spec
        return json.loads(gzip.decompress((tmp_path/spec["path"]).read_bytes()))
    for module in (runner, auditor):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "OUT", tmp_path/"results")
        monkeypatch.setattr(module, "BULK", tmp_path/"outputs", raising=False)
        monkeypatch.setattr(module, "artifact", artifact)
        monkeypatch.setattr(module, "save_new", save)
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
    monkeypatch.setattr(auditor, "load_archive", archive)
    monkeypatch.setattr(auditor, "limit_resources", lambda *_: None)
    return save, artifact


def make_auditable_campaign(tmp_path, monkeypatch):
    save, artifact = configure(tmp_path, monkeypatch)
    source = tmp_path/"scripts/audit_shared_prefix_systems001.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("artificial auditor source fixture\n")
    monkeypatch.setattr(runner, "SHAPES", ((2, 1, 2), (2, 2, 2), (2, 3, 2)))
    items = runner.workloads()
    save(auditor.OUT/"campaign-started.json", {"freeze": "frozen", "arms": runner.ARMS})
    processes = []
    for arm in runner.ARMS:
        inputs = save(auditor.BULK/f"{arm}-inputs.json.gz", {"items": items}, compressed=True)
        rows, traces = [], []
        for seed in runner.SEEDS:
            for index, item in enumerate(items):
                compatible = []
                for k, key in enumerate(item["keys"]):
                    table = dict(zip(ALPHABET, key, strict=True))
                    if ["".join(table[c] for c in text) for text in item["plain"]] == item["records"]:
                        compatible.append(k)
                reading = {"plaintexts": item["plain"], "compatible_key_indices": compatible, "joint_log_probability": -10,
                           "support_empty": False, "interval_certificate": False, "exact_evidence_computed": False,
                           "expanded_prefixes": 5, "distinct_source_prefixes": 4, "source_cache_hits": 1,
                           "channel_cells": item["key_count"]*sum(len(r)+1 for r in item["records"])}
                checks = {"maximum_delta": 0, "full_sequence_cpu_float64_score": -10}
                spec = save(auditor.BULK/f"{arm}-{seed}-{index}.json.gz", {"reading": reading, "checks": checks}, compressed=True)
                rows.append({"seed": seed, "item": index, "parameters": 7_405_079, "checks": checks, "reading": spec,
                             "key_count": item["key_count"], "beam": item["beam"], "initial_weights_sha256": str(seed), "wall_seconds": 1})
                traces.append({"stage": "transition", "seed": seed, "item": index, "calls": 1, "real_rows": 4,
                               "padded_rows": 4 if arm == "fixed8-host" else 0, "shape": [8 if arm == "fixed8-host" else 4, 1],
                               "driver_bytes": 100, "tensor_bytes": 20, "peak_rss_bytes": 200})
        trace = auditor.BULK/f"{arm}-trace.jsonl"
        trace.write_text("\n".join(json.dumps(row) for row in traces)+"\n")
        save(auditor.OUT/f"{arm}.json", {"arm": arm, "freeze": "frozen", "status": "PASS", "inputs": inputs,
                                      "trace": artifact(trace), "completed": rows, "sampled_driver_peak": 100, "sampled_tensor_peak": 20})
        log = auditor.BULK/f"{arm}.log"
        log.write_text("terminal\n")
        processes.append({"arm": arm, "returncode": 0, "outer_timeout": False, "log": artifact(log)})
    save(auditor.OUT/"campaign.json", {"freeze": "frozen", "processes": processes, "paid_spend_usd": 0,
                                    "only_artificial_inputs_and_untrained_models": True})


def test_all_workloads_and_matched_arms_accounted(tmp_path, monkeypatch):
    make_auditable_campaign(tmp_path, monkeypatch)
    auditor.audit()
    result = json.loads((auditor.OUT/"audit.json").read_text())
    assert result["status"] == "PASS" and len(result["comparisons"]) == 6
    assert all(row["completed_workloads"] == 6 for row in result["arms"].values())
    with pytest.raises(FileExistsError):
        auditor.audit()


@pytest.mark.parametrize("fault", ["missing", "parameters", "initialization", "process", "archive"])
def test_audit_rejects_corrupted_workload_process_and_archive_claims(tmp_path, monkeypatch, fault):
    make_auditable_campaign(tmp_path, monkeypatch)
    path = auditor.OUT/"fixed8-host.json"
    value = json.loads(path.read_text())
    if fault == "missing":
        value["completed"].pop()
    elif fault == "parameters":
        value["completed"][0]["parameters"] -= 1
    elif fault == "initialization":
        value["completed"][0]["initial_weights_sha256"] = "wrong"
    elif fault == "archive":
        value["trace"]["sha256"] = "wrong"
    else:
        campaign_path = auditor.OUT/"campaign.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["processes"][1]["returncode"] = -9
        campaign_path.write_text(json.dumps(campaign))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        auditor.audit()


@pytest.mark.parametrize("driver,host", [(2*1024**3+1, 1), (1, 4*1024**3+1)])
def test_both_memory_domains_enforced(driver, host):
    with pytest.raises(MemoryError):
        runner.check_memory({"driver_bytes": driver, "peak_rss_bytes": host})
