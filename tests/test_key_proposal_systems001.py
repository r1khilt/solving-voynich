import copy
import json
import subprocess
from dataclasses import asdict

import pytest
import torch

from scripts import audit_key_proposal_systems001 as audit
from scripts import benchmark_key_proposal_systems001 as bench
from scripts import run_blind_channel_dev004 as storage
from scripts import run_latin_source_model001 as resources
from scripts.run_blind_channel_dev004 import save_new
from scripts.run_latin_source_model001 import artifact
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig


@pytest.fixture(autouse=True)
def restore_attention_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def test_artificial_generator_complete_sizes_shared_dictionary_and_determinism():
    rows = bench.artificial_episodes()
    assert rows == bench.artificial_episodes() and len(rows) == 4
    for row in rows:
        assert len(row["literal_units"]) == 23
        assert len(row["observed_records"]) == 4
        for plain, observed in zip(row["source_indices"], row["observed_records"], strict=True):
            assert len(plain) == 224 and 224 <= len(observed) <= 448
            assert "".join(row["literal_units"][c] for c in plain) == observed
    one, keys = bench.pack(rows, 1)
    four, more = bench.pack(rows, 4)
    assert one.shape == (1, 4, 448) and keys.shape == (1, 23)
    assert torch.equal(one[0], four[0]) and torch.equal(keys[0], more[0])


def test_actual_registered_parameter_counts():
    # Meta allocation counts the actual constructor without allocating ~400MB.
    for scale, config in bench.CONFIGS.items():
        with torch.device("meta"):
            model = JointKeyProposal(config)
        assert sum(p.numel() for p in model.parameters()) == bench.PARAMETERS[scale]


def paths(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(resources, "ROOT", tmp_path)
    for module in (bench, audit):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "OUT", tmp_path/"results")
        monkeypatch.setattr(module, "BULK", tmp_path/"outputs")
        monkeypatch.setattr(module, "require_frozen", lambda *args: None)
        monkeypatch.setattr(module, "limit_resources", lambda *args: None)
    return tmp_path/"results", tmp_path/"outputs"


def test_all_campaign_outcomes_retained_no_retry_and_exclusive_start(monkeypatch, tmp_path):
    out, _ = paths(monkeypatch, tmp_path)
    calls = []

    def child(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(argv, 315)
        if len(calls) == 3:
            raise OSError("deliberate missing subprocess")
        return subprocess.CompletedProcess(argv, 0 if len(calls) == 1 else 1)

    monkeypatch.setattr(bench.subprocess, "run", child)
    bench.campaign("frozen")
    result = json.loads((out/"campaign.json").read_text())
    assert len(calls) == 4 and [r["arm"] for r in result["processes"]] == list(bench.ARMS)
    assert [r["returncode"] for r in result["processes"]] == [0, None, None, 1]
    assert result["processes"][1]["outer_timeout"]
    with pytest.raises(FileExistsError):
        bench.campaign("frozen")
    assert len(calls) == 4


def cpu_arm(monkeypatch, tmp_path):
    out, bulk = paths(monkeypatch, tmp_path)
    config = KeyProposalConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1)
    monkeypatch.setattr(bench, "CONFIGS", {"small": config})
    monkeypatch.setattr(audit, "CONFIGS", {"small": config})
    count = sum(p.numel() for p in JointKeyProposal(config).parameters())
    monkeypatch.setattr(bench, "PARAMETERS", {"small": count})
    monkeypatch.setattr(audit, "PARAMETERS", {"small": count})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name in ("synchronize", "driver_allocated_memory", "current_allocated_memory"):
        monkeypatch.setattr(torch.mps, name, lambda: 0)
    original = torch.Tensor.to

    def to_cpu(self, *args, **kwargs):
        args = tuple("cpu" if str(v) == "mps" else v for v in args)
        if str(kwargs.get("device")) == "mps":
            kwargs["device"] = "cpu"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        bench.run_arm("small-b1", "frozen")
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    process = {"arm": "small-b1", "returncode": 0, "outer_timeout": False}
    log = bulk/"small-b1.log"
    log.write_text("CPU-only mocked transport integration test\n")
    process["log"] = artifact(log)
    return out, bulk, process


def test_actual_optimizer_serialization_checkpoint_and_complete_audit_cpu_transport(monkeypatch, tmp_path):
    out, bulk, process = cpu_arm(monkeypatch, tmp_path)
    torch.backends.mha.set_fastpath_enabled(False)
    result = audit.inventory(process, "frozen")
    assert result["status"] == "PASS" and result["optimizer_updates"] == 15
    value = json.loads((out/"small-b1.json").read_text())
    assert len(value["proposed_keys"]) == 9
    assert len(value["optimizer_steps"]) == 15
    assert len([s for s in value["optimizer_steps"] if s["timed"]]) == 12
    assert result["replay"]["full_proposal_delta"] <= .002
    pristine = copy.deepcopy(value)
    for field, replacement in (("parameters", -1), ("proposed_logq", [1]*9),
                               ("mean_timed_step_seconds", -1), ("batch", 4)):
        altered = copy.deepcopy(pristine)
        altered[field] = replacement
        (out/"small-b1.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            audit.inventory(process, "frozen")
    (out/"small-b1.json").write_text(json.dumps(pristine))
    initial = torch.load(bulk/"small-b1-initial.pt", weights_only=True)
    assert initial["config"] == asdict(bench.CONFIGS["small"])
    with (bulk/"small-b1-initial.pt").open("ab") as stream:
        stream.write(b"corrupt checksum")
    with pytest.raises(ValueError, match="checksum"):
        audit.inventory(process, "frozen")


def test_failed_partial_inventory_preserves_orphan_checkpoint(monkeypatch, tmp_path):
    out, bulk = paths(monkeypatch, tmp_path)
    bulk.mkdir()
    (bulk/"small-b1-initial.pt").write_bytes(b"partial checkpoint")
    log = bulk/"small-b1.log"
    log.write_text("terminal child failure\n")
    process = {"arm": "small-b1", "returncode": 1, "outer_timeout": False, "log": artifact(log)}
    result = audit.inventory(process, "frozen")
    assert result["status"] == "missing_after_terminal_failure"
    assert result["retained_partial_artifacts"] == [artifact(bulk/"small-b1-initial.pt")]
    save_new(out/"small-b1.json", {"status": "PASS"})
    with pytest.raises(ValueError, match="success"):
        audit.inventory(process, "frozen")


def test_sampled_resource_guards():
    bench.check_memory({"driver_bytes": 16*1024**3, "peak_rss_bytes": 8*1024**3})
    for driver, host in ((16*1024**3+1, 1), (1, 8*1024**3+1)):
        with pytest.raises(MemoryError):
            bench.check_memory({"driver_bytes": driver, "peak_rss_bytes": host})


def test_complete_audit_retains_all_terminal_missing_arms_and_cannot_repeat(monkeypatch, tmp_path):
    out, bulk = paths(monkeypatch, tmp_path)
    bulk.mkdir()
    script = tmp_path/"scripts/audit_key_proposal_systems001.py"
    script.parent.mkdir()
    script.write_text("test-only source artifact\n")
    rows = []
    for arm in bench.ARMS:
        log = bulk/f"{arm}.log"
        log.write_text("deliberate child failure\n")
        rows.append({"arm": arm, "returncode": 1, "outer_timeout": False, "log": artifact(log)})
    save_new(out/"campaign-started.json", {"freeze": "frozen", "arms": list(bench.ARMS)})
    save_new(out/"campaign.json", {"freeze": "frozen", "processes": rows,
             "language_training_or_cipher_panel_files_opened": False, "paid_spend_usd": 0})
    audit.audit()
    value = json.loads((out/"audit.json").read_text())
    assert len(value["arms"]) == 4
    assert value["random_optimizer_updates"] is None
    assert all(r["status"] == "missing_after_terminal_failure" for r in value["arms"].values())
    with pytest.raises(FileExistsError):
        audit.audit()
