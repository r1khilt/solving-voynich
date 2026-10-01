import copy
import gzip
import hashlib
import json
import subprocess

import numpy as np
import pytest
import torch

from scripts import run_joint_key_train001 as run
from scripts import audit_joint_key_train001 as audit
from scripts import run_blind_channel_dev004 as storage
from scripts import run_latin_source_model001 as resources
from scripts import benchmark_key_proposal_systems001 as bench
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig
from voynich.joint_key_training import EpisodeSampler, metadata_bytes, pack_episodes, validation
from voynich.recurrent_latin_source import ALPHABET


@pytest.fixture(autouse=True)
def restore_backend():
    initial = torch.backends.mha.get_fastpath_enabled()
    yield
    torch.backends.mha.set_fastpath_enabled(initial)


def paths(monkeypatch, tmp_path):
    for module in (run, audit, storage, resources):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (run, audit):
        monkeypatch.setattr(module, "OUT", tmp_path/"results")
        monkeypatch.setattr(module, "BULK", tmp_path/"outputs")
        monkeypatch.setattr(module, "require_frozen", lambda *args: None)
        monkeypatch.setattr(module, "limit_resources", lambda *args: None)
    return tmp_path/"results", tmp_path/"outputs"


def sources(monkeypatch):
    training, valid = [ALPHABET*30], [ALPHABET[::-1]*25]
    monkeypatch.setattr(run, "load_inputs", lambda dataset: (training, valid,
        {"dataset": dataset, "training_characters": sum(map(len, training)), "validation_characters": sum(map(len, valid))}))
    monkeypatch.setattr(run, "VALIDATION_COUNT", 2)


def test_prepare_audit_exact_source_episodes_hashes_exclusive_and_corruption(monkeypatch, tmp_path):
    out, _ = paths(monkeypatch, tmp_path)
    sources(monkeypatch)
    run.prepare("frozen")
    run.audit_inputs("frozen")
    manifest, texts, valid, episodes = run.load_prepared()
    assert texts != valid and len(episodes) == 2
    assert json.loads((out/"input-audit.json").read_text())["status"] == "PASS"
    with pytest.raises(FileExistsError):
        run.prepare("frozen")
    with (tmp_path/manifest["training"]["path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="identity"):
        run.load_prepared()


def cpu_transport(monkeypatch):
    # Initialize the optimizer's lazy framework imports before replacing a
    # cached backend function whose __wrapped__ attribute Dynamo inspects.
    torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    original = torch.Tensor.to

    def cpu(self, *args, **kwargs):
        args = tuple("cpu" if str(v) == "mps" else v for v in args)
        if str(kwargs.get("device")) == "mps":
            kwargs["device"] = "cpu"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name in ("synchronize", "driver_allocated_memory", "current_allocated_memory"):
        monkeypatch.setattr(torch.mps, name, lambda: 0)


def test_real_micro_fit_serialization_gzip_ledger_selected_cpu_replay_and_corruptions(monkeypatch, tmp_path):
    out, bulk = paths(monkeypatch, tmp_path)
    sources(monkeypatch)
    run.prepare("frozen")
    run.audit_inputs("frozen")
    cpu_transport(monkeypatch)
    config = KeyProposalConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1)
    count = sum(p.numel() for p in JointKeyProposal(config).parameters())
    for module in (run, bench):
        monkeypatch.setattr(module, "CONFIGS", {"small": config, "large": config})
    for module in (run, audit):
        monkeypatch.setattr(module, "PARAMETERS", {"small": count, "large": count})
    for module in (run, audit):
        monkeypatch.setattr(module, "STEPS", 3)
        monkeypatch.setattr(module, "CHECKPOINTS", (0, 1, 3))
    run.fit("small-72203", "frozen")
    value = json.loads((out/"small-72203.json").read_text())
    assert value["updates"] == 3 and value["episodes_seen"] == 12
    with gzip.open(bulk/"small-72203-episodes.jsonl.gz", "rt") as stream:
        lines = list(stream)
    assert len(lines) == 3 and [json.loads(line)["step"] for line in lines] == [1, 2, 3]
    _, texts, _, episodes = run.load_prepared()
    prefixes, counts = audit.ledger_account(value, texts, episodes)
    assert len(prefixes) == 3 and prefixes[3][0] == value["data_sha256"]
    assert counts["distinct_literal_dictionaries"] == 12
    selected = value["selected"]
    model = audit.checkpoint(selected["weights"], config.__dict__, selected["weights_sha256"], selected["step"], "frozen")
    score = storage.load_archive(selected["validation"])["score"]
    check = audit.replay_selected(model, score, episodes)
    assert check["whole_logq_max_delta"] <= .002
    broken = copy.deepcopy(score)
    broken["joint_logq"][0] += 1
    with pytest.raises(ValueError):
        audit.replay_selected(model, broken, episodes)
    with pytest.raises(FileExistsError):
        run.fit("small-72203", "frozen")
    processes = []
    for arm in run.ARMS:
        if arm != "small-72203":
            run.fit(arm, "frozen")
        log = bulk/(arm+".log")
        log.write_text("CPU mocked-transport integration fixture\n")
        processes.append({"arm": arm, "returncode": 0, "outer_timeout": False, "log": resources.artifact(log)})
    storage.save_new(out/"campaign.json", {"freeze": "frozen", "processes": processes, "paid_spend_usd": 0})
    script = tmp_path/"scripts/audit_joint_key_train001.py"
    script.parent.mkdir()
    script.write_text("fixture source artifact\n")
    audit.audit()
    complete = json.loads((out/"audit.json").read_text())
    assert len(complete["arms"]) == 4
    assert all(r["status"] == "PASS" for r in complete["arms"].values())
    with pytest.raises(FileExistsError):
        audit.audit()


def test_all_terminal_campaign_outcomes_preserved_without_retry(monkeypatch, tmp_path):
    out, bulk = paths(monkeypatch, tmp_path)
    bulk.mkdir()
    called = []

    def child(argv, **kwargs):
        called.append(argv)
        if len(called) == 2:
            raise subprocess.TimeoutExpired(argv, 21720)
        if len(called) == 3:
            raise OSError("fixture failure")
        return subprocess.CompletedProcess(argv, 0 if len(called) == 1 else 1)

    monkeypatch.setattr(run.subprocess, "run", child)
    run.campaign("frozen")
    value = json.loads((out/"campaign.json").read_text())
    assert len(called) == 4 and [r["arm"] for r in value["processes"]] == list(run.ARMS)
    assert [r["returncode"] for r in value["processes"]] == [0, None, None, 1]
    with pytest.raises(FileExistsError):
        run.campaign("frozen")
    assert len(called) == 4


def test_seeded_ledger_rejects_changed_source_and_sampling(monkeypatch, tmp_path):
    _, bulk = paths(monkeypatch, tmp_path)
    bulk.mkdir()
    source = EpisodeSampler([ALPHABET*30])
    episodes = [EpisodeSampler([ALPHABET[::-1]*30]).sample(np.random.default_rng(72401))]
    restricted = EpisodeSampler([ALPHABET*30], forbidden_raw=[], forbidden_canonical=[])
    metadata = [restricted.sample(np.random.default_rng(172206))[2] for _ in range(4)]
    encoded = metadata_bytes({"step": 1, "episodes": metadata})
    path = bulk/"ledger.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(encoded)
    value = {"seed": 72203, "ledger": resources.artifact(path), "updates": 1,
             "data_sha256": hashlib.sha256(encoded).hexdigest(), "source_characters": 0,
             "episodes_seen": 4, "assigned_key_rows": 92, "rejected_disjointness_keys": 0}
    with pytest.raises(ValueError, match="replay"):
        audit.ledger_account(value, [ALPHABET*30], episodes)
    assert source.records


def test_actual_greedy_replay_and_probability_tampering_detection():
    torch.manual_seed(72403)
    model = JointKeyProposal(KeyProposalConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1)).double().eval()
    episode = EpisodeSampler([ALPHABET*30]).sample(np.random.default_rng(72405))
    score = validation(model, [episode], device="cpu", batch=1)
    audit.replay_selected(model, score, [episode])
    broken = copy.deepcopy(score)
    broken["used_correct_rows"] += 1
    with pytest.raises(ValueError):
        audit.replay_selected(model, broken, [episode])
    records, _ = pack_episodes([episode])
    assert records.shape == (1, 4, 448)
