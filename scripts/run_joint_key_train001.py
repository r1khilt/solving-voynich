"""Finite paired conditional-key training; no prior cipher panel access."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.benchmark_key_proposal_systems001 import CONFIGS, PARAMETERS, weights_digest
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, CORPUS, load_inputs, artifact, limit_resources
from voynich.joint_key_proposal import JointKeyProposal
from voynich.joint_key_training import (EpisodeSampler, MAX_LENGTH, dictionary_code, metadata_bytes,
    pack_episodes, schedule, validation)

EXP = "JOINT-KEY-TRAIN-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
SEEDS = (72203, 72209)
ARMS = ("small-72203", "large-72203", "small-72209", "large-72209")
VALIDATION_SEED, VALIDATION_COUNT = 72221, 64
STEPS, BATCH = 20_000, 4
CHECKPOINTS = (0, 200, 1000, 4000, 10000, 20000)
BASE_PATHS = ["src/voynich/joint_key_proposal.py", "src/voynich/joint_key_training.py",
    "src/voynich/recurrent_latin_source.py", "scripts/run_joint_key_train001.py",
    "scripts/audit_joint_key_train001.py", "tests/test_joint_key_training.py", "tests/test_joint_key_train001.py",
    "docs/experiments/JOINT-KEY-TRAIN-001.md", "docs/research/joint-key-training-design-2026-09-30.md",
    "scripts/benchmark_key_proposal_systems001.py", "scripts/run_latin_source_model001.py",
    "scripts/run_blind_channel_dev001.py", "scripts/run_blind_channel_dev004.py", "uv.lock", CORPUS,
    "results/LATIN-SOURCE-001/corpus_audit.json", "results/LATIN-SOURCE-001/overlap_audit.json",
    "results/KEY-PROPOSAL-SYSTEMS-001/audit.json", "results/KEY-PROPOSAL-SYSTEMS-001/publication-accounting.json"]
INPUT_PATHS = [f"results/{EXP}/{name}.json" for name in ("inputs", "prepare", "input-audit")]


def prepared_payload():
    training, original_validation, identity = load_inputs("large")
    valid = [r for r in original_validation if len(r) >= MAX_LENGTH]
    sampler, rng = EpisodeSampler(valid), np.random.default_rng(VALIDATION_SEED)
    episodes = [sampler.sample(rng) for _ in range(VALIDATION_COUNT)]
    metadata = [e[2] for e in episodes]
    if (len({dictionary_code(e["raw_indices"]) for e in metadata}) != VALIDATION_COUNT
            or len({dictionary_code(e["canonical_key"]) for e in metadata}) != VALIDATION_COUNT):
        raise ValueError("Validation dictionary collision; no silent redraw")
    identity = {**identity, "validation_eligible_characters": sum(map(len, valid)),
                "validation_eligible_segments": len(valid), "validation_minimum_segment": MAX_LENGTH,
                "validation_eligible_text_sha256": hashlib.sha256("\n".join(valid).encode()).hexdigest()}
    return training, valid, metadata, identity


def prepare(freeze):
    require_frozen(freeze, BASE_PATHS)
    save_new(OUT/"prepare-started.json", {"freeze": freeze, "start_unix": time.time()})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        training, valid, metadata, identity = prepared_payload()
        train_spec = save_new(BULK/"training.json.gz", {"texts": training, "role": "training_pool"}, compressed=True)
        valid_spec = save_new(BULK/"validation-source.json.gz", {"texts": valid, "role": "source_validation"}, compressed=True)
        keys_spec = save_new(BULK/"validation-episodes.json.gz", {"metadata": metadata}, compressed=True)
        if resource_report(wall, cpu)["peak_rss_bytes"] > 4*1024**3:
            raise MemoryError("Preparation 4GiB host guard")
        save_new(OUT/"inputs.json", {"freeze": freeze, "identity": identity,
            "training": train_spec, "validation_source": valid_spec, "validation_episodes": keys_spec,
            "validation_seed": VALIDATION_SEED, "validation_count": VALIDATION_COUNT,
            "alphabet": "abcdefghiklmnopqrstuxyz", "glyphs": "ABCDEF",
            "distinct_records_per_episode": 2, "fixed_shape_records": 4})
        save_new(OUT/"prepare.json", {"status": "PASS", "freeze": freeze,
                 "inputs": artifact(OUT/"inputs.json"), "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


def load_prepared():
    manifest = json.loads((OUT/"inputs.json").read_text())
    training = load_archive(manifest["training"])
    valid = load_archive(manifest["validation_source"])
    metadata = load_archive(manifest["validation_episodes"])["metadata"]
    if training["role"] != "training_pool" or valid["role"] != "source_validation":
        raise ValueError("Prepared source roles differ")
    sampler = EpisodeSampler(valid["texts"])
    episodes = [sampler.make(e["windows"], e["raw_indices"]) for e in metadata]
    if [e[2] for e in episodes] != metadata:
        raise ValueError("Prepared validation episode replay differs")
    return manifest, training["texts"], valid["texts"], episodes


def audit_inputs(freeze):
    require_frozen(freeze, BASE_PATHS)
    save_new(OUT/"input-audit-started.json", {"freeze": freeze, "start_unix": time.time()})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest, training, valid, episodes = load_prepared()
        expected_train, expected_valid, metadata, identity = prepared_payload()
        if (manifest["freeze"] != freeze or training != expected_train or valid != expected_valid
                or metadata != [e[2] for e in episodes] or manifest["identity"] != identity):
            raise ValueError("Original-author/source-window/seed replay differs")
        if resource_report(wall, cpu)["peak_rss_bytes"] > 4*1024**3:
            raise MemoryError("Input audit 4GiB host guard")
        save_new(OUT/"input-audit.json", {"status": "PASS", "freeze": freeze, "inputs": artifact(OUT/"inputs.json"),
            "scope": "allowed source roles, exact original-source text, full episode windows/keys/canonicalization",
            "independent_agent_review": False, "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


def guard(row):
    if row["driver_bytes"] > 16*1024**3 or row["peak_rss_bytes"] > 8*1024**3:
        raise MemoryError("Sampled 16GiB Metal driver / 8GiB host guard")


def short_check(model, episode):
    """Explicit CPU double check and two/four-record equality, all decoder rows."""
    with torch.inference_mode():
        records, keys = pack_episodes([episode])
        records = records[:, :, :16]
        four = model(records.to("mps"), keys.to("mps")).cpu().double()
        two = model(records[:, :2].to("mps"), keys.to("mps")).cpu().double()
        reference = copy.deepcopy(model).cpu().double().eval()
        expected = reference(records, keys)
        cpu_two = reference(records[:, :2], keys)
        delta, duplicated = float((four-expected).abs().max()), float((four-two).abs().max())
        cpu_delta = float((expected-cpu_two).abs().max())
        if (not torch.isfinite(four).all() or not torch.isfinite(two).all()
                or delta > 5e-5 or duplicated > 5e-5 or cpu_delta > 1e-10):
            raise ValueError("Short device reference or record-duplication invariant failed")
    return {"mps_logits": four.tolist(), "mps_cpu_max_delta": delta,
            "mps_two_four_delta": duplicated, "cpu_two_four_delta": cpu_delta}


def fit(arm, freeze):
    require_frozen(freeze, [*BASE_PATHS, *INPUT_PATHS])
    if arm not in ARMS or not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("Registered arm/MPS/disabled CPU fallback required")
    save_new(OUT/(arm+"-started.json"), {"freeze": freeze, "start_unix": time.time(), "torch": str(torch.__version__),
             "numpy": np.__version__, "cpu_threads": 2, "cpu_mha_fastpath": False})
    scale, seed = arm.split("-")
    seed = int(seed)
    limit_resources(5400 if scale == "small" else 21600, 4500 if scale == "small" else 18000)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    BULK.mkdir(parents=True, exist_ok=True)
    checkpoint_rows, last_step, validation_keys, source_characters, episodes_seen = [], 0, None, 0, 0
    data_digest = hashlib.sha256()
    ledger_path, trace_path = BULK/(arm+"-episodes.jsonl.gz"), BULK/(arm+"-trace.jsonl")
    with ledger_path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as ledger, trace_path.open("x") as trace:
        def sample(info):
            torch.mps.synchronize()
            row = {**info, "driver_bytes": torch.mps.driver_allocated_memory(),
                   "tensor_bytes": torch.mps.current_allocated_memory(), **resource_report(wall, cpu)}
            trace.write(json.dumps(row, allow_nan=False)+"\n")
            trace.flush()
            guard(row)
        try:
            manifest, texts, _, validation_keys = load_prepared()
            input_audit = json.loads((OUT/"input-audit.json").read_text())
            if input_audit["status"] != "PASS" or input_audit["inputs"] != artifact(OUT/"inputs.json"):
                raise ValueError("Complete input audit required")
            raw_keys = [dictionary_code(e[2]["raw_indices"]) for e in validation_keys]
            canonical_keys = [dictionary_code(e[1]) for e in validation_keys]
            sampler = EpisodeSampler(texts, forbidden_raw=raw_keys, forbidden_canonical=canonical_keys)
            rng = np.random.default_rng(seed+100003)
            torch.manual_seed(seed)
            model = JointKeyProposal(CONFIGS[scale]).to("mps")
            if sum(p.numel() for p in model.parameters()) != PARAMETERS[scale]:
                raise ValueError("Instantiated parameter count differs")
            initial_digest = weights_digest(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, foreach=False)

            def snapshot(step):
                model.eval()
                check = short_check(model, validation_keys[0])
                sample({"stage": "short_reference", "step": step})
                score = validation(model, validation_keys, device="mps", batch=BATCH,
                                   progress=lambda info: sample({"stage": "validation", "step": step, **info}))
                path = BULK/f"{arm}-step{step}.pt"
                with path.open("xb") as handle:
                    torch.save({"config": model.config, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step, "seed": seed, "freeze": freeze, "inputs": artifact(OUT/"inputs.json")}, handle)
                score_spec = save_new(BULK/f"{arm}-step{step}-validation.json.gz", {"score": score, "check": check}, compressed=True)
                row = {"step": step, "weights": artifact(path), "weights_sha256": weights_digest(model),
                       "validation": score_spec, "mean_nats_per_row": score["mean_nats_per_row"],
                       "used_correct_rows": score["used_correct_rows"], "used_rows": score["used_rows"]}
                checkpoint_rows.append(row)
                save_new(OUT/f"{arm}-step{step}.json", row)
                sample({"stage": "checkpoint", "step": step})
                print(json.dumps({"arm": arm, **{k: row[k] for k in ("step", "mean_nats_per_row", "used_correct_rows", "used_rows")}}), flush=True)
                model.train()

            snapshot(0)
            for step in range(1, STEPS+1):
                began = time.monotonic()
                episodes = [sampler.sample(rng) for _ in range(BATCH)]
                metadata = [e[2] for e in episodes]
                encoded = metadata_bytes({"step": step, "episodes": metadata})
                ledger.write(encoded)
                data_digest.update(encoded)
                episodes_seen += len(episodes)
                source_characters += sum(s["length"] for e in metadata for s in e["windows"])
                records, targets = pack_episodes(episodes)
                optimizer.zero_grad(set_to_none=True)
                lr = schedule(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                logits = model(records.to("mps"), targets.to("mps"))
                loss = F.cross_entropy(logits.reshape(-1, 42), targets.to("mps").reshape(-1))
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite full-row training loss")
                loss.backward()
                if step in (1, STEPS) and any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise ValueError("Missing/nonfinite training gradients")
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
                if not math.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                torch.mps.synchronize()
                last_step = step
                trace.write(json.dumps({"stage": "optimizer", "step": step, "loss_nats_per_row": float(loss.item()),
                    "preclip_norm": norm, "lr": lr, "episodes_seen": episodes_seen,
                    "source_characters": source_characters, "data_sha256": data_digest.hexdigest(),
                    "step_wall_seconds": time.monotonic()-began}, allow_nan=False)+"\n")
                if step % 100 == 0:
                    ledger.flush()
                    sample({"stage": "resource", "step": step})
                if step in CHECKPOINTS:
                    snapshot(step)
            selected = min(checkpoint_rows, key=lambda r: (r["mean_nats_per_row"], r["step"]))
            ledger.close()
            raw.flush()
            trace.flush()
            save_new(OUT/(arm+".json"), {"status": "PASS", "arm": arm, "seed": seed, "freeze": freeze,
                "inputs": artifact(OUT/"inputs.json"), "parameters": PARAMETERS[scale], "updates": last_step,
                "episodes_seen": episodes_seen, "source_characters": source_characters,
                "assigned_key_rows": episodes_seen*23, "data_sha256": data_digest.hexdigest(),
                "rejected_disjointness_keys": sampler.rejected_keys, "initial_weights_sha256": initial_digest,
                "checkpoints": checkpoint_rows, "selected": selected, "trace": artifact(trace_path),
                "ledger": artifact(ledger_path), "corpus_identity": manifest["identity"],
                "reserved_authors_or_prior_cipher_panel_opened": False, "resources": resource_report(wall, cpu)})
        except Exception as error:
            signal.alarm(0)
            ledger.close()
            raw.flush()
            trace.flush()
            save_new(OUT/(arm+"-failure.json"), {"arm": arm, "freeze": freeze, "error": str(error),
                "type": type(error).__name__, "completed_updates": last_step, "generated_episodes": episodes_seen,
                "source_characters": source_characters, "checkpoints": checkpoint_rows,
                "trace": artifact(trace_path), "ledger": artifact(ledger_path), "resources": resource_report(wall, cpu)})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    require_frozen(freeze, [*BASE_PATHS, *INPUT_PATHS])
    save_new(OUT/"campaign-started.json", {"freeze": freeze, "arms": ARMS, "start_unix": time.time()})
    wall, before = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
    processes = []
    for arm in ARMS:
        began = time.monotonic()
        with (BULK/(arm+".log")).open("x") as log:
            try:
                process = subprocess.run([sys.executable, "-u", __file__, "fit", "--arm", arm, "--freeze", freeze],
                    cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                    timeout=(5400 if arm.startswith("small") else 21600)+120, check=False)
                code, timeout = process.returncode, False
            except subprocess.TimeoutExpired:
                code, timeout = None, True
            except OSError as error:
                code, timeout = None, False
                log.write(str(error)+"\n")
        row = {"arm": arm, "returncode": code, "outer_timeout": timeout,
               "wall_seconds": time.monotonic()-began, "log": artifact(BULK/(arm+".log"))}
        processes.append(row)
        print(json.dumps(row), flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/"campaign.json", {"freeze": freeze, "processes": processes, "wall_seconds": time.monotonic()-wall,
        "all_children_cpu_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        "paid_spend_usd": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "audit-inputs", "fit", "campaign"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    {"prepare": prepare, "audit-inputs": audit_inputs, "campaign": campaign}.get(args.mode,
        lambda freeze: fit(args.arm, freeze))(args.freeze)
