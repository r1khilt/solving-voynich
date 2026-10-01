"""Bounded random-only whole-key network optimizer and proposal cost measurement."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig, canonicalize_records, canonical_key_indices, unit_pool

EXP = "KEY-PROPOSAL-SYSTEMS-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
ARMS = ("small-b1", "small-b4", "large-b1", "large-b4")
CONFIGS = {"small": KeyProposalConfig(width=256, heads=8, encoder_layers=4, decoder_layers=2),
           "large": KeyProposalConfig()}
PARAMETERS = {"small": 5_423_146, "large": 94_981_674}
MODEL_SEED, DATA_SEED, SAMPLE_SEED = 72131, 72121, 72139
STEPS, WARMUP = 15, 3
PATHS = ["src/voynich/joint_key_proposal.py", "scripts/benchmark_key_proposal_systems001.py",
         "scripts/audit_key_proposal_systems001.py", "tests/test_joint_key_proposal.py",
         "tests/test_key_proposal_systems001.py", "docs/experiments/KEY-PROPOSAL-SYSTEMS-001.md",
         "docs/research/joint-key-proposal-implementation-2026-09-30.md", "uv.lock",
         "scripts/run_blind_channel_dev001.py", "scripts/run_blind_channel_dev004.py",
         "scripts/run_latin_source_model001.py"]


def artificial_episodes():
    rng = np.random.default_rng(DATA_SEED)
    pool = unit_pool(6)
    rows = []
    for _ in range(4):
        indices = rng.integers(len(pool), size=23).tolist()
        units = tuple("".join("ABCDEF"[i] for i in pool[u]) for u in indices)
        plain = [rng.integers(23, size=224).tolist() for _ in range(4)]
        records = ["".join(units[c] for c in text) for text in plain]
        canonical = canonicalize_records(records, "ABCDEF")
        rows.append({"source_indices": plain, "literal_units": units, "observed_records": records,
                     "canonical_records": canonical.records, "canonical_key": canonical_key_indices(units, canonical),
                     "observed_symbols": canonical.observed, "unseen_symbols": canonical.unseen})
    return rows


def pack(episodes, batch):
    records = torch.full((batch, 4, 448), 6, dtype=torch.long)
    for i, episode in enumerate(episodes[:batch]):
        for j, record in enumerate(episode["canonical_records"]):
            records[i, j, :len(record)] = torch.tensor(record)
    keys = torch.tensor([r["canonical_key"] for r in episodes[:batch]])
    return records, keys


def weights_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save({"config": model.config, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, stream)
    return artifact(path)


def check_memory(row):
    if row["driver_bytes"] > 16*1024**3 or row["peak_rss_bytes"] > 8*1024**3:
        raise MemoryError("Sampled 16GiB driver / 8GiB host guard")


def short_reference(model, records, keys):
    with torch.inference_mode():
        model.eval()
        actual = model(records[:1, :, :16].to("mps"), keys[:1].to("mps")).cpu().double()
        reference = copy.deepcopy(model).to("cpu").double().eval()
        expected = reference(records[:1, :, :16], keys[:1])
        delta = float((actual-expected).abs().max())
    if delta > 5e-5:
        raise ValueError("Short complete decoder CPUfloat64 logit reference differs")
    return {"maximum_logit_delta": delta, "mps_logits": actual.tolist(), "record_glyphs": 16,
            "rows": 23, "units": 42}


def run_arm(arm, freeze):
    require_frozen(freeze, PATHS)
    if arm not in ARMS or not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("Registered arm, MPS and explicitly disabled CPU fallback required")
    save_new(OUT/f"{arm}-started.json", {"freeze": freeze, "start_unix": time.time(), "torch": str(torch.__version__),
             "numpy": np.__version__, "cpu_threads": 2, "model_seed": MODEL_SEED,
             "data_seed": DATA_SEED, "sample_seed": SAMPLE_SEED, "cpu_mha_fastpath": False})
    limit_resources(300, 240)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    scale, batch = arm.split("-b")
    batch = int(batch)
    episodes = artificial_episodes()
    records, keys = pack(episodes, batch)
    input_spec = save_new(BULK/f"{arm}-inputs.json.gz", {"seed": DATA_SEED, "episodes": episodes}, compressed=True)
    trace, steps = [], []
    BULK.mkdir(parents=True, exist_ok=True)
    trace_path = BULK/f"{arm}-trace.jsonl"
    with trace_path.open("x") as stream:
        def sample(info):
            torch.mps.synchronize()
            row = {**info, "driver_bytes": torch.mps.driver_allocated_memory(),
                   "tensor_bytes": torch.mps.current_allocated_memory(), **resource_report(wall, cpu)}
            trace.append(row)
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            check_memory(row)
        try:
            torch.manual_seed(MODEL_SEED)
            model = JointKeyProposal(CONFIGS[scale]).to("mps")
            count = sum(p.numel() for p in model.parameters())
            if count != PARAMETERS[scale]:
                raise ValueError("Registered exact parameter count differs")
            initial_digest = weights_digest(model)
            initial = save_model(model, BULK/f"{arm}-initial.pt")
            sample({"stage": "initialized"})
            initial_check = short_reference(model, records, keys)
            sample({"stage": "initial_reference"})
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, foreach=False)
            inputs, targets = records.to("mps"), keys.to("mps")
            for step in range(STEPS):
                began = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, targets)
                loss = F.cross_entropy(logits.reshape(-1, 42), targets.reshape(-1))
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite random-only full-row loss")
                loss.backward()
                if step in (0, STEPS-1) and any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise ValueError("Missing/nonfinite model gradient")
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
                if not math.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                torch.mps.synchronize()
                elapsed = time.monotonic()-began
                steps.append({"step": step+1, "loss_nats_per_row": float(loss.item()), "preclip_gradient_norm": norm,
                              "wall_seconds": elapsed, "timed": step >= WARMUP})
                sample({"stage": "optimizer", "step": step+1})
            model.eval()
            final_check = short_reference(model, records, keys)
            final_digest = weights_digest(model)
            final = save_model(model, BULK/f"{arm}-final.pt")
            sample({"stage": "final_reference"})
            with torch.inference_mode():
                began = time.monotonic()
                memory, padding = model.encode_records(inputs[:1])
                sampled, logq = model.propose(memory, padding, samples=8, generator=torch.Generator().manual_seed(SAMPLE_SEED),
                             after_step=lambda info: sample({"stage": "sampling", **info}))
                greedy, greedy_logq = model.propose(memory, padding, greedy=True,
                             after_step=lambda info: sample({"stage": "greedy", **info}))
                proposal_seconds = time.monotonic()-began
                proposed = torch.cat((sampled[0], greedy[0])).cpu()
                proposed_logq = torch.cat((logq[0], greedy_logq[0]))
                reference = copy.deepcopy(model).to("cpu").double().eval()
                ref_memory, ref_padding = reference.encode_records(records[:1])
                replay = reference.joint_log_probability(ref_memory.expand(9, -1, -1), ref_padding.expand(9, -1), proposed)
                proposal_delta = float((replay-proposed_logq).abs().max())
                if proposal_delta > .002:
                    raise ValueError("Full-record autoregressive proposal CPUfloat64 replay differs")
            sample({"stage": "full_proposals_verified"})
            timed = [r["wall_seconds"] for r in steps if r["timed"]]
            save_new(OUT/f"{arm}.json", {"status": "PASS", "arm": arm, "freeze": freeze, "config": asdict(CONFIGS[scale]),
                "parameters": count, "batch": batch, "inputs": input_spec, "initial": initial, "final": final,
                "initial_weights_sha256": initial_digest, "final_weights_sha256": final_digest,
                "initial_check": initial_check, "final_check": final_check, "optimizer_steps": steps,
                "mean_timed_step_seconds": math.fsum(timed)/len(timed), "proposal_wall_seconds": proposal_seconds,
                "proposed_keys": proposed.tolist(), "proposed_logq": proposed_logq.tolist(),
                "maximum_full_proposal_cpu_float64_delta": proposal_delta,
                "sampled_driver_peak": max(r["driver_bytes"] for r in trace), "sampled_tensor_peak": max(r["tensor_bytes"] for r in trace),
                "trace": artifact(trace_path), "random_optimizer_updates": STEPS,
                "language_training_or_cipher_panel_files_opened": False, "resources": resource_report(wall, cpu)})
            print(json.dumps({"arm": arm, "parameters": count, "mean_step_seconds": math.fsum(timed)/len(timed),
                              "proposal_seconds": proposal_seconds, "proposal_delta": proposal_delta}), flush=True)
        except Exception as error:
            signal.alarm(0)
            save_new(OUT/f"{arm}-failure.json", {"arm": arm, "freeze": freeze, "type": type(error).__name__,
                "error": str(error), "completed_optimizer_steps": steps, "inputs": input_spec, "trace": artifact(trace_path),
                "last_sample": trace[-1] if trace else None, "resources": resource_report(wall, cpu)})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/"campaign-started.json", {"freeze": freeze, "start_unix": time.time(), "arms": ARMS})
    BULK.mkdir(parents=True, exist_ok=True)
    wall = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    rows = []
    for arm in ARMS:
        began = time.monotonic()
        with (BULK/f"{arm}.log").open("x") as log:
            try:
                result = subprocess.run([sys.executable, "-u", __file__, "arm", "--arm", arm, "--freeze", freeze],
                          cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, timeout=315, check=False)
                code, timeout = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timeout = None, True
            except OSError as error:
                code, timeout = None, False
                log.write(str(error)+"\n")
        row = {"arm": arm, "returncode": code, "outer_timeout": timeout,
               "wall_seconds": time.monotonic()-began, "log": artifact(BULK/f"{arm}.log")}
        rows.append(row)
        print(json.dumps(row), flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/"campaign.json", {"freeze": freeze, "processes": rows, "wall_seconds": time.monotonic()-wall,
              "all_children_cpu_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
              "language_training_or_cipher_panel_files_opened": False, "paid_spend_usd": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("campaign", "arm"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    if args.mode == "campaign":
        campaign(args.freeze)
    else:
        run_arm(args.arm, args.freeze)
