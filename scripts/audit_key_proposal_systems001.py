"""Complete frozen resource inventory and separate CPU-double checkpoint replay."""
from __future__ import annotations

import hashlib
import json
import math
import signal
import time

import torch
from torch.nn import functional as F

from scripts.benchmark_key_proposal_systems001 import (ARMS, CONFIGS, DATA_SEED, OUT, BULK, PARAMETERS,
    PATHS, STEPS, WARMUP, artificial_episodes, pack)
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig


def verify(spec):
    if artifact(ROOT/spec["path"]) != spec:
        raise ValueError("Artifact checksum/size/path differs")


def checkpoint(spec, config, digest, count):
    verify(spec)
    raw = torch.load(ROOT/spec["path"], map_location="cpu", weights_only=True)
    if raw["config"] != config:
        raise ValueError("Checkpoint configuration differs")
    expected = hashlib.sha256()
    for name, tensor in raw["state_dict"].items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError("Nonfinite/wrong-dtype checkpoint")
        expected.update(name.encode())
        expected.update(str(tuple(tensor.shape)).encode())
        expected.update(tensor.numpy().tobytes())
    if expected.hexdigest() != digest:
        raise ValueError("Checkpoint state digest differs")
    model = JointKeyProposal(KeyProposalConfig(**config))
    if sum(p.numel() for p in model.parameters()) != count:
        raise ValueError("Actual parameter count differs")
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.double().eval()


def replay(value, records, keys):
    deltas = {}
    with torch.inference_mode():
        for stage in ("initial", "final"):
            model = checkpoint(value[stage], value["config"], value[stage+"_weights_sha256"], value["parameters"])
            check = value[stage+"_check"]
            actual = torch.tensor(check["mps_logits"], dtype=torch.float64)
            expected = model(records[:1, :, :16], keys[:1])
            delta = float((actual-expected).abs().max())
            if (actual.shape != (1, 23, 42) or not torch.isfinite(actual).all().item()
                    or (check["record_glyphs"], check["rows"], check["units"]) != (16, 23, 42)
                    or delta > 5e-5 or abs(delta-check["maximum_logit_delta"]) > 1e-12):
                raise ValueError("Saved short-logit reference differs")
            deltas[stage+"_logit_delta"] = delta
            if stage == "final":
                proposed = torch.tensor(value["proposed_keys"], dtype=torch.long)
                if proposed.shape != (9, 23) or (proposed < 0).any() or (proposed >= 42).any():
                    raise ValueError("Whole-key proposal dimensions/inventory differ")
                # Teacher-forced full-sequence replay uses explicit normalized row
                # factors, not the proposal routine or its accumulated score.
                memory, padding = model.encode_records(records[:1])
                logits = model.decode_partial(memory.expand(9, -1, -1), padding.expand(9, -1), proposed[:, :-1])
                scores = F.log_softmax(logits, -1).gather(-1, proposed[..., None]).squeeze(-1).sum(-1)
                saved = torch.tensor(value["proposed_logq"], dtype=torch.float64)
                if saved.shape != (9,) or not torch.isfinite(saved).all() or (saved > 0).any():
                    raise ValueError("Proposal log probabilities differ")
                delta = float((scores-saved).abs().max())
                if delta > .002 or abs(delta-value["maximum_full_proposal_cpu_float64_delta"]) > 1e-10:
                    raise ValueError("Full-record joint law replay differs")
                deltas["full_proposal_delta"] = delta
            del model
    return deltas


def inventory(process, freeze, *, replay_models=True):
    arm = process["arm"]
    verify(process["log"])
    result, failure = OUT/f"{arm}.json", OUT/f"{arm}-failure.json"
    if result.exists() and failure.exists():
        raise ValueError("Contradictory result/failure")
    success = process["returncode"] == 0 and not process["outer_timeout"]
    if success != result.exists():
        raise ValueError("Terminal process/result success differs")
    available = [artifact(p) for p in sorted(BULK.glob(arm+"-*")) if p.is_file()]
    if not result.exists() and not failure.exists():
        return {"status": "missing_after_terminal_failure", "process": process,
                "retained_partial_artifacts": available}
    path = result if success else failure
    value = json.loads(path.read_text())
    if value["arm"] != arm or value["freeze"] != freeze:
        raise ValueError("Arm/source identity differs")
    verify(value["inputs"])
    inputs = load_archive(value["inputs"])
    expected = json.loads(json.dumps({"seed": DATA_SEED, "episodes": artificial_episodes()}))
    if inputs != expected:
        raise ValueError("Artificial generation differs")
    verify(value["trace"])
    trace = [json.loads(line) for line in (ROOT/value["trace"]["path"]).read_text().splitlines()]
    steps = value["optimizer_steps"] if success else value["completed_optimizer_steps"]
    if [s["step"] for s in steps] != list(range(1, len(steps)+1)) or len(steps) > STEPS:
        raise ValueError("Random optimizer work differs")
    for row in steps:
        if (row["timed"] != (row["step"] > WARMUP)
                or not all(math.isfinite(row[k]) for k in ("loss_nats_per_row", "preclip_gradient_norm", "wall_seconds"))
                or min(row["loss_nats_per_row"], row["preclip_gradient_norm"], row["wall_seconds"]) < 0):
            raise ValueError("Optimizer arithmetic differs")
    optimizer_trace = [t["step"] for t in trace if t["stage"] == "optimizer"]
    if optimizer_trace != [s["step"] for s in steps]:
        raise ValueError("Trace/optimizer work differs")
    checked = {}
    if success:
        scale, batch = arm.split("-b")
        config = CONFIGS[scale].__dict__
        timed = [s["wall_seconds"] for s in steps if s["timed"]]
        if (value["status"] != "PASS" or len(steps) != STEPS or value["random_optimizer_updates"] != STEPS
                or value["language_training_or_cipher_panel_files_opened"] is not False
                or value["config"] != config or value["parameters"] != PARAMETERS[scale] or value["batch"] != int(batch)
                or abs(value["mean_timed_step_seconds"]-math.fsum(timed)/len(timed)) > 1e-12
                or value["sampled_driver_peak"] != max(t["driver_bytes"] for t in trace)
                or value["sampled_tensor_peak"] != max(t["tensor_bytes"] for t in trace)
                or max(t["driver_bytes"] for t in trace) > 16*1024**3
                or max(t["peak_rss_bytes"] for t in trace) > 8*1024**3):
            raise ValueError("Successful scope/config/cost/guard differs")
        for stage, count in (("sampling", 8), ("greedy", 1)):
            rows = [t for t in trace if t["stage"] == stage]
            if [t["choice"] for t in rows] != list(range(1, 24)) or any(t["keys"] != count for t in rows):
                raise ValueError("Autoregressive proposal trace differs")
        if replay_models:
            records, keys = pack(inputs["episodes"], int(batch))
            checked = replay(value, records, keys)
    return {"status": "PASS" if success else "retained_failure", "result": artifact(path),
            "optimizer_updates": len(steps), "retained_partial_artifacts": available, "replay": checked,
            "initial_weights_sha256": value.get("initial_weights_sha256"),
            "sampled_driver_peak": max((t["driver_bytes"] for t in trace), default=0)}


def audit():
    path = OUT/"campaign.json"
    campaign = json.loads(path.read_text())
    require_frozen(campaign["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": campaign["freeze"], "start_unix": time.time()})
    started = json.loads((OUT/"campaign-started.json").read_text())
    if (started["freeze"] != campaign["freeze"] or started["arms"] != list(ARMS)
            or [p["arm"] for p in campaign["processes"]] != list(ARMS)
            or campaign["language_training_or_cipher_panel_files_opened"] is not False or campaign["paid_spend_usd"] != 0):
        raise ValueError("Complete fixed four-arm terminal campaign required")
    limit_resources(450, 400)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        rows = {p["arm"]: inventory(p, campaign["freeze"]) for p in campaign["processes"]}
        for scale in ("small", "large"):
            a, b = rows[scale+"-b1"], rows[scale+"-b4"]
            if a["status"] == b["status"] == "PASS" and a["initial_weights_sha256"] != b["initial_weights_sha256"]:
                raise ValueError("Matched initialization differs")
        if resource_report(wall, cpu)["peak_rss_bytes"] > 8*1024**3:
            raise MemoryError("Auditor 8GiB sampled host guard")
        save_new(OUT/"audit.json", {"status": "PASS", "campaign": artifact(path), "arms": rows,
            "random_optimizer_updates": (sum(r["optimizer_updates"] for r in rows.values())
                 if all("optimizer_updates" in r for r in rows.values()) else None),
            "known_completed_random_optimizer_updates": sum(r.get("optimizer_updates", 0) for r in rows.values()),
            "auditor": artifact(ROOT/"scripts/audit_key_proposal_systems001.py"),
            "scope": "complete inventory, hashes, random work, costs and separate CPUfloat64 checkpoint probability replay",
            "independent_agent_review": False, "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
