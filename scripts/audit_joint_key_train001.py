"""Complete training ledger/resource accounting and selected-checkpoint replay."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_joint_key_train001 import (ARMS, BASE_PATHS, BATCH, BULK, CHECKPOINTS, INPUT_PATHS,
    OUT, PARAMETERS, ROOT, STEPS, load_prepared)
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig
from voynich.joint_key_training import EpisodeSampler, dictionary_code, metadata_bytes, pack_episodes, schedule


def verify(spec):
    if artifact(ROOT/spec["path"]) != spec:
        raise ValueError("Training artifact binding differs")


def checkpoint(spec, config, state_digest, expected_step, freeze):
    verify(spec)
    value = torch.load(ROOT/spec["path"], map_location="cpu", weights_only=True)
    if value["step"] != expected_step or value["freeze"] != freeze or value["config"] != config:
        raise ValueError("Checkpoint identity/config differs")
    digest = hashlib.sha256()
    for name, tensor in value["state_dict"].items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite checkpoint")
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    if digest.hexdigest() != state_digest:
        raise ValueError("Raw state digest differs")
    model = JointKeyProposal(KeyProposalConfig(**config))
    model.load_state_dict(value["state_dict"], strict=True)
    return model.double().eval()


def replay_selected(model, score, episodes):
    if (score["episodes"] != len(episodes) or score["rows"] != len(episodes)*23
            or len(score["joint_logq"]) != len(episodes) or score["used_mask_is_diagnostic_only"] is not True):
        raise ValueError("Fixed full-row validation scope differs")
    truth_values, greedy_values, maxima, matched_keys = [], [], [], []
    with torch.inference_mode():
        for index, episode in enumerate(episodes):
            records, keys = pack_episodes([episode])
            memory, mask = model.encode_records(records)
            logits = model.decode_partial(memory, mask, keys[:, :-1])
            logq = F.log_softmax(logits, -1).gather(-1, keys[..., None]).squeeze(-1).sum(-1)
            chosen = torch.tensor([score["free_running_greedy_keys"][index]], dtype=torch.long)
            if chosen.shape != (1, 23) or (chosen < 0).any() or (chosen >= 42).any():
                raise ValueError("Greedy key inventory differs")
            guess_logits = model.decode_partial(memory, mask, chosen[:, :-1])
            guess_score = F.log_softmax(guess_logits, -1).gather(-1, chosen[..., None]).squeeze(-1).sum(-1)
            margins = guess_logits.max(-1).values-guess_logits.gather(-1, chosen[..., None]).squeeze(-1)
            maxima.append(float(margins.max()))
            truth_values.append(float(logq[0]))
            greedy_values.append(float(guess_score[0]))
            matched_keys.append((chosen == keys)[0])
    deltas = [abs(a-b) for a, b in zip(truth_values, score["joint_logq"], strict=True)]
    greedy_deltas = [abs(a-b) for a, b in zip(greedy_values, score["greedy_logq"], strict=True)]
    correct = sum(int(row.sum()) for row in matched_keys)
    used_correct = sum(int(row[e[2]["used_rows"]].sum()) for row, e in zip(matched_keys, episodes, strict=True))
    used_count = sum(len(e[2]["used_rows"]) for e in episodes)
    mean = -math.fsum(score["joint_logq"])/(len(episodes)*23)
    if (max(deltas+greedy_deltas) > .002 or max(maxima) > 1e-4 or not math.isfinite(mean)
            or abs(mean-score["mean_nats_per_row"]) > 1e-12 or score["correct_rows"] != correct
            or (score["used_correct_rows"], score["used_rows"]) != (used_correct, used_count)
            or score["whole_keys_exact"] != sum(bool(r.all()) for r in matched_keys)):
        raise ValueError("CPU probability/greedy choice or diagnostic accounting differs")
    return {"whole_logq_max_delta": max(deltas), "greedy_logq_max_delta": max(greedy_deltas),
            "cpu_argmax_max_deficit": max(maxima), "episodes": len(episodes)}


def ledger_account(value, texts, episodes):
    seed = value["seed"]
    sampler = EpisodeSampler(texts, forbidden_raw=[dictionary_code(e[2]["raw_indices"]) for e in episodes],
                             forbidden_canonical=[dictionary_code(e[1]) for e in episodes])
    rng, digest = np.random.default_rng(seed+100003), hashlib.sha256()
    characters, count, prefixes = 0, 0, {}
    raw_keys, canonical_keys = set(), set()
    verify(value["ledger"])
    with gzip.open(ROOT/value["ledger"]["path"], "rb") as stream:
        for step, encoded in enumerate(stream, start=1):
            if step > STEPS:
                raise ValueError("Extra training episode batches")
            metadata = [sampler.sample(rng)[2] for _ in range(BATCH)]
            raw_keys.update(dictionary_code(e["raw_indices"]) for e in metadata)
            canonical_keys.update(dictionary_code(e["canonical_key"]) for e in metadata)
            expected = metadata_bytes({"step": step, "episodes": metadata})
            if encoded != expected:
                raise ValueError("Boundary/window/key/disjointness/seed replay differs")
            digest.update(encoded)
            characters += sum(w["length"] for e in metadata for w in e["windows"])
            count += BATCH
            prefixes[step] = (digest.hexdigest(), characters, count)
    if (len(prefixes) != value["updates"] or digest.hexdigest() != value["data_sha256"]
            or characters != value["source_characters"] or count != value["episodes_seen"]
            or count*23 != value["assigned_key_rows"] or sampler.rejected_keys != value["rejected_disjointness_keys"]):
        raise ValueError("Complete training exposure differs")
    return prefixes, {"distinct_literal_dictionaries": len(raw_keys), "distinct_canonical_dictionaries": len(canonical_keys)}


def audit():
    campaign_path = OUT/"campaign.json"
    campaign = json.loads(campaign_path.read_text())
    freeze = campaign["freeze"]
    require_frozen(freeze, [*BASE_PATHS, *INPUT_PATHS])
    save_new(OUT/"audit-started.json", {"freeze": freeze, "start_unix": time.time()})
    limit_resources(3600, 3000)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    outcomes = {}
    try:
        if [p["arm"] for p in campaign["processes"]] != list(ARMS) or campaign["paid_spend_usd"] != 0:
            raise ValueError("Complete terminal four-arm inventory required")
        manifest, texts, _, episodes = load_prepared()
        for process in campaign["processes"]:
            arm = process["arm"]
            verify(process["log"])
            path, failure = OUT/(arm+".json"), OUT/(arm+"-failure.json")
            if path.exists() and failure.exists():
                raise ValueError("Contradictory result/failure")
            passed = process["returncode"] == 0 and not process["outer_timeout"]
            if passed != path.exists():
                raise ValueError("Successful process/result differs")
            if not passed:
                if failure.exists():
                    partial = json.loads(failure.read_text())
                    for field in ("trace", "ledger"):
                        verify(partial[field])
                outcomes[arm] = {"status": "retained_failure", "process": process,
                    "partial_artifacts": [artifact(p) for p in sorted(BULK.glob(arm+"-*")) if p.is_file()]}
                continue
            value = json.loads(path.read_text())
            scale, seed = arm.split("-")
            from scripts.benchmark_key_proposal_systems001 import CONFIGS
            config = CONFIGS[scale].__dict__
            if (value["status"] != "PASS" or value["arm"] != arm or value["freeze"] != freeze
                    or value["seed"] != int(seed) or value["updates"] != STEPS
                    or value["parameters"] != PARAMETERS[scale] or value["corpus_identity"] != manifest["identity"]
                    or value["reserved_authors_or_prior_cipher_panel_opened"] is not False
                    or value["inputs"] != artifact(OUT/"inputs.json")):
                raise ValueError("Completed arm scope differs")
            prefixes, distinct = ledger_account(value, texts, episodes)
            verify(value["trace"])
            trace = [json.loads(line) for line in (ROOT/value["trace"]["path"]).read_text().splitlines()]
            optimizer = [row for row in trace if row["stage"] == "optimizer"]
            if [r["step"] for r in optimizer] != list(range(1, STEPS+1)):
                raise ValueError("All optimizer updates required")
            for r in optimizer:
                if ((r["data_sha256"], r["source_characters"], r["episodes_seen"]) != prefixes[r["step"]]
                        or r["lr"] != schedule(r["step"])
                        or not all(math.isfinite(r[k]) and r[k] >= 0 for k in ("loss_nats_per_row", "preclip_norm", "step_wall_seconds"))):
                    raise ValueError("Trace/data/optimizer schedule differs")
            samples = [r for r in trace if "driver_bytes" in r]
            if (max(r["driver_bytes"] for r in samples) > 16*1024**3
                    or max(r["peak_rss_bytes"] for r in samples) > 8*1024**3):
                raise ValueError("Training sampled resource guard differs")
            rows = value["checkpoints"]
            if [r["step"] for r in rows] != list(CHECKPOINTS):
                raise ValueError("Every declared checkpoint required")
            selected = min(rows, key=lambda r: (r["mean_nats_per_row"], r["step"]))
            if selected != value["selected"]:
                raise ValueError("Fixed validation-only selection differs")
            replay = {}
            for row in rows:
                if json.loads((OUT/f"{arm}-step{row['step']}.json").read_text()) != row:
                    raise ValueError("Compact checkpoint publication differs")
                verify(row["validation"])
                saved = load_archive(row["validation"])
                score = saved["score"]
                if (row["mean_nats_per_row"] != score["mean_nats_per_row"]
                        or abs(score["mean_nats_per_row"]+math.fsum(score["joint_logq"])/(len(episodes)*23)) > 1e-12
                        or len(score["joint_logq"]) != len(episodes)
                        or (row["step"] == 0 and row["weights_sha256"] != value["initial_weights_sha256"])):
                    raise ValueError("Published validation score differs")
                model = checkpoint(row["weights"], config, row["weights_sha256"], row["step"], freeze)
                if sum(p.numel() for p in model.parameters()) != value["parameters"]:
                    raise ValueError("Actual parameter count differs")
                if row["step"] in (0, selected["step"]):
                    replay[str(row["step"])] = replay_selected(model, score, episodes)
                del model
            outcomes[arm] = {"status": "PASS", "result": artifact(path), "updates": STEPS,
                "data_sha256": value["data_sha256"], "initial_weights_sha256": value["initial_weights_sha256"],
                "selected_step": selected["step"], "selected_nats_per_row": selected["mean_nats_per_row"],
                "replay": replay, **distinct}
        for seed in (72203, 72209):
            a, b = outcomes["small-"+str(seed)], outcomes["large-"+str(seed)]
            if a["status"] == b["status"] == "PASS" and a["data_sha256"] != b["data_sha256"]:
                raise ValueError("Paired capacity data exposure differs")
        if resource_report(wall, cpu)["peak_rss_bytes"] > 8*1024**3:
            raise MemoryError("Auditor 8GiB sampled host guard")
        save_new(OUT/"audit.json", {"status": "PASS", "arms": outcomes, "campaign": artifact(campaign_path),
            "independent_agent_review": False, "auditor": artifact(ROOT/"scripts/audit_joint_key_train001.py"),
            "scope": "complete episode/optimizer/checkpoint inventory and initial/selected CPUdouble joint law and greedy replay",
            "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
