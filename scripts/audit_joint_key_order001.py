"""Replay the complete input-control grid and arithmetic without new inference."""
from __future__ import annotations

import json
import math
import signal
import time

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_joint_key_order001 import ARM, BULK, CONDITIONS, OUT, PATHS, ROOT
from scripts.run_joint_key_train001 import load_prepared
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.joint_key_order_controls import controlled_episode, summarize


def audit_rows(rows, episodes, texts, saved_by_step):
    expected = [(step, i, condition) for step in (0, 20000)
                for i in range(len(episodes)) for condition in CONDITIONS]
    if [(r["checkpoint"], r["episode"], r["condition"]) for r in rows] != expected:
        raise ValueError("Complete ordered diagnostic grid differs")
    for row in rows:
        episode = episodes[row["episode"]]
        if row["condition"] == "original":
            if row["construction"] is not None:
                raise ValueError("Original input acquired a transformation")
            saved = saved_by_step[row["checkpoint"]]
            delta = abs(row["joint_logq"]-saved["joint_logq"][row["episode"]])
            numeric = row["numeric"]
            if (numeric["true_logq_delta"] != delta or delta > .002
                    or not 0 <= numeric["saved_greedy_logq_delta"] <= .002
                    or not 0 <= numeric["saved_greedy_max_cpu_deficit"] <= 1e-4
                    or numeric["cpu_vs_mps_greedy_changed_rows"] != sum(a != b for a, b in zip(row["greedy_key"], saved["free_running_greedy_keys"][row["episode"]], strict=True))):
                raise ValueError("Recorded original numeric gate differs")
        else:
            _, construction = controlled_episode(episode, texts, condition=row["condition"], seed=72341+row["episode"])
            if construction != row["construction"] or "numeric" in row:
                raise ValueError("Exact construction replay differs")
        q, guess, truth = row["row_logq"], row["greedy_key"], episode[1]
        if (len(q) != 23 or len(guess) != 23 or any(type(i) is not int or not 0 <= i < 42 for i in guess)
                or not all(math.isfinite(v) and v <= 1e-10 for v in [*q, row["joint_logq"], row["greedy_logq"]])
                or abs(math.fsum(q)-row["joint_logq"]) > 1e-10):
            raise ValueError("Whole-key probability/row inventory differs")
        used = episode[2]["used_rows"]
        if (row["correct_rows"] != sum(a == b for a, b in zip(guess, truth, strict=True))
                or row["used_correct_rows"] != sum(guess[i] == truth[i] for i in used)
                or row["used_rows"] != len(used) or row["whole_key_exact"] != (tuple(guess) == tuple(truth))):
            raise ValueError("Greedy/used-row diagnostics differ")
    return summarize(rows)


def audit():
    result = json.loads((OUT/"result.json").read_text())
    require_frozen(result["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time(),
        "model_inference_repeated": False})
    limit_resources(120, 100)
    wall, cpu = time.monotonic(), time.process_time()
    if (result["status"] != "PASS_complete_numeric_and_construction_checks"
            or result["completed_cells"] != 384 or result["cpu_only"] is not True
            or result["training_final_audit_pending"] is not True or result["arm"] != artifact(ROOT/ARM)
            or result["resources"]["paid_spend_usd"] != 0 or result["resources"]["peak_rss_bytes"] > 8*1024**3):
        raise ValueError("Registered diagnostic scope/resources differ")
    value = json.loads((ROOT/ARM).read_text())
    if result["training_freeze"] != value["freeze"]:
        raise ValueError("Training identity differs")
    inputs, _, texts, episodes = load_prepared()
    if (len(episodes) != 64 or result["records"] != artifact(BULK/"records.jsonl")
            or result["validation_source"] != inputs["validation_source"]
            or result["validation_episodes"] != inputs["validation_episodes"]):
        raise ValueError("Result source/record binding differs")
    saved = {}
    for snapshot in (value["checkpoints"][0], value["selected"]):
        if snapshot["weights"] != artifact(ROOT/snapshot["weights"]["path"]):
            raise ValueError("Frozen weight archive differs")
        saved[snapshot["step"]] = load_archive(snapshot["validation"])["score"]
    rows = [json.loads(line) for line in (BULK/"records.jsonl").read_text().splitlines()]
    summary = audit_rows(rows, episodes, texts, saved)
    if summary != result["summary"]:
        raise ValueError("Complete paired statistics/bootstrap differs")
    for step, record in summary["checkpoints"].items():
        for condition, contrast in record["comparisons"].items():
            original = [r for r in rows if str(r["checkpoint"]) == step and r["condition"] == "original"]
            changed = [r for r in rows if str(r["checkpoint"]) == step and r["condition"] == condition]
            manual = math.fsum((a["joint_logq"]-b["joint_logq"])/23 for a, b in zip(original, changed, strict=True))/64
            if abs(manual-contrast["mean_control_minus_original_nats_per_row"]) > 1e-12:
                raise ValueError("Alternate scalar paired mean differs")
    for name, field in (("numeric_true_logq_max_delta", "true_logq_delta"),
                        ("numeric_saved_greedy_logq_max_delta", "saved_greedy_logq_delta"),
                        ("numeric_saved_greedy_max_deficit", "saved_greedy_max_cpu_deficit")):
        if result[name] != max(r["numeric"][field] for r in rows if "numeric" in r):
            raise ValueError("Numeric maximum differs")
    resources = resource_report(wall, cpu)
    if resources["peak_rss_bytes"] > 2*1024**3:
        raise MemoryError("2GiB sampled metadata auditor guard")
    report = {"status": "PASS_artifact_construction_and_arithmetic_only", "result": artifact(OUT/"result.json"),
        "records": result["records"], "cells": len(rows), "construction_replays": 256,
        "model_inference_repeated": False, "independent_agent_review": False, "paid_spend_usd": 0,
        "resources": resources}
    save_new(OUT/"audit.json", report)
    signal.alarm(0)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    audit()
