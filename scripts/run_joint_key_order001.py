"""One finite CPU-only order diagnostic on the already completed first fit."""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time
from pathlib import Path

import torch

from scripts.audit_joint_key_train001 import checkpoint
from scripts.benchmark_key_proposal_systems001 import CONFIGS
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_joint_key_train001 import BASE_PATHS, INPUT_PATHS, load_prepared
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.joint_key_order_controls import controlled_episode, score_episode, summarize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "JOINT-KEY-ORDER-001"
OUT = ROOT/"results"/EXPERIMENT
BULK = ROOT/"outputs"/EXPERIMENT
ARM = "results/JOINT-KEY-TRAIN-001/small-72203.json"
PATHS = [*BASE_PATHS, *INPUT_PATHS, ARM, "src/voynich/joint_key_order_controls.py",
    "scripts/run_joint_key_order001.py", "scripts/audit_joint_key_order001.py", "tests/test_joint_key_order_controls.py",
    "docs/experiments/JOINT-KEY-ORDER-001.md"]
CONDITIONS = ("original", "unit_shuffle", "glyph_shuffle")


def run(freeze):
    require_frozen(freeze, PATHS)
    value = json.loads((ROOT/ARM).read_text())
    if (value["arm"] != "small-72203" or value["status"] != "PASS"
            or value["updates"] != 20000 or value["parameters"] != 5423146):
        raise ValueError("Only fixed completed first control admitted")
    initial, selected = value["checkpoints"][0], value["selected"]
    if initial["step"] != 0 or selected["step"] != 20000:
        raise ValueError("Frozen initial/selected checkpoint pair differs")
    inputs, _, texts, episodes = load_prepared()
    if len(episodes) != 64 or value["inputs"] != artifact(ROOT/"results/JOINT-KEY-TRAIN-001/inputs.json"):
        raise ValueError("Fixed development episodes/input binding differs")
    OUT.mkdir(parents=True, exist_ok=True)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(),
        "arm": artifact(ROOT/ARM), "cpu_only": True, "training_final_audit_pending": True,
        "torch": torch.__version__, "cpu_threads": 2, "paid_spend_usd": 0})
    limit_resources(1200, 1000)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    wall, cpu = time.monotonic(), time.process_time()
    BULK.mkdir(parents=True, exist_ok=True)
    trace_path = BULK/"records.jsonl"
    rows = []
    try:
        with trace_path.open("x") as trace:
            for saved in (initial, selected):
                model = checkpoint(saved["weights"], CONFIGS["small"].__dict__, saved["weights_sha256"], saved["step"], value["freeze"])
                scores = load_archive(saved["validation"])["score"]
                for index, episode in enumerate(episodes):
                    for condition in CONDITIONS:
                        control, construction = episode, None
                        original = None
                        if condition == "original":
                            original = {"key": scores["free_running_greedy_keys"][index],
                                "joint_logq": scores["joint_logq"][index], "greedy_logq": scores["greedy_logq"][index]}
                        else:
                            control, construction = controlled_episode(episode, texts, condition=condition, seed=72341+index)
                        row = {"checkpoint": saved["step"], "episode": index, "condition": condition,
                            "construction": construction, **score_episode(model, control, saved_original=original)}
                        rows.append(row)
                        trace.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                        trace.flush()
                        current = resource_report(wall, cpu)
                        if current["peak_rss_bytes"] > 8*1024**3:
                            raise MemoryError("8GiB sampled host guard")
                        if len(rows)%24 == 0:
                            print(json.dumps({"completed_cells": len(rows), "scheduled_cells": 384,
                                "checkpoint": saved["step"], "resources": current}), flush=True)
                del model
                gc.collect()
        if len(rows) != 384:
            raise ValueError("Complete fixed diagnostic grid required")
        result = {"status": "PASS_complete_numeric_and_construction_checks",
            "experiment": EXPERIMENT, "freeze": freeze, "arm": artifact(ROOT/ARM),
            "training_freeze": value["freeze"], "training_final_audit_pending": True,
            "validation_source": inputs["validation_source"], "validation_episodes": inputs["validation_episodes"],
            "records": artifact(trace_path), "completed_cells": len(rows), "summary": summarize(rows),
            "numeric_true_logq_max_delta": max(r["numeric"]["true_logq_delta"] for r in rows if "numeric" in r),
            "numeric_saved_greedy_logq_max_delta": max(r["numeric"]["saved_greedy_logq_delta"] for r in rows if "numeric" in r),
            "numeric_saved_greedy_max_deficit": max(r["numeric"]["saved_greedy_max_cpu_deficit"] for r in rows if "numeric" in r),
            "resources": resource_report(wall, cpu), "cpu_only": True,
            "scope": "Exploratory exposed first-fit input-order intervention, not neuron/circuit/recovery qualification; final training ledger audit pending."}
        save_new(OUT/"result.json", result)
        print(json.dumps({"status": result["status"], "diagnostic": result["summary"]["exploratory_unit_order_diagnostic"],
            "resources": result["resources"]}), flush=True)
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_cells": len(rows), "freeze": freeze,
            "partial_records": artifact(trace_path) if trace_path.exists() else None,
            "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    run(args.freeze)
