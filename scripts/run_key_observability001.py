"""One frozen known-plaintext oracle diagnostic; not a ciphertext-only solver."""
from __future__ import annotations

import argparse
import json
import signal
import time
from fractions import Fraction
from pathlib import Path

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_joint_key_train001 import BASE_PATHS, INPUT_PATHS
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.joint_key_proposal import unit_pool
from voynich.joint_key_training import EpisodeSampler
from voynich.known_plaintext_keys import oracle_statistics, solve_known_plaintext

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/KEY-OBSERVABILITY-001"
BULK = ROOT/"outputs/KEY-OBSERVABILITY-001"
PATHS = [*BASE_PATHS, *INPUT_PATHS, "src/voynich/known_plaintext_keys.py",
         "scripts/run_key_observability001.py", "scripts/audit_key_observability001.py",
         "tests/test_known_plaintext_keys.py", "docs/experiments/KEY-OBSERVABILITY-001.md"]


def fixed_inputs():
    manifest = json.loads((ROOT/"results/JOINT-KEY-TRAIN-001/inputs.json").read_text())
    valid = load_archive(manifest["validation_source"])
    metadata = load_archive(manifest["validation_episodes"])["metadata"]
    if valid["role"] != "source_validation" or len(metadata) != 64:
        raise ValueError("Only fixed exposed validation source/64 episodes admitted")
    sampler = EpisodeSampler(valid["texts"])
    cases = []
    pool = unit_pool(6)
    for e in metadata:
        cipher, target, replayed = sampler.make(e["windows"], e["raw_indices"])
        if replayed != e:
            raise ValueError("Fixed episode provenance replay differs")
        source = tuple(tuple(int(c) for c in sampler.records[w["segment"]][w["start"]:w["start"]+w["length"]]) for w in e["windows"])
        truth = tuple(pool[k] for k in target)
        if tuple(tuple(g for row in s for g in truth[row]) for s in source) != cipher:
            raise ValueError("Gold source/cipher provenance reconstruction differs")
        cases.append((source, cipher, truth))
    return manifest, cases


def summary(rows):
    complete = [r for r in rows if r["solver"]["complete"]]
    stats = [r["statistics"] for r in complete]
    if any(s is None for s in stats):
        raise ValueError("Generated positive must have a compatible solution")
    return {"cases": len(rows), "complete_cases": len(complete), "capped_cases": len(rows)-len(complete),
            "known_plaintext_only": True, "unique_used_dictionary_cases": sum(s["used_solution_count"] == 1 for s in stats),
            "used_rows_in_complete_cases": sum(len(r["solver"]["used_rows"]) for r in complete),
            "determined_used_rows": sum(len(s["determined_used_rows"]) for s in stats),
            "bayes_expected_used_matches_sum": str(sum((Fraction(s["bayes_expected_used_row_matches"]) for s in stats), Fraction(0))),
            "bayes_expected_full_key_matches_sum": str(sum((Fraction(s["full_key_map_probability"]) for s in stats), Fraction(0))),
            "unused_row_histogram": {str(n): sum(s["unused_rows"] == n for s in stats) for n in range(24)},
            "total_search_nodes": sum(r["solver"]["nodes"] for r in rows),
            "scope": "Exposed development, source-supplied observability calibration. Oracle probabilities use iid validation prior, not learned q or a ciphertext-only qualification."}


def run(freeze):
    require_frozen(freeze, PATHS)
    OUT.mkdir(parents=True, exist_ok=True)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    trace = BULK/"cases.jsonl"
    BULK.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        manifest, cases = fixed_inputs()
        with trace.open("x") as stream:
            for i, (source, cipher, truth) in enumerate(cases):
                solved = solve_known_plaintext(source, cipher)
                compatible = tuple(truth[r] if r in solved["used_rows"] else None for r in range(23))
                contained = compatible in solved["solutions"]
                if solved["complete"] and not contained:
                    raise ValueError("Complete oracle missed compatible gold used dictionary")
                row = {"case": i, "solver": solved, "statistics": oracle_statistics(solved),
                       "gold_used_dictionary_found": contained, "gold_key_is_solver_input": False}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
                rows.append(row)
                resources = resource_report(wall, cpu)
                if resources["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("2GiB sampled host cap")
                print(json.dumps({"completed_cases": len(rows), "nodes": solved["nodes"],
                                  "complete": solved["complete"]}), flush=True)
        result = {"status": "PASS_complete_fixed_grid", "freeze": freeze,
                  "source": manifest["validation_source"], "episodes": manifest["validation_episodes"],
                  "records": artifact(trace), "summary": summary(rows), "resources": resource_report(wall, cpu),
                  "training_final_audit_pending": True, "gpu_inference": False, "training_updates": 0}
        save_new(OUT/"result.json", result)
        print(json.dumps(result["summary"], sort_keys=True), flush=True)
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_cases": len(rows),
            "records": artifact(trace) if trace.exists() else None, "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
