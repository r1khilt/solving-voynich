"""Alternate complete inventory/literal/accounting audit, no cipher panel access."""
import json
import signal
import time

from scripts.benchmark_shared_prefix_systems001 import ARMS, OUT, PATHS, SEEDS, workloads
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.recurrent_latin_source import ALPHABET


def verify(spec):
    if artifact(ROOT/spec["path"]) != spec:
        raise ValueError("Benchmark archive binding differs")


def audit():
    campaign_path = OUT/"campaign.json"
    campaign = json.loads(campaign_path.read_text())
    require_frozen(campaign["freeze"], PATHS)
    started = json.loads((OUT/"campaign-started.json").read_text())
    if (started["freeze"] != campaign["freeze"] or started["arms"] != list(ARMS)
            or [p["arm"] for p in campaign["processes"]] != list(ARMS)
            or campaign["only_artificial_inputs_and_untrained_models"] is not True
            or campaign["paid_spend_usd"] != 0):
        raise ValueError("Entire two-arm terminal campaign required")
    limit_resources(120, 100)
    wall, cpu = time.monotonic(), time.process_time()
    outcomes, row_sets = {}, {}
    try:
        for process in campaign["processes"]:
            arm = process["arm"]
            verify(process["log"])
            result, failure = OUT/f"{arm}.json", OUT/f"{arm}-failure.json"
            if result.exists() and failure.exists():
                raise ValueError("Contradictory success/failure")
            if process["returncode"] == 0 and (not result.exists() or failure.exists() or process["outer_timeout"]):
                raise ValueError("False successful process label")
            if process["returncode"] != 0 and result.exists():
                raise ValueError("Failed process cannot be admitted as success")
            if not result.exists() and not failure.exists():
                outcomes[arm] = {"status": "missing_after_terminal_failure", "process": process}
                continue
            path = result if result.exists() else failure
            value = json.loads(path.read_text())
            if value["arm"] != arm or value["freeze"] != campaign["freeze"]:
                raise ValueError("Arm/freeze identity differs")
            verify(value["inputs"])
            items = load_archive(value["inputs"])["items"]
            expected = json.loads(json.dumps(workloads()))
            if items != expected:
                raise ValueError("Artificial generation replay differs")
            verify(value["trace"])
            trace = [json.loads(line) for line in (ROOT/value["trace"]["path"]).read_text().splitlines()]
            rows = value["completed"]
            expected_order = [(seed, i) for seed in SEEDS for i in range(3)]
            order = [(row["seed"], row["item"]) for row in rows]
            if order != expected_order[:len(order)] or (result.exists() and order != expected_order):
                raise ValueError("Missing, reordered or duplicated workload outcomes")
            for row in rows:
                item = items[row["item"]]
                verify(row["reading"])
                reading = load_archive(row["reading"])
                checks, decoded = reading["checks"], reading["reading"]
                if checks != row["checks"] or checks["maximum_delta"] > .002:
                    raise ValueError("Independent full-history numeric check failed")
                texts = decoded["plaintexts"]
                compatible = []
                for k, key in enumerate(item["keys"]):
                    if ["".join(key[ALPHABET.index(c)] for c in text) for text in texts] == item["records"]:
                        compatible.append(k)
                if compatible != decoded["compatible_key_indices"] or not compatible:
                    raise ValueError("Literal global-key replay failed")
                if (row["parameters"] != 7_405_079 or row["key_count"] != item["key_count"]
                        or row["beam"] != item["beam"] or decoded["support_empty"]
                        or decoded["interval_certificate"] or decoded["exact_evidence_computed"]
                        or decoded["expanded_prefixes"] > 100_000 or decoded["distinct_source_prefixes"] > 100_000
                        or decoded["channel_cells"] != len(item["keys"])*sum(len(r)+1 for r in item["records"])
                        or decoded["source_cache_hits"]+decoded["distinct_source_prefixes"] != decoded["expanded_prefixes"]
                        or abs(abs(checks["full_sequence_cpu_float64_score"]-decoded["joint_log_probability"])
                               -checks["maximum_delta"]) > 1e-12):
                    raise ValueError("Dimension, bound, work or score arithmetic differs")
                transitions = [r for r in trace if r.get("stage") == "transition" and r.get("seed") == row["seed"]
                               and r.get("item") == row["item"]]
                if (not transitions or transitions[-1]["real_rows"] != decoded["distinct_source_prefixes"]
                        or transitions[-1]["calls"] != len(transitions)
                        or any(r["calls"] != i+1 for i, r in enumerate(transitions))):
                    raise ValueError("Provider transition accounting differs")
                if arm == "fixed8-host" and (any(r["shape"] != [8, 1] for r in transitions)
                        or transitions[-1]["real_rows"]+transitions[-1]["padded_rows"] != 8*len(transitions)):
                    raise ValueError("Dummy transition rows leaked or fixed shape differs")
            if result.exists() and (value["status"] != "PASS"
                    or value["sampled_driver_peak"] != max(r["driver_bytes"] for r in trace)
                    or value["sampled_tensor_peak"] != max(r["tensor_bytes"] for r in trace)
                    or max(r["driver_bytes"] for r in trace) > 2*1024**3
                    or max(r["peak_rss_bytes"] for r in trace) > 4*1024**3):
                raise ValueError("Successful arm violated sampled resources")
            outcomes[arm] = {"status": "PASS" if result.exists() else "retained_failure",
                             "result": artifact(path), "completed_workloads": len(rows),
                             "sampled_driver_peak": max((r["driver_bytes"] for r in trace), default=0),
                             "sampled_tensor_peak": max((r["tensor_bytes"] for r in trace), default=0)}
            row_sets[arm] = rows
        comparisons = []
        if len(row_sets) == 2:
            for a, b in zip(row_sets[ARMS[0]], row_sets[ARMS[1]], strict=False):
                if (a["seed"], a["item"], a["initial_weights_sha256"]) != (b["seed"], b["item"], b["initial_weights_sha256"]):
                    raise ValueError("Paired random initialization differs")
                av, bv = load_archive(a["reading"])["reading"], load_archive(b["reading"])["reading"]
                comparisons.append({"seed": a["seed"], "item": a["item"], "same_text_tuple": av["plaintexts"] == bv["plaintexts"],
                                    "variable_seconds": a["wall_seconds"], "fixed_seconds": b["wall_seconds"],
                                    "score_difference": bv["joint_log_probability"]-av["joint_log_probability"]})
        save_new(OUT/"audit.json", {"status": "PASS", "campaign": artifact(campaign_path), "arms": outcomes,
                 "comparisons": comparisons, "auditor": artifact(ROOT/"scripts/audit_shared_prefix_systems001.py"),
                 "scope": "inventory, hashes, whole-key literal support, saved independent CPU score arithmetic, work and sampled resources",
                 "new_model_scores_computed": False, "independent_agent_review": False, "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
