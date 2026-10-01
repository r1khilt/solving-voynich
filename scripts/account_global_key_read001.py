"""Read-only publication accounting of every terminal prediction and archive."""
import json
import signal
import time
from collections import Counter

from scripts.global_key_read001_common import NAMES, OUT, ROOT, fit_status
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources


def main():
    campaign = json.loads((OUT/"campaign.json").read_text())
    panel, _, _, _ = fit_status(campaign["freeze"])
    audit_path = OUT/"audit.json"
    audit = json.loads(audit_path.read_text())
    if (audit["status"] != "PASS" or audit["campaign"] != artifact(OUT/"campaign.json")
            or audit["auditor"] != artifact(ROOT/"scripts/audit_global_key_read001.py")
            or audit["no_gold_opened"] is not True or set(audit["cases"]) != set(NAMES)
            or [row["case"] for row in campaign["results"]] != list(NAMES)
            or campaign["status"] != "complete" or any(row["returncode"] != 0 for row in campaign["results"])):
        raise ValueError("Matching complete no-answer audit/campaign required")
    limit_resources(300, 250)
    wall, cpu = time.monotonic(), time.process_time()
    rows, bindings, statuses = [], 0, Counter()
    try:
        for name in NAMES:
            path = OUT/f"{name}.json"
            result = json.loads(path.read_text())
            report = audit["cases"][name]
            positive = panel["cases"][name]["positive"]
            if (result["case"] != name or result["freeze"] != campaign["freeze"]
                    or report["result"] != artifact(path) or report["evidence"] != artifact(OUT/f"{name}-evidence.json")
                    or result["status"] != ("complete" if positive else "complete_evidence_only_by_design")
                    or bool(result["readings"]) != positive
                    or result["resources"]["paid_spend_usd"] != 0
                    or result["resources"]["peak_rss_bytes"] > 4*1024**3):
                raise ValueError("False complete prediction/resource binding")
            for field in ("evidence", "readings", "points", "frequency_order3", "transfer", "bank", "parent"):
                spec = result.get(field)
                if spec is not None:
                    if artifact(ROOT/spec["path"]) != spec:
                        raise ValueError("Prediction archive binding differs: "+name+"/"+field)
                    bindings += 1
            rows.append({"case": name, "positive": positive, "result": artifact(path),
                         "resources": result["resources"], "numeric_audit": report.get("reading")})
            statuses[result["status"]] += 1
        if statuses != {"complete": 16, "complete_evidence_only_by_design": 16}:
            raise ValueError("Entire positive/null scope changed")
        if list(OUT.glob("*-failure.json")):
            raise ValueError("Unaccounted contradictory failure files")
        save_new(OUT/"publication-accounting.json", {
            "status": "PASS", "freeze": campaign["freeze"], "cases": rows,
            "campaign": artifact(OUT/"campaign.json"), "audit": artifact(audit_path),
            "code": artifact(ROOT/"scripts/account_global_key_read001.py"),
            "result_archive_bindings": bindings, "outcome_counts": dict(statuses),
            "all_child_cpu_seconds": campaign["all_children_cpu_seconds_including_failed"],
            "summed_prediction_cpu_seconds": sum(r["resources"]["cpu_seconds"] for r in rows),
            "summed_prediction_wall_seconds": sum(r["resources"]["wall_seconds"] for r in rows),
            "maximum_prediction_rss_bytes": max(r["resources"]["peak_rss_bytes"] for r in rows),
            "backward_records": sum(r.get("backward_records", 0) for r in audit["cases"].values()),
            "point_record_replays": sum(r.get("point_record_replays", 0) for r in audit["cases"].values()),
            "frequency_record_replays": sum(r.get("frequency_record_replays", 0) for r in audit["cases"].values()),
            "literal_candidate_key_checks": sum(r.get("reading", {}).get("literal_candidate_key_checks", 0) for r in audit["cases"].values()),
            "sampled_candidate_tuples": sum(r.get("reading", {}).get("sampled_candidate_tuples", 0) for r in audit["cases"].values()),
            "maximum_backward_delta": max(r.get("maximum_delta", 0) for r in audit["cases"].values()),
            "answer_files_semantically_opened": False, "new_model_scores_computed": False,
            "independent_agent_review": False, "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
