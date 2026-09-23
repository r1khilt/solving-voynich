#!/usr/bin/env python3
"""Independent artifact and decision audit for TEACH-0005."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from voynich.workspace.teacher2_train import wilson_95
from voynich.workspace.teacher4_tasks import symbolic_oracle
from voynich.workspace.teacher5_intervene import Config, build_groups, stable_json


ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rate(correct, total):
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "wilson_95": wilson_95(correct, total)}


def independent_summary(rows):
    groups = sorted({row["group"] for row in rows})
    eligible_groups = [group for group in groups if all(
        row["eligible_group"] for row in rows if row["group"] == group)]
    eligible = [row for row in rows if row["group"] in eligible_groups]
    summary = {"clean_groups": rate(len(eligible_groups), len(groups))}
    if not eligible:
        return summary
    targets = {
        "donor_first": "donor_answer", "random_first": "donor_answer",
        "zero_first": "base_answer", "same_key_first": "base_answer",
        "attention_only": "base_answer", "rescue_base_first": "base_answer",
        "full_second": "donor_reference_answer", "clean_base": "base_answer",
        "clean_donor": "donor_answer", "clean_direct": "direct_answer",
        "patched_direct": "direct_answer", "clean_copy": "copy_answer",
        "patched_copy": "copy_answer",
    }
    for condition, target in targets.items():
        correct = sum(row[condition]["prediction"] == row[target] for row in eligible)
        summary[condition] = rate(correct, len(eligible))
    group_exact = sum(all(row["donor_first"]["prediction"] == row["donor_answer"]
                          for row in eligible if row["group"] == group)
                      for group in eligible_groups)
    summary["donor_first_groups"] = rate(group_exact, len(eligible_groups))
    cross = [row for row in eligible if row["g_index"] in (1, 2)]
    summary["donor_first_cross_g_non_injection"] = rate(sum(
        row["donor_first"]["prediction"] != row["donor_reference_answer"] for row in cross),
        len(cross))
    deltas = [row["donor_first"]["donor_target_probability"]
              - row["clean_base"]["donor_target_probability"] for row in eligible]
    summary["donor_target_probability_delta"] = {
        "mean": sum(deltas) / len(deltas), "minimum": min(deltas), "maximum": max(deltas)}
    summary["zero_first_base_accuracy_drop"] = (
        summary["clean_base"]["accuracy"] - summary["zero_first"]["accuracy"])
    summary["donor_first_random_advantage"] = (
        summary["donor_first"]["accuracy"] - summary["random_first"]["accuracy"])
    summary["direct_accuracy_drop"] = (
        summary["clean_direct"]["accuracy"] - summary["patched_direct"]["accuracy"])
    summary["copy_accuracy_drop"] = (
        summary["clean_copy"]["accuracy"] - summary["patched_copy"]["accuracy"])
    return summary


def independent_decision(summaries, numerical):
    clauses = {}
    for rep, summary in summaries.items():
        clauses[rep] = {
            "clean_groups_at_least_0.95": summary["clean_groups"]["accuracy"] >= .95,
            "donor_first_groups_at_least_0.80": summary.get("donor_first_groups", {}).get(
                "accuracy", 0) >= .80,
            "donor_first_items_at_least_0.90": summary.get("donor_first", {}).get(
                "accuracy", 0) >= .90,
            "advantage_over_random_at_least_0.40": summary.get(
                "donor_first_random_advantage", -1) >= .40,
            "probability_delta_at_least_0.40": summary.get(
                "donor_target_probability_delta", {}).get("mean", -1) >= .40,
            "cross_g_non_injection_at_least_0.90": summary.get(
                "donor_first_cross_g_non_injection", {}).get("accuracy", 0) >= .90,
            "full_second_positive_at_least_0.80": summary.get("full_second", {}).get(
                "accuracy", 0) >= .80,
            "zero_necessity_drop_at_least_0.30": summary.get(
                "zero_first_base_accuracy_drop", -1) >= .30,
            "rescue_at_least_0.95": summary.get("rescue_base_first", {}).get(
                "accuracy", 0) >= .95,
            "same_key_at_least_0.90": summary.get("same_key_first", {}).get(
                "accuracy", 0) >= .90,
            "attention_only_at_least_0.90": summary.get("attention_only", {}).get(
                "accuracy", 0) >= .90,
            "direct_drop_at_most_0.05": summary.get("direct_accuracy_drop", 1) <= .05,
            "copy_drop_at_most_0.05": summary.get("copy_accuracy_drop", 1) <= .05,
            "numerical_error_below_1e-6": numerical[rep] < 1e-6,
        }
    if not all(row["clean_groups_at_least_0.95"] and
               row["full_second_positive_at_least_0.80"] for row in clauses.values()):
        verdict = "inconclusive"
    else:
        verdict = "causal_key_supported" if all(all(row.values()) for row in clauses.values()) \
            else "not_supported"
    return {"verdict": verdict, "clauses_by_seed": clauses}


def verify_source(report):
    revision = report["source_git_head"]
    for path, expected in report["source_sha256"].items():
        raw = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                             capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AssertionError(f"Frozen source mismatch: {path}")


def validate_rows(rows, groups, config):
    if len(rows) != config.groups * config.g_tables:
        raise AssertionError("Wrong row count")
    condition_names = (
        "clean_base", "clean_donor", "donor_first", "random_first", "zero_first",
        "same_key_first", "attention_only", "rescue_base_first", "full_second",
        "clean_direct", "patched_direct", "clean_copy", "patched_copy",
    )
    for index, row in enumerate(rows):
        group_index, g_index = divmod(index, config.g_tables)
        group = groups[group_index]
        if row["group"] != group_index or row["g_index"] != g_index:
            raise AssertionError("Row ordering changed")
        for field, source in (("base_tokens", "base"), ("donor_tokens", "donor"),
                              ("direct_tokens", "direct"), ("copy_tokens", "copy")):
            if row[field] != group[source][g_index]:
                raise AssertionError(f"Token mismatch: {field}")
        if row["same_key_tokens"] != group["same_key"]:
            raise AssertionError("Same-key token mismatch")
        labels = {
            "base_answer": group["base_answers"][g_index],
            "donor_answer": group["donor_answers"][g_index],
            "donor_reference_answer": group["donor_answers"][0],
            "direct_answer": group["direct_answers"][g_index],
            "copy_answer": group["copy_answers"][g_index],
        }
        if any(row[key] != value for key, value in labels.items()):
            raise AssertionError("Stored answer mismatch")
        if any(symbolic_oracle(tuple(row[field])) != row[label] for field, label in (
                ("base_tokens", "base_answer"), ("donor_tokens", "donor_answer"),
                ("direct_tokens", "direct_answer"), ("copy_tokens", "copy_answer"))):
            raise AssertionError("Independent oracle mismatch")
        for condition in condition_names:
            payload = row[condition]
            if not isinstance(payload["prediction"], int) or not 0 <= payload["prediction"] < 45:
                raise AssertionError("Invalid prediction")
            if not 0 <= payload["target_probability"] <= 1:
                raise AssertionError("Invalid probability")
        expected_eligible = all(
            rows[group_index * config.g_tables + j][condition]["prediction"] == target[j]
            for j in range(config.g_tables)
            for condition, target in (("clean_base", group["base_answers"]),
                                      ("clean_donor", group["donor_answers"])))
        if row["eligible_group"] is not expected_eligible:
            raise AssertionError("Eligibility mismatch")


def audit(result_dir=ROOT / "results" / "TEACH-0005",
          checkpoint_dir=ROOT / "outputs" / "TEACH-0004"):
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    config = Config(**report["config"])
    config.validate()
    if report.get("experiment") != "TEACH-0005" or report.get("status") != "complete":
        raise AssertionError("Incomplete TEACH-0005 report")
    verify_source(report)
    groups = build_groups(config.suite_seed, config.groups)
    if hashlib.sha256(stable_json(groups).encode()).hexdigest() != report["suite_sha256"]:
        raise AssertionError("Suite hash mismatch")
    summaries = {}
    for rep in ("0", "1"):
        checkpoint = checkpoint_dir / f"rep{rep}-two_read.pt"
        if sha_file(checkpoint) != report["artifacts"][rep]["checkpoint_sha256"]:
            raise AssertionError("Checkpoint hash mismatch")
        path = result_dir / f"rows-rep{rep}.json.gz"
        if sha_file(path) != report["artifacts"][rep]["rows_sha256"]:
            raise AssertionError("Row archive hash mismatch")
        rows = json.loads(gzip.decompress(path.read_bytes()))
        validate_rows(rows, groups, config)
        summaries[rep] = independent_summary(rows)
    if stable_json(summaries) != stable_json(report["summaries"]):
        raise AssertionError("Summary mismatch")
    decision = independent_decision(summaries, report["numerical_max_recompute_error"])
    if decision != report["decision"]:
        raise AssertionError("Decision mismatch")
    payload = {"experiment": "TEACH-0005", "audit": "pass",
               "source_git_head": report["source_git_head"],
               "suite_sha256": report["suite_sha256"], "summaries": summaries,
               "decision": decision, "report_sha256": sha_file(report_path),
               "rows_sha256": {rep: report["artifacts"][rep]["rows_sha256"]
                               for rep in ("0", "1")}}
    (result_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    try:
        result = audit()
    except Exception as exc:
        print(json.dumps({"audit": "fail", "reason": f"{type(exc).__name__}: {exc}"},
                         indent=2))
        raise
    print(json.dumps({"audit": result["audit"],
                      "decision": result["decision"]["verdict"]}, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
