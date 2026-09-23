#!/usr/bin/env python3
"""Independent compact-artifact audit for TEACH-0007."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from voynich.workspace.teacher2_train import wilson_95
from voynich.workspace.teacher5_intervene import build_groups, sha_file, stable_json
from voynich.workspace.teacher7_dense_mechanism import Config


ROOT = Path(__file__).resolve().parents[1]


def rate(correct, total):
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "wilson_95": wilson_95(correct, total)}


def score(rows, condition, g_tables=3):
    field = f"{condition}_prediction"
    donor = sum(row[field] == row["donor_answer"] for row in rows)
    base = sum(row[field] == row["base_answer"] for row in rows)
    groups = sorted({row["group"] for row in rows})
    exact = sum(all(row[field] == row["donor_answer"]
                    for row in rows if row["group"] == group) for group in groups)
    cross = [row for row in rows if row["g_index"]]
    non_injection = sum(row[field] != next(
        candidate["donor_answer"] for candidate in rows
        if candidate["group"] == row["group"] and candidate["g_index"] == 0)
                        for row in cross)
    return {"donor_items": rate(donor, len(rows)), "base_items": rate(base, len(rows)),
            "donor_groups": rate(exact, len(groups)),
            "cross_g_non_injection": rate(non_injection, len(cross))}


def independent_site(screens):
    candidates = []
    for cut in range(5):
        for slot in range(6):
            key = f"{cut}:{slot}"
            candidates.append((cut, slot,
                               min(row[key]["donor_groups"]["accuracy"]
                                   for row in screens.values()),
                               min(row[key]["donor_items"]["accuracy"]
                                   for row in screens.values()),
                               min(row[key]["cross_g_non_injection"]["accuracy"]
                                   for row in screens.values())))
    qualified = [row for row in candidates if row[2] >= .65 and row[4] >= .90]
    if qualified:
        earliest = min(row[0] for row in qualified)
        chosen = sorted((row for row in qualified if row[0] == earliest),
                        key=lambda row: (-row[2], -row[3], row[1]))[0]
        flag = True
    else:
        chosen = sorted(candidates, key=lambda row: (-row[2], -row[3], row[0], row[1]))[0]
        flag = False
    return {"cut": chosen[0], "slot": chosen[1], "discovery_qualified": flag,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
            "minimum_cross_g_non_injection": chosen[4]}


def independent_decision(site, metrics):
    site_clauses, subspace_clauses = {}, {}
    for rep, row in metrics.items():
        scores = row["scores"]
        site_clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "selected_items_at_least_0.80": scores["selected"]["donor_items"]["accuracy"] >= .80,
            "selected_groups_at_least_0.65": scores["selected"]["donor_groups"]["accuracy"] >= .65,
            "selected_non_injection_at_least_0.90": scores["selected"][
                "cross_g_non_injection"]["accuracy"] >= .90,
            "random_advantage_at_least_0.40": scores["selected"]["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40,
            "same_key_preserves_base_at_least_0.90": scores["same_key"]["base_items"][
                "accuracy"] >= .90,
            "numerical_error_below_1e-6": row["numerical_error"] < 1e-6,
        }
        subspace_clauses[rep] = {
            "rank_at_most_11": row["rank"] <= 11,
            "key_span_items_at_least_0.75": scores["key_span"]["donor_items"]["accuracy"] >= .75,
            "key_span_groups_at_least_0.60": scores["key_span"]["donor_groups"]["accuracy"] >= .60,
            "complement_preserves_base_at_least_0.90": scores["complement"]["base_items"][
                "accuracy"] >= .90,
            "complement_donor_at_most_0.10": scores["complement"]["donor_items"][
                "accuracy"] <= .10,
            "random_p95_advantage_at_least_0.40": scores["key_span"]["donor_items"]["accuracy"]
                - row["random_subspace_p95"] >= .40,
            "nearest_centroid_at_least_0.90": row[
                "nearest_centroid_confirmation_accuracy"] >= .90,
        }
    site_pass = site["discovery_qualified"] and all(
        all(row.values()) for row in site_clauses.values())
    subspace_pass = site_pass and all(all(row.values()) for row in subspace_clauses.values())
    return {"emergent_key_site": "supported" if site_pass else "not_supported",
            "emergent_key_site_clauses": site_clauses,
            "dense_causal_subspace": "supported" if subspace_pass else "not_supported",
            "dense_causal_subspace_clauses": subspace_clauses}


def verify_sources(report):
    for path, expected in report["source_sha256"].items():
        raw = subprocess.run(["git", "show", f"{report['source_git_head']}:{path}"],
                             cwd=ROOT, capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AssertionError(f"Source mismatch: {path}")


def audit(result_dir=ROOT / "results" / "TEACH-0007",
          checkpoint_dir=ROOT / "outputs" / "TEACH-0004"):
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    config = Config(**report["config"])
    config.validate()
    if report.get("experiment") != "TEACH-0007" or report.get("status") != "complete":
        raise AssertionError("Incomplete report")
    verify_sources(report)
    groups = build_groups(config.suite_seed,
                          config.discovery_groups + config.confirmation_groups)
    if hashlib.sha256(stable_json(groups).encode()).hexdigest() != report["suite_sha256"]:
        raise AssertionError("Suite mismatch")
    if independent_site(report["discovery_screen"]) != report["site"]:
        raise AssertionError("Site selection mismatch")
    confirmation = groups[config.discovery_groups:]
    rescored = {}
    conditions = ("clean_base", "clean_donor", "selected", "random", "same_key",
                  "neighbor", "f_positive", "all_slots", "key_span", "complement")
    for rep in ("0", "1"):
        checkpoint = checkpoint_dir / f"rep{rep}-dense_row.pt"
        if sha_file(checkpoint) != report["artifacts"][rep]["checkpoint_sha256"]:
            raise AssertionError("Checkpoint mismatch")
        path = result_dir / f"rows-rep{rep}.json.gz"
        if sha_file(path) != report["artifacts"][rep]["rows_sha256"]:
            raise AssertionError("Rows mismatch")
        rows = json.loads(gzip.decompress(path.read_bytes()))
        if len(rows) != config.confirmation_groups * config.g_tables:
            raise AssertionError("Row count mismatch")
        for index, row in enumerate(rows):
            group, g_index = divmod(index, config.g_tables)
            if row["group"] != group or row["g_index"] != g_index or \
                    row["base_answer"] != confirmation[group]["base_answers"][g_index] or \
                    row["donor_answer"] != confirmation[group]["donor_answers"][g_index]:
                raise AssertionError("Row order/label mismatch")
        rescored[rep] = {name: score(rows, name, config.g_tables) for name in conditions}
        if stable_json(rescored[rep]) != stable_json(report["metrics"][rep]["scores"]):
            raise AssertionError("Score mismatch")
        metric = report["metrics"][rep]
        if not 1 <= metric["rank"] <= 11 or len(metric["random_subspace_accuracies"]) != 32 \
                or len(metric["rotated_top_rank_accuracies"]) != 16:
            raise AssertionError("Geometry-control mismatch")
        expected = sorted(metric["random_subspace_accuracies"])[math.ceil(.95 * 32) - 1]
        if expected != metric["random_subspace_p95"]:
            raise AssertionError("Random percentile mismatch")
    decision = independent_decision(report["site"], report["metrics"])
    if decision != report["decision"]:
        raise AssertionError("Decision mismatch")
    payload = {"experiment": "TEACH-0007", "audit": "pass",
               "source_git_head": report["source_git_head"], "site": report["site"],
               "primary_scores": rescored, "decision": decision,
               "suite_sha256": report["suite_sha256"],
               "report_sha256": sha_file(report_path)}
    (result_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    result = audit()
    print(json.dumps({"audit": result["audit"], "site": result["site"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
