#!/usr/bin/env python3
"""Independent compact-artifact audit for TEACH-0006."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from voynich.workspace.teacher2_train import wilson_95
from voynich.workspace.teacher5_intervene import build_groups, sha_file, stable_json
from voynich.workspace.teacher6_geometry import Config


ROOT = Path(__file__).resolve().parents[1]


def rate(correct, total):
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "wilson_95": wilson_95(correct, total)}


def score(rows, prediction):
    donor = sum(row[prediction] == row["donor_answer"] for row in rows)
    base = sum(row[prediction] == row["base_answer"] for row in rows)
    groups = sorted({row["group"] for row in rows})
    exact = sum(all(row[prediction] == row["donor_answer"]
                    for row in rows if row["group"] == group) for group in groups)
    return {"donor_items": rate(donor, len(rows)), "base_items": rate(base, len(rows)),
            "donor_groups": rate(exact, len(groups))}


def independent_decision(metrics, cross):
    clauses, axes = {}, {}
    for rep, row in metrics.items():
        scores = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean"]["base_items"]["accuracy"] >= .95,
            "full_patch_at_least_0.95": scores["full"]["donor_items"]["accuracy"] >= .95,
            "rank_at_most_11": row["rank"] <= 11,
            "key_span_items_at_least_0.80": scores["key_span"]["donor_items"]["accuracy"] >= .80,
            "key_span_groups_at_least_0.65": scores["key_span"]["donor_groups"]["accuracy"] >= .65,
            "complement_preserves_base_at_least_0.90": scores["complement"]["base_items"][
                "accuracy"] >= .90,
            "complement_donor_at_most_0.10": scores["complement"]["donor_items"][
                "accuracy"] <= .10,
            "random_p95_advantage_at_least_0.40": scores["key_span"]["donor_items"]["accuracy"]
                - row["random_subspace_p95"] >= .40,
            "nearest_centroid_at_least_0.90": row[
                "nearest_centroid_confirmation_accuracy"] >= .90,
        }
        native = row["native_neuron_scores"][str(row["rank"])]["donor_items"]["accuracy"]
        axes[rep] = {"native_top_rank_at_least_0.80": native >= .80,
                     "rotation_p95_advantage_at_least_0.30":
                         native - row["rotated_top_rank_p95"] >= .30}
    subspace = "causal_subspace_supported" if all(all(row.values()) for row in clauses.values()) \
        else "not_supported"
    alignment_clauses = {name: {
        "donor_accuracy_at_least_0.80": cross[name]["donor_accuracy"] >= .80,
        "shuffled_advantage_at_least_0.40": cross[name]["advantage"] >= .40,
    } for name in ("0_to_1", "1_to_0")}
    alignment = "cross_seed_causal_alignment" if all(
        all(row.values()) for row in alignment_clauses.values()) else "not_supported"
    axis = "native_axis_concentrated" if all(all(row.values()) for row in axes.values()) \
        else "not_supported"
    return {"causal_subspace": subspace, "causal_subspace_clauses": clauses,
            "cross_seed_alignment": alignment, "cross_seed_clauses": alignment_clauses,
            "native_axis": axis, "native_axis_clauses": axes}


def verify_sources(report):
    revision = report["source_git_head"]
    for path, expected in report["source_sha256"].items():
        raw = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                             capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AssertionError(f"Source mismatch: {path}")


def validate_metrics(row, config):
    rank = row["rank"]
    if not 1 <= rank <= 11 or len(row["singular_values"]) != 12:
        raise AssertionError("Invalid centroid rank/spectrum")
    energy = row["centroid_energy_fraction"]
    if len(energy) != 12 or not math.isclose(sum(energy), 1.0, abs_tol=1e-5):
        raise AssertionError("Invalid centroid energy fractions")
    if sum(row["key_counts_discovery"].values()) != 2 * config.discovery_groups or \
            sum(row["key_counts_confirmation"].values()) != 2 * config.confirmation_groups:
        raise AssertionError("Invalid key counts")
    leverage = row["leverage"]
    if len(leverage) != 128 or not math.isclose(sum(leverage), rank, abs_tol=1e-4):
        raise AssertionError("Invalid projector leverage")
    if len(row["random_subspace_donor_accuracies"]) != config.random_subspaces or \
            len(row["rotated_top_rank_donor_accuracies"]) != config.rotations:
        raise AssertionError("Wrong control counts")
    expected_p95 = sorted(row["random_subspace_donor_accuracies"])[
        math.ceil(.95 * config.random_subspaces) - 1]
    expected_rotated = sorted(row["rotated_top_rank_donor_accuracies"])[
        math.ceil(.95 * config.rotations) - 1]
    if row["random_subspace_p95"] != expected_p95 or \
            row["rotated_top_rank_p95"] != expected_rotated:
        raise AssertionError("Control percentile mismatch")
    if not set(("0.0", "0.25", "0.5", "0.75", "1.0")) <= set(row["trajectory"]):
        raise AssertionError("Incomplete interpolation trajectory")
    if row["locality"]["items"] != 128:
        raise AssertionError("Wrong locality item count")


def audit(result_dir=ROOT / "results" / "TEACH-0006",
          checkpoint_dir=ROOT / "outputs" / "TEACH-0004"):
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    config = Config(**report["config"])
    config.validate()
    if report.get("experiment") != "TEACH-0006" or report.get("status") != "complete":
        raise AssertionError("Incomplete TEACH-0006 report")
    verify_sources(report)
    groups = build_groups(config.suite_seed,
                          config.discovery_groups + config.confirmation_groups)
    if hashlib.sha256(stable_json(groups).encode()).hexdigest() != report["suite_sha256"]:
        raise AssertionError("Suite hash mismatch")
    confirmation = groups[config.discovery_groups:]
    rescored = {}
    for rep in ("0", "1"):
        checkpoint = checkpoint_dir / f"rep{rep}-two_read.pt"
        if sha_file(checkpoint) != report["artifacts"][rep]["checkpoint_sha256"]:
            raise AssertionError("Checkpoint hash mismatch")
        path = result_dir / f"rows-rep{rep}.json.gz"
        if sha_file(path) != report["artifacts"][rep]["rows_sha256"]:
            raise AssertionError("Rows hash mismatch")
        rows = json.loads(gzip.decompress(path.read_bytes()))
        if len(rows) != config.confirmation_groups * config.g_tables:
            raise AssertionError("Wrong row count")
        for index, row in enumerate(rows):
            group_index, g_index = divmod(index, config.g_tables)
            group = confirmation[group_index]
            if row["group"] != group_index or row["g_index"] != g_index or \
                    row["base_answer"] != group["base_answers"][g_index] or \
                    row["donor_answer"] != group["donor_answers"][g_index]:
                raise AssertionError("Row label/order mismatch")
            for field in ("clean_prediction", "full_prediction", "key_span_prediction",
                          "complement_prediction"):
                if not isinstance(row[field], int) or not 0 <= row[field] < 45:
                    raise AssertionError("Invalid prediction")
        rescored[rep] = {
            "clean": score(rows, "clean_prediction"),
            "full": score(rows, "full_prediction"),
            "key_span": score(rows, "key_span_prediction"),
            "complement": score(rows, "complement_prediction"),
        }
        if stable_json(rescored[rep]) != stable_json(report["metrics"][rep]["scores"]):
            raise AssertionError("Primary score mismatch")
        validate_metrics(report["metrics"][rep], config)
    decision = independent_decision(report["metrics"], report["cross_seed"])
    if decision != report["decision"]:
        raise AssertionError("Decision mismatch")
    payload = {"experiment": "TEACH-0006", "audit": "pass",
               "source_git_head": report["source_git_head"],
               "suite_sha256": report["suite_sha256"], "primary_scores": rescored,
               "decision": decision, "report_sha256": sha_file(report_path),
               "rows_sha256": {rep: report["artifacts"][rep]["rows_sha256"]
                               for rep in ("0", "1")}}
    (result_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    result = audit()
    print(json.dumps({"audit": result["audit"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
