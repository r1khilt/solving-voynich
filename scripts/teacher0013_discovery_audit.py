#!/usr/bin/env python3
"""Independent CPU audit of TEACH-0013 compact residual-discovery maps.

This module intentionally imports neither the neural model nor TEACH-0013 generator,
intervention, scoring, or campaign code.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECONDARY_CONDITIONS = {
    "f_binding": 1, "g_content": 1, "g_binding": 1,
    "order": 2, "format": 1, "distractor": 1,
}
THRESHOLDS = {
    "item_accuracy": .75, "group_accuracy": .60,
    "non_injection": .90, "mean_probability_gain": .35,
}
STRATA = ("all", "marked", "marker_free", "first_half", "second_half")


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_rows(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def effect(rows):
    need(rows, "Empty intervention cell")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["group_id"], row["recipient"])
        need(key not in seen, "Duplicate group/recipient row")
        seen.add(key)
        need(row["recipient"] in (0, 1, 2), "Bad recipient index")
        grouped[row["group_id"]].append(row)
    need(all({row["recipient"] for row in group} == {0, 1, 2}
             for group in grouped.values()), "Incomplete recipient group")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    exact = sum(all(row["prediction"] == row["target"] for row in group)
                for group in grouped.values())
    changed = [row for row in rows if row["recipient"] != 0]
    non_injected = sum(row["prediction"] != row["fixed_donor_answer"] for row in changed)
    gains = [row["target_probability"] - row["base_target_probability"] for row in rows]
    return {"items": len(rows), "correct": correct, "item_accuracy": correct / len(rows),
            "groups": len(grouped), "exact_groups": exact,
            "group_accuracy": exact / len(grouped),
            "changed_recipient_items": len(changed), "non_injected": non_injected,
            "non_injection": non_injected / len(changed),
            "mean_probability_gain": sum(gains) / len(gains)}


def stratum(rows, name):
    if name == "all":
        return rows
    if name in ("marked", "marker_free"):
        return [row for row in rows if row["render_stratum"] == name]
    return [row for row in rows if row["position_stratum"] == name]


def primary_selection(rows, labels, expected_groups):
    label_index = {label: index for index, label in enumerate(labels)}
    cells = defaultdict(list)
    for row in rows:
        cells[(row["replicate"], row["cut_index"], row["semantic_label"])].append(row)
    table, qualified = {}, []
    for cut, label in sorted({(cut, label) for _, cut, label in cells}):
        by_seed = {}
        for replicate in (0, 1):
            need((replicate, cut, label) in cells, "Missing primary seed cell")
            by_seed[str(replicate)] = {
                name: effect(selected) if (selected := stratum(
                    cells[(replicate, cut, label)], name)) else None
                for name in STRATA
            }
        complete = all(
            metrics["all"] is not None and metrics["all"]["groups"] == 2 * expected_groups
            and metrics["marked"] is not None
            and metrics["marked"]["groups"] == expected_groups
            and metrics["marker_free"] is not None
            and metrics["marker_free"]["groups"] == expected_groups
            for metrics in by_seed.values())
        qualifies = complete and all(
            value is not None
            and value["item_accuracy"] >= THRESHOLDS["item_accuracy"]
            and value["group_accuracy"] >= THRESHOLDS["group_accuracy"]
            and value["non_injection"] >= THRESHOLDS["non_injection"]
            and value["mean_probability_gain"] >= THRESHOLDS["mean_probability_gain"]
            for metrics in by_seed.values() for value in metrics.values())
        existing = [value for metrics in by_seed.values() for value in metrics.values()
                    if value is not None]
        entry = {"cut_index": cut, "semantic_label": label, "qualified": qualifies,
                 "by_seed": by_seed,
                 "minimum_group_accuracy": min(x["group_accuracy"] for x in existing),
                 "minimum_item_accuracy": min(x["item_accuracy"] for x in existing)}
        table[f"{cut}:{label}"] = entry
        if qualifies:
            qualified.append(entry)
    if qualified:
        earliest = min(row["cut_index"] for row in qualified)
        chosen = sorted((row for row in qualified if row["cut_index"] == earliest),
                        key=lambda row: (-row["minimum_group_accuracy"],
                                         -row["minimum_item_accuracy"],
                                         label_index[row["semantic_label"]]))[0]
        selection = {key: value for key, value in chosen.items() if key != "by_seed"}
    else:
        selection = None
    return {"thresholds": THRESHOLDS, "strata": list(STRATA),
            "expected_groups_per_render": expected_groups,
            "semantic_label_order": list(labels), "candidates": table,
            "selection": selection}


def secondary_selection(rows, labels, expected_groups, family):
    label_index = {label: index for index, label in enumerate(labels)}
    cells = defaultdict(list)
    for row in rows:
        cells[(row["replicate"], row["cut_index"], row["semantic_label"])].append(row)
    table = {}
    for cut, label in sorted({(cut, label) for _, cut, label in cells}):
        scores = []
        by_seed = {}
        for replicate in (0, 1):
            need((replicate, cut, label) in cells, "Missing secondary seed cell")
            score = effect(cells[(replicate, cut, label)])
            by_seed[str(replicate)] = score
            scores.append(score)
        entry = {"cut_index": cut, "semantic_label": label,
                 "complete": all(row["groups"] == expected_groups for row in scores),
                 "by_seed": by_seed,
                 "minimum_group_accuracy": min(row["group_accuracy"] for row in scores),
                 "minimum_item_accuracy": min(row["item_accuracy"] for row in scores),
                 "minimum_mean_probability_gain": min(
                     row["mean_probability_gain"] for row in scores)}
        table[f"{cut}:{label}"] = entry
    eligible = [row for row in table.values() if row["complete"]]
    if eligible:
        chosen = sorted(eligible, key=lambda row: (
            -row["minimum_group_accuracy"], -row["minimum_item_accuracy"],
            -row["minimum_mean_probability_gain"], row["cut_index"],
            label_index[row["semantic_label"]]))[0]
        selection = {key: value for key, value in chosen.items() if key != "by_seed"}
    else:
        selection = None
    return {"family": family, "expected_groups": expected_groups,
            "semantic_label_order": list(labels),
            "ranking_rule": ("max_min_group_then_item_then_probability_gain_then_earlier_cut_"
                             "then_semantic_label"),
            "candidates": table, "selection": selection}


def validate_row(row, *, family, replicate, labels, group_ids, sites):
    required = {"family", "replicate", "cut_index", "site", "semantic_label",
                "group_id", "logical_group_id", "recipient", "render_stratum",
                "position_stratum", "target", "prediction", "base_prediction",
                "base_answer", "base_correct", "preserves_base_prediction",
                "preserves_base_answer", "fixed_donor_answer", "candidate_member",
                "wrong_destination", "target_probability", "base_target_probability",
                "target_logit", "base_target_logit", "prediction_logit"}
    need(set(row) == required, "Discovery row fields changed")
    need(row["family"] == family and row["replicate"] == replicate,
         "Row family/replicate mismatch")
    need(type(row["cut_index"]) is int and 0 <= row["cut_index"] < len(sites)
         and row["site"] == sites[row["cut_index"]], "Cut/site mismatch")
    need(row["semantic_label"] in labels and row["logical_group_id"] in group_ids,
         "Unregistered label or group")
    need(row["group_id"].startswith(row["logical_group_id"] + ":"), "Bad grouped ID")
    need(row["recipient"] in (0, 1, 2)
         and row["position_stratum"] in ("first_half", "second_half"),
         "Bad recipient or position stratum")
    need(row["base_correct"] == (row["base_prediction"] == row["base_answer"])
         and row["preserves_base_prediction"] == (row["prediction"] == row["base_prediction"])
         and row["preserves_base_answer"] == (row["prediction"] == row["base_answer"]),
         "Derived base flags disagree")
    need(row["wrong_destination"] in {
        "target", "base_answer", "fixed_donor_answer", "other_recipient_target",
        "other_legal_candidate", "outside_legal_candidates"}, "Unknown destination label")
    for name in ("target_probability", "base_target_probability"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name])
             and 0 <= row[name] <= 1, f"Bad probability: {name}")
    for name in ("target_logit", "base_target_logit", "prediction_logit"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name]),
             f"Bad logit: {name}")


def audit(manifest_path: Path, discovery_path: Path, output_path: Path):
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(discovery_path.read_text())
    need(manifest.get("experiment") == report.get("experiment") == "TEACH-0013",
         "Wrong experiment identity")
    need(report.get("status") in {
        "stage_b_single_site_complete", "stage_b_residual_complete_path_pending"},
         "Discovery status is not an auditable residual outcome")
    labels = tuple(report["residual_selection"]["semantic_label_order"])
    need(list(labels) == sorted(labels) and len(labels) == len(set(labels)),
         "Semantic label order is not fixed and unique")
    group_ids = set(manifest["group_ids"]["discovery"])
    expected_groups = manifest["split_counts"]["discovery"]
    sites = tuple(["embed"] + [f"blocks.{index}.resid_post" for index in range(12)])
    by_family = defaultdict(list)
    verified = {}
    for replicate in ("0", "1"):
        qualification = report["numerical_qualification"][replicate]
        need(qualification["qualified"] is True
             and qualification["finite"] is True
             and qualification["maximum_identity_logit_error"] < 1e-6
             and qualification["maximum_probability_normalization_error"] <= 1e-6
             and set(qualification["identity_errors"]) == set(sites),
             f"Numerical qualification failed for replicate {replicate}")
        for family in ("f_content", *SECONDARY_CONDITIONS):
            metadata = report["artifacts"][replicate][family]
            path = ROOT / metadata["path"]
            need(path.is_file() and path.stat().st_size == metadata["bytes"]
                 and file_digest(path) == metadata["sha256"], "Row artifact mismatch")
            rows = load_rows(path)
            need(len(rows) == metadata["rows"], "Row artifact count mismatch")
            for row in rows:
                validate_row(row, family=family, replicate=int(replicate), labels=set(labels),
                             group_ids=group_ids, sites=sites)
            by_family[family].extend(rows)
            verified[f"{replicate}:{family}"] = metadata["sha256"]
    primary = primary_selection(by_family["f_content"], labels, expected_groups)
    need(primary == report["residual_selection"], "Primary residual selection does not recompute")
    secondary = {family: secondary_selection(
        by_family[family], labels, expected_groups * condition_count, family)
        for family, condition_count in SECONDARY_CONDITIONS.items()}
    need(secondary == report["secondary_residual_selections"],
         "Secondary residual selections do not recompute")
    expected_status = ("stage_b_single_site_complete" if primary["selection"]
                       else "stage_b_residual_complete_path_pending")
    need(report["status"] == expected_status, "Discovery status disagrees with selection")
    result = {"audit": "pass", "experiment": "TEACH-0013",
              "scope": "residual_discovery", "status": expected_status,
              "suite_gzip_sha256": report["suite_gzip_sha256"],
              "discovery_decision_sha256": file_digest(discovery_path),
              "verified_row_archives": verified,
              "primary_selection": primary["selection"],
              "secondary_selections": {name: row["selection"]
                                       for name, row in secondary.items()}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "results/TEACH-0013/suite-manifest.json")
    parser.add_argument("--discovery", type=Path,
                        default=ROOT / "results/TEACH-0013/discovery-decision.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/TEACH-0013/discovery-audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.discovery, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
