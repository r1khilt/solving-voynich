#!/usr/bin/env python3
"""Independent no-model audit of sealed TEACH-0013 Stage-A/C confirmation.

This file imports neither the model, generator, intervention code, campaign runner nor
experiment scoring functions. It rescales every verdict from retained item rows and
checks predictions/probabilities against the exact stored symbol-logit tensors.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
SYMBOL_START = 16
MAX_STAGE_C_SECONDS = 5400.0
MAX_CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
CONTROL_CONDITIONS = (
    "norm_matched_gaussian", "norm_matched_cyclic",
    "norm_matched_wrong_position", "norm_matched_wrong_key",
)


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


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_row(row: dict, group_ids: set[str], *, stage_a: bool) -> None:
    required = {
        "condition", "direction", "logical_group_id", "group_id", "recipient",
        "target", "prediction", "base_answer", "clean_prediction",
        "fixed_donor_answer", "candidate_member", "wrong_destination",
        "target_probability", "base_target_probability", "target_logit",
        "base_target_logit", "prediction_logit",
    }
    optional = {"pair_id"} if stage_a else {"corruption_changed_clean_prediction"}
    need(required <= set(row) and set(row) <= required | optional,
         "Confirmation row fields changed")
    need(row["logical_group_id"] in group_ids and row["recipient"] in (0, 1, 2),
         "Unknown confirmation group or recipient")
    need(row["group_id"] == f"{row['logical_group_id']}:{row['direction']}",
         "Grouped row identifier changed")
    if stage_a:
        need(row["direction"] == "stage_a" and isinstance(row.get("pair_id"), str),
             "Malformed Stage-A row")
    if "corruption_changed_clean_prediction" in row:
        need(type(row["corruption_changed_clean_prediction"]) is bool,
             "Malformed corruption flag")
    need(row["wrong_destination"] in {
        "target", "base_answer", "fixed_donor_answer", "other_recipient_target",
        "other_legal_candidate", "outside_legal_candidates"},
        "Unknown wrong-destination label")
    for name in ("target_probability", "base_target_probability"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name])
             and 0 <= row[name] <= 1, f"Bad probability: {name}")
    for name in ("target_logit", "base_target_logit", "prediction_logit"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name]),
             f"Bad logit: {name}")


def verify_logit_archive(metadata: dict, rows: list[dict]) -> dict:
    path = ROOT / metadata["path"]
    need(path.is_file() and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], "Exact-logit artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") in ("TEACH-0013-symbol-logits-v1",
                                   "TEACH-0013-symbol-logits-v2")
         and isinstance(payload.get("records"), list), "Unknown exact-logit format")
    records = payload["records"]
    need(len(records) == metadata["records"], "Exact-logit record count mismatch")
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["condition"], row["direction"])].append(row)
    offsets = defaultdict(int)
    checked = 0
    for record in records:
        required = {"condition", "direction", "logical_group_ids", "recipients", "targets",
                    "edited_symbol_logits", "clean_symbol_logits"}
        need(set(record) in (required, required | {"item_ids"}),
             "Exact-logit record fields changed")
        key = (record["condition"], record["direction"])
        edited, clean = record["edited_symbol_logits"], record["clean_symbol_logits"]
        need(isinstance(edited, torch.Tensor) and isinstance(clean, torch.Tensor)
             and edited.ndim == clean.ndim == 2 and edited.shape == clean.shape
             and edited.dtype == clean.dtype == torch.float32
             and torch.isfinite(edited).all().item()
             and torch.isfinite(clean).all().item(), "Malformed exact-logit tensor")
        offset, count = offsets[key], edited.shape[0]
        selected = by_key[key][offset:offset + count]
        need(len(selected) == count, "Exact-logit rows exceed compact rows")
        need(tuple(row["logical_group_id"] for row in selected)
             == tuple(record["logical_group_ids"])
             and tuple(row["recipient"] for row in selected) == tuple(record["recipients"])
             and tuple(row["target"] for row in selected) == tuple(record["targets"]),
             "Exact-logit metadata does not align with compact rows")
        if "item_ids" in record:
            need(tuple(row.get("item_id", f"{row['condition']}:{row['direction']}:"
                                          f"{row['logical_group_id']}:{row['recipient']}")
                       for row in selected) == tuple(record["item_ids"]),
                 "Exact-logit item IDs do not align with compact rows")
        probability = edited.softmax(-1)
        clean_probability = clean.softmax(-1)
        need(torch.allclose(probability.sum(-1), torch.ones(count), atol=1e-6, rtol=1e-6)
             and torch.allclose(clean_probability.sum(-1), torch.ones(count),
                                atol=1e-6, rtol=1e-6),
             "Exact-logit probabilities are not normalized")
        predictions = edited.argmax(-1) + SYMBOL_START
        clean_predictions = clean.argmax(-1) + SYMBOL_START
        for index, row in enumerate(selected):
            need(SYMBOL_START <= row["target"] < SYMBOL_START + edited.shape[1]
                 and SYMBOL_START <= row["prediction"] < SYMBOL_START + edited.shape[1]
                 and SYMBOL_START <= row["clean_prediction"] < SYMBOL_START + edited.shape[1],
                 "Compact symbol is outside exact-logit vocabulary")
            target_index = row["target"] - SYMBOL_START
            prediction_index = int(predictions[index].item()) - SYMBOL_START
            need(int(predictions[index].item()) == row["prediction"]
                 and int(clean_predictions[index].item()) == row["clean_prediction"],
                 "Compact prediction disagrees with exact logits")
            need(math.isclose(float(probability[index, target_index].item()),
                              row["target_probability"], abs_tol=1e-6)
                 and math.isclose(float(clean_probability[index, target_index].item()),
                                  row["base_target_probability"], abs_tol=1e-6)
                 and math.isclose(float(edited[index, target_index].item()),
                                  row["target_logit"], abs_tol=1e-7)
                 and math.isclose(float(clean[index, target_index].item()),
                                  row["base_target_logit"], abs_tol=1e-7)
                 and math.isclose(float(edited[index, prediction_index].item()),
                                  row["prediction_logit"], abs_tol=1e-7),
                 "Compact probability/logit disagrees with exact tensor")
        offsets[key] += count
        checked += count
    need(checked == len(rows) == metadata["rows"]
         and all(offsets[key] == len(value) for key, value in by_key.items()),
         "Exact-logit archive does not exhaust compact rows")
    return {"sha256": metadata["sha256"], "records": len(records), "rows": checked}


def fresh_scores(rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    required = {
        "stage_a_marked_f0", "stage_a_marked_f1",
        "stage_a_marker_free_f0", "stage_a_marker_free_f1",
        "stage_a_reordered_f0", "stage_a_reordered_f1",
        "stage_a_distractor_f0", "stage_a_distractor_f1",
        "stage_a_first_hop_f0", "stage_a_first_hop_f1",
        "stage_a_direct_f0", "stage_a_direct_f1", "stage_a_copy",
    }
    need(set(by_condition) == required, "Stage-A conditions are incomplete")

    def correct(row):
        return row["prediction"] == row["target"]

    def accuracy(names):
        selected = [row for name in names for row in by_condition[name]]
        return sum(correct(row) for row in selected) / len(selected)

    def groups(name):
        grouped = defaultdict(list)
        for row in by_condition[name]:
            grouped[row["logical_group_id"]].append(row)
        need(all({row["recipient"] for row in group} == {0, 1, 2}
                 for group in grouped.values()), "Incomplete Stage-A recipient group")
        return sum(all(correct(row) for row in group) for group in grouped.values()) \
            / len(grouped)

    def pairs(left_names, right_names):
        left = {row["pair_id"]: row for name in left_names for row in by_condition[name]}
        right = {row["pair_id"]: row for name in right_names for row in by_condition[name]}
        need(left.keys() == right.keys()
             and len(left) == sum(len(by_condition[name]) for name in left_names),
             "Incomplete or duplicate Stage-A pairs")
        return sum(correct(left[key]) and correct(right[key]) for key in left) / len(left)

    marked = ("stage_a_marked_f0", "stage_a_marked_f1")
    return {
        "composed_items": accuracy(marked),
        "base_recipient_groups": groups("stage_a_marked_f0"),
        "donor_recipient_groups": groups("stage_a_marked_f1"),
        "marker_pairs": pairs(marked, (
            "stage_a_marker_free_f0", "stage_a_marker_free_f1")),
        "order_pairs": pairs(marked, (
            "stage_a_reordered_f0", "stage_a_reordered_f1")),
        "distractor_pairs": pairs(
            marked, ("stage_a_distractor_f0", "stage_a_distractor_f1")),
        "first_hop": accuracy(("stage_a_first_hop_f0", "stage_a_first_hop_f1")),
        "direct": accuracy(("stage_a_direct_f0", "stage_a_direct_f1")),
        "copy": accuracy(("stage_a_copy",)),
    }


def fresh_decision(scores: dict) -> dict:
    thresholds = {
        "composed_items": .90, "base_recipient_groups": .80,
        "donor_recipient_groups": .80, "marker_pairs": .85,
        "order_pairs": .85, "distractor_pairs": .85,
        "first_hop": .95, "direct": .95, "copy": .98,
    }
    clauses = {replicate: {f"{name}_at_least_{threshold:.2f}": row[name] >= threshold
                           for name, threshold in thresholds.items()}
               for replicate, row in scores.items()}
    passed = set(scores) == {"0", "1"} and all(all(row.values()) for row in clauses.values())
    return {"label": "pass" if passed else "inconclusive_fresh_panel_competence",
            "thresholds": thresholds, "clauses_by_seed": clauses}


def effect(rows: list[dict]) -> dict:
    need(rows, "Empty Stage-C effect cell")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["group_id"], row["recipient"])
        need(key not in seen, "Duplicate Stage-C group/recipient")
        seen.add(key)
        grouped[row["group_id"]].append(row)
    need(all({row["recipient"] for row in group} == {0, 1, 2}
             for group in grouped.values()), "Incomplete Stage-C recipient group")
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


def _item_accuracy(rows, condition):
    selected = [row for row in rows if row["condition"] == condition]
    need(selected, f"Missing Stage-C condition: {condition}")
    return sum(row["prediction"] == row["target"] for row in selected) / len(selected)


def _specificity_loss(rows, task):
    losses = []
    for family in ("format", "order", "distractor"):
        selected = [row for row in rows
                    if row["condition"] == f"{task}_same_key_{family}"]
        need(selected, "Missing task-specificity cell")
        clean = sum(row["clean_prediction"] == row["target"] for row in selected) \
            / len(selected)
        edited = sum(row["prediction"] == row["target"] for row in selected) \
            / len(selected)
        losses.append(max(0.0, clean - edited))
    return max(losses)


def stage_c_scores(rows: list[dict]) -> dict:
    expected_conditions = {
        "clean_base", "clean_donor_input_f_replacement",
        "final_state_fixed_answer_injection", "sufficiency_marked",
        "sufficiency_marker_free", "same_key_format", "same_key_order",
        "same_key_distractor", *CONTROL_CONDITIONS, "norm_matched_corruption",
        "native_state_rescue",
        *(f"{task}_same_key_{family}" for task in ("direct", "copy")
          for family in ("format", "order", "distractor")),
    }
    need({row["condition"] for row in rows} == expected_conditions,
         "Stage-C condition set changed")
    def selected(condition, direction):
        return [row for row in rows if row["condition"] == condition
                and row["direction"] == direction]

    true = {direction: effect(selected("sufficiency_marked", direction))
            for direction in ("forward", "reverse")}
    controls = {condition: {direction: effect(selected(condition, direction))
                            for direction in ("forward", "reverse")}
                for condition in CONTROL_CONDITIONS}
    advantages = [true[direction]["item_accuracy"]
                  - controls[condition][direction]["item_accuracy"]
                  for condition in CONTROL_CONDITIONS
                  for direction in ("forward", "reverse")]
    corrupt = {(row["group_id"], row["recipient"]): row for row in rows
               if row["condition"] == "norm_matched_corruption"}
    rescue = {(row["group_id"], row["recipient"]): row for row in rows
              if row["condition"] == "native_state_rescue"}
    need(corrupt and corrupt.keys() == rescue.keys(), "Unpaired corruption/rescue rows")
    changed = [key for key, row in corrupt.items()
               if row["prediction"] != row["clean_prediction"]]
    restored = sum(rescue[key]["prediction"] == rescue[key]["clean_prediction"]
                   for key in changed)
    rescue_summary = {"paired_items": len(corrupt), "changed_cases": len(changed),
                      "restored_cases": restored,
                      "rescue_rate": None if not changed else restored / len(changed)}
    reverse = true["reverse"]
    return {
        "clean_base": _item_accuracy(rows, "clean_base"),
        "clean_donor": _item_accuracy(rows, "clean_donor_input_f_replacement"),
        "input_f_replacement": _item_accuracy(rows, "clean_donor_input_f_replacement"),
        "sufficiency_items_forward": true["forward"]["item_accuracy"],
        "sufficiency_groups_forward": true["forward"]["group_accuracy"],
        "sufficiency_items_reverse": reverse["item_accuracy"],
        "sufficiency_groups_reverse": reverse["group_accuracy"],
        "changed_non_injection_forward": true["forward"]["non_injection"],
        "changed_non_injection_reverse": reverse["non_injection"],
        "necessity_items": reverse["item_accuracy"],
        "necessity_groups": reverse["group_accuracy"],
        "rescue": 0.0 if rescue_summary["rescue_rate"] is None
        else rescue_summary["rescue_rate"],
        "rescue_changed_cases": rescue_summary["changed_cases"],
        "same_key_format": _item_accuracy(rows, "same_key_format"),
        "same_key_order": _item_accuracy(rows, "same_key_order"),
        "same_key_distractor": _item_accuracy(rows, "same_key_distractor"),
        "minimum_control_advantage": min(advantages),
        "mean_probability_gain": min(row["mean_probability_gain"] for row in true.values()),
        "direct_loss": _specificity_loss(rows, "direct"),
        "copy_loss": _specificity_loss(rows, "copy"),
        "final_answer_injection": _item_accuracy(rows, "final_state_fixed_answer_injection"),
        "marked_summaries": true, "control_summaries": controls,
        "marker_free_summaries": {direction: effect(
            selected("sufficiency_marker_free", direction))
            for direction in ("forward", "reverse")},
        "rescue_summary": rescue_summary,
    }


def transfer_decision(scores, numerical_qualified):
    floors = {
        "clean_base": .95, "clean_donor": .95, "input_f_replacement": .95,
        "sufficiency_items_forward": .85, "sufficiency_groups_forward": .70,
        "sufficiency_items_reverse": .85, "sufficiency_groups_reverse": .70,
        "changed_non_injection_forward": .95, "changed_non_injection_reverse": .95,
        "necessity_items": .80, "necessity_groups": .65, "rescue": .95,
        "same_key_format": .95, "same_key_order": .95, "same_key_distractor": .95,
        "minimum_control_advantage": .40, "mean_probability_gain": .40,
        "final_answer_injection": .95,
    }
    ceilings = {"direct_loss": .05, "copy_loss": .05}
    clauses = {replicate: {
        **{f"{name}_at_least_{threshold:.2f}": row[name] >= threshold
           for name, threshold in floors.items()},
        **{f"{name}_at_most_{threshold:.2f}": row[name] <= threshold
           for name, threshold in ceilings.items()},
    } for replicate, row in scores.items()}
    supported = numerical_qualified and set(scores) == {"0", "1"} \
        and all(all(row.values()) for row in clauses.values())
    return {"label": "recipient_key_transfer_supported" if supported else "not_supported",
            "discovery_selected": True, "numerical_qualified": numerical_qualified,
            "floor_thresholds": floors, "ceiling_thresholds": ceilings,
            "clauses_by_seed": clauses}


def auditor_provenance() -> dict:
    relative = "scripts/teacher0013_confirmation_audit.py"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    need(not status, "Commit independent confirmation auditor before audit")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {"revision": revision, "sha256": {relative: file_digest(ROOT / relative)}}


def verify_discovery_prerequisite(report: dict) -> None:
    prerequisite = report["discovery_prerequisite"]
    decision_path, audit_path = (ROOT / prerequisite[name]
                                 for name in ("decision_path", "audit_path"))
    need(decision_path.is_file() and audit_path.is_file()
         and file_digest(decision_path) == prerequisite["decision_sha256"]
         and file_digest(audit_path) == prerequisite["audit_sha256"],
         "Frozen discovery prerequisite artifact mismatch")
    decision, independent = (json.loads(path.read_text())
                             for path in (decision_path, audit_path))
    need(independent.get("audit") == "pass", "Discovery prerequisite audit did not pass")
    mediator = report["mediator"]
    sites = tuple(["embed"] + [f"blocks.{index}.resid_post" for index in range(12)])
    if prerequisite["kind"] == "single":
        selected = decision.get("residual_selection", {}).get("selection")
        need(independent.get("scope") == "residual_discovery"
             and independent.get("primary_selection") == selected
             and mediator == {"kind": "single", "site": sites[selected["cut_index"]],
                              "label": selected["semantic_label"], "early_site": None,
                              "source_label": None, "late_site": None,
                              "destination_label": None},
             "Single-site prerequisite does not reproduce the frozen mediator")
    else:
        selected = decision.get("path_selection", {}).get("selection")
        need(prerequisite["kind"] == "path" and independent.get("scope") == "path_discovery"
             and independent.get("selection") == selected
             and mediator == {"kind": "path", "site": None, "label": None,
                              "early_site": sites[selected["early_cut_index"]],
                              "source_label": selected["source_label"],
                              "late_site": sites[selected["late_cut_index"]],
                              "destination_label": selected["destination_label"]},
             "Two-site prerequisite does not reproduce the frozen mediator")


def validate_resources(report: dict) -> None:
    prerequisite = report["discovery_prerequisite"]
    need(report["cumulative_campaign_seconds"]
         == prerequisite["stage_b_seconds"] + report["elapsed_seconds"]
         and report["cumulative_materialized_bytes"]
         == prerequisite["stage_b_materialized_bytes"]
         + report["materialized_activation_bytes"], "Cumulative resources do not reconcile")
    need(report["elapsed_seconds"] <= MAX_STAGE_C_SECONDS
         and report["cumulative_campaign_seconds"] <= MAX_CAMPAIGN_SECONDS
         and report["peak_sampled_current_allocated_bytes"] <= MAX_CURRENT_BYTES
         and report["cumulative_materialized_bytes"] <= MAX_TRAFFIC_BYTES,
         "Confirmation resource ceiling exceeded")


def audit(manifest_path: Path, report_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    need(manifest.get("experiment") == report.get("experiment") == "TEACH-0013",
         "Wrong experiment identity")
    need(report.get("suite_gzip_sha256") == manifest.get("suite_gzip_sha256"),
         "Suite hash mismatch")
    need(report.get("confirmation_source_git_head") == manifest.get("source_git_head")
         and report.get("confirmation_source_sha256") == manifest.get("source_sha256"),
         "Confirmation source provenance does not match suite freeze")
    verify_discovery_prerequisite(report)
    group_ids = set(manifest["group_ids"]["confirmation"])
    need(len(group_ids) == manifest["split_counts"]["confirmation"],
         "Confirmation group manifest is duplicated")
    verified, clean_scores = {}, {}
    for replicate in ("0", "1"):
        artifacts = report["artifacts"][replicate]
        row_meta = artifacts["stage_a_rows"]
        row_path = ROOT / row_meta["path"]
        need(row_path.is_file() and row_path.stat().st_size == row_meta["bytes"]
             and file_digest(row_path) == row_meta["sha256"], "Stage-A row artifact mismatch")
        rows = load_rows(row_path)
        need(len(rows) == row_meta["rows"] == 29 * len(group_ids),
             "Stage-A row count mismatch")
        for row in rows:
            validate_row(row, group_ids, stage_a=True)
        clean_scores[replicate] = fresh_scores(rows)
        logits = verify_logit_archive(artifacts["stage_a_exact_symbol_logits"], rows)
        verified[f"{replicate}:stage_a"] = {
            "rows_sha256": row_meta["sha256"], "logits": logits}
    need(clean_scores == report["clean_confirmation_scores"],
         "Stage-A scores do not independently recompute")
    competence = fresh_decision(clean_scores)
    need(competence == report["fresh_panel_decision"],
         "Stage-A decision does not independently recompute")
    validate_resources(report)

    if competence["label"] != "pass":
        need(report.get("status") == "inconclusive_fresh_panel_competence",
             "Fresh-panel failure status mismatch")
        result = {"audit": "pass", "experiment": "TEACH-0013",
                  "scope": "stage_a_confirmation", "status": report["status"],
                  "confirmation_decision_sha256": file_digest(report_path),
                  "verified_artifacts": verified, "fresh_panel_decision": competence,
                  "auditor": auditor_provenance()}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    scores = {}
    for replicate in ("0", "1"):
        artifacts = report["artifacts"][replicate]
        row_meta = artifacts["compact_rows"]
        row_path = ROOT / row_meta["path"]
        need(row_path.is_file() and row_path.stat().st_size == row_meta["bytes"]
             and file_digest(row_path) == row_meta["sha256"], "Stage-C row artifact mismatch")
        rows = load_rows(row_path)
        need(len(rows) == row_meta["rows"] == 72 * len(group_ids),
             "Stage-C row count mismatch")
        for row in rows:
            validate_row(row, group_ids, stage_a=False)
        scores[replicate] = stage_c_scores(rows)
        logits = verify_logit_archive(artifacts["exact_symbol_logits"], rows)
        verified[f"{replicate}:stage_c"] = {
            "rows_sha256": row_meta["sha256"], "logits": logits}
    need(scores == report["scores"], "Stage-C scores do not independently recompute")
    numerical = report["numerical_qualification"]
    numerical_pass = True
    for replicate in ("0", "1"):
        row = numerical[replicate]
        finite = all(math.isfinite(row[name]) for name in (
            "maximum_fused_instrumented_logit_error", "maximum_identity_logit_error"))
        qualified = finite and row["maximum_fused_instrumented_logit_error"] < 1e-6 \
            and row["maximum_identity_logit_error"] < 1e-6
        need(row["finite"] == finite and row["qualified"] == qualified,
             "Numerical qualification does not recompute")
        numerical_pass = numerical_pass and qualified
    decision = transfer_decision(scores, numerical_pass)
    need(decision == report["recipient_transfer_decision"],
         "Recipient-transfer verdict does not independently recompute")
    expected_status = "stage_c_complete" if numerical_pass else "stage_c_numerical_inconclusive"
    need(report["status"] == expected_status, "Stage-C status mismatch")
    result = {"audit": "pass", "experiment": "TEACH-0013",
              "scope": "stage_c_confirmation", "status": expected_status,
              "confirmation_decision_sha256": file_digest(report_path),
              "verified_artifacts": verified, "fresh_panel_decision": competence,
              "recipient_transfer_decision": decision,
              "auditor": auditor_provenance()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "results/TEACH-0013/suite-manifest.json")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results/TEACH-0013/confirmation-decision.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/TEACH-0013/confirmation-audit.json")
    args = parser.parse_args()
    result = audit(args.manifest, args.report, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
