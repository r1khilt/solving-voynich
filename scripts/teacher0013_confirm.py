#!/usr/bin/env python3
"""Run sealed TEACH-0013 Stage-A/C confirmation after audited discovery."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from voynich.workspace.teacher13_confirm import (
    MediatorSpec,
    bidirectional_sufficiency,
    corruption_rescue_rows,
    fresh_panel_confirmation_rows,
    identity_error,
    matched_sufficiency_controls,
    nuisance_preservation,
    positive_control_rows,
    summarize_confirmation_rows,
    summarize_fresh_panel_rows,
    summarize_rescue_rows,
    task_specificity_rows,
)
from voynich.workspace.teacher13_discovery import (
    episode_from_record,
    load_suite,
)
from voynich.workspace.teacher13_intervene import (
    load_raw_checkpoint,
    materialized_tensor_counter,
    residual_sites,
)
from voynich.workspace.teacher13_score import (
    fresh_panel_decision,
    recipient_transfer_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MAX_STAGE_C_SECONDS = 5400.0
MAX_CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3
CONTROL_CONDITIONS = (
    "norm_matched_gaussian", "norm_matched_cyclic",
    "norm_matched_wrong_position", "norm_matched_wrong_key",
)


class ConfirmationError(RuntimeError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) \
        if path.exists() else 0


def write_rows(path: Path, rows: list[dict]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9,
                           mtime=0) as compressed:
            for row in rows:
                compressed.write(stable_json(row) + b"\n")
    return sha_file(path)


def load_manifest(result_dir: Path) -> dict:
    path = result_dir / "suite-manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != "TEACH-0013" \
            or manifest.get("status") != "suite_frozen":
        raise ConfirmationError("Frozen TEACH-0013 suite manifest required")
    return manifest


def verify_frozen_source(manifest: dict) -> dict:
    paths = tuple(sorted(manifest["source_sha256"]))
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise ConfirmationError("Commit all registered source before confirmation")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    hashes = {}
    for relative in paths:
        raw = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True,
            capture_output=True).stdout
        hashes[relative] = sha_bytes(raw)
        if hashes[relative] != manifest["source_sha256"][relative]:
            raise ConfirmationError(f"Frozen source changed after suite freeze: {relative}")
    if revision != manifest["source_git_head"]:
        raise ConfirmationError("Confirmation must use the suite-freeze source revision")
    return {"confirmation_source_git_head": revision,
            "confirmation_source_sha256": hashes}


def _checked_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfirmationError(f"Required frozen artifact missing: {path.name}")
    return json.loads(path.read_text())


def _source_matches_manifest(report: dict, manifest: dict) -> bool:
    hashes = report.get("campaign_source_sha256", {})
    return all(hashes.get(path) == digest
               for path, digest in manifest["source_sha256"].items())


def load_discovery_prerequisite(result_dir: Path, manifest: dict
                                ) -> tuple[MediatorSpec, dict]:
    """Resolve one audited, immutable mediator before suite confirmation access."""
    residual_path = result_dir / "discovery-decision.json"
    residual_audit_path = result_dir / "discovery-audit.json"
    residual = _checked_json(residual_path)
    residual_audit = _checked_json(residual_audit_path)
    if residual.get("suite_gzip_sha256") != manifest["suite_gzip_sha256"] \
            or not _source_matches_manifest(residual, manifest):
        raise ConfirmationError("Residual discovery provenance does not match the suite")
    if residual_audit.get("audit") != "pass" \
            or residual_audit.get("scope") != "residual_discovery" \
            or residual_audit.get("discovery_decision_sha256") != sha_file(residual_path):
        raise ConfirmationError("Passing matching residual-discovery audit required")
    selected = residual.get("residual_selection", {}).get("selection")
    sites = residual_sites("raw_deep")
    if residual.get("status") == "stage_b_single_site_complete" and selected:
        if residual_audit.get("primary_selection") != selected:
            raise ConfirmationError("Residual auditor selection mismatch")
        cut = selected["cut_index"]
        if type(cut) is not int or not 0 <= cut < len(sites):
            raise ConfirmationError("Selected residual cut is invalid")
        return MediatorSpec(kind="single", site=sites[cut],
                            label=selected["semantic_label"]), {
            "kind": "single", "decision_path": str(residual_path.relative_to(ROOT)),
            "decision_sha256": sha_file(residual_path),
            "audit_path": str(residual_audit_path.relative_to(ROOT)),
            "audit_sha256": sha_file(residual_audit_path),
            "stage_b_seconds": residual["elapsed_seconds"],
            "stage_b_materialized_bytes": residual["materialized_activation_bytes"],
        }
    if residual.get("status") != "stage_b_residual_complete_path_pending" or selected:
        raise ConfirmationError("Residual discovery has no valid confirmation transition")

    path_decision_path = result_dir / "path-decision.json"
    path_audit_path = result_dir / "path-audit.json"
    path_report = _checked_json(path_decision_path)
    path_audit = _checked_json(path_audit_path)
    path_selection = path_report.get("path_selection", {}).get("selection")
    if path_report.get("status") != "stage_b_path_complete" or not path_selection:
        raise ConfirmationError("No discovery-qualified static or two-site mediator")
    if path_report.get("residual_decision_sha256") != sha_file(residual_path) \
            or path_report.get("residual_audit_sha256") != sha_file(residual_audit_path) \
            or path_audit.get("audit") != "pass" \
            or path_audit.get("scope") != "path_discovery" \
            or path_audit.get("path_decision_sha256") != sha_file(path_decision_path) \
            or path_audit.get("selection") != path_selection:
        raise ConfirmationError("Passing matching path-discovery audit required")
    early, late = path_selection["early_cut_index"], path_selection["late_cut_index"]
    if type(early) is not int or type(late) is not int \
            or not 0 <= early < late < len(sites):
        raise ConfirmationError("Selected path cuts are invalid")
    return MediatorSpec(
        kind="path", early_site=sites[early], source_label=path_selection["source_label"],
        late_site=sites[late], destination_label=path_selection["destination_label"]), {
            "kind": "path", "decision_path": str(path_decision_path.relative_to(ROOT)),
            "decision_sha256": sha_file(path_decision_path),
            "audit_path": str(path_audit_path.relative_to(ROOT)),
            "audit_sha256": sha_file(path_audit_path),
            "stage_b_seconds": path_report["cumulative_stage_b_seconds"],
            "stage_b_materialized_bytes": path_report[
                "cumulative_stage_b_materialized_bytes"],
        }


def load_confirmation_suite(output_dir: Path, manifest: dict) -> dict:
    path = output_dir / "suite.json.gz"
    if not path.is_file() or sha_file(path) != manifest["suite_gzip_sha256"]:
        raise ConfirmationError("Frozen suite gzip hash mismatch")
    raw = gzip.decompress(path.read_bytes())
    if sha_bytes(raw) != manifest["suite_uncompressed_sha256"]:
        raise ConfirmationError("Frozen suite content hash mismatch")
    suite = load_suite(path)
    if [group["group_id"] for group in suite["splits"]["confirmation"]] \
            != manifest["group_ids"]["confirmation"]:
        raise ConfirmationError("Confirmation group order or membership changed")
    return suite


def checkpoint_path(manifest: dict, replicate: str) -> Path:
    metadata = manifest["checkpoints"][replicate]["8000"]
    path = ROOT / metadata["path"]
    if not path.is_file() or path.stat().st_size != metadata["bytes"] \
            or sha_file(path) != metadata["sha256"]:
        raise ConfirmationError(f"Checkpoint mismatch for replicate {replicate}")
    return path


class LogitCapture:
    """Retain exact float tensors for every frozen Stage-C primary/control cell."""

    def __init__(self):
        self.records = []

    def __call__(self, condition, direction, metadata, logits, clean_logits):
        self.records.append({
            "condition": condition, "direction": direction,
            "logical_group_ids": tuple(row["logical_group_id"] for row in metadata),
            "recipients": tuple(row["recipient"] for row in metadata),
            "targets": tuple(row["target"] for row in metadata),
            "item_ids": tuple(row.get(
                "item_id", f"{condition}:{direction}:{row['logical_group_id']}:{row['recipient']}")
                for row in metadata),
            "edited_symbol_logits": logits.contiguous(),
            "clean_symbol_logits": clean_logits.contiguous(),
        })

    def write(self, path: Path) -> dict:
        payload = {"format": "TEACH-0013-symbol-logits-v2", "records": self.records}
        torch.save(payload, path)
        return {"path": str(path.relative_to(ROOT)), "sha256": sha_file(path),
                "bytes": path.stat().st_size, "records": len(self.records),
                "rows": sum(record["edited_symbol_logits"].shape[0]
                            for record in self.records)}


def _item_accuracy(rows: list[dict], condition: str) -> float:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ConfirmationError(f"Missing confirmation condition: {condition}")
    return sum(row["prediction"] == row["target"] for row in selected) / len(selected)


def _specificity_loss(rows: list[dict], task: str) -> float:
    losses = []
    for family in ("format", "order", "distractor"):
        condition = f"{task}_same_key_{family}"
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            raise ConfirmationError(f"Missing specificity condition: {condition}")
        clean = sum(row["clean_prediction"] == row["target"] for row in selected) \
            / len(selected)
        edited = sum(row["prediction"] == row["target"] for row in selected) \
            / len(selected)
        losses.append(max(0.0, clean - edited))
    return max(losses)


def score_confirmation_rows(rows: list[dict]) -> dict:
    """Reduce all registered Stage-C conditions without choosing a favorable slice."""
    true = {direction: summarize_confirmation_rows(
        [row for row in rows if row["condition"] == "sufficiency_marked"],
        direction=direction) for direction in ("forward", "reverse")}
    controls = {}
    for condition in CONTROL_CONDITIONS:
        controls[condition] = {direction: summarize_confirmation_rows(
            [row for row in rows if row["condition"] == condition],
            direction=direction) for direction in ("forward", "reverse")}
    advantages = [true[direction]["item_accuracy"]
                  - controls[condition][direction]["item_accuracy"]
                  for condition in CONTROL_CONDITIONS
                  for direction in ("forward", "reverse")]
    reverse = true["reverse"]
    rescue = summarize_rescue_rows(rows)
    return {
        "clean_base": _item_accuracy(rows, "clean_base"),
        "clean_donor": _item_accuracy(rows, "clean_donor_input_f_replacement"),
        "input_f_replacement": _item_accuracy(
            rows, "clean_donor_input_f_replacement"),
        "sufficiency_items_forward": true["forward"]["item_accuracy"],
        "sufficiency_groups_forward": true["forward"]["group_accuracy"],
        "sufficiency_items_reverse": reverse["item_accuracy"],
        "sufficiency_groups_reverse": reverse["group_accuracy"],
        "changed_non_injection_forward": true["forward"]["non_injection"],
        "changed_non_injection_reverse": reverse["non_injection"],
        "necessity_items": reverse["item_accuracy"],
        "necessity_groups": reverse["group_accuracy"],
        "rescue": 0.0 if rescue["rescue_rate"] is None else rescue["rescue_rate"],
        "rescue_changed_cases": rescue["changed_cases"],
        "same_key_format": _item_accuracy(rows, "same_key_format"),
        "same_key_order": _item_accuracy(rows, "same_key_order"),
        "same_key_distractor": _item_accuracy(rows, "same_key_distractor"),
        "minimum_control_advantage": min(advantages),
        "mean_probability_gain": min(true[direction]["mean_probability_gain"]
                                     for direction in ("forward", "reverse")),
        "direct_loss": _specificity_loss(rows, "direct"),
        "copy_loss": _specificity_loss(rows, "copy"),
        "final_answer_injection": _item_accuracy(
            rows, "final_state_fixed_answer_injection"),
        "marked_summaries": true, "control_summaries": controls,
        "marker_free_summaries": {direction: summarize_confirmation_rows(
            [row for row in rows if row["condition"] == "sufficiency_marker_free"],
            direction=direction) for direction in ("forward", "reverse")},
        "rescue_summary": rescue,
    }


def _run_seed(net, groups: list[dict], spec: MediatorSpec, *, device: str,
              capture: LogitCapture) -> tuple[list[dict], dict]:
    rows = []
    sink = capture
    rows.extend(positive_control_rows(
        net, groups, episode_loader=episode_from_record,
        final_site=residual_sites("raw_deep")[-1], device=device, logit_sink=sink))
    for render in ("marked", "marker_free"):
        rows.extend(bidirectional_sufficiency(
            net, groups, spec, episode_loader=episode_from_record,
            device=device, render=render, logit_sink=sink))
    for family in ("format", "order", "distractor"):
        rows.extend(nuisance_preservation(
            net, groups, spec, family=family, episode_loader=episode_from_record,
            device=device, logit_sink=sink))
    rows.extend(matched_sufficiency_controls(
        net, groups, spec, episode_loader=episode_from_record,
        device=device, logit_sink=sink))
    rows.extend(corruption_rescue_rows(
        net, groups, spec, episode_loader=episode_from_record,
        device=device, logit_sink=sink))
    rows.extend(task_specificity_rows(
        net, groups, spec, episode_loader=episode_from_record,
        device=device, logit_sink=sink))
    identity_episodes = tuple(episode_from_record(row)
                              for group in groups for row in group["base"])
    numerical = identity_error(net, identity_episodes, spec, device=device)
    return rows, numerical


def confirmation(result_dir: Path, output_dir: Path, device: str) -> dict:
    report_path = result_dir / "confirmation-decision.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0013 confirmation overwrite")
    manifest = load_manifest(result_dir)
    provenance = verify_frozen_source(manifest)
    # The mediator and independent audit are resolved before suite decompression or
    # confirmation inference, preventing outcome-dependent selection.
    spec, prerequisite = load_discovery_prerequisite(result_dir, manifest)
    suite = load_confirmation_suite(output_dir, manifest)
    groups = suite["splits"]["confirmation"]
    if len(groups) != manifest["split_counts"]["confirmation"]:
        raise ConfirmationError("Confirmation split count changed")

    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak = {"current": 0, "driver": 0}
    traffic = {"bytes": 0}
    artifacts, clean_scores = {}, {}

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        elapsed = time.monotonic() - start
        if elapsed > MAX_STAGE_C_SECONDS \
                or elapsed + prerequisite["stage_b_seconds"] > MAX_CAMPAIGN_SECONDS:
            raise ConfirmationError("TEACH-0013 Stage-C/campaign wall-time ceiling exceeded")
        if peak["current"] > MAX_CURRENT_BYTES:
            raise ConfirmationError("TEACH-0013 Stage-C sampled allocation ceiling exceeded")
        if traffic["bytes"] + prerequisite["stage_b_materialized_bytes"] \
                > MAX_TRAFFIC_BYTES:
            raise ConfirmationError("TEACH-0013 cumulative materialized-traffic ceiling exceeded")

    def count_traffic(value):
        traffic["bytes"] += int(value)

    for replicate in ("0", "1"):
        net, metadata = load_raw_checkpoint(
            checkpoint_path(manifest, replicate), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise ConfirmationError("Confirmation requires final raw-deep checkpoints")
        capture = LogitCapture()
        with materialized_tensor_counter(count_traffic):
            stage_a_rows = fresh_panel_confirmation_rows(
                net, groups, episode_loader=episode_from_record,
                device=device, logit_sink=capture)
        clean_scores[replicate] = summarize_fresh_panel_rows(stage_a_rows)
        compact_path = result_dir / f"stage-a-rows-rep{replicate}.jsonl.gz"
        logits_path = output_dir / f"stage-a-logits-rep{replicate}.pt"
        if compact_path.exists() or logits_path.exists():
            raise FileExistsError("No automatic TEACH-0013 Stage-A artifact overwrite")
        artifacts[replicate] = {
            "stage_a_rows": {
                "path": str(compact_path.relative_to(ROOT)),
                "sha256": write_rows(compact_path, stage_a_rows),
                "bytes": compact_path.stat().st_size, "rows": len(stage_a_rows)},
            "stage_a_exact_symbol_logits": capture.write(logits_path),
        }
        if device == "mps":
            torch.mps.synchronize()
            del net
            torch.mps.empty_cache()
        probe()
    competence = fresh_panel_decision(clean_scores)
    if competence["label"] != "pass":
        elapsed = time.monotonic() - start
        report = {
            "experiment": "TEACH-0013", "mode": "confirmation",
            "status": "inconclusive_fresh_panel_competence",
            "mediator": asdict(spec), "discovery_prerequisite": prerequisite,
            "clean_confirmation_scores": clean_scores,
            "fresh_panel_decision": competence, "artifacts": artifacts,
            "elapsed_seconds": elapsed,
            "cumulative_campaign_seconds": prerequisite["stage_b_seconds"] + elapsed,
            "materialized_activation_bytes": traffic["bytes"],
            "cumulative_materialized_bytes": (
                prerequisite["stage_b_materialized_bytes"] + traffic["bytes"]),
            "peak_sampled_current_allocated_bytes": peak["current"],
            "peak_sampled_driver_allocated_bytes": peak["driver"],
            "ignored_output_bytes": tree_bytes(output_dir),
            "tracked_result_bytes": tree_bytes(result_dir),
            "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    numerical, scores = {}, {}

    for replicate in ("0", "1"):
        net, _ = load_raw_checkpoint(checkpoint_path(manifest, replicate), device=device)
        capture = LogitCapture()
        with materialized_tensor_counter(count_traffic):
            seed_rows, numerical[replicate] = _run_seed(
                net, groups, spec, device=device, capture=capture)
        probe()
        compact_path = result_dir / f"confirmation-rows-rep{replicate}.jsonl.gz"
        logits_path = output_dir / f"confirmation-logits-rep{replicate}.pt"
        if compact_path.exists() or logits_path.exists():
            raise FileExistsError("No automatic TEACH-0013 confirmation artifact overwrite")
        compact = {"path": str(compact_path.relative_to(ROOT)),
                   "sha256": write_rows(compact_path, seed_rows),
                   "bytes": compact_path.stat().st_size, "rows": len(seed_rows)}
        logits = capture.write(logits_path)
        artifacts[replicate].update(
            {"compact_rows": compact, "exact_symbol_logits": logits})
        scores[replicate] = score_confirmation_rows(seed_rows)
        if device == "mps":
            torch.mps.synchronize()
            del net
            torch.mps.empty_cache()
        probe()
    numerical_pass = all(row["qualified"] for row in numerical.values())
    decision = recipient_transfer_decision(
        scores, discovery_selected=True, numerical_qualified=numerical_pass)
    elapsed = time.monotonic() - start
    output_bytes, result_bytes = tree_bytes(output_dir), tree_bytes(result_dir)
    if output_bytes > MAX_OUTPUT_BYTES or result_bytes > MAX_RESULT_BYTES:
        raise ConfirmationError("TEACH-0013 confirmation artifact ceiling exceeded")
    report = {
        "experiment": "TEACH-0013", "mode": "confirmation",
        "status": "stage_c_complete" if numerical_pass else "stage_c_numerical_inconclusive",
        "mediator": asdict(spec), "discovery_prerequisite": prerequisite,
        "clean_confirmation_scores": clean_scores, "fresh_panel_decision": competence,
        "numerical_qualification": numerical, "scores": scores,
        "recipient_transfer_decision": decision, "artifacts": artifacts,
        "elapsed_seconds": elapsed,
        "cumulative_campaign_seconds": prerequisite["stage_b_seconds"] + elapsed,
        "materialized_activation_bytes": traffic["bytes"],
        "cumulative_materialized_bytes": (
            prerequisite["stage_b_materialized_bytes"] + traffic["bytes"]),
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "ignored_output_bytes": output_bytes, "tracked_result_bytes": result_bytes,
        "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path,
                        default=ROOT / "results/TEACH-0013")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    result = confirmation(args.result_dir, args.output_dir, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
