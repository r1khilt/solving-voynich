"""Replay sampled gate interventions from audited TEACH-0014 checkpoints."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.teacher0014_diagnostic_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, audit_diagnostic,
)
from voynich.workspace.teacher14_diagnose import CONDITIONS, evaluate_batch
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_tasks import evaluation_suite


ABS_TOLERANCE = 2e-3
REL_TOLERANCE = 2e-3
ARM = "latent_rows_answer"


def _compare(actual: list[float], expected: list[float]) -> float:
    if len(actual) != 2064 or len(expected) != 2064:
        raise ValueError("Diagnostic sampled logit shape mismatch")
    if any(not math.isfinite(value) for value in actual + expected):
        raise ValueError("Nonfinite diagnostic replay vector")
    errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
    if any(error > ABS_TOLERANCE + REL_TOLERANCE * abs(reference)
           for error, reference in zip(errors, expected, strict=True)):
        raise ValueError(f"Diagnostic intervention replay differs: {max(errors)}")
    return max(errors)


def replay(primary_dir: Path, result_dir: Path, root: Path) -> dict:
    audited = audit_diagnostic(primary_dir, result_dir, root)
    report = json.loads((result_dir / "report.json").read_text())
    primary_report = json.loads((primary_dir / "report.json").read_text())
    for name, expected in primary_report["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Primary replay model source changed: {name}")
    if (report.get("status") != "complete" or
            report.get("decision_audit_sha256") != hashlib.sha256(
                (result_dir / "decision-audit.json").read_bytes()).hexdigest() or
            json.loads((result_dir / "decision-audit.json").read_text()) !=
            audited):
        raise ValueError("Diagnostic archive audit differs from saved decision")
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if (report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or artifact_bytes > MAX_ARTIFACT_BYTES):
        raise ValueError("Diagnostic resource cap exceeded")
    suite = evaluation_suite(84311, 128)
    max_error = 0.0
    sample_count = 0
    condition_count = 0
    per_run = {}
    with torch.no_grad():
        for rep in ("0", "1"):
            checkpoint = root / "outputs/TEACH-0014-v3" / (
                f"rep{rep}-{ARM}-step6000.pt")
            expected_checkpoint = report["runs"][rep]["checkpoint_sha256"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != (
                    expected_checkpoint):
                raise ValueError("Diagnostic replay checkpoint hash differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = CandidateEdgeWorkspace()
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            archive = json.loads(gzip.decompress(
                (result_dir / f"diagnostic-rep{rep}.json.gz").read_bytes()))
            count = 0
            run_error = 0.0
            for name, episodes in suite.items():
                positions = (0, len(episodes) // 2, len(episodes) - 1)
                observed = evaluate_batch(
                    model, [episodes[index] for index in positions],
                    device="cpu")
                for local, index in enumerate(positions):
                    saved_row = archive["panels"][name][index]
                    actual_row = observed[local]
                    if (saved_row["render_id"] != actual_row["render_id"] or
                            saved_row["random_selected_indices"] != actual_row[
                                "random_selected_indices"] or
                            saved_row["path_candidate_indices"] != actual_row[
                                "path_candidate_indices"]):
                        raise ValueError("Diagnostic replay item/mask mismatch")
                    for condition in CONDITIONS:
                        original_gate = saved_row["gate_assignment_logits"][
                            condition]
                        replay_gate = actual_row["gate_assignment_logits"][
                            condition]
                        if len(original_gate) != len(replay_gate) or any(
                                abs(a - b) > ABS_TOLERANCE + REL_TOLERANCE *
                                abs(b) for a, b in zip(original_gate,
                                                       replay_gate, strict=True)):
                            raise ValueError("Diagnostic replay gate mask differs")
                    for condition in CONDITIONS:
                        actual = actual_row["conditions"][condition]
                        expected = saved_row["conditions"][condition]
                        if actual["prediction"] != expected["prediction"]:
                            raise ValueError("Diagnostic replay prediction differs")
                        if (any(abs(a - b) > ABS_TOLERANCE for key in (
                                    "target_attention", "false_attention")
                                    for a, b in zip(actual[key], expected[key],
                                                    strict=True)) or any(
                                    (actual[key] is None) != (expected[key] is None)
                                    or (actual[key] is not None and
                                        abs(actual[key] - expected[key]) >
                                        ABS_TOLERANCE + REL_TOLERANCE * abs(
                                            expected[key]))
                                    for key in ("first_value_norm",
                                                "first_state_norm"))):
                            raise ValueError("Diagnostic replay readout differs")
                        error = _compare(actual["logits"],
                                         expected["sampled_logits"])
                        run_error = max(run_error, error)
                        condition_count += 1
                    count += 1
            per_run[rep] = {"samples": count,
                            "max_abs_logit_error": run_error}
            sample_count += count
            max_error = max(max_error, run_error)
            del model, saved
    final_label = ("PARSER-GATING BOTTLENECK SUPPORTED" if
                   audited["candidate_label_pending_numerical_replay"] ==
                   "PARSER-GATING BOTTLENECK CANDIDATE" else
                   "NO REGISTERED PARSER-GATE SUPPORT")
    return {"audit": "pass", "scope": "sampled_intervention_checkpoint_replay",
            "manifest_sha256": audited["manifest_sha256"],
            "samples": sample_count, "condition_vectors": condition_count,
            "max_abs_logit_error": max_error,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "per_run": per_run, "final_diagnostic_label": final_label}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014-diagnostic"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.primary_dir, args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
