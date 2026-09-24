"""CPU checkpoint replay for the sampled TEACH-0015 clean answers."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.teacher0015_clean_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SAMPLE_INDICES,
    SPLITS, audit_screen, read_manifests,
)
from scripts.teacher0015_clean_run import _episodes
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_objectives import padded_tokens
from voynich.workspace.teacher14_tasks import SYMBOL_START


ABS_TOLERANCE = 2e-3
REL_TOLERANCE = 2e-3


def _compare(actual: list[float], expected: list[float],
             prediction: int) -> float:
    if len(actual) != 2064 or len(expected) != 2064:
        raise ValueError("TEACH-0015 sampled logit shape differs")
    if any(not math.isfinite(value) for value in actual + expected):
        raise ValueError("TEACH-0015 nonfinite sampled logits")
    errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
    if any(error > ABS_TOLERANCE + REL_TOLERANCE * abs(reference)
           for error, reference in zip(errors, expected, strict=True)):
        raise ValueError(f"TEACH-0015 checkpoint replay differs: {max(errors)}")
    if max(range(SYMBOL_START, 2064), key=lambda token: actual[token]) != (
            prediction):
        raise ValueError("TEACH-0015 replay answer differs")
    return max(errors)


def replay(primary_dir: Path, suite_dir: Path, result_dir: Path,
           root: Path) -> dict:
    audited = audit_screen(primary_dir, suite_dir, result_dir, root)
    report = json.loads((result_dir / "report.json").read_text())
    decision_path = result_dir / "decision-audit.json"
    if (report.get("status") != "complete" or
            report.get("decision_audit_sha256") != hashlib.sha256(
                decision_path.read_bytes()).hexdigest() or
            json.loads(decision_path.read_text()) != audited):
        raise ValueError("TEACH-0015 clean archive audit differs")
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if (report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or artifact_bytes > MAX_ARTIFACT_BYTES):
        raise ValueError("TEACH-0015 screen resource cap exceeded")
    primary_report = json.loads((primary_dir / "report.json").read_text())
    for name, expected in primary_report["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Primary replay model source changed: {name}")
    manifests = read_manifests(suite_dir)
    max_error = 0.0
    vectors = 0
    per_run = {}
    with torch.no_grad():
        for arm in audited["eligible_arms"]:
            per_run[arm] = {}
            for rep in ("0", "1"):
                checkpoint = root / "outputs/TEACH-0014-v3" / (
                    f"rep{rep}-{arm}-step6000.pt")
                expected_checkpoint = report["runs"][arm][rep][
                    "checkpoint_sha256"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != (
                        expected_checkpoint):
                    raise ValueError("TEACH-0015 replay checkpoint hash differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model = CandidateEdgeWorkspace()
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                archive = json.loads(gzip.decompress((
                    result_dir / f"predictions-{arm}-rep{rep}.json.gz").read_bytes()))
                run_error = 0.0
                count = 0
                for split in SPLITS:
                    episodes = _episodes(manifests[split])
                    selected = [episodes[index] for index in SAMPLE_INDICES]
                    output = model(padded_tokens(selected, "cpu"))
                    saved_samples = archive["splits"][split]["samples"]
                    for local, index in enumerate(SAMPLE_INDICES):
                        row = saved_samples[local]
                        if row["render_id"] != selected[local].render_id:
                            raise ValueError("TEACH-0015 replay render ID differs")
                        prediction = archive["splits"][split]["rows"][index][
                            "prediction"]
                        error = _compare(output.logits[local].float().tolist(),
                                         row["logits"], prediction)
                        run_error = max(run_error, error)
                        count += 1
                per_run[arm][rep] = {"vectors": count,
                                     "max_abs_logit_error": run_error}
                max_error = max(max_error, run_error)
                vectors += count
                del model, saved
    return {"audit": "pass", "scope": "sampled_clean_checkpoint_replay",
            "manifest_sha256": audited["manifest_sha256"],
            "vectors": vectors, "max_abs_logit_error": max_error,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "per_run": per_run,
            "clean_competent_arms": audited[
                "candidate_competent_arms_pending_replay"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0015-screen"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.primary_dir, args.suite_dir,
                    args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
