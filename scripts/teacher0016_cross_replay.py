"""Independent sampled CPU checkpoint replay for TEACH-0016 causal patches."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.teacher0015_clean_audit import SPLITS
from scripts.teacher0015_finite_replay import _compare
from scripts.teacher0016_clean_audit import read_manifests
from scripts.teacher0016_clean_replay import replay as replay_screen
from scripts.teacher0016_cross_audit import audit_control_permutations
from scripts.teacher0016_cross_result_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SAMPLE_SURFACES,
    VECTOR_NAMES, audit_cross,
)
from scripts.teacher0016_cross_run import _surface
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher16_tasks import generate_split


ABS_TOLERANCE = 2e-3
REL_TOLERANCE = 2e-3


def replay(primary_dir: Path, suite_dir: Path, finite_dir: Path,
           screen_dir: Path, result_dir: Path, root: Path) -> dict:
    screened = replay_screen(primary_dir, suite_dir, finite_dir,
                             screen_dir, root)
    screen_archive = json.loads((screen_dir / "replay-audit.json").read_text())
    audited = audit_cross(primary_dir, suite_dir, finite_dir,
                          screen_dir, result_dir, root)
    archive_audit = json.loads((result_dir / "decision-audit.json").read_text())
    report = json.loads((result_dir / "report.json").read_text())
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if (screened != screen_archive or screened.get("audit") != "pass" or
            audited != archive_audit or audited.get("audit") != "pass" or
            report.get("status") != "complete" or
            report.get("decision_audit_sha256") != hashlib.sha256((
                result_dir / "decision-audit.json").read_bytes()).hexdigest() or
            report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or artifact_bytes > MAX_ARTIFACT_BYTES):
        raise ValueError("TEACH-0016 causal archive/resource audit differs")
    primary = json.loads((primary_dir / "report.json").read_text())
    for name, expected in primary["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"TEACH-0016 checkpoint source changed: {name}")
    manifests = read_manifests(suite_dir, root)
    groups = {split: generate_split(split) for split in SPLITS}
    controls = {split: audit_control_permutations(manifests[split]["groups"])[0]
                for split in SPLITS}
    for split in SPLITS:
        if [group.group_id for group in groups[split]] != [
                record["group_id"] for record in manifests[split]["groups"]]:
            raise ValueError("TEACH-0016 replay group order differs")
    vector_count = 0
    logit_count = 0
    max_vector = 0.0
    max_logit = 0.0
    per_run = {}
    with torch.no_grad():
        for arm in audited["eligible_arms"]:
            per_run[arm] = {}
            for rep in ("0", "1"):
                checkpoint = root / "outputs/TEACH-0014-v3" / (
                    f"rep{rep}-{arm}-step6000.pt")
                expected = report["runs"][arm][rep]["checkpoint_sha256"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
                    raise ValueError("TEACH-0016 checkpoint bytes differ")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model = CandidateEdgeWorkspace()
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                local_vectors = 0
                local_logits = 0
                local_vector_error = 0.0
                local_logit_error = 0.0
                for split in SPLITS:
                    rows_path = result_dir / f"rows-{arm}-rep{rep}-{split}.json.gz"
                    vec_path = result_dir / f"vectors-{arm}-rep{rep}-{split}.npy"
                    archive = json.loads(gzip.decompress(rows_path.read_bytes()))
                    vectors = np.load(vec_path, mmap_mode="r", allow_pickle=False)
                    for position in SAMPLE_SURFACES:
                        group_index, surface_index = divmod(position, 8)
                        actual = _surface(
                            model, groups[split], group_index, surface_index,
                            controls[split], device="cpu", full_logits=True)
                        stored = archive["rows"][position]
                        for attempt, (computed, row) in enumerate(zip(
                                actual, stored, strict=True)):
                            if set(row) != set(computed) - {"replacement_vectors"}:
                                raise ValueError("TEACH-0016 replay field set differs")
                            for key, value in computed.items():
                                if key == "replacement_vectors":
                                    continue
                                if key.endswith("_logits"):
                                    error = _compare(value, row[key], label=key)
                                    local_logit_error = max(local_logit_error,
                                                            error)
                                    local_logits += 1
                                elif isinstance(value, float):
                                    if (abs(value - row[key]) > ABS_TOLERANCE +
                                            REL_TOLERANCE * abs(row[key])):
                                        raise ValueError(
                                            f"TEACH-0016 replay {key} differs")
                                elif value != row[key]:
                                    raise ValueError(
                                        f"TEACH-0016 replay {key} differs")
                            for index, name in enumerate(VECTOR_NAMES):
                                archived_vector = vectors[
                                    group_index, surface_index, attempt,
                                    index].tolist()
                                error = _compare(
                                    computed["replacement_vectors"][name],
                                    archived_vector, label=name)
                                local_vector_error = max(local_vector_error,
                                                         error)
                                local_vectors += 1
                per_run[arm][rep] = {
                    "vectors": local_vectors, "logits": local_logits,
                    "max_abs_vector_error": local_vector_error,
                    "max_abs_logit_error": local_logit_error}
                vector_count += local_vectors
                logit_count += local_logits
                max_vector = max(max_vector, local_vector_error)
                max_logit = max(max_logit, local_logit_error)
                del model, saved
    return {"audit": "pass", "scope": "sampled_crossed_state_checkpoint_replay",
            "manifest_sha256": audited["manifest_sha256"],
            "vectors": vector_count, "logits": logit_count,
            "max_abs_vector_error": max_vector,
            "max_abs_logit_error": max_logit,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "per_run": per_run,
            "order_robust_portable_state_supported_arms": audited[
                "candidate_arms_pending_checkpoint_replay"],
            "inconclusive_interface_arms": audited[
                "inconclusive_interface_arms"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    parser.add_argument("--finite-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("results/TEACH-0016-screen"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0016-cross"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.primary_dir, args.suite_dir, args.finite_dir,
                    args.screen_dir, args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
