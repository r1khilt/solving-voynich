"""CPU replay of sampled TEACH-0015 finite vectors and logits."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.teacher0015_clean_audit import SPLITS, read_manifests
from scripts.teacher0015_finite_result_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SAMPLE_SURFACES,
    VECTOR_NAMES, audit_finite,
)
from scripts.teacher0015_finite_run import _surface
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher15_pairs import control_permutations
from voynich.workspace.teacher15_tasks import generate_split


ABS_TOLERANCE = 2e-3
REL_TOLERANCE = 2e-3


def _compare(actual: list[float], archived: list[float],
             *, label: str) -> float:
    if len(actual) != len(archived) or any(
            not math.isfinite(value) for value in actual + archived):
        raise ValueError(f"Finite sampled {label} shape/finite failure")
    differences = [abs(a - b) for a, b in zip(actual, archived, strict=True)]
    if any(error > ABS_TOLERANCE + REL_TOLERANCE * abs(reference)
           for error, reference in zip(differences, archived, strict=True)):
        raise ValueError(f"Finite sampled {label} checkpoint replay differs")
    return max(differences, default=0.0)


def replay(primary_dir: Path, suite_dir: Path, screen_dir: Path,
           result_dir: Path, root: Path) -> dict:
    audited = audit_finite(primary_dir, suite_dir, screen_dir, result_dir, root)
    archive_audit = json.loads((result_dir / "decision-audit.json").read_text())
    report = json.loads((result_dir / "report.json").read_text())
    if (audited != archive_audit or report.get("status") != "complete" or
            report.get("decision_audit_sha256") != hashlib.sha256((
                result_dir / "decision-audit.json").read_bytes()).hexdigest() or
            report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or
            sum(path.stat().st_size for path in result_dir.rglob("*")
                if path.is_file()) > MAX_ARTIFACT_BYTES):
        raise ValueError("Finite archive/resource audit differs")
    primary = json.loads((primary_dir / "report.json").read_text())
    for name, expected in primary["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Finite replay model source changed: {name}")
    manifests = read_manifests(suite_dir)
    groups = {split: generate_split(split) for split in SPLITS}
    controls = {split: control_permutations(groups[split]) for split in SPLITS}
    for split in SPLITS:
        if [group.group_id for group in groups[split]] != [
                item["group_id"] for item in manifests[split]["groups"]]:
            raise ValueError("Finite replay generated suite differs")
    count = 0
    max_vector = 0.0
    max_logit = 0.0
    per_run = {}
    with torch.no_grad():
        for arm in audited["eligible_arms"]:
            per_run[arm] = {}
            for rep in ("0", "1"):
                checkpoint = root / "outputs/TEACH-0014-v3" / (
                    f"rep{rep}-{arm}-step6000.pt")
                expected_sha = report["runs"][arm][rep]["checkpoint_sha256"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha:
                    raise ValueError("Finite replay checkpoint hash differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model = CandidateEdgeWorkspace()
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                local_count = 0
                local_vector = 0.0
                local_logit = 0.0
                for split in SPLITS:
                    rows_path = result_dir / f"rows-{arm}-rep{rep}-{split}.json.gz"
                    vec_path = result_dir / f"vectors-{arm}-rep{rep}-{split}.npy"
                    archive = json.loads(gzip.decompress(rows_path.read_bytes()))
                    vectors = np.load(vec_path, mmap_mode="r", allow_pickle=False)
                    for position in SAMPLE_SURFACES:
                        group_index, surface_index = divmod(position, 8)
                        actual = _surface(model, groups[split], group_index,
                                          surface_index, controls[split],
                                          device="cpu", full_logits=True)
                        stored = archive["rows"][position]
                        for recipient, (computed, row) in enumerate(zip(
                                actual, stored, strict=True)):
                            for key, value in computed.items():
                                if key == "replacement_vectors":
                                    continue
                                if key.endswith("_logits"):
                                    error = _compare(value, row[key], label=key)
                                    local_logit = max(local_logit, error)
                                elif isinstance(value, float):
                                    if (abs(value - row[key]) > ABS_TOLERANCE +
                                            REL_TOLERANCE * abs(row[key])):
                                        raise ValueError(
                                            f"Finite sampled {key} replay differs")
                                elif value != row[key]:
                                    raise ValueError(
                                        f"Finite sampled {key} replay differs")
                            if set(row) != set(computed) - {"replacement_vectors"}:
                                raise ValueError("Finite sampled field set differs")
                            for vector_index, name in enumerate(VECTOR_NAMES):
                                archived_vector = vectors[
                                    group_index, surface_index, recipient,
                                    vector_index].tolist()
                                error = _compare(
                                    computed["replacement_vectors"][name],
                                    archived_vector, label=name)
                                local_vector = max(local_vector, error)
                            local_count += 1
                count += local_count
                max_vector = max(max_vector, local_vector)
                max_logit = max(max_logit, local_logit)
                per_run[arm][rep] = {"recipient_rows": local_count,
                                     "max_abs_vector_error": local_vector,
                                     "max_abs_logit_error": local_logit}
                del model, saved
    return {"audit": "pass", "scope": "sampled_finite_checkpoint_replay",
            "manifest_sha256": audited["manifest_sha256"],
            "recipient_rows": count, "max_abs_vector_error": max_vector,
            "max_abs_logit_error": max_logit,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "per_run": per_run,
            "reusable_key_state_supported_arms": audited[
                "candidate_arms_pending_checkpoint_replay"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("results/TEACH-0015-screen"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.primary_dir, args.suite_dir, args.screen_dir,
                    args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
