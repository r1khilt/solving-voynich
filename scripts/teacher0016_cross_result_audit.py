"""No-checkpoint audit of the complete TEACH-0016 crossed-state archive."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

from scripts.teacher0015_clean_audit import ARMS, SPLITS, _sha
from scripts.teacher0016_clean_audit import (
    EXPECTED_MANIFESTS, audit_screen, read_manifests,
)
from scripts.teacher0016_cross_audit import (
    VECTOR_NAMES, audit_control_permutations, audit_cross_surface,
)
from scripts.teacher0016_cross_score import score_split


SOURCE_PATHS = (
    "docs/experiments/TEACH-0016-cross-order-registration.md",
    "docs/experiments/TEACH-0016-cross-order-execution-registration.md",
    "src/voynich/workspace/teacher16_intervene.py",
    "scripts/teacher0016_cross_audit.py",
    "scripts/teacher0016_cross_score.py",
    "scripts/teacher0016_cross_result_audit.py",
    "scripts/teacher0016_cross_run.py",
    "scripts/teacher0016_cross_replay.py",
    "tests/test_teacher0016_intervene.py",
    "tests/test_teacher0016_cross_audit.py",
    "tests/test_teacher0016_cross_score.py",
    "tests/test_teacher0016_cross_result_audit.py",
    "tests/test_teacher0016_cross_replay.py",
)
SURFACES = tuple((d, marked, order) for d in (0, 1)
                 for marked in (True, False) for order in (0, 1))
SAMPLE_SURFACES = (0, 512, 1023)
MAX_SECONDS = 4 * 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 3 * 1024**3
TOTAL_WORST_SURFACES = len(ARMS) * 2 * len(SPLITS) * 128 * 8


def _source_check(root: Path, status: dict) -> None:
    hashes = status.get("cross_source_sha256")
    head = status.get("cross_git_head")
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS) or
            not isinstance(head, str) or len(head) != 40):
        raise ValueError("TEACH-0016 causal source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"TEACH-0016 causal source changed: {name}")


def _benchmark_check(benchmark: dict, hashes: dict,
                     suite_dir: Path) -> None:
    timings = benchmark.get("timings_seconds")
    if (not isinstance(timings, dict) or set(timings) != {
            str(index) for index in range(8)} or any(
            not isinstance(values, list) or len(values) != 3 or any(
                type(value) not in (int, float) or not math.isfinite(value)
                or value <= 0 for value in values)
            for values in timings.values())):
        raise ValueError("TEACH-0016 causal benchmark timings invalid")
    slowest = max(statistics.median(values) for values in timings.values())
    projected = 1.75 * slowest * TOTAL_WORST_SURFACES + 300
    if (benchmark.get("cross_source_sha256") != hashes or
            benchmark.get("manifest_sha256") != EXPECTED_MANIFESTS or
            benchmark.get("suite_file_sha256") != {
                split: _sha(suite_dir / f"{split}.json") for split in SPLITS} or
            benchmark.get("vector_names") != list(VECTOR_NAMES) or
            benchmark.get("warmup_surfaces") != 8 or
            benchmark.get("timed_surfaces") != 24 or
            benchmark.get("worst_case_surfaces") != TOTAL_WORST_SURFACES or
            benchmark.get("slowest_type_median_seconds") != slowest or
            not math.isclose(benchmark.get("conservative_projected_seconds",
                                           math.inf), projected, rel_tol=1e-9) or
            benchmark.get("max_seconds") != MAX_SECONDS or
            benchmark.get("max_mps_bytes") != MAX_MPS_BYTES or
            benchmark.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            benchmark.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or
            benchmark.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            benchmark.get("artifact_bytes", math.inf) > MAX_ARTIFACT_BYTES or
            projected >= MAX_SECONDS or benchmark.get("admitted") is not True):
        raise ValueError("TEACH-0016 causal resource gate failed")


def audit_cross(primary_dir: Path, suite_dir: Path, finite_dir: Path,
                screen_dir: Path, result_dir: Path, root: Path,
                *, runs: dict | None = None) -> dict:
    status = json.loads((result_dir / "status.json").read_text())
    _source_check(root, status)
    manifests = read_manifests(suite_dir, root)
    benchmark = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark, status["cross_source_sha256"], suite_dir)
    screen = audit_screen(primary_dir, suite_dir, finite_dir, screen_dir, root)
    archived_screen = json.loads((screen_dir / "decision-audit.json").read_text())
    replayed_screen = json.loads((screen_dir / "replay-audit.json").read_text())
    arms = list(replayed_screen.get("clean_competent_arms", []))
    if (screen != archived_screen or screen.get("audit") != "pass" or
            replayed_screen.get("audit") != "pass" or
            arms != screen["candidate_competent_arms_pending_replay"] or
            not arms or any(arm not in ARMS for arm in arms) or
            status.get("eligible_arms") != arms or
            status.get("manifest_sha256") != EXPECTED_MANIFESTS or
            status.get("suite_file_sha256") != {
                split: _sha(suite_dir / f"{split}.json") for split in SPLITS} or
            status.get("screen_report_sha256") != _sha(screen_dir / "report.json") or
            status.get("screen_decision_audit_sha256") != _sha(
                screen_dir / "decision-audit.json") or
            status.get("screen_replay_audit_sha256") != _sha(
                screen_dir / "replay-audit.json") or
            status.get("benchmark_sha256") != _sha(result_dir / "benchmark.json") or
            status.get("primary_report_sha256") != _sha(primary_dir / "report.json")):
        raise ValueError("TEACH-0016 causal entry provenance differs")
    if runs is None:
        runs = json.loads((result_dir / "report.json").read_text())["runs"]
    if set(runs) != set(arms):
        raise ValueError("TEACH-0016 causal arm coverage differs")
    primary = json.loads((primary_dir / "report.json").read_text())
    scored = {}
    max_identity = 0.0
    max_final = 0.0
    for arm in arms:
        if set(runs[arm]) != {"0", "1"}:
            raise ValueError("TEACH-0016 causal replicate coverage differs")
        scored[arm] = {}
        for rep in ("0", "1"):
            record = runs[arm][rep]
            if record.get("checkpoint_sha256") != primary[
                    "training"][arm][rep]["final_checkpoint"]["sha256"]:
                raise ValueError("TEACH-0016 causal checkpoint hash differs")
            if set(record.get("splits", {})) != set(SPLITS):
                raise ValueError("TEACH-0016 causal split coverage differs")
            scored[arm][rep] = {}
            for split in SPLITS:
                row_path = result_dir / f"rows-{arm}-rep{rep}-{split}.json.gz"
                vector_path = result_dir / f"vectors-{arm}-rep{rep}-{split}.npy"
                files = record["splits"][split]
                if (files.get("rows_sha256") != _sha(row_path) or
                        files.get("vectors_sha256") != _sha(vector_path) or
                        files.get("rows_bytes") != row_path.stat().st_size or
                        files.get("vectors_bytes") != vector_path.stat().st_size):
                    raise ValueError("TEACH-0016 causal archive hash/size differs")
                archive = json.loads(gzip.decompress(row_path.read_bytes()))
                if (archive.get("split") != split or
                        archive.get("checkpoint_sha256") != record[
                            "checkpoint_sha256"] or
                        len(archive.get("rows", [])) != 1024):
                    raise ValueError("TEACH-0016 causal surface archive incomplete")
                vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
                if (vectors.dtype != np.float32 or
                        vectors.shape != (128, 8, 9, len(VECTOR_NAMES), 512) or
                        not np.isfinite(vectors).all()):
                    raise ValueError("TEACH-0016 causal vector archive invalid")
                groups = manifests[split]["groups"]
                controls, _ = audit_control_permutations(groups)
                for group_index, group in enumerate(groups):
                    for surface_index, (d, marked, source_order) in enumerate(
                            SURFACES):
                        position = group_index * 8 + surface_index
                        rows = archive["rows"][position]
                        check = audit_cross_surface(
                            group, groups[controls[group_index]], rows,
                            vectors[group_index, surface_index], distractor=d,
                            marked=marked, source_order=source_order,
                            full_logits=position in SAMPLE_SURFACES)
                        if check["audit"] != "pass":
                            raise ValueError("TEACH-0016 causal surface audit failed")
                        max_identity = max(max_identity, max(
                            row["identity_max_abs_logit_error"] for row in rows))
                        max_final = max(max_final, max(
                            row["final_donor_max_abs_logit_error"]
                            for row in rows))
                scored[arm][rep][split] = score_split(
                    manifests[split], archive["rows"],
                    ["TEACH-0016-cross", arm, rep, split])
                expected_shifted = json.loads((root / (
                    "results/TEACH-0016/suite-audit.json")).read_text())[
                        split]["shifted_slot_pairs"]
                if scored[arm][rep][split]["shifted_surface_pairs"] != expected_shifted:
                    raise ValueError("TEACH-0016 shifted denominator differs")
    candidates = [arm for arm in arms if all(
        scored[arm][rep]["confirmation"][
            "candidate_order_robust_state_pending_replay"]
        for rep in ("0", "1"))]
    inconclusive = [arm for arm in arms if any(
        not scored[arm][rep]["confirmation"]["same_key_positive_control_pass"]
        for rep in ("0", "1"))]
    return {"audit": "pass", "scope": "all_crossed_order_state_transfers",
            "manifest_sha256": EXPECTED_MANIFESTS,
            "eligible_arms": arms, "scores": scored,
            "max_identity_logit_error": max_identity,
            "max_final_donor_logit_error": max_final,
            "candidate_arms_pending_checkpoint_replay": candidates,
            "inconclusive_interface_arms": inconclusive,
            "sampled_checkpoint_replay": "pending"}
