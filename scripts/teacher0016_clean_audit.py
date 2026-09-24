"""No-model fresh-answer audit before the TEACH-0016 cross-order assay."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from scripts.teacher0015_clean_audit import (
    ARMS, SPLITS, _competent, _sha, score_split,
)
from scripts.teacher0016_suite_audit import audit_splits


EXPECTED_MANIFESTS = {
    "discovery": "57e95bf3c56083055f896be005dc5da05eb764021ad0817a784ff46c0cb82139",
    "confirmation": "102e3d902dd36882d0901b5d51ba09bbb09af5bf5422e3e2cc90fbb1359ea55f",
}
SOURCE_PATHS = (
    "docs/experiments/TEACH-0016-clean-screen-registration.md",
    "scripts/teacher0016_clean_audit.py",
    "scripts/teacher0016_clean_run.py",
    "scripts/teacher0016_clean_replay.py",
    "scripts/teacher0016_suite_audit.py",
    "src/voynich/workspace/teacher16_tasks.py",
    "scripts/teacher0015_clean_audit.py",
    "scripts/teacher0015_clean_run.py",
    "scripts/teacher0015_clean_replay.py",
    "tests/test_teacher0016_clean.py",
)
MAX_SECONDS = 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 1024**3
SAMPLE_INDICES = (0, 4224, 8447)
BATCH_SIZE = 32


def read_manifests(suite_dir: Path, root: Path) -> dict:
    manifests = {split: json.loads((suite_dir / f"{split}.json").read_text())
                 for split in SPLITS}
    prior = tuple(json.loads((root / "outputs/TEACH-0015" /
                             f"{split}.json").read_text()) for split in SPLITS)
    teach14 = tuple(json.loads(path.read_text()) for path in (
        suite_dir / "teach14-74111.json",
        suite_dir / "teach14-74117.json",
        root / "results/TEACH-0014-v3/suite.json"))
    checked = audit_splits(manifests["discovery"],
                           manifests["confirmation"], prior, teach14)
    saved = json.loads((root / "results/TEACH-0016/suite-audit.json").read_text())
    if (checked["discovery"] != saved["discovery"] or
            checked["confirmation"] != saved["confirmation"] or
            checked["teacher14_exposure_sha256"] != saved[
                "teacher14_exposure_sha256"] or
            checked["teacher15_exposure_sha256"] != saved[
                "teacher15_exposure_sha256"] or
            {split: checked[split]["manifest_sha256"]
             for split in SPLITS} != EXPECTED_MANIFESTS or
            saved.get("suite_file_sha256") != {
                split: _sha(suite_dir / f"{split}.json") for split in SPLITS}):
        raise ValueError("TEACH-0016 fresh suite audit/hash differs")
    head = saved.get("source_git_head")
    hashes = saved.get("source_sha256")
    if not isinstance(head, str) or len(head) != 40 or not isinstance(hashes, dict):
        raise ValueError("TEACH-0016 suite source provenance missing")
    for name, expected in hashes.items():
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != expected or
                _sha(root / name) != expected):
            raise ValueError(f"TEACH-0016 suite source changed: {name}")
    return manifests


def _source_check(root: Path, status: dict) -> None:
    hashes = status.get("screen_source_sha256")
    head = status.get("screen_git_head")
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS) or
            not isinstance(head, str) or len(head) != 40):
        raise ValueError("TEACH-0016 screen source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"TEACH-0016 screen source changed: {name}")


def _benchmark_check(benchmark: dict, hashes: dict, suite_dir: Path) -> None:
    timings = benchmark.get("timings_seconds")
    if (not isinstance(timings, list) or len(timings) != 24 or
            any(type(value) not in (int, float) or not math.isfinite(value)
                or value <= 0 for value in timings)):
        raise ValueError("TEACH-0016 screen benchmark timings invalid")
    median = statistics.median(timings)
    projected = 1.75 * median * 528 * len(ARMS) * 2 + 300
    if (benchmark.get("screen_source_sha256") != hashes or
            benchmark.get("manifest_sha256") != EXPECTED_MANIFESTS or
            benchmark.get("suite_file_sha256") != {
                split: _sha(suite_dir / f"{split}.json") for split in SPLITS} or
            benchmark.get("batch_size") != BATCH_SIZE or
            benchmark.get("batches_per_arm_seed") != 528 or
            benchmark.get("warmup_batches") != 6 or
            benchmark.get("timed_batches") != 24 or
            benchmark.get("median_batch_seconds") != median or
            not math.isclose(benchmark.get("conservative_projected_seconds",
                                           math.inf), projected, rel_tol=1e-9) or
            benchmark.get("max_seconds") != MAX_SECONDS or
            benchmark.get("max_mps_bytes") != MAX_MPS_BYTES or
            benchmark.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            benchmark.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or
            benchmark.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            benchmark.get("artifact_bytes", math.inf) > MAX_ARTIFACT_BYTES or
            projected >= MAX_SECONDS or
            benchmark.get("admitted") is not True):
        raise ValueError("TEACH-0016 screen resource gate failed")


def eligible_arms(finite_dir: Path) -> list[str]:
    report = json.loads((finite_dir / "report.json").read_text())
    decision_path = finite_dir / "decision-audit.json"
    replay = json.loads((finite_dir / "replay-audit.json").read_text())
    decision = json.loads(decision_path.read_text())
    arms = replay.get("portable_state_supported_arms")
    if (report.get("status") != "complete" or
            report.get("decision_audit_sha256") != _sha(decision_path) or
            decision.get("audit") != "pass" or
            replay.get("audit") != "pass" or
            arms != decision.get("candidate_arms_pending_checkpoint_replay") or
            not isinstance(arms, list) or any(arm not in ARMS for arm in arms)):
        raise ValueError("TEACH-0016 prior portable-state gate differs")
    return arms


def audit_screen(primary_dir: Path, suite_dir: Path, finite_dir: Path,
                 result_dir: Path, root: Path, *, runs: dict | None = None
                 ) -> dict:
    status = json.loads((result_dir / "status.json").read_text())
    _source_check(root, status)
    manifests = read_manifests(suite_dir, root)
    benchmark = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark, status["screen_source_sha256"], suite_dir)
    suite_hashes = {split: _sha(suite_dir / f"{split}.json")
                    for split in SPLITS}
    if (status.get("manifest_sha256") != EXPECTED_MANIFESTS or
            status.get("suite_file_sha256") != suite_hashes or
            status.get("benchmark_sha256") != _sha(result_dir / "benchmark.json") or
            status.get("suite_audit_sha256") != _sha(
                root / "results/TEACH-0016/suite-audit.json") or
            status.get("finite_report_sha256") != _sha(finite_dir / "report.json") or
            status.get("finite_decision_audit_sha256") != _sha(
                finite_dir / "decision-audit.json") or
            status.get("finite_replay_audit_sha256") != _sha(
                finite_dir / "replay-audit.json") or
            status.get("primary_report_sha256") != _sha(
                primary_dir / "report.json")):
        raise ValueError("TEACH-0016 screen provenance differs")
    arms = eligible_arms(finite_dir)
    if not arms or status.get("eligible_arms") != arms:
        raise ValueError("TEACH-0016 screen entry differs")
    if runs is None:
        runs = json.loads((result_dir / "report.json").read_text())["runs"]
    if set(runs) != set(arms):
        raise ValueError("TEACH-0016 screen arm coverage differs")
    primary = json.loads((primary_dir / "report.json").read_text())
    scores = {}
    for arm in arms:
        if set(runs[arm]) != {"0", "1"}:
            raise ValueError("TEACH-0016 screen seed coverage differs")
        scores[arm] = {}
        for rep in ("0", "1"):
            detail = runs[arm][rep]
            path = result_dir / f"predictions-{arm}-rep{rep}.json.gz"
            checkpoint_sha = primary["training"][arm][rep][
                "final_checkpoint"]["sha256"]
            if (detail.get("sha256") != _sha(path) or
                    detail.get("bytes") != path.stat().st_size or
                    detail.get("checkpoint_sha256") != checkpoint_sha):
                raise ValueError("TEACH-0016 screen artifact/checkpoint differs")
            archive = json.loads(gzip.decompress(path.read_bytes()))
            if archive.get("checkpoint_sha256") != checkpoint_sha or (
                    set(archive.get("splits", {})) != set(SPLITS)):
                raise ValueError("TEACH-0016 screen split archive incomplete")
            scores[arm][rep] = {
                split: score_split(manifests[split],
                                   archive["splits"][split]["rows"],
                                   archive["splits"][split]["samples"],
                                   ["TEACH-0016", arm, rep, split])
                for split in SPLITS}
    return {"audit": "pass", "scope": "all_fresh_cross_order_clean_answers",
            "manifest_sha256": EXPECTED_MANIFESTS,
            "eligible_arms": arms, "scores": scores,
            "candidate_competent_arms_pending_replay": [
                arm for arm in arms if all(_competent(
                    scores[arm][rep]["confirmation"])
                    for rep in ("0", "1"))],
            "sampled_checkpoint_replay": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    parser.add_argument("--finite-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0016-screen"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_screen(args.primary_dir, args.suite_dir,
                          args.finite_dir, args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
