"""Bounded fresh-answer screen before the TEACH-0016 causal challenge."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from scripts.teacher0015_clean_audit import ARMS, SPLITS, _sha
from scripts.teacher0015_clean_run import (
    _episodes, _load_model, _predict, _prepare_mps, _primary_gate, _resource,
    _write,
)
from scripts.teacher0015_finite_replay import replay as replay_finite
from scripts.teacher0016_clean_audit import (
    BATCH_SIZE, EXPECTED_MANIFESTS, MAX_ARTIFACT_BYTES, MAX_MPS_BYTES,
    MAX_SECONDS, SOURCE_PATHS, _benchmark_check, audit_screen,
    eligible_arms, read_manifests,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace


def _provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0016 screen sources before benchmark")
    return {"screen_git_head": head,
            "screen_source_sha256": {
                name: _sha(root / name) for name in SOURCE_PATHS}}


def benchmark(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "benchmark.json").exists():
        raise FileExistsError("TEACH-0016 screen benchmark already exists")
    provenance = _provenance(root)
    manifests = read_manifests(suite_dir, root)
    _prepare_mps()
    model = CandidateEdgeWorkspace().to("mps").eval()
    episodes = _episodes(manifests["confirmation"])
    choices = (0, 32, 4224, 8416)
    timings = []
    start = time.monotonic()
    peak = 0
    for turn in range(30):
        offset = choices[turn % len(choices)]
        begin = time.monotonic()
        _predict(model, episodes[offset:offset + BATCH_SIZE], "mps")
        sampled = _resource(start, result_dir)
        peak = max(peak, sampled["sampled_mps_allocated_bytes"])
        if turn >= 6:
            timings.append(time.monotonic() - begin)
    median = statistics.median(timings)
    projected = 1.75 * median * 528 * len(ARMS) * 2 + 300
    result = {"experiment": "TEACH-0016-clean-screen",
              "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "batch_size": BATCH_SIZE, "batches_per_arm_seed": 528,
              "warmup_batches": 6, "timed_batches": 24,
              "timings_seconds": timings, "median_batch_seconds": median,
              "conservative_projected_seconds": projected,
              "max_seconds": MAX_SECONDS, "max_mps_bytes": MAX_MPS_BYTES,
              "max_artifact_bytes": MAX_ARTIFACT_BYTES,
              "peak_sampled_mps_allocated_bytes": peak,
              "admitted": projected < MAX_SECONDS,
              **_resource(start, result_dir), **provenance}
    _write(result_dir / "benchmark.json", result)
    return result


def _entry(root: Path, primary_dir: Path, suite_dir: Path,
           finite_dir: Path) -> tuple[dict, list[str]]:
    primary_report, _ = _primary_gate(primary_dir, root)
    read_manifests(suite_dir, root)
    if not (finite_dir / "report.json").exists():
        status_path = finite_dir / "status.json"
        if status_path.exists():
            raise ValueError("Prior finite status exists without complete report")
        return primary_report, []
    replayed = replay_finite(primary_dir, root / "outputs/TEACH-0015",
                             root / "results/TEACH-0015-screen",
                             finite_dir, root)
    archived = json.loads((finite_dir / "replay-audit.json").read_text())
    if replayed != archived or replayed.get("audit") != "pass":
        raise ValueError("TEACH-0015 finite replay gate differs")
    arms = eligible_arms(finite_dir)
    if arms != replayed["portable_state_supported_arms"]:
        raise ValueError("TEACH-0016 prior arm entry differs")
    return primary_report, arms


def run(root: Path, primary_dir: Path, suite_dir: Path,
        finite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "status.json").exists() or (
            result_dir / "report.json").exists():
        raise FileExistsError("No automatic TEACH-0016 screen rerun")
    provenance = _provenance(root)
    primary, arms = _entry(root, primary_dir, suite_dir, finite_dir)
    if not arms:
        result = {"experiment": "TEACH-0016-clean-screen",
                  "status": "not_entered_prior_portable_state",
                  "finite_report_sha256": (_sha(finite_dir / "report.json")
                                           if (finite_dir / "report.json").exists()
                                           else None),
                  "finite_entry_sha256": (_sha(finite_dir / "entry.json")
                                          if (finite_dir / "entry.json").exists()
                                          else None),
                  "primary_report_sha256": _sha(primary_dir / "report.json"),
                  **provenance}
        _write(result_dir / "entry.json", result)
        return result
    benchmark_row = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark_row, provenance["screen_source_sha256"], suite_dir)
    manifests = read_manifests(suite_dir, root)
    _prepare_mps()
    start = time.monotonic()
    status = {"experiment": "TEACH-0016-clean-screen", "status": "running",
              "eligible_arms": arms, "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "suite_audit_sha256": _sha(root / "results/TEACH-0016/suite-audit.json"),
              "benchmark_sha256": _sha(result_dir / "benchmark.json"),
              "finite_report_sha256": _sha(finite_dir / "report.json"),
              "finite_decision_audit_sha256": _sha(
                  finite_dir / "decision-audit.json"),
              "finite_replay_audit_sha256": _sha(
                  finite_dir / "replay-audit.json"),
              "primary_report_sha256": _sha(primary_dir / "report.json"),
              "torch_version": torch.__version__,
              "platform": platform.platform(),
              "peak_sampled_mps_allocated_bytes": 0, **provenance}
    _write(result_dir / "status.json", status)
    runs = {}
    try:
        for arm in arms:
            runs[arm] = {}
            for rep in (0, 1):
                model, checkpoint_sha = _load_model(
                    root, primary, arm, rep, "mps")
                splits = {}
                for split in SPLITS:
                    episodes = _episodes(manifests[split])
                    def health() -> None:
                        sampled = _resource(start, result_dir)
                        status["peak_sampled_mps_allocated_bytes"] = max(
                            status["peak_sampled_mps_allocated_bytes"],
                            sampled["sampled_mps_allocated_bytes"])
                    rows, samples = _predict(model, episodes, "mps", health)
                    splits[split] = {"rows": rows, "samples": samples}
                    sampled = _resource(start, result_dir)
                    status["peak_sampled_mps_allocated_bytes"] = max(
                        status["peak_sampled_mps_allocated_bytes"],
                        sampled["sampled_mps_allocated_bytes"])
                    status["progress"] = {"arm": arm, "replicate": rep,
                                          "split": split}
                    status.update(sampled)
                    _write(result_dir / "status.json", status)
                path = result_dir / f"predictions-{arm}-rep{rep}.json.gz"
                path.write_bytes(gzip.compress(json.dumps(
                    {"checkpoint_sha256": checkpoint_sha, "splits": splits},
                    sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode(), compresslevel=9, mtime=0))
                runs[arm][str(rep)] = {"sha256": _sha(path),
                                       "bytes": path.stat().st_size,
                                       "checkpoint_sha256": checkpoint_sha}
                del model
                torch.mps.empty_cache()
                _resource(start, result_dir)
        decision = audit_screen(primary_dir, suite_dir, finite_dir,
                                result_dir, root, runs=runs)
        _write(result_dir / "decision-audit.json", decision)
        sampled = _resource(start, result_dir)
        final = {**status, "status": "complete", "runs": runs,
                 "decision_audit_sha256": _sha(
                     result_dir / "decision-audit.json"), **sampled}
        _write(result_dir / "report.json", final)
        _write(result_dir / "status.json", {**status, "status": "complete",
                                             **sampled})
        return final
    except Exception as exc:
        _write(result_dir / "status.json", {**status, "status": "stopped",
                                             "reason": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    parser.add_argument("--finite-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0016-screen"))
    args = parser.parse_args()
    result = (benchmark(args.root, args.suite_dir, args.result_dir)
              if args.mode == "benchmark" else
              run(args.root, args.primary_dir, args.suite_dir,
                  args.finite_dir, args.result_dir))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
