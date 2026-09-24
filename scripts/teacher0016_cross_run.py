"""Resource-gated TEACH-0016 crossed-order state interchange on local MPS."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from scripts.teacher0015_clean_audit import SPLITS, _sha
from scripts.teacher0015_clean_run import _load_model, _write
from scripts.teacher0016_clean_audit import (
    EXPECTED_MANIFESTS, audit_screen, read_manifests,
)
from scripts.teacher0016_clean_replay import replay as replay_screen
from scripts.teacher0016_cross_audit import audit_control_permutations
from scripts.teacher0016_cross_result_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SAMPLE_SURFACES,
    SOURCE_PATHS, SURFACES, TOTAL_WORST_SURFACES, VECTOR_NAMES,
    _benchmark_check, audit_cross,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher16_intervene import evaluate_cross_surface
from voynich.workspace.teacher16_tasks import generate_split


def _provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0016 causal sources before benchmark")
    return {"cross_git_head": head,
            "cross_source_sha256": {name: _sha(root / name)
                                    for name in SOURCE_PATHS}}


def _prepare_mps() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0016 causal assay")
    recommended = torch.mps.recommended_max_memory()
    if recommended <= 0:
        raise RuntimeError("Mac GPU memory recommendation unavailable")
    torch.mps.set_per_process_memory_fraction(min(
        .40, MAX_MPS_BYTES / recommended))


def _resource(start: float, result_dir: Path) -> dict:
    torch.mps.synchronize()
    elapsed = time.monotonic() - start
    allocated = torch.mps.current_allocated_memory()
    bytes_used = sum(path.stat().st_size for path in result_dir.rglob("*")
                     if path.is_file()) if result_dir.exists() else 0
    if (elapsed > MAX_SECONDS or allocated > MAX_MPS_BYTES or
            bytes_used > MAX_ARTIFACT_BYTES):
        raise RuntimeError("TEACH-0016 causal resource cap reached")
    return {"elapsed_seconds": elapsed,
            "sampled_mps_allocated_bytes": allocated,
            "artifact_bytes": bytes_used}


def _surface(model, groups, group_index: int, surface_index: int,
             controls: tuple[int, ...], *, device: str,
             full_logits: bool) -> list[dict]:
    d, marked, source_order = SURFACES[surface_index]
    return evaluate_cross_surface(
        model, groups[group_index], groups[controls[group_index]],
        distractor=d, marked=marked, source_order=source_order,
        device=device, full_logits=full_logits)


def _pack_surface(rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    vectors = np.asarray([[row["replacement_vectors"][name]
                           for name in VECTOR_NAMES] for row in rows],
                         dtype=np.float32)
    if (vectors.shape != (9, len(VECTOR_NAMES), 512) or
            not np.isfinite(vectors).all()):
        raise ValueError("TEACH-0016 vector surface shape/finite failure")
    compact = [{key: value for key, value in row.items()
                if key != "replacement_vectors"} for row in rows]
    return compact, vectors


def _groups(suite_dir: Path, root: Path) -> tuple[dict, dict]:
    manifests = read_manifests(suite_dir, root)
    groups = {split: generate_split(split) for split in SPLITS}
    for split in SPLITS:
        if [group.group_id for group in groups[split]] != [
                record["group_id"] for record in manifests[split]["groups"]]:
            raise ValueError("TEACH-0016 generated group order differs")
    return manifests, groups


def benchmark(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "benchmark.json").exists():
        raise FileExistsError("TEACH-0016 causal benchmark already exists")
    provenance = _provenance(root)
    _, groups = _groups(suite_dir, root)
    controls, _ = audit_control_permutations(
        json.loads((suite_dir / "confirmation.json").read_text())["groups"])
    _prepare_mps()
    model = CandidateEdgeWorkspace().to("mps").eval()
    timings = {str(index): [] for index in range(8)}
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch_rows = result_dir / "benchmark-scratch.json.gz"
    scratch_vectors = result_dir / "benchmark-scratch.bin"
    peak = 0
    start = time.monotonic()
    try:
        for turn in range(4):
            for surface_index in range(8):
                begin = time.monotonic()
                rows = _surface(model, groups["confirmation"], turn % 2,
                                surface_index, controls, device="mps",
                                full_logits=False)
                compact, vectors = _pack_surface(rows)
                scratch_rows.write_bytes(gzip.compress(json.dumps(
                    compact, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode(), compresslevel=6, mtime=0))
                scratch_vectors.write_bytes(vectors.tobytes())
                sampled = _resource(start, result_dir)
                peak = max(peak, sampled["sampled_mps_allocated_bytes"])
                if turn:
                    timings[str(surface_index)].append(time.monotonic() - begin)
    finally:
        scratch_rows.unlink(missing_ok=True)
        scratch_vectors.unlink(missing_ok=True)
    slowest = max(statistics.median(values) for values in timings.values())
    projected = 1.75 * slowest * TOTAL_WORST_SURFACES + 300
    result = {"experiment": "TEACH-0016-cross-order",
              "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "vector_names": list(VECTOR_NAMES),
              "warmup_surfaces": 8, "timed_surfaces": 24,
              "timings_seconds": timings,
              "slowest_type_median_seconds": slowest,
              "worst_case_surfaces": TOTAL_WORST_SURFACES,
              "conservative_projected_seconds": projected,
              "max_seconds": MAX_SECONDS,
              "max_mps_bytes": MAX_MPS_BYTES,
              "max_artifact_bytes": MAX_ARTIFACT_BYTES,
              "peak_sampled_mps_allocated_bytes": peak,
              "admitted": projected < MAX_SECONDS,
              **_resource(start, result_dir), **provenance}
    _write(result_dir / "benchmark.json", result)
    return result


def _entry(primary_dir: Path, suite_dir: Path, finite_dir: Path,
           screen_dir: Path, root: Path) -> tuple[dict, list[str]]:
    if not (screen_dir / "report.json").exists():
        entry_path = screen_dir / "entry.json"
        if not entry_path.exists():
            raise ValueError("TEACH-0016 prior clean screen incomplete")
        entry = json.loads(entry_path.read_text())
        if entry.get("status") != "not_entered_prior_portable_state":
            raise ValueError("TEACH-0016 prior clean screen entry differs")
        return json.loads((primary_dir / "report.json").read_text()), []
    screened = audit_screen(primary_dir, suite_dir, finite_dir, screen_dir, root)
    archived = json.loads((screen_dir / "decision-audit.json").read_text())
    replayed = replay_screen(primary_dir, suite_dir, finite_dir,
                             screen_dir, root)
    archived_replay = json.loads((screen_dir / "replay-audit.json").read_text())
    report = json.loads((screen_dir / "report.json").read_text())
    if (screened != archived or replayed != archived_replay or
            screened.get("audit") != "pass" or
            replayed.get("audit") != "pass" or
            report.get("status") != "complete"):
        raise ValueError("TEACH-0016 fresh clean audit/replay gate failed")
    primary = json.loads((primary_dir / "report.json").read_text())
    return primary, list(replayed["clean_competent_arms"])


def run(root: Path, primary_dir: Path, suite_dir: Path, finite_dir: Path,
        screen_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "status.json").exists() or (
            result_dir / "report.json").exists():
        raise FileExistsError("No automatic TEACH-0016 causal rerun")
    provenance = _provenance(root)
    primary, arms = _entry(primary_dir, suite_dir, finite_dir,
                           screen_dir, root)
    if not arms:
        result = {"experiment": "TEACH-0016-cross-order",
                  "status": "not_entered_fresh_competence",
                  "screen_report_sha256": (_sha(screen_dir / "report.json")
                                           if (screen_dir / "report.json").exists()
                                           else None), **provenance}
        _write(result_dir / "entry.json", result)
        return result
    benchmark_row = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark_row, provenance["cross_source_sha256"], suite_dir)
    manifests, groups = _groups(suite_dir, root)
    controls = {split: audit_control_permutations(manifests[split]["groups"])[0]
                for split in SPLITS}
    _prepare_mps()
    start = time.monotonic()
    status = {"experiment": "TEACH-0016-cross-order", "status": "running",
              "eligible_arms": arms, "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "benchmark_sha256": _sha(result_dir / "benchmark.json"),
              "screen_report_sha256": _sha(screen_dir / "report.json"),
              "screen_decision_audit_sha256": _sha(
                  screen_dir / "decision-audit.json"),
              "screen_replay_audit_sha256": _sha(
                  screen_dir / "replay-audit.json"),
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
                record = {"checkpoint_sha256": checkpoint_sha, "splits": {}}
                runs[arm][str(rep)] = record
                for split in SPLITS:
                    row_path = result_dir / f"rows-{arm}-rep{rep}-{split}.json.gz"
                    vec_path = result_dir / f"vectors-{arm}-rep{rep}-{split}.npy"
                    vectors = np.lib.format.open_memmap(
                        vec_path, mode="w+", dtype=np.float32,
                        shape=(128, 8, 9, len(VECTOR_NAMES), 512))
                    all_rows = []
                    for group_index in range(128):
                        for surface_index in range(8):
                            position = group_index * 8 + surface_index
                            rows = _surface(
                                model, groups[split], group_index,
                                surface_index, controls[split], device="mps",
                                full_logits=position in SAMPLE_SURFACES)
                            compact, array = _pack_surface(rows)
                            vectors[group_index, surface_index] = array
                            all_rows.append(compact)
                            if position % 8 == 0:
                                sampled = _resource(start, result_dir)
                                status["peak_sampled_mps_allocated_bytes"] = max(
                                    status["peak_sampled_mps_allocated_bytes"],
                                    sampled["sampled_mps_allocated_bytes"])
                                status["progress"] = {"arm": arm,
                                                      "replicate": rep,
                                                      "split": split,
                                                      "surface": position}
                                status.update(sampled)
                                _write(result_dir / "status.json", status)
                    vectors.flush()
                    del vectors
                    row_path.write_bytes(gzip.compress(json.dumps(
                        {"split": split, "checkpoint_sha256": checkpoint_sha,
                         "rows": all_rows}, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode(),
                        compresslevel=6, mtime=0))
                    record["splits"][split] = {
                        "rows_sha256": _sha(row_path),
                        "vectors_sha256": _sha(vec_path),
                        "rows_bytes": row_path.stat().st_size,
                        "vectors_bytes": vec_path.stat().st_size}
                    _resource(start, result_dir)
                del model
                torch.mps.empty_cache()
        decision = audit_cross(primary_dir, suite_dir, finite_dir,
                               screen_dir, result_dir, root, runs=runs)
        _write(result_dir / "decision-audit.json", decision)
        sampled = _resource(start, result_dir)
        status["peak_sampled_mps_allocated_bytes"] = max(
            status["peak_sampled_mps_allocated_bytes"],
            sampled["sampled_mps_allocated_bytes"])
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
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("results/TEACH-0016-screen"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0016-cross"))
    args = parser.parse_args()
    result = (benchmark(args.root, args.suite_dir, args.result_dir)
              if args.mode == "benchmark" else
              run(args.root, args.primary_dir, args.suite_dir,
                  args.finite_dir, args.screen_dir, args.result_dir))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
