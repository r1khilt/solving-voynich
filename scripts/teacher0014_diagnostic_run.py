"""Bounded post-audit TEACH-0014 candidate-gate diagnosis.

Benchmarking uses random weights and the public frozen suite. The scientific
run refuses trained checkpoints until the primary artifact and numerical
replay audits exist and agree. This is a diagnostic, not a training rerun.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch

from voynich.workspace.teacher14_diagnose import evaluate_batch
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_tasks import evaluation_suite


ARM = "latent_rows_answer"
BATCH_SIZE = 32
RANDOM_REP = 0
MAX_SECONDS = 2 * 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
SOURCE_PATHS = (
    "docs/experiments/TEACH-0014-diagnostic-preregistration.md",
    "src/voynich/workspace/teacher14_diagnose.py",
    "scripts/teacher0014_diagnostic_run.py",
    "scripts/teacher0014_diagnostic_audit.py",
    "scripts/teacher0014_diagnostic_replay.py",
    "tests/test_teacher0014_diagnose.py",
    "tests/test_teacher0014_diagnostic_audit.py",
    "tests/test_teacher0014_diagnostic_replay.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit diagnostic sources before benchmark")
    return {"diagnostic_git_head": head,
            "diagnostic_source_sha256": {
                name: _sha(root / name) for name in SOURCE_PATHS}}


def _resource(start: float, directory: Path) -> dict:
    torch.mps.synchronize()
    elapsed = time.monotonic() - start
    allocated = torch.mps.current_allocated_memory()
    artifact_bytes = sum(path.stat().st_size for path in directory.rglob("*")
                         if path.is_file()) if directory.exists() else 0
    if elapsed > MAX_SECONDS or allocated > MAX_MPS_BYTES or (
            artifact_bytes > MAX_ARTIFACT_BYTES):
        raise RuntimeError("TEACH-0014 diagnostic finite resource cap reached")
    return {"elapsed_seconds": elapsed,
            "sampled_mps_allocated_bytes": allocated,
            "artifact_bytes": artifact_bytes}


def _batch_count(suite: dict) -> int:
    return sum(math.ceil(len(episodes) / BATCH_SIZE)
               for episodes in suite.values())


def _prepare_mps() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for diagnostic benchmark and run")
    recommended = torch.mps.recommended_max_memory()
    if recommended <= 0:
        raise RuntimeError("Mac GPU memory recommendation unavailable")
    torch.mps.set_per_process_memory_fraction(min(
        .40, MAX_MPS_BYTES / recommended))


def benchmark(root: Path, primary_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "benchmark.json").exists():
        raise FileExistsError("Diagnostic benchmark already exists")
    provenance = _provenance(root)
    _prepare_mps()
    suite = evaluation_suite(84311, 128)
    names = ("factorial", "boundary_groups", "long_ood",
             "hop_4_long_ood", "copy_confirm", "alias_inner")
    model = CandidateEdgeWorkspace().to("mps").eval()
    start = time.monotonic()
    times = {name: [] for name in names}
    peak_mps = 0
    for turn in range(30):
        episodes = suite[names[turn % len(names)]][:BATCH_SIZE]
        begin = time.monotonic()
        evaluate_batch(model, episodes, device="mps", random_rep=RANDOM_REP)
        torch.mps.synchronize()
        if turn >= 6:
            times[names[turn % len(names)]].append(time.monotonic() - begin)
        sampled = _resource(start, result_dir)
        peak_mps = max(peak_mps, sampled["sampled_mps_allocated_bytes"])
    panel_medians = {name: statistics.median(values)
                     for name, values in times.items()}
    slowest_median = max(panel_medians.values())
    projected = 1.75 * slowest_median * 2 * _batch_count(suite) + 300
    result = {"experiment": "TEACH-0014-gate-diagnostic",
              "benchmark": "random_weight_full_condition_batches",
              "primary_suite_sha256": _sha(primary_dir / "suite.json"),
              "batch_size": BATCH_SIZE, "random_rep": RANDOM_REP,
              "warmup_batches": 6, "timed_batches": 24,
              "timings_seconds": times,
              "panel_median_seconds": panel_medians,
              "slowest_panel_median_seconds": slowest_median,
              "batches_per_seed": _batch_count(suite),
              "conservative_projected_seconds": projected,
              "max_seconds": MAX_SECONDS,
              "max_mps_bytes": MAX_MPS_BYTES,
              "max_artifact_bytes": MAX_ARTIFACT_BYTES,
              "admitted": projected < MAX_SECONDS,
              "peak_sampled_mps_allocated_bytes": peak_mps,
              **_resource(start, result_dir), **provenance}
    _write(result_dir / "benchmark.json", result)
    return result


def _primary_gate(primary_dir: Path, root: Path) -> tuple[dict, dict]:
    from scripts.teacher0014_artifact_audit import audit_artifacts

    primary_outputs = root / "outputs/TEACH-0014-v3"
    audited = audit_artifacts(primary_dir, primary_outputs, root)
    archived_audit = json.loads((primary_dir / "artifact-audit.json").read_text())
    replay = json.loads((primary_dir / "replay-audit.json").read_text())
    report = json.loads((primary_dir / "report.json").read_text())
    if (audited != archived_audit or audited.get("audit") != "pass" or
            replay.get("audit") != "pass" or
            replay.get("source_git_head") != report.get("source_git_head") or
            replay.get("manifest_sha256") != audited.get("manifest_sha256") or
            report.get("status") != "complete"):
        raise ValueError("Primary TEACH-0014 artifact/replay gate failed")
    for name, expected in report["source_sha256"].items():
        if _sha(root / name) != expected:
            raise ValueError(f"Primary model source changed: {name}")
    behavior = json.loads((primary_dir / "behavior-audit.json").read_text())
    parser = json.loads((primary_dir / "parser-audit.json").read_text())
    if not (behavior["decisions"]["null_valid"] and
            behavior["decisions"]["oracle_twohop"] and
            all(not behavior["scores"][ARM][str(rep)]["twohop_absolute"]
                for rep in (0, 1)) and
            all(not parser["runs"][ARM][str(rep)]["parser_qualified"]
                for rep in (0, 1))):
        raise ValueError("Parser-rescue diagnostic not entered by primary gates")
    return report, audited


def _read_benchmark(result_dir: Path, primary_dir: Path,
                    provenance: dict) -> dict:
    row = json.loads((result_dir / "benchmark.json").read_text())
    suite = evaluation_suite(84311, 128)
    names = {"factorial", "boundary_groups", "long_ood",
             "hop_4_long_ood", "copy_confirm", "alias_inner"}
    timings = row.get("timings_seconds")
    if (not isinstance(timings, dict) or set(timings) != names or any(
            not isinstance(values, list) or len(values) != 4 or any(
                type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0 for value in values)
            for values in timings.values())):
        raise ValueError("Diagnostic benchmark timing grid invalid")
    medians = {name: statistics.median(values)
               for name, values in timings.items()}
    slowest = max(medians.values())
    projection = 1.75 * slowest * 2 * _batch_count(suite) + 300
    if (row.get("admitted") is not True or
            row.get("benchmark") != "random_weight_full_condition_batches" or
            row.get("warmup_batches") != 6 or row.get("timed_batches") != 24 or
            row.get("batches_per_seed") != _batch_count(suite) or
            row.get("panel_median_seconds") != medians or
            row.get("slowest_panel_median_seconds") != slowest or
            not math.isclose(row.get("conservative_projected_seconds", math.inf),
                             projection, rel_tol=1e-9) or
            projection >= MAX_SECONDS or
            row.get("max_seconds") != MAX_SECONDS or
            row.get("max_mps_bytes") != MAX_MPS_BYTES or
            row.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            row.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or
            row.get("diagnostic_source_sha256") != provenance[
                "diagnostic_source_sha256"] or
            row.get("primary_suite_sha256") != _sha(primary_dir / "suite.json") or
            row.get("batch_size") != BATCH_SIZE or
            row.get("random_rep") != RANDOM_REP):
        raise ValueError("Source-matched diagnostic benchmark failed")
    return row


def _load_model(primary_dir: Path, root: Path, report: dict,
                replicate: int) -> tuple[CandidateEdgeWorkspace, str]:
    path = root / "outputs/TEACH-0014-v3" / (
        f"rep{replicate}-{ARM}-step6000.pt")
    expected = report["training"][ARM][str(replicate)]["final_checkpoint"]
    if _sha(path) != expected["sha256"]:
        raise ValueError("Diagnostic checkpoint hash differs from primary archive")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("arm") != ARM or saved.get("replicate") != replicate or
            saved.get("step") != 6000):
        raise ValueError("Diagnostic checkpoint metadata mismatch")
    model = CandidateEdgeWorkspace().to("mps")
    model.load_state_dict(saved["model"], strict=True)
    return model.eval(), expected["sha256"]


def run(root: Path, primary_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "report.json").exists():
        raise FileExistsError("No automatic diagnostic scientific rerun")
    provenance = _provenance(root)
    report, primary_audit = _primary_gate(primary_dir, root)
    _read_benchmark(result_dir, primary_dir, provenance)
    _prepare_mps()
    suite = evaluation_suite(84311, 128)
    manifest = json.loads((primary_dir / "suite.json").read_text())
    if set(suite) != set(manifest["panels"]) or any(
            [episode.render_id for episode in suite[name]] !=
            [episode["render_id"] for episode in manifest["panels"][name]]
            for name in suite):
        raise ValueError("Diagnostic suite differs from primary manifest")
    start = time.monotonic()
    status_path = result_dir / "status.json"
    status = {"status": "running", "experiment": "TEACH-0014-gate-diagnostic",
              "primary_report_sha256": _sha(primary_dir / "report.json"),
              "primary_artifact_audit_sha256": _sha(
                  primary_dir / "artifact-audit.json"),
              "primary_replay_audit_sha256": _sha(
                  primary_dir / "replay-audit.json"),
              "benchmark_sha256": _sha(result_dir / "benchmark.json"),
              "manifest_sha256": primary_audit["manifest_sha256"],
              "primary_source_git_head": report["source_git_head"],
              "arm": ARM, "random_rep": RANDOM_REP,
              "peak_sampled_mps_allocated_bytes": 0, **provenance}
    _write(status_path, status)
    runs = {}
    try:
        for replicate in (0, 1):
            model, checkpoint_sha = _load_model(
                primary_dir, root, report, replicate)
            panels = {}
            for name, episodes in suite.items():
                rows = []
                samples = {0, len(episodes) // 2, len(episodes) - 1}
                for offset in range(0, len(episodes), BATCH_SIZE):
                    observed = evaluate_batch(
                        model, episodes[offset:offset + BATCH_SIZE],
                        device="mps", random_rep=RANDOM_REP)
                    for local, item in enumerate(observed):
                        index = offset + local
                        row = {key: value for key, value in item.items()
                               if key != "conditions"}
                        row["conditions"] = {}
                        for condition, detail in item["conditions"].items():
                            row["conditions"][condition] = {
                                key: value for key, value in detail.items()
                                if key != "logits"}
                            if index in samples:
                                row["conditions"][condition]["sampled_logits"] = (
                                    detail["logits"])
                        rows.append(row)
                    sampled = _resource(start, result_dir)
                    status["peak_sampled_mps_allocated_bytes"] = max(
                        status["peak_sampled_mps_allocated_bytes"],
                        sampled["sampled_mps_allocated_bytes"])
                panels[name] = rows
                status["progress"] = {"replicate": replicate, "panel": name,
                                      "panels_complete": len(panels)}
                status.update(_resource(start, result_dir))
                _write(status_path, status)
            path = result_dir / f"diagnostic-rep{replicate}.json.gz"
            raw = {"checkpoint_sha256": checkpoint_sha, "panels": panels}
            path.write_bytes(gzip.compress(json.dumps(
                raw, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode(), compresslevel=9, mtime=0))
            runs[str(replicate)] = {"path": str(path), "sha256": _sha(path),
                                    "bytes": path.stat().st_size,
                                    "checkpoint_sha256": checkpoint_sha}
            del model
            torch.mps.empty_cache()
            _resource(start, result_dir)
        from scripts.teacher0014_diagnostic_audit import audit_diagnostic

        decision = audit_diagnostic(primary_dir, result_dir, root,
                                    runs=runs)
        _write(result_dir / "decision-audit.json", decision)
        final = {**status, "status": "complete", "runs": runs,
                 "decision_audit_sha256": _sha(result_dir / "decision-audit.json"),
                 **_resource(start, result_dir)}
        _write(result_dir / "report.json", final)
        _write(status_path, {**status, "status": "complete",
                             **_resource(start, result_dir)})
        return final
    except Exception as exc:
        _write(status_path, {**status, "status": "stopped",
                             "reason": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014-diagnostic"))
    args = parser.parse_args()
    result = (benchmark(args.root, args.primary_dir, args.result_dir)
              if args.mode == "benchmark" else
              run(args.root, args.primary_dir, args.result_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
