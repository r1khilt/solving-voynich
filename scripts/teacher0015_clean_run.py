"""Bounded clean-competence stage before any TEACH-0015 intervention."""

import argparse
import gzip
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Callable

import torch

from scripts.teacher0015_clean_audit import (
    ARMS, EXPECTED_MANIFESTS, MAX_ARTIFACT_BYTES, MAX_MPS_BYTES,
    MAX_SECONDS, SOURCE_PATHS, SPLITS, _sha,
    audit_screen, eligible_arms, read_manifests,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_objectives import padded_tokens
from voynich.workspace.teacher14_tasks import Episode, SYMBOL_START


BATCH_SIZE = 32


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
        raise RuntimeError("Commit TEACH-0015 screen sources before benchmark")
    return {"screen_git_head": head,
            "screen_source_sha256": {
                name: _sha(root / name) for name in SOURCE_PATHS}}


def _prepare_mps() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0015 clean screen")
    recommended = torch.mps.recommended_max_memory()
    if recommended <= 0:
        raise RuntimeError("Mac GPU memory recommendation unavailable")
    torch.mps.set_per_process_memory_fraction(min(
        .40, MAX_MPS_BYTES / recommended))


def _resource(start: float, result_dir: Path) -> dict:
    torch.mps.synchronize()
    elapsed = time.monotonic() - start
    allocated = torch.mps.current_allocated_memory()
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file()) if result_dir.exists() else 0
    if (elapsed > MAX_SECONDS or allocated > MAX_MPS_BYTES or
            artifact_bytes > MAX_ARTIFACT_BYTES):
        raise RuntimeError("TEACH-0015 screen resource cap reached")
    return {"elapsed_seconds": elapsed,
            "sampled_mps_allocated_bytes": allocated,
            "artifact_bytes": artifact_bytes}


def _episodes(manifest: dict) -> list[Episode]:
    return [Episode(**{**cell["episode"],
                       "tokens": tuple(cell["episode"]["tokens"])})
            for group in manifest["groups"]
            for cell in group["cells"]]


@torch.no_grad()
def _predict(model: CandidateEdgeWorkspace, episodes: list[Episode],
             device: str, health: Callable[[], None] | None = None
             ) -> tuple[list[dict], list[dict]]:
    if not episodes:
        raise ValueError("TEACH-0015 clean screen requires episodes")
    model.eval()
    rows = []
    samples = []
    sample_indices = (0, len(episodes) // 2, len(episodes) - 1)
    for offset in range(0, len(episodes), BATCH_SIZE):
        chunk = episodes[offset:offset + BATCH_SIZE]
        output = model(padded_tokens(chunk, device))
        logits = output.logits
        guesses = (logits[:, SYMBOL_START:].argmax(dim=-1) +
                   SYMBOL_START).tolist()
        rows.extend({"render_id": episode.render_id, "prediction": prediction}
                    for episode, prediction in zip(chunk, guesses, strict=True))
        for index in sample_indices:
            if offset <= index < offset + len(chunk):
                samples.append({"index": index,
                                "render_id": episodes[index].render_id,
                                "logits": logits[index - offset].float().cpu().tolist()})
        if health is not None and (offset // BATCH_SIZE) % 8 == 0:
            health()
    return rows, samples


def benchmark(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "benchmark.json").exists():
        raise FileExistsError("TEACH-0015 screen benchmark already exists")
    provenance = _provenance(root)
    manifests = read_manifests(suite_dir)
    _prepare_mps()
    model = CandidateEdgeWorkspace().to("mps").eval()
    episodes = _episodes(manifests["confirmation"])
    # Alternate complete composed, auxiliary and mixed batches; only random
    # weights are used. Every timed batch follows the same forward/archive path.
    choices = (0, 32, 4224, 8416)
    start = time.monotonic()
    timings = []
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
    batches_per_arm_seed = sum(math.ceil(len(_episodes(manifests[split])) /
                                         BATCH_SIZE) for split in SPLITS)
    projection = 1.75 * median * batches_per_arm_seed * len(ARMS) * 2 + 300
    result = {"experiment": "TEACH-0015-clean-screen",
              "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "batch_size": BATCH_SIZE,
              "warmup_batches": 6, "timed_batches": 24,
              "timings_seconds": timings, "median_batch_seconds": median,
              "batches_per_arm_seed": batches_per_arm_seed,
              "conservative_projected_seconds": projection,
              "max_seconds": MAX_SECONDS,
              "max_mps_bytes": MAX_MPS_BYTES,
              "max_artifact_bytes": MAX_ARTIFACT_BYTES,
              "peak_sampled_mps_allocated_bytes": peak,
              "admitted": projection < MAX_SECONDS,
              **_resource(start, result_dir), **provenance}
    _write(result_dir / "benchmark.json", result)
    return result


def _primary_gate(primary_dir: Path, root: Path) -> tuple[dict, tuple[str, ...]]:
    from scripts.teacher0014_artifact_audit import audit_artifacts

    checked = audit_artifacts(primary_dir, root / "outputs/TEACH-0014-v3", root)
    archived = json.loads((primary_dir / "artifact-audit.json").read_text())
    replay = json.loads((primary_dir / "replay-audit.json").read_text())
    report = json.loads((primary_dir / "report.json").read_text())
    if (checked != archived or checked.get("audit") != "pass" or
            replay.get("audit") != "pass" or
            replay.get("manifest_sha256") != checked["manifest_sha256"] or
            replay.get("source_git_head") != report["source_git_head"] or
            report.get("status") != "complete"):
        raise ValueError("TEACH-0014 primary artifact/replay gate failed")
    for name, expected in report["source_sha256"].items():
        if _sha(root / name) != expected:
            raise ValueError(f"Primary model source changed: {name}")
    return report, eligible_arms(primary_dir)


def _benchmark_gate(result_dir: Path, suite_dir: Path,
                    provenance: dict) -> dict:
    from scripts.teacher0015_clean_audit import _benchmark_check

    benchmark_row = json.loads((result_dir / "benchmark.json").read_text())
    manifests = read_manifests(suite_dir)
    _benchmark_check(benchmark_row, manifests,
                     provenance["screen_source_sha256"])
    if benchmark_row.get("suite_file_sha256") != {
            split: _sha(suite_dir / f"{split}.json") for split in SPLITS}:
        raise ValueError("TEACH-0015 screen suite file changed")
    return benchmark_row


def _load_model(root: Path, primary_report: dict, arm: str,
                rep: int, device: str) -> tuple[CandidateEdgeWorkspace, str]:
    path = root / "outputs/TEACH-0014-v3" / f"rep{rep}-{arm}-step6000.pt"
    expected = primary_report["training"][arm][str(rep)]["final_checkpoint"]
    checkpoint_sha = _sha(path)
    if checkpoint_sha != expected["sha256"]:
        raise ValueError("TEACH-0015 screen checkpoint hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("arm") != arm or saved.get("replicate") != rep or
            saved.get("step") != 6000):
        raise ValueError("TEACH-0015 screen checkpoint metadata differs")
    model = CandidateEdgeWorkspace().to(device)
    model.load_state_dict(saved["model"], strict=True)
    if model.parameter_count != primary_report["training"][arm][str(rep)][
            "parameters"]:
        raise ValueError("TEACH-0015 screen model parameter count differs")
    return model.eval(), checkpoint_sha


def run(root: Path, primary_dir: Path, suite_dir: Path,
        result_dir: Path) -> dict:
    if (result_dir / "report.json").exists():
        raise FileExistsError("No automatic TEACH-0015 clean-screen rerun")
    provenance = _provenance(root)
    primary_report, arms = _primary_gate(primary_dir, root)
    if not arms:
        result = {"experiment": "TEACH-0015-clean-screen",
                  "status": "not_entered_primary_competence",
                  "primary_report_sha256": _sha(primary_dir / "report.json"),
                  **provenance}
        _write(result_dir / "entry.json", result)
        return result
    _benchmark_gate(result_dir, suite_dir, provenance)
    manifests = read_manifests(suite_dir)
    _prepare_mps()
    start = time.monotonic()
    status_path = result_dir / "status.json"
    status = {"experiment": "TEACH-0015-clean-screen",
              "status": "running", "eligible_arms": list(arms),
              "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(
                  suite_dir / f"{split}.json") for split in SPLITS},
              "benchmark_sha256": _sha(result_dir / "benchmark.json"),
              "primary_report_sha256": _sha(primary_dir / "report.json"),
              "primary_artifact_audit_sha256": _sha(
                  primary_dir / "artifact-audit.json"),
              "primary_replay_audit_sha256": _sha(
                  primary_dir / "replay-audit.json"),
              "primary_source_git_head": primary_report["source_git_head"],
              "torch_version": torch.__version__,
              "platform": platform.platform(),
              "peak_sampled_mps_allocated_bytes": 0, **provenance}
    _write(status_path, status)
    runs = {}
    def health() -> None:
        sampled = _resource(start, result_dir)
        status["peak_sampled_mps_allocated_bytes"] = max(
            status["peak_sampled_mps_allocated_bytes"],
            sampled["sampled_mps_allocated_bytes"])

    try:
        for arm in arms:
            runs[arm] = {}
            for rep in (0, 1):
                model, checkpoint_sha = _load_model(
                    root, primary_report, arm, rep, "mps")
                splits = {}
                for split in SPLITS:
                    episodes = _episodes(manifests[split])
                    rows, samples = _predict(model, episodes, "mps", health)
                    splits[split] = {"rows": rows, "samples": samples}
                    sampled = _resource(start, result_dir)
                    status["peak_sampled_mps_allocated_bytes"] = max(
                        status["peak_sampled_mps_allocated_bytes"],
                        sampled["sampled_mps_allocated_bytes"])
                    status["progress"] = {"arm": arm, "replicate": rep,
                                          "split": split}
                    status.update(sampled)
                    _write(status_path, status)
                path = result_dir / f"predictions-{arm}-rep{rep}.json.gz"
                raw = {"checkpoint_sha256": checkpoint_sha,
                       "splits": splits}
                path.write_bytes(gzip.compress(json.dumps(
                    raw, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode(), compresslevel=9, mtime=0))
                runs[arm][str(rep)] = {
                    "path": str(path), "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                    "checkpoint_sha256": checkpoint_sha}
                del model
                torch.mps.empty_cache()
                _resource(start, result_dir)
        decision = audit_screen(primary_dir, suite_dir, result_dir,
                                root, runs=runs)
        _write(result_dir / "decision-audit.json", decision)
        final = {**status, "status": "complete", "runs": runs,
                 "decision_audit_sha256": _sha(
                     result_dir / "decision-audit.json"),
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
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0015-screen"))
    args = parser.parse_args()
    result = (benchmark(args.root, args.suite_dir, args.result_dir)
              if args.mode == "benchmark" else
              run(args.root, args.primary_dir, args.suite_dir,
                  args.result_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
