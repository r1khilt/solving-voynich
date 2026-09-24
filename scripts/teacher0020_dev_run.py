"""Bounded development-only screen of completed interrupted TEACH-0014 arms."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from scripts.teacher0014_artifact_audit import audit_trace
from scripts.teacher0014_suite_audit import audit_manifest
from voynich.workspace.teacher14_tasks import Episode, SYMBOL_START
from voynich.workspace.teacher14_train import (
    ARMS, Config, SOURCE_PATHS as PRIMARY_SOURCES, _set_mps_cap,
    model_output, new_model,
)


ARMS_DONE = ARMS[:6]
EXPECTED_MANIFEST = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
SAMPLE_COUNT = 3
MAX_SECONDS = 3600.0
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 1024**3
SOURCE_PATHS = (
    "docs/experiments/TEACH-0020-interrupted-development-screen.md",
    "scripts/teacher0020_dev_run.py",
    "scripts/teacher0020_dev_audit.py",
    "scripts/teacher0020_dev_replay.py",
    "tests/test_teacher0020_dev.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def episode_from_json(value: dict) -> Episode:
    """Restore tuple-typed fields required by the frozen model input path."""
    return Episode(**{**value,
                      "tokens": tuple(value["tokens"]),
                      "signal_paths": tuple(tuple(path) for path in value[
                          "signal_paths"]),
                      "distractor_paths": tuple(tuple(path) for path in value[
                          "distractor_paths"]),
                      "serialized_rows": tuple(tuple(row) for row in value[
                          "serialized_rows"]),
                      "row_positions": tuple(tuple(row) for row in value[
                          "row_positions"]),
                      "stage_partitions": tuple(value["stage_partitions"])})


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0020 sources before inference")
    return {"screen_source_git_head": head,
            "screen_source_sha256": {name: sha(root / name)
                                     for name in SOURCE_PATHS}}


def preflight(root: Path, primary_dir: Path, output_dir: Path,
              suite_path: Path) -> tuple[dict, dict, dict, dict]:
    status_path = primary_dir / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "running" or (
            primary_dir / "report.json").exists():
        raise ValueError("Original TEACH-0014 campaign is not interrupted")
    head = status.get("source_git_head")
    hashes = status.get("source_sha256")
    if (head != "37288feb7769ef62317cd3325ea1375d729f7d46" or
            not isinstance(hashes, dict) or set(hashes) != set(PRIMARY_SOURCES)):
        raise ValueError("Original source provenance differs")
    for name, expected in hashes.items():
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != expected or (
                sha(root / name) != expected):
            raise ValueError(f"Original TEACH-0014 source changed: {name}")
    benchmark = json.loads((primary_dir / "benchmark.json").read_text())
    if (benchmark.get("status") != "pass" or
            benchmark.get("source_git_head") != head or
            benchmark.get("source_sha256") != hashes or
            sha(primary_dir / "benchmark.json") != status.get("benchmark_sha256")):
        raise ValueError("Original TEACH-0014 benchmark differs")
    manifest = json.loads(suite_path.read_text())
    checked = audit_manifest(manifest)
    if checked["manifest_sha256"] != EXPECTED_MANIFEST:
        raise ValueError("Exposed development manifest differs")
    if set(manifest["panels"]) != set(json.loads(
            (primary_dir / "suite.json").read_text())["panels"]):
        raise ValueError("Development panel set differs")
    files = {}
    fingerprints = {}
    for arm in ARMS_DONE:
        files[arm] = {}
        fingerprints[arm] = {}
        for rep in (0, 1):
            checkpoint = output_dir / f"rep{rep}-{arm}-step6000.pt"
            losses = primary_dir / f"losses-rep{rep}-{arm}.json.gz"
            if not checkpoint.is_file() or not losses.is_file():
                raise ValueError(f"Completed {arm}/{rep} files missing")
            archive = json.loads(gzip.decompress(losses.read_bytes()))
            fingerprints[arm][rep] = audit_trace(archive, arm, steps=6000)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (saved.get("arm") != arm or saved.get("replicate") != rep or
                    saved.get("step") != 6000 or
                    saved.get("config") != Config().__dict__ or
                    saved.get("parameters") != (
                        24_766_465)):
                raise ValueError(f"Completed {arm}/{rep} checkpoint identity")
            files[arm][str(rep)] = {
                "checkpoint_sha256": sha(checkpoint),
                "loss_sha256": sha(losses),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "loss_bytes": losses.stat().st_size,
            }
            del saved
    for rep in (0, 1):
        oracle = fingerprints["oracle_rows_workspace"][rep]
        for arm in ARMS_DONE:
            if (fingerprints[arm][rep]["answer_inputs"] !=
                    oracle["answer_inputs"] or
                    fingerprints[arm][rep]["answer_labels"] !=
                    oracle["answer_labels"]):
                raise ValueError(f"Completed {arm}/{rep} input-label drift")
        if (fingerprints["latent_rows_causal"][rep]["causal_inputs"] !=
                fingerprints["latent_rows_wrong_causal"][rep]["causal_inputs"]):
            raise ValueError("Completed causal objective input drift")
    return status, benchmark, manifest, files


def _resource(start: float, result_dir: Path) -> dict:
    elapsed = time.monotonic() - start
    used = torch.mps.current_allocated_memory()
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if elapsed > MAX_SECONDS or used > MAX_MPS_BYTES or (
            artifact_bytes > MAX_ARTIFACT_BYTES):
        raise RuntimeError("TEACH-0020 resource cap exceeded")
    return {"elapsed_seconds": elapsed,
            "sampled_mps_allocated_bytes": used,
            "artifact_bytes": artifact_bytes}


@torch.no_grad()
def predict(model: torch.nn.Module, arm: str,
            panels: dict[str, list[Episode]], start: float,
            result_dir: Path) -> dict:
    model.eval()
    rows = {}
    samples = {}
    for name, episodes in panels.items():
        indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
        records = []
        selected = []
        for offset in range(0, len(episodes), 32):
            batch = episodes[offset:offset + 32]
            output = model_output(model, arm, batch, "mps")
            logits = output.logits.float()
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite development logits")
            guesses = (logits[:, SYMBOL_START:].argmax(dim=-1)
                       + SYMBOL_START).tolist()
            records.extend({"render_id": item.render_id, "prediction": int(guess)}
                           for item, guess in zip(batch, guesses, strict=True))
            for index in indices:
                if offset <= index < offset + len(batch):
                    selected.append({"index": index,
                                     "render_id": episodes[index].render_id,
                                     "logits": logits[index - offset].cpu().tolist()})
            _resource(start, result_dir)
        rows[name] = records
        samples[name] = sorted(selected, key=lambda item: item["index"])
    return {"panels": rows, "samples": samples}


def run(root: Path, primary_dir: Path, output_dir: Path,
        suite_path: Path, result_dir: Path) -> dict:
    if (result_dir / "status.json").exists() or (
            result_dir / "report.json").exists():
        raise FileExistsError("No automatic TEACH-0020 rerun")
    source = provenance(root)
    primary, benchmark, manifest, files = preflight(
        root, primary_dir, output_dir, suite_path)
    worst = max(benchmark["measured"][arm]["median_timed_step_seconds"]
                for arm in ARMS_DONE)
    projected = 1.5 * worst * len(ARMS_DONE) * 2 * 148 + 300
    if projected >= MAX_SECONDS:
        raise RuntimeError("TEACH-0020 development resource projection failed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0020")
    fraction, recommended = _set_mps_cap(Config())
    panels = {name: [episode_from_json(item) for item in episodes]
              for name, episodes in manifest["panels"].items()}
    start = time.monotonic()
    status = {"experiment": "TEACH-0020", "status": "running",
              "primary_status_sha256": sha(primary_dir / "status.json"),
              "primary_source_git_head": primary["source_git_head"],
              "primary_source_sha256": primary["source_sha256"],
              "primary_benchmark_sha256": sha(primary_dir / "benchmark.json"),
              "manifest_sha256": EXPECTED_MANIFEST,
              "manifest_file_sha256": sha(suite_path),
              "mps_fraction": fraction, "recommended_memory": recommended,
              "projected_seconds": projected, "max_seconds": MAX_SECONDS,
              "max_mps_bytes": MAX_MPS_BYTES,
              "max_artifact_bytes": MAX_ARTIFACT_BYTES,
              "peak_sampled_mps_allocated_bytes": 0,
              "completed_checkpoint_files": files, **source}
    _write(result_dir / "status.json", status)
    archives = {}
    try:
        for arm in ARMS_DONE:
            archives[arm] = {}
            for rep in (0, 1):
                checkpoint = output_dir / f"rep{rep}-{arm}-step6000.pt"
                model, optimizer = new_model(Config(), arm, rep, "mps")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model.load_state_dict(saved["model"], strict=True)
                payload = predict(model, arm, panels, start, result_dir)
                path = result_dir / f"rows-{arm}-rep{rep}.json.gz"
                path.write_bytes(gzip.compress(json.dumps(
                    payload, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode(), compresslevel=9, mtime=0))
                archives[arm][str(rep)] = {"sha256": sha(path),
                                           "bytes": path.stat().st_size}
                resource = _resource(start, result_dir)
                status["progress"] = {"arm": arm, "replicate": rep}
                status["peak_sampled_mps_allocated_bytes"] = max(
                    status["peak_sampled_mps_allocated_bytes"],
                    resource["sampled_mps_allocated_bytes"])
                status.update(resource)
                _write(result_dir / "status.json", status)
                del model, optimizer, saved, payload
                torch.mps.empty_cache()
        report = {**status, "status": "complete", "archives": archives,
                  **_resource(start, result_dir)}
        _write(result_dir / "report.json", report)
        _write(result_dir / "status.json", {**status, "status": "complete"})
        return report
    except Exception as exc:
        _write(result_dir / "status.json", {**status, "status": "stopped",
                                             "reason": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    root = Path.cwd()
    run(root, root / "results/TEACH-0014-v3",
        root / "outputs/TEACH-0014-v3",
        root / "outputs/TEACH-0016/teach14-74111.json",
        root / "results/TEACH-0020-interrupted-dev-v2")
