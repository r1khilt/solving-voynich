"""Bounded, resumable linked-reader factorial campaign."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from scripts.teacher0022_suite_audit import audit as audit_suites
from voynich.workspace.teacher14_objectives import answer_loss
from voynich.workspace.teacher14_tasks import training_batch
from voynich.workspace.teacher22_models import (
    make_reader, reader_output, route_loss,
)


ARMS = ("unlinked_route", "linked_answer", "linked_route")
STEPS = 6000
BATCH = 32
INIT_SEEDS = (84121, 84131)
TRAIN_SEEDS = (84221, 84231)
ROUTE_WEIGHT = 0.2
LR = 3e-4
MAX_SECONDS = 4 * 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
SOURCE_PATHS = (
    "docs/experiments/TEACH-0022-linked-reader.md",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
    "src/voynich/workspace/teacher14_objectives.py",
    "src/voynich/workspace/teacher14_train.py",
    "src/voynich/workspace/teacher22_models.py",
    "scripts/teacher0014_suite_audit.py",
    "scripts/teacher0014_behavior_audit.py",
    "scripts/teacher0014_artifact_audit.py",
    "scripts/teacher0020_dev_run.py",
    "scripts/teacher0022_suite.py",
    "scripts/teacher0022_suite_audit.py",
    "scripts/teacher0022_train.py",
    "scripts/teacher0022_score.py",
    "scripts/teacher0022_audit.py",
    "scripts/teacher0022_replay.py",
    "tests/test_teacher0022_models.py",
    "tests/test_teacher0022_campaign.py",
)


@dataclass(frozen=True)
class Config:
    steps: int = STEPS
    batch: int = BATCH
    init_seeds: tuple[int, int] = INIT_SEEDS
    train_seeds: tuple[int, int] = TRAIN_SEEDS
    route_weight: float = ROUTE_WEIGHT
    learning_rate: float = LR
    max_seconds: int = MAX_SECONDS
    max_mps_bytes: int = MAX_MPS_BYTES
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0022 config differs")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(root: Path) -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *SOURCE_PATHS], cwd=root, check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit all TEACH-0022 sources before execution")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    return {"source_head": head,
            "source_sha256": {name: sha(root / name) for name in SOURCE_PATHS}}


def new_arm(arm: str, rep: int, device: str):
    if arm not in ARMS or rep not in (0, 1):
        raise ValueError("Unknown TEACH-0022 arm/replicate")
    model = make_reader(linked=arm.startswith("linked_"),
                        seed=INIT_SEEDS[rep], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    return model, optimizer


def update(model, optimizer, arm: str, rep: int, step: int,
           device: str) -> dict:
    episodes, answers = training_batch(TRAIN_SEEDS[rep] + step, BATCH,
                                       step=step)
    optimizer.zero_grad(set_to_none=True)
    output, left, right, mask = reader_output(
        model, episodes, device, capture=arm.endswith("_route"))
    primary = answer_loss(output, answers)
    components = {"answer": primary}
    if arm.endswith("_route"):
        components["route"] = route_loss(output, episodes, left, right, mask)
    total = primary + ROUTE_WEIGHT * components.get(
        "route", primary.new_zeros(()))
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("Nonfinite TEACH-0022 loss")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("Nonfinite TEACH-0022 gradient")
    optimizer.step()
    return {"step": step, "arm": arm, "replicate": rep,
            "losses": {name: float(value.detach())
                       for name, value in components.items()},
            "gradient_norm": float(norm),
            "answer_input_sha256": canonical([
                episode.render_id for episode in episodes]),
            "answer_label_sha256": canonical(answers)}


def _resource(root: Path, start: float) -> dict:
    torch.mps.synchronize()
    allocated = torch.mps.current_allocated_memory()
    artifact_bytes = sum(path.stat().st_size for folder in (
        root / "outputs/TEACH-0022", root / "results/TEACH-0022")
                         if folder.exists() for path in folder.rglob("*")
                         if path.is_file())
    elapsed = time.monotonic() - start
    if elapsed > MAX_SECONDS or allocated > MAX_MPS_BYTES or (
            artifact_bytes > MAX_ARTIFACT_BYTES):
        raise RuntimeError("TEACH-0022 resource limit")
    return {"elapsed_seconds": elapsed, "sampled_mps_bytes": allocated,
            "artifact_bytes": artifact_bytes}


def _baseline(root: Path) -> dict:
    from scripts.teacher0014_artifact_audit import audit_trace

    primary_dir = root / "results/TEACH-0014-v3"
    status = json.loads((primary_dir / "status.json").read_text())
    if status.get("source_git_head") != (
            "37288feb7769ef62317cd3325ea1375d729f7d46"):
        raise ValueError("Original baseline source differs")
    files = {}
    fingerprints = {}
    for rep in (0, 1):
        path = primary_dir / f"losses-rep{rep}-oracle_rows_workspace.json.gz"
        checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
        archive = json.loads(gzip.decompress(path.read_bytes()))
        fingerprints[str(rep)] = audit_trace(
            archive, "oracle_rows_workspace", steps=STEPS)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["arm"] != "oracle_rows_workspace" or
                saved["replicate"] != rep or saved["step"] != STEPS):
            raise ValueError("Original baseline checkpoint identity")
        files[str(rep)] = {
            "loss_sha256": sha(path), "checkpoint_sha256": sha(checkpoint),
            "parameters": saved["parameters"]}
    return {"source_head": status["source_git_head"],
            "files": files, "fingerprints": fingerprints}


def benchmark(root: Path) -> dict:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    source = provenance(root)
    result_dir = root / "results/TEACH-0022"
    path = result_dir / "benchmark.json"
    if path.exists():
        raise FileExistsError("No automatic benchmark rerun")
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"experiment": "TEACH-0022", "mode": "benchmark",
              "status": "running", "config": asdict(Config()),
              "torch_version": torch.__version__,
              "platform": platform.platform(), **source}
    write_json(path, report)
    measured = {}
    try:
        for arm in ARMS:
            model, optimizer = new_arm(arm, 0, "mps")
            times = []
            for step in range(24):
                before = time.monotonic()
                update(model, optimizer, arm, 0, step, "mps")
                torch.mps.synchronize()
                if step >= 4:
                    times.append(time.monotonic() - before)
                resource = _resource(root, start)
            measured[arm] = {
                "parameters": model.parameter_count,
                "median_timed_step_seconds": statistics.median(times),
                "timed_steps": 20}
            report.update({"measured": measured, **resource})
            write_json(path, report)
            del model, optimizer
            torch.mps.empty_cache()
        worst = max(row["median_timed_step_seconds"]
                    for row in measured.values())
        projection = 1.5 * worst * len(ARMS) * 2 * STEPS + 900
        report.update({"status": "pass" if projection < MAX_SECONDS else "stop",
                       "worst_median_seconds": worst,
                       "conservative_projected_seconds": projection,
                       **_resource(root, start)})
        write_json(path, report)
        return report
    except Exception as exc:
        report.update({"status": "stop",
                       "reason": f"{type(exc).__name__}: {exc}"})
        write_json(path, report)
        raise


def _atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(root: Path, *, resume: bool = False) -> dict:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    source = provenance(root)
    result_dir = root / "results/TEACH-0022"
    output_dir = root / "outputs/TEACH-0022"
    benchmark_path = result_dir / "benchmark.json"
    bench = json.loads(benchmark_path.read_text())
    if (bench.get("status") != "pass" or bench["source_sha256"] != source[
            "source_sha256"] or bench["config"] != json.loads(json.dumps(
                asdict(Config()))) or bench[
                    "conservative_projected_seconds"] >= MAX_SECONDS):
        raise RuntimeError("Source-matched resource gate failed")
    suite_audit = audit_suites(root)
    stored_suite = json.loads((result_dir / "suite-audit.json").read_text())
    if suite_audit != stored_suite or stored_suite["audit"] != "pass":
        raise RuntimeError("Fresh suite audit differs")
    baseline = _baseline(root)
    status_path = result_dir / "status.json"
    if status_path.exists() != resume:
        raise FileExistsError("Explicit resume required for existing campaign")
    if resume:
        status = json.loads(status_path.read_text())
        if status["source_sha256"] != source["source_sha256"] or status[
                "baseline"] != baseline:
            raise ValueError("Resume provenance differs")
        if status["status"] == "complete":
            raise ValueError("Campaign already complete")
    else:
        status = {"experiment": "TEACH-0022", "status": "running",
                  "source_head": source["source_head"],
                  "source_sha256": source["source_sha256"],
                  "suite_audit_sha256": sha(result_dir / "suite-audit.json"),
                  "benchmark_sha256": sha(benchmark_path),
                  "baseline": baseline, "runs": {}}
        write_json(status_path, status)
    start = time.monotonic()
    try:
        for arm in ARMS:
            status["runs"].setdefault(arm, {})
            for rep in (0, 1):
                key = str(rep)
                if status["runs"][arm].get(key, {}).get("status") == "complete":
                    continue
                model, optimizer = new_arm(arm, rep, "mps")
                progress_path = output_dir / f"current-rep{rep}-{arm}.pt"
                trace_path = result_dir / f"losses-rep{rep}-{arm}.json.gz"
                if resume:
                    if not progress_path.exists():
                        raise ValueError("Interrupted run lacks optimizer checkpoint")
                    saved = torch.load(progress_path, map_location="cpu",
                                       weights_only=True)
                    if (saved["arm"] != arm or saved["replicate"] != rep or
                            saved["source_sha256"] != source["source_sha256"]):
                        raise ValueError("Resume checkpoint identity differs")
                    model.load_state_dict(saved["model"], strict=True)
                    optimizer.load_state_dict(saved["optimizer"])
                    trace = saved["trace"]
                    next_step = saved["next_step"]
                else:
                    trace, next_step = [], 0
                model.train()
                for step in range(next_step, STEPS):
                    row = update(model, optimizer, arm, rep, step, "mps")
                    trace.append(row)
                    if step % 100 == 99 or step == STEPS - 1:
                        resource = _resource(root, start)
                        status["runs"][arm][key] = {
                            "status": "running", "next_step": step + 1,
                            **resource}
                        write_json(status_path, status)
                    if step % 1000 == 999 and step != STEPS - 1:
                        _atomic_checkpoint(progress_path, {
                            "model": {name: tensor.detach().cpu()
                                      for name, tensor in model.state_dict().items()},
                            "optimizer": optimizer.state_dict(),
                            "trace": trace, "next_step": step + 1,
                            "arm": arm, "replicate": rep,
                            "source_sha256": source["source_sha256"]})
                final = output_dir / f"rep{rep}-{arm}-step6000.pt"
                _atomic_checkpoint(final, {
                    "model": {name: tensor.detach().cpu()
                              for name, tensor in model.state_dict().items()},
                    "arm": arm, "replicate": rep, "step": STEPS,
                    "parameters": model.parameter_count,
                    "source_sha256": source["source_sha256"]})
                trace_path.write_bytes(gzip.compress(
                    json.dumps(trace, separators=(",", ":"),
                               allow_nan=False).encode(), compresslevel=9))
                if progress_path.exists():
                    progress_path.unlink()
                status["runs"][arm][key] = {
                    "status": "complete", "checkpoint_sha256": sha(final),
                    "loss_sha256": sha(trace_path),
                    "parameters": model.parameter_count,
                    **_resource(root, start)}
                write_json(status_path, status)
                del model, optimizer
                torch.mps.empty_cache()
        status.update({"status": "complete", **_resource(root, start)})
        write_json(status_path, status)
        return status
    except Exception as exc:
        status.update({"status": "stopped",
                       "reason": f"{type(exc).__name__}: {exc}"})
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "run", "resume"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.mode == "benchmark":
        result = benchmark(root)
    else:
        result = train(root, resume=args.mode == "resume")
    print(json.dumps(result, indent=2, sort_keys=True))
