"""Bounded TEACH-0014 trainer draft; campaign admission remains closed.

The benchmark and run interfaces are implemented to make cost measurable and
artifacts auditable. Scientific launch requires a frozen full outcome auditor,
source registration and an explicit resource amendment if the benchmark stops.
"""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from .teacher14_models import (
    CandidateEdgeWorkspace, DenseEpisodeClassifier, public_edge_loss,
    public_row_tensors,
)
from .teacher14_objectives import (
    answer_loss, interchange_loss, matched_refinement_objectives, padded_tokens,
)
from .teacher14_tasks import (
    SYMBOL_START, causal_training_batch, evaluation_suite, suite_manifest,
    training_batch,
)


ARMS = (
    "oracle_rows_workspace", "latent_rows_answer", "latent_rows_edge_aux",
    "latent_rows_causal", "latent_rows_wrong_causal", "latent_rows_one_read",
    "latent_rows_mean_address", "raw_dense_matched",
    "latent_rows_recurrent4", "latent_rows_diffuse", "raw_null",
)
EXPECTED_PARAMETERS = {
    arm: (24_757_760 if arm == "raw_dense_matched" else
          27_390_465 if arm in ("latent_rows_recurrent4", "latent_rows_diffuse")
          else 24_766_465) for arm in ARMS
}
SOURCE_PATHS = (
    "docs/experiments/TEACH-0014-design.md",
    "docs/experiments/TEACH-0014-benchmark-registration.md",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
    "src/voynich/workspace/teacher14_objectives.py",
    "src/voynich/workspace/teacher14_train.py",
    "scripts/teacher0014_suite_audit.py",
    "scripts/teacher0014_behavior_audit.py",
    "scripts/teacher0014_artifact_audit.py",
    "scripts/teacher0014_replay.py",
    "tests/test_workspace_teacher14_tasks.py",
    "tests/test_workspace_teacher14_models.py",
    "tests/test_workspace_teacher14_objectives.py",
    "tests/test_teacher0014_suite_audit.py",
    "tests/test_teacher0014_behavior_audit.py",
    "tests/test_workspace_teacher14_train.py",
    "tests/test_teacher0014_artifact_audit.py",
    "tests/test_teacher0014_replay.py",
)
BENCHMARK_ADMITTED = True
LAUNCH_ADMITTED = False
EXPECTED_SUITE_SHA256 = "6af176d376921caccc0d40641002b94a462b42670fc4794f5b354457d17fa827"


@dataclass(frozen=True)
class Config:
    init_seeds: tuple[int, int] = (84121, 84131)
    train_seeds: tuple[int, int] = (84221, 84231)
    eval_seed: int = 84311
    steps_per_arm: int = 6000
    batch_size: int = 32
    causal_groups: int = 4
    eval_groups: int = 128
    learning_rate: float = 3e-4
    edge_weight: float = .2
    causal_weight: float = .2
    max_seconds: float = 8 * 3600.0
    max_mps_bytes: int = 12 * 1024**3
    max_artifact_bytes: int = 4 * 1024**3
    benchmark_steps: int = 24
    benchmark_warmup_steps: int = 4
    benchmark_max_seconds: float = 1800.0

    def validate(self) -> None:
        if self != Config():
            raise ValueError("Frozen TEACH-0014 configuration changed")


class ResourceStop(RuntimeError):
    """The registered local-resource bound was reached."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def source_provenance(config: Config) -> dict:
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True,
                          timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit TEACH-0014 sources before benchmark: {status}")
    return {"source_git_head": head, "source_worktree_status": status,
            "source_sha256": {path: _sha_file(root / path) for path in SOURCE_PATHS},
            "config_sha256": _canonical_sha(asdict(config))}


def _artifact_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def resource_check(config: Config, start: float, resource: dict,
                   output_dir: Path, result_dir: Path, *, benchmark: bool) -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        used = torch.mps.current_allocated_memory()
        resource["peak_sampled_mps_allocated_bytes"] = max(
            resource["peak_sampled_mps_allocated_bytes"], used)
        if used > config.max_mps_bytes:
            raise ResourceStop("TEACH-0014 sampled MPS allocation exceeded 12 GiB")
    ceiling = config.benchmark_max_seconds if benchmark else config.max_seconds
    if time.monotonic() - start > ceiling:
        raise ResourceStop("TEACH-0014 wall-time ceiling reached")
    if (_artifact_bytes(output_dir) + _artifact_bytes(result_dir)
            > config.max_artifact_bytes):
        raise ResourceStop("TEACH-0014 artifact ceiling exceeded 4 GiB")


def _set_mps_cap(config: Config) -> tuple[float, int]:
    recommended = torch.mps.recommended_max_memory()
    if recommended <= 0:
        raise RuntimeError("Mac GPU memory recommendation unavailable")
    fraction = min(.40, config.max_mps_bytes / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    return fraction, recommended


def new_model(config: Config, arm: str, replicate: int,
              device: str) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    if arm not in ARMS or replicate not in (0, 1):
        raise ValueError("Unknown arm or replicate")
    torch.manual_seed(config.init_seeds[replicate])
    if arm == "raw_dense_matched":
        model = DenseEpisodeClassifier()
    else:
        model = CandidateEdgeWorkspace(
            oracle_rows=arm == "oracle_rows_workspace",
            one_read=arm == "latent_rows_one_read",
            mean_address=arm == "latent_rows_mean_address",
            refinement=("recurrent4" if arm == "latent_rows_recurrent4" else
                        "diffusion4" if arm == "latent_rows_diffuse" else "standard"))
    model = model.to(device)
    if model.parameter_count != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"{arm}: parameter count drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=.01)
    return model, optimizer


def model_output(model: torch.nn.Module, arm: str, episodes: list,
                 device: str):
    ids = padded_tokens(episodes, device)
    if arm == "oracle_rows_workspace":
        left, right, mask = public_row_tensors(ids)
        return model(ids, row_left=left, row_right=right, row_mask=mask)
    return model(ids)


def one_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               config: Config, replicate: int, arm: str, step: int,
               device: str) -> dict:
    episodes, answers = training_batch(
        config.train_seeds[replicate] + step, config.batch_size, step=step,
        null_composed=arm == "raw_null")
    optimizer.zero_grad(set_to_none=True)
    output = model_output(model, arm, episodes, device)
    components = {"answer": answer_loss(output, answers)}
    causal_input_sha = None
    if arm == "latent_rows_edge_aux":
        components["edge"] = public_edge_loss(output)
    elif arm in ("latent_rows_causal", "latent_rows_wrong_causal"):
        pairs = causal_training_batch(
            config.train_seeds[replicate] ^ 0x14CA50 ^ step,
            config.causal_groups, step=step)
        causal_input_sha = _canonical_sha([
            (pair.donor.render_id, tuple(base.render_id for base in pair.bases))
            for pair in pairs])
        components["causal"], _ = interchange_loss(
            model, pairs, device,
            wrong_targets=arm == "latent_rows_wrong_causal")
    elif arm in ("latent_rows_recurrent4", "latent_rows_diffuse"):
        mask = output.auxiliary["candidate_mask"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train_seeds[replicate] ^ 0x14DE00 ^ step)
        uniforms = torch.rand(mask.shape, generator=generator).to(device)
        components = matched_refinement_objectives(
            model, output, answers, time_step=step % 4 + 1,
            uniform=uniforms)
    loss = components["answer"] + config.edge_weight * sum(
        value for key, value in components.items() if key.startswith("edge"))
    if "causal" in components:
        loss = loss + config.causal_weight * components["causal"]
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"Nonfinite {arm} loss at step {step}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError(f"Nonfinite {arm} gradients at step {step}")
    optimizer.step()
    return {"step": step, "losses": {
        key: float(value.detach().item()) for key, value in components.items()},
        "gradient_norm": float(norm.item()),
        "answer_input_sha256": _canonical_sha(
            [episode.render_id for episode in episodes]),
        "answer_label_sha256": _canonical_sha(answers),
        "causal_input_sha256": causal_input_sha}


def benchmark(config: Config, result_dir: Path, output_dir: Path) -> dict:
    config.validate()
    if not BENCHMARK_ADMITTED:
        raise RuntimeError("TEACH-0014 benchmark admission closed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for the TEACH-0014 benchmark")
    provenance = source_provenance(config)
    path = result_dir / "benchmark.json"
    if path.exists():
        raise FileExistsError("No automatic TEACH-0014 benchmark rerun")
    mps_fraction, recommended_memory = _set_mps_cap(config)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    report = {"experiment": "TEACH-0014", "mode": "benchmark",
              "status": "running", "config": asdict(config),
              "torch_version": torch.__version__, "platform": platform.platform(),
              "machine": platform.machine(), "device": "mps",
              "mps_memory_fraction": mps_fraction,
              "mps_recommended_max_memory": recommended_memory, **provenance,
              **resource}
    _write_json(path, report)
    try:
        measured = {}
        for arm in ARMS:
            model, optimizer = new_model(config, arm, 0, "mps")
            model.train()
            timings = []
            for step in range(config.benchmark_steps):
                before = time.monotonic()
                one_update(model, optimizer, config, 0, arm, step, "mps")
                torch.mps.synchronize()
                if step >= config.benchmark_warmup_steps:
                    timings.append(time.monotonic() - before)
                resource_check(config, start, resource, output_dir, result_dir,
                               benchmark=True)
            measured[arm] = {
                "parameters": model.parameter_count,
                "warmup_steps": config.benchmark_warmup_steps,
                "timed_steps": len(timings),
                "timed_step_seconds": timings,
                "median_timed_step_seconds": statistics.median(timings),
            }
            report.update({"measured": measured, **resource})
            _write_json(path, report)
            del model, optimizer
            torch.mps.synchronize()
            torch.mps.empty_cache()
        projection = sum(row["median_timed_step_seconds"]
                         for row in measured.values()) * config.steps_per_arm * 2
        # Reserve 30 minutes for suite generation, evaluation, checkpoint I/O
        # and audit; the training projection itself receives a 1.5x factor.
        conservative = projection * 1.5 + 1800.0
        report.update({"projected_training_seconds": projection,
                       "conservative_projected_seconds": conservative,
                       "elapsed_seconds": time.monotonic() - start,
                       "status": "pass" if conservative <= config.max_seconds
                       else "stop", **resource})
        if report["status"] == "stop":
            report["reason"] = "Conservative projection exceeds eight-hour cap"
    except Exception as exc:
        report.update({"status": "stop", "reason": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.monotonic() - start, **resource})
        _write_json(path, report)
        raise
    _write_json(path, report)
    return report


def check_benchmark(config: Config, result_dir: Path,
                    provenance: dict) -> dict:
    path = result_dir / "benchmark.json"
    if not path.exists():
        raise RuntimeError("Passing TEACH-0014 benchmark required")
    report = json.loads(path.read_text())
    valid = (report.get("status") == "pass" and
             report.get("config") == json.loads(json.dumps(asdict(config))) and
             report.get("source_git_head") == provenance["source_git_head"] and
             report.get("source_sha256") == provenance["source_sha256"] and
             report.get("config_sha256") == provenance["config_sha256"] and
             report.get("conservative_projected_seconds", math.inf)
             <= config.max_seconds and
             report.get("peak_sampled_mps_allocated_bytes", math.inf)
             <= config.max_mps_bytes)
    if valid:
        measured = report.get("measured", {})
        valid = set(measured) == set(ARMS) and all(
            measured[arm].get("parameters") == EXPECTED_PARAMETERS[arm]
            and measured[arm].get("timed_steps") == 20 for arm in ARMS)
    if not valid:
        raise RuntimeError("TEACH-0014 benchmark did not qualify or source changed")
    return report


def _save_checkpoint(model: torch.nn.Module, config: Config, replicate: int,
                     arm: str, output_dir: Path) -> dict:
    path = output_dir / f"rep{replicate}-{arm}-step{config.steps_per_arm}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model": {key: value.detach().cpu()
                          for key, value in model.state_dict().items()},
                "config": asdict(config), "replicate": replicate, "arm": arm,
                "step": config.steps_per_arm, "parameters": model.parameter_count},
               temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": _sha_file(path),
            "bytes": path.stat().st_size}


@torch.no_grad()
def _predict_panels(model: torch.nn.Module, arm: str, suite: dict,
                    device: str) -> tuple[dict, dict]:
    model.eval()
    panels = {}
    replay = {}
    for name, episodes in suite.items():
        rows = []
        replay_rows = []
        replay_indices = {0, len(episodes) // 2, len(episodes) - 1}
        for offset in range(0, len(episodes), 32):
            chunk = episodes[offset:offset + 32]
            logits = model_output(model, arm, chunk, device).logits
            guesses = (logits[:, SYMBOL_START:].argmax(dim=-1) + SYMBOL_START).tolist()
            rows.extend({"render_id": episode.render_id, "prediction": guess}
                        for episode, guess in zip(chunk, guesses, strict=True))
            for global_index in sorted(replay_indices & set(range(
                    offset, offset + len(chunk)))):
                local_index = global_index - offset
                replay_rows.append({
                    "index": global_index,
                    "render_id": episodes[global_index].render_id,
                    "logits": logits[local_index].float().cpu().tolist(),
                })
        panels[name] = rows
        replay[name] = replay_rows
    return {"panels": panels}, replay


def run(config: Config, result_dir: Path, output_dir: Path) -> dict:
    """Scientific run remains closed until the complete auditor is frozen."""
    config.validate()
    if not LAUNCH_ADMITTED:
        raise RuntimeError("TEACH-0014 launch closed: full outcome auditor pending")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for the TEACH-0014 campaign")
    provenance = source_provenance(config)
    gate = check_benchmark(config, result_dir, provenance)
    report_path = result_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0014 scientific rerun")
    mps_fraction, recommended_memory = _set_mps_cap(config)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    suite = evaluation_suite(config.eval_seed, config.eval_groups)
    manifest = suite_manifest(suite, seed=config.eval_seed,
                              group_count=config.eval_groups)
    from scripts.teacher0014_suite_audit import audit_manifest

    suite_audit = audit_manifest(manifest)
    if suite_audit["manifest_sha256"] != EXPECTED_SUITE_SHA256:
        raise RuntimeError("Frozen TEACH-0014 confirmation suite hash changed")
    manifest_path = result_dir / "suite.json"
    _write_json(manifest_path, manifest)
    status_path = result_dir / "status.json"
    status = {"experiment": "TEACH-0014", "status": "running",
              "config": asdict(config), "device": "mps",
              "suite_sha256": suite_audit["manifest_sha256"],
              "benchmark_sha256": _sha_file(result_dir / "benchmark.json"),
              "benchmark_projection_seconds": gate["conservative_projected_seconds"],
              "mps_memory_fraction": mps_fraction,
              "mps_recommended_max_memory": recommended_memory,
              "torch_version": torch.__version__, "platform": platform.platform(),
              **provenance, **resource}
    _write_json(status_path, status)
    predictions = {"experiment": "TEACH-0014",
                   "manifest_sha256": suite_audit["manifest_sha256"],
                   "runs": {}}
    replay_logits = {"experiment": "TEACH-0014",
                     "manifest_sha256": suite_audit["manifest_sha256"],
                     "runs": {}}
    training = {}
    try:
        for arm in ARMS:
            predictions["runs"][arm] = {}
            replay_logits["runs"][arm] = {}
            training[arm] = {}
            for replicate in (0, 1):
                model, optimizer = new_model(config, arm, replicate, "mps")
                model.train()
                losses = []
                for step in range(config.steps_per_arm):
                    parts = one_update(model, optimizer, config, replicate, arm,
                                       step, "mps")
                    losses.append(parts)
                    if (step + 1) % 250 == 0:
                        status["progress"] = {"arm": arm, "replicate": replicate,
                                              "step": step + 1,
                                              "latest_losses": parts["losses"],
                                              "elapsed_seconds": time.monotonic() - start}
                        resource_check(config, start, resource, output_dir,
                                       result_dir,
                                       benchmark=False)
                        status.update(resource)
                        _write_json(status_path, status)
                checkpoint = _save_checkpoint(model, config, replicate, arm,
                                              output_dir)
                loss_path = result_dir / f"losses-rep{replicate}-{arm}.json.gz"
                raw = json.dumps(losses, separators=(",", ":"),
                                 allow_nan=False).encode()
                loss_path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
                predicted, sampled_logits = _predict_panels(
                    model, arm, suite, "mps")
                predictions["runs"][arm][str(replicate)] = predicted
                replay_logits["runs"][arm][str(replicate)] = sampled_logits
                training[arm][str(replicate)] = {
                    "final_checkpoint": checkpoint,
                    "loss_archive_sha256": _sha_file(loss_path),
                    "loss_count": len(losses), "parameters": model.parameter_count}
                del model, optimizer
                torch.mps.synchronize()
                torch.mps.empty_cache()
                resource_check(config, start, resource, output_dir, result_dir,
                               benchmark=False)
        predictions_path = result_dir / "predictions.json.gz"
        raw = json.dumps(predictions, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        predictions_path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        replay_path = result_dir / "replay-logits.json.gz"
        replay_raw = json.dumps(replay_logits, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode()
        replay_path.write_bytes(gzip.compress(replay_raw, compresslevel=9, mtime=0))
        from scripts.teacher0014_behavior_audit import audit_behavior

        behavior = audit_behavior(manifest, predictions)
        behavior_path = result_dir / "behavior-audit.json"
        _write_json(behavior_path, behavior)
        resource_check(config, start, resource, output_dir, result_dir,
                       benchmark=False)
        report = {**status, "status": "complete", "training": training,
                  "predictions_sha256": _sha_file(predictions_path),
                  "replay_logits_sha256": _sha_file(replay_path),
                  "behavior_audit_sha256": _sha_file(behavior_path),
                  "manifest_file_sha256": _sha_file(manifest_path),
                  "elapsed_seconds": time.monotonic() - start,
                  "artifact_bytes": (_artifact_bytes(output_dir)
                                     + _artifact_bytes(result_dir)), **resource}
        _write_json(report_path, report)
        _write_json(status_path, {**status, "status": "complete", **resource})
        return report
    except Exception as exc:
        _write_json(status_path, {**status, "status": "stopped",
                                  "reason": f"{type(exc).__name__}: {exc}",
                                  "elapsed_seconds": time.monotonic() - start,
                                  **resource})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/TEACH-0014"))
    args = parser.parse_args()
    if args.mode == "benchmark":
        result = benchmark(Config(), args.result_dir, args.output_dir)
    else:
        result = run(Config(), args.result_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
