"""Bounded local compositional teacher and matched independent-label null."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer
from voynich.workspace.teacher1_tasks import (
    OBJECTS, SEQUENCE_LENGTH, VOCAB_SIZE, evaluation_suite, symbolic_oracle, training_batch,
)


@dataclass(frozen=True)
class TeacherConfig:
    seed: int = 61101
    train_seed: int = 61102
    eval_seed: int = 61103
    steps_per_arm: int = 2500
    batch_size: int = 128
    eval_size: int = 256
    learning_rate: float = 3e-4
    max_seconds: float = 3600.0
    max_mps_bytes: int = 8 * 1024**3

    def validate(self) -> None:
        if self.steps_per_arm <= 0 or self.batch_size <= 0 or self.eval_size <= 0:
            raise ValueError("Positive steps, batch size and eval size required")
        if self.eval_size % 2:
            raise ValueError("Eval size must be even for factorial bundles")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid learning rate")
        if not 0 < self.max_seconds <= 3600 or not 0 < self.max_mps_bytes <= 8 * 1024**3:
            raise ValueError("Time and MPS bounds exceed the registered ceiling")


class BudgetExceeded(RuntimeError):
    pass


def _model() -> VoynichTransformer:
    return VoynichTransformer(ModelConfig(
        vocab_size=VOCAB_SIZE, pad_id=0, d_model=128, n_layers=4, n_heads=4,
        d_ff=256, context_length=SEQUENCE_LENGTH, dropout=0.0,
    ))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(config: TeacherConfig) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    registration = repo_root / "docs/experiments/TEACH-0001.md"
    command = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
        text=True, check=True, timeout=5,
    )
    source_paths = [
        "docs/experiments/TEACH-0001.md",
        "src/voynich/workspace/teacher1_tasks.py",
        "src/voynich/workspace/teacher1_train.py",
        "src/voynich/model.py",
    ]
    source_status = subprocess.run(
        ["git", "status", "--porcelain", "--", *source_paths], cwd=repo_root,
        capture_output=True, text=True, check=True, timeout=5,
    )
    serialized_config = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()
    return {
        "source_git_head": command.stdout.strip(),
        "source_worktree_status": source_status.stdout.splitlines(),
        "registration_sha256": _hash_file(registration),
        "config_sha256": hashlib.sha256(serialized_config).hexdigest(),
        "source_sha256": {
            "tasks": _hash_file(Path(__file__).with_name("teacher1_tasks.py")),
            "trainer": _hash_file(Path(__file__)),
            "model": _hash_file(Path(__file__).parents[1] / "model.py"),
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _record_progress(status: dict, status_path: Path, resources: dict[str, int],
                     arm: str, step: int, elapsed: float, latest_loss: float,
                     recent_loss: float, peak_mps: int) -> None:
    status["progress"] = {
        "arm": arm, "step": step, "elapsed_seconds": elapsed,
        "latest_loss": latest_loss, "last_100_loss": recent_loss,
        "peak_sampled_mps_allocated_bytes": peak_mps,
    }
    status.update(resources)
    _write_json(status_path, status)


def _check_budget(config: TeacherConfig, start: float, device: str,
                  resources: dict[str, int]) -> None:
    if device == "mps":
        torch.mps.synchronize()
        allocated = torch.mps.current_allocated_memory()
        resources["peak_sampled_mps_allocated_bytes"] = max(
            resources["peak_sampled_mps_allocated_bytes"], allocated)
        if allocated > config.max_mps_bytes:
            raise BudgetExceeded("MPS allocation exceeded registered 8 GiB ceiling")
    if time.monotonic() - start > config.max_seconds:
        raise BudgetExceeded("Registered 3600-second campaign time ceiling exceeded")


def numerical_qualification(device: str) -> dict:
    """Check no-hook/cached logits and one finite gradient on the actual device."""
    torch.manual_seed(61999)
    model = _model().to(device)
    episodes, labels = training_batch(61998, 4)
    tokens = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    targets = torch.tensor([x - OBJECTS[0] for x in labels], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        clean = model(tokens).logits[:, -1, :]
        hooked = model(tokens, cache_names=["blocks.1.resid_post"]).logits[:, -1, :]
        error = float((clean - hooked).abs().max().item())
    model.train()
    logits = model(tokens).logits[:, -1, OBJECTS[0]:OBJECTS[-1] + 1]
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item())
                           for p in model.parameters())
    if not math.isfinite(loss.item()) or not gradients_finite or error > 0.002:
        raise RuntimeError(f"Numerical qualification failed: loss={loss.item()}, error={error}")
    del model
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return {"max_cached_logit_error": error, "finite_loss": True, "finite_gradients": True}


def _batch(seed: int, batch_size: int, null_labels: bool, device: str
           ) -> tuple[torch.Tensor, torch.Tensor]:
    episodes, labels = training_batch(seed, batch_size, null_labels=null_labels)
    tokens = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    targets = torch.tensor([x - OBJECTS[0] for x in labels], dtype=torch.long, device=device)
    return tokens, targets


def _train_arm(config: TeacherConfig, *, null_labels: bool, device: str,
               start: float, output_dir: Path,
               resources: dict[str, int], progress_callback) -> tuple[VoynichTransformer, dict]:
    torch.manual_seed(config.seed)
    model = _model().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    losses = []
    checkpoints = []
    for step in range(config.steps_per_arm):
        _check_budget(config, start, device, resources)
        inputs, targets = _batch(config.train_seed + step, config.batch_size, null_labels, device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(inputs).logits[:, -1, OBJECTS[0]:OBJECTS[-1] + 1]
        loss = F.cross_entropy(scores, targets)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        _check_budget(config, start, device, resources)
        losses.append(float(loss.item()))
        if (step + 1) % 250 == 0 or step + 1 == config.steps_per_arm:
            recent_loss = sum(losses[-100:]) / min(100, len(losses))
            checkpoints.append({"step": step + 1, "last_100_loss": recent_loss})
            progress_callback(
                "null" if null_labels else "primary", step + 1,
                time.monotonic() - start, losses[-1], recent_loss,
                resources["peak_sampled_mps_allocated_bytes"],
            )
    arm = "null" if null_labels else "primary"
    checkpoint = output_dir / f"{arm}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(config), "null_labels": null_labels}, temporary)
    temporary.replace(checkpoint)
    return model, {"steps": config.steps_per_arm, "history": checkpoints,
                   "checkpoint_sha256": _hash_file(checkpoint), "parameter_count": model.parameter_count}


@torch.no_grad()
def _score(model: VoynichTransformer, episodes: list, device: str) -> dict:
    model.eval()
    correct = 0
    predictions = []
    for start in range(0, len(episodes), 256):
        chunk = episodes[start:start + 256]
        inputs = torch.tensor([ep.tokens for ep in chunk], dtype=torch.long, device=device)
        scores = model(inputs).logits[:, -1, OBJECTS[0]:OBJECTS[-1] + 1]
        predictions.extend((scores.argmax(-1) + OBJECTS[0]).cpu().tolist())
    for ep, predicted in zip(episodes, predictions, strict=True):
        correct += int(predicted == ep.answer)
    return {"correct": correct, "total": len(episodes), "accuracy": correct / len(episodes),
            "wilson_95": wilson_95(correct, len(episodes)),
            "predictions": predictions}


def wilson_95(correct: int, total: int) -> list[float]:
    if not 0 <= correct <= total or total <= 0:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    lower = 0.0 if correct == 0 else max(0.0, center - radius)
    upper = 1.0 if correct == total else min(1.0, center + radius)
    return [lower, upper]


def evaluate(model: VoynichTransformer, suite: dict[str, list], device: str) -> dict:
    result = {}
    for name, episodes in suite.items():
        raw = _score(model, episodes, device)
        if name == "factorial":
            fours = [raw["predictions"][i:i + 4] for i in range(0, len(episodes), 4)]
            truths = [[ep.answer for ep in episodes[i:i + 4]] for i in range(0, len(episodes), 4)]
            exact = sum(pred == expected for pred, expected in zip(fours, truths, strict=True))
            raw["exact_quartets"] = exact
            raw["total_quartets"] = len(fours)
            raw["quartet_accuracy"] = exact / len(fours)
            raw["quartet_wilson_95"] = wilson_95(exact, len(fours))
        del raw["predictions"]
        result[name] = raw
    return result


def decide(primary: dict, null: dict) -> dict:
    p = primary
    n = null
    clauses = {
        "both_heldout_composed_at_least_0.90": p["composed_holdout_holdout"]["accuracy"] >= 0.90,
        "crossed_heldout_composed_each_at_least_0.90": all(
            p[name]["accuracy"] >= 0.90 for name in
            ("composed_holdout_train", "composed_train_holdout")),
        "factorial_exact_quartets_at_least_0.70": p["factorial"]["quartet_accuracy"] >= 0.70,
        "direct_and_copy_each_at_least_0.95": all(p[name]["accuracy"] >= 0.95
                                                   for name in ("direct", "copy")),
        "null_both_heldout_at_most_0.25": n["composed_holdout_holdout"]["accuracy"] <= 0.25,
        "primary_minus_null_at_least_0.60": (
            p["composed_holdout_holdout"]["accuracy"] -
            n["composed_holdout_holdout"]["accuracy"] >= 0.60),
    }
    return {"verdict": "qualified" if all(clauses.values()) else "not_qualified", "clauses": clauses}


def run(config: TeacherConfig, output_dir: Path, result_dir: Path, device: str) -> dict:
    config.validate()
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in ("mps", "cpu"):
        raise ValueError("TEACH-0001 supports local CPU or MPS only")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.16)
    start = time.monotonic()
    resources = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0001", "status": "running", "config": asdict(config),
              "device": device, "torch_version": torch.__version__, **_provenance(config), **resources}
    status_path = result_dir / "status.json"
    _write_json(status_path, status)

    def record_progress(arm: str, step: int, elapsed: float, latest_loss: float,
                        recent_loss: float, peak_mps: int) -> None:
        _record_progress(status, status_path, resources, arm, step, elapsed,
                         latest_loss, recent_loss, peak_mps)

    try:
        qualification = numerical_qualification(device)
        _check_budget(config, start, device, resources)
        suite = evaluation_suite(config.eval_seed, config.eval_size)
        if any(symbolic_oracle(ep.tokens) != ep.answer for rows in suite.values() for ep in rows):
            raise AssertionError("Evaluation generator/oracle mismatch")
        suite_hash = hashlib.sha256(json.dumps(
            {k: [(ep.tokens, ep.answer) for ep in v] for k, v in suite.items()},
            sort_keys=True).encode()).hexdigest()
        status.update({"qualification": qualification, "eval_suite_sha256": suite_hash})
        arm_reports = {}
        scores = {}
        for arm, null_labels in (("primary", False), ("null", True)):
            model, arm_reports[arm] = _train_arm(
                config, null_labels=null_labels, device=device, start=start,
                output_dir=output_dir, resources=resources, progress_callback=record_progress)
            _check_budget(config, start, device, resources)
            scores[arm] = evaluate(model, suite, device)
            _check_budget(config, start, device, resources)
            del model
            if device == "mps":
                torch.mps.empty_cache()
            status["completed_arms"] = list(arm_reports)
            status.update(resources)
            _write_json(status_path, status)
        decision = decide(scores["primary"], scores["null"])
        status.update({"status": "complete", "elapsed_seconds": time.monotonic() - start,
                       "arms": arm_reports, "scores": scores, "decision": decision,
                       **resources})
        _write_json(result_dir / "report.json", status)
        _write_json(status_path, {"experiment": "TEACH-0001", "status": "complete",
                                  "decision": decision, "report": "report.json",
                                  "source_git_head": status["source_git_head"],
                                  "registration_sha256": status["registration_sha256"],
                                  "config_sha256": status["config_sha256"], **resources})
        return status
    except (BudgetExceeded, RuntimeError, AssertionError) as exc:
        status.update({"status": "stopped", "reason": type(exc).__name__ + ": " + str(exc),
                       "elapsed_seconds": time.monotonic() - start, **resources})
        _write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/TEACH-0001"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0001"))
    args = parser.parse_args()
    report = run(TeacherConfig(), args.output_dir, args.result_dir, args.device)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
