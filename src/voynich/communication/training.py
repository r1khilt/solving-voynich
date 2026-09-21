"""Finite, resumable joint-inference training with isolated synthetic system holdouts."""

from dataclasses import asdict, dataclass
import json
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from ..runtime import resolve_device
from .denoiser import ConditionalDenoiser, DenoiserConfig, masked_loss
from .pipeline import HELD_OUT, collate, episode_identity, make_episode
from .schema import JointLayout, object_digest, save_json


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2000
    batch_size: int = 16
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_steps: int = 100
    eval_interval: int = 100
    validation_examples: int = 64
    seed: int = 41021
    max_seconds: float = 1200.0
    threads: int = 4
    anchor_count: int = 2
    gradient_clip: float = 1.0

    def __post_init__(self):
        for name in ("steps", "batch_size", "eval_interval", "validation_examples", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "seed", "anchor_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "max_seconds", "gradient_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.weight_decay) not in (int, float)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid weight decay")
        if self.steps * self.batch_size >= 1_000_000:
            raise ValueError(
                "This registered episode domain supports fewer than one million examples per run"
            )


def source_manifest():
    base = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.glob("*.py"))}


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "git_commit": commit,
        "source_sha256": source_manifest(),
    }


def read_configuration(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"schema_version", "layout", "model", "training"} or payload["schema_version"] != 1:
        raise ValueError("Unsupported training configuration")
    layout = JointLayout(**payload["layout"])
    model = DenoiserConfig(
        vocab_size=layout.vocab_size,
        condition_vocab_size=layout.condition_vocab_size,
        max_latent_length=layout.latent_length,
        max_condition_length=layout.condition_length,
        **payload["model"],
    )
    if model.type_vocab_size < 6 or layout.alphabet_size < 19:
        raise ValueError("Generated-world training requires at least 6 latent types and 19 alphabet slots")
    training = TrainingConfig(**payload["training"])
    if training.anchor_count > layout.max_anchors:
        raise ValueError("Anchor capacity insufficient")
    return layout, model, training


def _snapshot(model, optimizer, layout, config, *, step, history, sampler, best, device, source):
    record = {
        "schema_version": 1,
        "model_config": model.config.to_dict(),
        "layout": asdict(layout),
        "training_config": asdict(config),
        "step": step,
        "history": history,
        "best_validation_loss": best,
        "device": device,
        "source_sha256": source,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if device == "mps":
        record["mps_rng"] = torch.mps.get_rng_state().cpu()
    elif device == "cuda":
        record["cuda_rng"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return record


def _save_checkpoint(path, record):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(record, temp)
    temp.replace(path)


def load_checkpoint(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    layout = JointLayout(**record["layout"])
    model = ConditionalDenoiser(DenoiserConfig(**record["model_config"]))
    if (
        model.config.vocab_size,
        model.config.max_latent_length,
        model.config.condition_vocab_size,
        model.config.max_condition_length,
    ) != (layout.vocab_size, layout.latent_length, layout.condition_vocab_size, layout.condition_length):
        raise ValueError("Checkpoint model/layout mismatch")
    model.load_state_dict(record["model"], strict=True)
    return model.to(device), layout, record


@torch.no_grad()
def validation_loss(
    model, layout, config, device, *, anchor_count=None, split="validation", allow_test=False
):
    if split == "test" and not allow_test:
        raise ValueError("Final synthetic holdout requires an explicit frozen evaluation")
    if split not in {"validation", "test"}:
        raise ValueError("Invalid validation split")
    anchors = config.anchor_count if anchor_count is None else anchor_count
    rng = torch.Generator().manual_seed(config.seed + 10007)
    was_training = model.training
    model.eval()
    losses, counts = 0.0, 0
    identities = []
    try:
        for start in range(0, config.validation_examples, config.batch_size):
            episodes = [
                make_episode(i, split, layout, anchor_count=anchors, seed=config.seed)
                for i in range(start, min(start + config.batch_size, config.validation_examples))
            ]
            batch = collate(episodes, layout, device)
            loss, metrics = masked_loss(
                model,
                batch["clean"],
                batch["condition"],
                generator=rng,
                latent_types=batch["latent_types"],
                allowed=batch["allowed"],
            )
            n = metrics["masked_tokens"]
            losses += float(loss) * n
            counts += n
            identities.extend(episode_identity(e) for e in episodes)
    finally:
        model.train(was_training)
    return {
        "loss": losses / counts,
        "masked_tokens": counts,
        "episodes": len(identities),
        "input_sha256": object_digest(identities),
        "anchor_count": anchors,
        "split": split,
        "held_out_combination": list(HELD_OUT[split]),
    }


def train(config_path, run_dir, *, device="auto", resume=None, stop_after=None):
    layout, model_config, cfg = read_configuration(config_path)
    device = resolve_device(device)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    if device == "mps":
        torch.mps.manual_seed(cfg.seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = ConditionalDenoiser(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sampler = torch.Generator().manual_seed(cfg.seed + 1)
    source = source_manifest()
    step, best, history, initial = 0, math.inf, [], None
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()) and resume is None:
        raise ValueError("Use a new output directory; no existing run is overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    if resume:
        restored, old_layout, record = load_checkpoint(resume, device)
        if (
            old_layout != layout
            or record["model_config"] != model_config.to_dict()
            or record["training_config"] != asdict(cfg)
            or record["device"] != device
            or record["source_sha256"] != source
        ):
            raise ValueError("Resume requires identical configuration, device, and source bytes")
        model.load_state_dict(restored.state_dict())
        del restored
        optimizer.load_state_dict(record["optimizer"])
        sampler.set_state(record["sampler_rng"])
        torch.set_rng_state(record["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(record["mps_rng"])
        elif device == "cuda":
            torch.cuda.set_rng_state_all(record["cuda_rng"])
        step, best, history = record["step"], record["best_validation_loss"], record["history"]
        initial = record.get("initial_validation")
    if step >= cfg.steps:
        raise ValueError("Checkpoint already completed its schedule")
    if stop_after is not None and (
        type(stop_after) is not int or stop_after <= step or stop_after > cfg.steps
    ):
        raise ValueError("stop_after must exceed current step without extending the registered budget")
    manifest = {
        "schema_version": 1,
        "configuration": json.loads(Path(config_path).read_text()),
        "environment": environment(),
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "supervision": "synthetic key, mask, boundary, grammar, morphology, family; optional oracle anchors",
        "data_policy": "fresh generated systems; held-out grammar/morph combinations; no manuscript data",
        "training_start_step": step,
        "resume_from": str(resume) if resume else None,
        "bounds": {"max_steps": cfg.steps, "max_seconds_this_invocation": cfg.max_seconds},
    }
    save_json(destination / "manifest.json", manifest)
    start = time.monotonic()
    if initial is None:
        initial = validation_loss(model, layout, cfg, device)
    stop_reason = "completed"
    while step < cfg.steps:
        if time.monotonic() - start >= cfg.max_seconds:
            stop_reason = "time_budget"
            break
        model.train()
        episodes = [
            make_episode(
                step * cfg.batch_size + i,
                "train",
                layout,
                anchor_count=(cfg.anchor_count if (step * cfg.batch_size + i) % 2 else 0),
                seed=cfg.seed,
            )
            for i in range(cfg.batch_size)
        ]
        batch = collate(episodes, layout, device)
        progress = (step + 1) / cfg.steps
        warmup = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        lr = cfg.learning_rate * warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = masked_loss(
            model,
            batch["clean"],
            batch["condition"],
            generator=sampler,
            latent_types=batch["latent_types"],
            allowed=batch["allowed"],
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss; training stopped")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        evaluate = step % cfg.eval_interval == 0 or step == cfg.steps or step == stop_after
        if evaluate:
            validation = validation_loss(model, layout, cfg, device)
            row = {
                "step": step,
                "train_loss": metrics["loss"],
                "validation": validation,
                "gradient_norm": float(norm),
                "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - start,
            }
            history.append(row)
            improved = validation["loss"] < best
            best = min(best, validation["loss"])
            snapshot = _snapshot(
                model,
                optimizer,
                layout,
                cfg,
                step=step,
                history=history,
                sampler=sampler,
                best=best,
                device=device,
                source=source,
            )
            snapshot["initial_validation"] = initial
            _save_checkpoint(destination / "last.pt", snapshot)
            if improved:
                _save_checkpoint(destination / "best.pt", snapshot)
            save_json(destination / "history.json", history)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": row["train_loss"],
                        "validation_loss": validation["loss"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
        if step == stop_after:
            stop_reason = "explicit_checkpoint_stop"
            break
    snapshot = _snapshot(
        model,
        optimizer,
        layout,
        cfg,
        step=step,
        history=history,
        sampler=sampler,
        best=best,
        device=device,
        source=source,
    )
    snapshot["initial_validation"] = initial
    _save_checkpoint(destination / "last.pt", snapshot)
    summary = {
        "schema_version": 1,
        "status": stop_reason,
        "steps": step,
        "parameters": manifest["parameters"],
        "initial_validation": initial,
        "best_validation_loss": None if math.isinf(best) else best,
        "elapsed_seconds": time.monotonic() - start,
        "fresh_training_episodes": step * cfg.batch_size,
        "claim": "synthetic joint-inference training only; no decipherment claim",
    }
    save_json(destination / "summary.json", summary)
    save_json(destination / "history.json", history)
    return summary
