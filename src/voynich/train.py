"""Bounded from-scratch training; validation chooses checkpoints, never test."""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import torch

from .model import ModelConfig, VoynichTransformer
from .runtime import PageWindows, corpus_identity, environment, evaluate_model, loss_sum, resolve_device, write_json
from .tokenizer import EVATokenizer


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = VoynichTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device), payload


def train(config_path, data_dir, run_dir, *, device="auto", steps=None, seed=None, resume=None, threads=4,
          preparation_manifest=None):
    torch.set_num_threads(threads)
    config = json.loads(Path(config_path).read_text())
    cfg = copy.deepcopy(config["training"])
    if steps is not None:
        cfg["steps"] = steps
    if seed is not None:
        cfg["seed"] = seed
    if min(cfg["steps"], cfg["batch_size"], cfg["eval_interval"]) < 1:
        raise ValueError("steps, batch size and evaluation interval must be positive")
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("invalid optimizer configuration")
    if cfg["warmup_steps"] < 0 or cfg["auxiliary_weight"] < 0 or cfg["patience"] < 1:
        raise ValueError("invalid schedule or auxiliary objective")
    device = resolve_device(device)
    torch.manual_seed(cfg["seed"])
    sampler = torch.Generator().manual_seed(cfg["seed"] + 1)
    tokenizer = EVATokenizer.load(Path(data_dir) / "tokenizer.json")
    model_cfg = ModelConfig(**{**config["model"], "vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id})
    model = VoynichTransformer(model_cfg).to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["learning_rate"], betas=(0.9, 0.95))
    identity = corpus_identity(data_dir, preparation_manifest)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Use a new run directory; artifacts are never silently overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data = PageWindows(data_dir, "train", model_cfg.context_length, tokenizer)
    validation = PageWindows(data_dir, "validation", model_cfg.context_length, tokenizer)
    horizons = (1, *model_cfg.auxiliary_horizons)
    start_step, best, stale = 0, float("inf"), 0
    best_state = None
    if resume:
        _, payload = load_checkpoint(resume)
        if payload["model_config"] != model_cfg.to_dict() or payload["corpus_identity"] != identity:
            raise ValueError("Resume requires identical model, tokenizer and corpus/splits")
        if payload["training_config"] != cfg:
            raise ValueError("Resume requires identical training schedule/configuration")
        if payload.get("device") != device:
            raise ValueError("Resume requires the same device family for RNG continuity")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step, best, stale = payload["step"], payload["best_validation_nll"], payload["stale_evaluations"]
        best_state = payload.get("best_checkpoint")
        if best_state is None:
            if payload.get("is_best_checkpoint"):
                best_state = copy.deepcopy(payload)
            else:
                raise ValueError("Checkpoint lacks a recoverable best state; use a checkpoint from this implementation")
        sampler.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if device == "mps" and "mps_rng" in payload:
            torch.mps.set_rng_state(payload["mps_rng"])
        if device == "cuda" and "cuda_rng" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        if start_step >= cfg["steps"]:
            raise ValueError("Checkpoint already completed this schedule")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "model_config": model_cfg.to_dict(), "training_config": cfg, "corpus_identity": identity,
        "environment": environment(), "device": device, "parameter_count": model.parameter_count,
        "resume_from": str(resume) if resume else None,
        "training_data": "Voynich train split only; random initialization; no pretrained weights",
        "selection_split": "validation", "test_evaluated": False,
        "sampler": "uniform page-window sampling with replacement; final short windows right padded",
    }
    write_json(run_dir / "manifest.json", metadata)
    tokenizer.save(run_dir / "tokenizer.json")
    started = time.monotonic()
    initial = evaluate_model(model, validation, device, cfg["batch_size"])
    history = [{"step": start_step, "validation": initial, "elapsed_seconds": 0.0}]
    def snapshot(step):
        state = {
            "model": model.state_dict(), "model_config": model_cfg.to_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "training_config": cfg,
            "corpus_identity": identity, "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "best_validation_nll": best, "stale_evaluations": stale, "device": device,
            "is_best_checkpoint": False,
        }
        if device == "mps":
            state["mps_rng"] = torch.mps.get_rng_state()
        if device == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        return state
    def save_state(path, state):
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)
    def checkpoint(path, step):
        save_state(path, {**snapshot(step), "best_checkpoint": best_state})
    if not resume:
        best = initial["nll_nats"]
        best_state = copy.deepcopy(snapshot(start_step))
        best_state["is_best_checkpoint"] = True
    save_state(run_dir / "best.pt", best_state)
    checkpoint(run_dir / "initial.pt", start_step)
    stop_reason = "step_budget"
    tokens_seen = 0
    for step in range(start_step + 1, cfg["steps"] + 1):
        model.train()
        indices = torch.randint(len(train_data.windows), (cfg["batch_size"],), generator=sampler).tolist()
        x, targets, _ = train_data.batch(indices, device, horizons)
        output = model(x)
        total, count = loss_sum(output.logits, targets[1])
        if int(count) == 0:
            raise ValueError("Training batch contains no scorable targets")
        loss = total / count
        auxiliary_loss = []
        for h, logits in output.auxiliary_logits.items():
            aux_total, aux_count = loss_sum(logits, targets[h])
            if int(aux_count):
                auxiliary_loss.append(aux_total / aux_count)
        if auxiliary_loss:
            loss = loss + cfg["auxiliary_weight"] * torch.stack(auxiliary_loss).mean()
        warmup = min(cfg["warmup_steps"], cfg["steps"])
        if warmup and step <= warmup:
            factor = step / warmup
        else:
            progress = (step - warmup) / max(cfg["steps"] - warmup, 1)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * factor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        tokens_seen += int(count)
        if step % cfg["eval_interval"] == 0 or step == cfg["steps"]:
            measured = evaluate_model(model, validation, device, cfg["batch_size"])
            item = {
                "step": step, "train_batch_loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.cpu()), "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": measured, "elapsed_seconds": time.monotonic() - started,
                "scored_training_tokens_this_invocation": tokens_seen,
            }
            history.append(item)
            print(json.dumps({"step": step, "validation_bits_per_token": measured["bits_per_token"],
                              "elapsed_seconds": item["elapsed_seconds"]}), flush=True)
            if measured["nll_nats"] < best:
                best, stale = measured["nll_nats"], 0
                best_state = copy.deepcopy(snapshot(step))
                best_state["is_best_checkpoint"] = True
                save_state(run_dir / "best.pt", best_state)
            else:
                stale += 1
            checkpoint(run_dir / "last.pt", step)
            write_json(run_dir / "history.json", history)
            if stale >= cfg["patience"]:
                stop_reason = "validation_patience"
                break
    summary = {
        "completed_steps": step, "stop_reason": stop_reason, "best_validation_bits_per_token": best / math.log(2),
        "initial_validation_bits_per_token": initial["bits_per_token"],
        "elapsed_seconds": time.monotonic() - started, "parameter_count": model.parameter_count,
        "best_step": best_state["step"],
        "test_evaluated": False, "scored_training_tokens_this_invocation": tokens_seen,
        "status": "predictive run; not evidence of decipherment",
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reference.json")
    parser.add_argument("--data", default="data/processed/zl3b")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--preparation-manifest", help="Optional explicit registered corpus manifest")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.data, args.run_dir, device=args.device,
                           steps=args.steps, seed=args.seed, resume=args.resume, threads=args.threads,
                           preparation_manifest=args.preparation_manifest), indent=2))


if __name__ == "__main__":
    main()
