"""Matched canonical Transformer fits with posterior training targets for EXP-0035."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from voynich.exact_count_decode import exact_count_keep_masks
from voynich.exp0032_model import (
    BATCH_SIZE,
    EVAL_INTERVAL,
    LEARNING_RATE,
    SEEDS,
    STEPS,
    ContextMaskModel,
    arrays,
    null_f1,
    probabilities,
)
from voynich.latent_recovery import WORLD_C
from voynich.runtime import write_json


EXPERIMENT = "EXP-0035"
VARIANTS = ("uniform", "weighted")
MAX_TRAIN_SECONDS = 600
PROBE_STEPS = 12
DATA_MANIFEST_SHA = "ca8e3e6c7dfab5df74f750b6434b9905629f2f629a9c886e8f780935e406ec44"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_weighted_bce(logits: torch.Tensor, keep_probability: torch.Tensor) -> torch.Tensor:
    """Expectation of EXP-0032's positive-keep/2x-negative-null BCE."""
    if logits.shape != keep_probability.shape:
        raise ValueError("soft-target/logit shape mismatch")
    return (keep_probability * F.softplus(-logits) + 2.0 * (1.0 - keep_probability) * F.softplus(logits)).mean()


def load_data(root: Path, variant: str, device: str) -> tuple[list[dict], list[dict], dict, torch.Tensor]:
    manifest_path = root / "data/manifests/exp0035_data.json"
    audit_path = root / "results/EXP-0035/data_audit.json"
    if digest(manifest_path) != DATA_MANIFEST_SHA:
        raise ValueError("EXP-0035 data manifest changed")
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["manifest_sha256"] != digest(manifest_path):
        raise ValueError("independent EXP-0035 data audit missing or mismatched")
    prior = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    paths = {name: root / f"data/processed/exp0032/{name}.jsonl" for name in ("train", "validation")}
    for name, path in paths.items():
        if digest(path) != prior["derived_sha256"][name]:
            raise ValueError(f"original {name} data changed")
    target_path = root / "data/processed/exp0035/train_soft_targets.npz"
    if digest(target_path) != manifest["derived_sha256"]["train_soft_targets"]:
        raise ValueError("training soft target checksum changed")
    with np.load(target_path) as archive:
        target = archive[variant].copy()
    if target.shape != (12000, 128) or target.dtype != np.float32:
        raise ValueError("soft-target array shape/dtype changed")
    train_rows = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    val_rows = [json.loads(line) for line in paths["validation"].read_text().splitlines()]
    data = arrays(train_rows, "canonical", device)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    return train_rows, val_rows, data, target_tensor


def selected_batch(rng: np.random.Generator, c_ids: np.ndarray, other_ids: np.ndarray) -> np.ndarray:
    selected = np.concatenate((rng.choice(c_ids, size=22, replace=True), rng.choice(other_ids, size=10, replace=True)))
    rng.shuffle(selected)
    if len(selected) != BATCH_SIZE:
        raise AssertionError("batch size drift")
    return selected


def train(root: Path, variant: str, seed: int, device: str, probe: bool = False) -> dict:
    if variant not in VARIANTS or seed not in SEEDS or device not in ("cpu", "mps"):
        raise ValueError("unregistered variant, seed or device")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(6)
    output = root / f"outputs/EXP-0035/{variant}_{seed}.pt"
    report_path = root / f"results/EXP-0035/train_{variant}_{seed}.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("training output already exists; refusing overwrite")
    train_rows, val_rows, train_data, target = load_data(root, variant, device)
    val_copy_rows = [row for row in val_rows if row["world"] == WORLD_C and row["filler_family"] == "copy_mutate"]
    if len(val_copy_rows) < 200:
        raise ValueError("too few fixed validation copy rows")
    val_data = arrays(val_copy_rows, "canonical", device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContextMaskModel().to(device)
    if sum(p.numel() for p in model.parameters()) != 10_746_629:
        raise AssertionError("baseline model capacity changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    c_ids = np.asarray([i for i, row in enumerate(train_rows) if row["world"] == WORLD_C])
    other_ids = np.asarray([i for i, row in enumerate(train_rows) if row["world"] != WORLD_C])
    if len(c_ids) < 6000 or len(other_ids) < 3000:
        raise ValueError("training world population drift")
    best_f1, best_step, best_state = -1.0, 0, None
    history: list[dict] = []
    start = time.monotonic()
    end_step = PROBE_STEPS if probe else STEPS
    for step in range(1, end_step + 1):
        selected = selected_batch(rng, c_ids, other_ids)
        idx = torch.tensor(selected, dtype=torch.long, device=device)
        logits, world_logits = model(train_data["x"][idx], train_data["features"][idx])
        loss_mask = expected_weighted_bce(logits, target[idx])
        loss_world = F.cross_entropy(world_logits, train_data["world"][idx])
        loss = loss_mask + 0.10 * loss_world
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not probe and step % EVAL_INTERVAL == 0:
            p = probabilities(model, val_data, list(range(len(val_copy_rows))))
            masks = exact_count_keep_masks([row for row in p])
            score = null_f1(val_copy_rows, masks)
            history.append({"step": step, "train_loss": float(loss.item()), "validation_copy_null_f1": score})
            if score > best_f1:
                best_f1, best_step = score, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.monotonic() - start
    if probe:
        return {"experiment": EXPERIMENT, "mode": "probe", "variant": variant, "seed": seed,
                "steps": PROBE_STEPS, "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / PROBE_STEPS,
                "estimated_four_run_seconds": elapsed / PROBE_STEPS * STEPS * 4,
                "mps_allocated_gib": float(torch.mps.current_allocated_memory() / (1024 ** 3)) if device == "mps" else None}
    if best_state is None or best_step not in range(EVAL_INTERVAL, STEPS + 1, EVAL_INTERVAL):
        raise AssertionError("no validation-selected checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment": EXPERIMENT, "variant": variant, "seed": seed,
                "model": best_state, "best_step": best_step,
                "validation_copy_null_f1": best_f1,
                "data_manifest_sha256": DATA_MANIFEST_SHA,
                "runner_sha256": digest(Path(__file__))}, output)
    report = {"experiment": EXPERIMENT, "variant": variant, "seed": seed, "device": device,
              "torch": str(torch.__version__), "parameter_count": sum(p.numel() for p in model.parameters()),
              "steps": STEPS, "best_step": best_step, "best_validation_copy_null_f1": best_f1,
              "validation_copy_n": len(val_copy_rows), "elapsed_seconds": elapsed,
              "history": history, "checkpoint_sha256": digest(output),
              "data_manifest_sha256": DATA_MANIFEST_SHA,
              "target_sha256": json.loads((root / "data/manifests/exp0035_data.json").read_text())["derived_sha256"]["train_soft_targets"],
              "runner_sha256": digest(Path(__file__))}
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"EXP-0035 {'probe' if args.probe else 'training'} exceeded {60 if args.probe else MAX_TRAIN_SECONDS} seconds")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(60 if args.probe else MAX_TRAIN_SECONDS)
    try:
        result = train(args.root, args.variant, args.seed, args.device, args.probe)
    finally:
        signal.alarm(0)
    if args.probe:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({key: result[key] for key in ("variant", "seed", "parameter_count", "best_step", "best_validation_copy_null_f1", "elapsed_seconds", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
