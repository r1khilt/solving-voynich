"""Bounded training and honest overlap-aware evaluation of synthetic action dynamics."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time

import torch

from ..runtime import resolve_device
from .dynamics import ActionWorldModel, DynamicsConfig, dynamics_loss, transition_batch
from .schema import object_digest, save_json
from .training import environment, source_manifest


def _validate_batch(batch):
    required = {"states", "actions", "kinds", "next_states", "valid"}
    if not isinstance(batch, dict) or set(batch) != required:
        raise ValueError("transition batch must contain exactly states/actions/kinds/next_states/valid")
    if any(not isinstance(value, torch.Tensor) for value in batch.values()):
        raise ValueError("transition batch fields must be tensors")
    states = batch["states"]
    if states.ndim != 2 or min(states.shape) < 1:
        raise ValueError("states must be a nonempty [batch, entities] tensor")
    count = states.shape[0]
    if (batch["next_states"].shape != states.shape or batch["kinds"].shape != states.shape or
            batch["actions"].shape != (count, 3) or batch["valid"].shape != (count,)):
        raise ValueError("incompatible transition batch shapes")
    if batch["valid"].dtype != torch.bool or any(batch[name].dtype != torch.long for name in required - {"valid"}):
        raise ValueError("valid must be boolean; state/action/kind/target fields must be Long")
    if (bool(((states < 0) | (states > 2)).any()) or
            bool(((batch["next_states"] < 0) | (batch["next_states"] > 2)).any()) or
            bool(((batch["kinds"] < 0) | (batch["kinds"] > 1)).any()) or
            bool(((batch["actions"][:, 0] < 0) | (batch["actions"][:, 0] > 2)).any()) or
            bool(((batch["actions"][:, 1:] < 0) | (batch["actions"][:, 1:] >= states.shape[1])).any())):
        raise ValueError("transition batch value outside its declared domain")


def transition_keys(batch):
    """State/action/type triples; future state and validity labels are excluded."""
    _validate_batch(batch)
    states, actions, kinds = (batch[name].detach().cpu().tolist() for name in ("states", "actions", "kinds"))
    return [(tuple(state), tuple(action), tuple(kind)) for state, action, kind in zip(states, actions, kinds)]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def evaluate_actions(model, batch, *, seen_transitions=(), batch_size=256):
    """Score future states only on executable actions, plus applicability on all.

    Independent generator seeds are NOT an unseen-state/action split in a small
    finite world. Both row-weighted and unique-pair overlap are explicitly counted.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    _validate_batch(batch)
    count = len(batch["valid"])
    if not count:
        raise ValueError("cannot evaluate an empty transition set")
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    predicted, applicability = [], []
    try:
        model.eval()
        for start in range(0, count, batch_size):
            state, action, kinds = (batch[name][start:start + batch_size].to(device)
                                    for name in ("states", "actions", "kinds"))
            logits, valid_logit = model(state, action, kinds)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(valid_logit).all()):
                raise FloatingPointError("non-finite action model prediction")
            predicted.append(logits.argmax(-1).cpu())
            applicability.append((valid_logit >= 0).cpu())
    finally:
        for module, training in modes:
            module.training = training
    predicted, applicability = torch.cat(predicted), torch.cat(applicability)
    target, valid = batch["next_states"].cpu(), batch["valid"].cpu()
    states = batch["states"].cpu()
    node_correct = predicted.eq(target)
    state_correct = node_correct.all(-1)
    validity_correct = applicability.eq(valid)
    changed = states.ne(target) & valid[:, None]
    true_count, false_count = int(valid.sum()), int((~valid).sum())
    true_recall = _ratio(int((applicability & valid).sum()), true_count)
    false_recall = _ratio(int((~applicability & ~valid).sum()), false_count)
    keys = transition_keys(batch)
    seen = set(seen_transitions)
    unseen = torch.tensor([key not in seen for key in keys], dtype=torch.bool)
    unseen_valid = unseen & valid
    unique = set(keys)
    full_correct = validity_correct & (~valid | state_correct)
    return {
        "examples": count, "valid_examples": true_count, "invalid_examples": false_count,
        "valid_node_accuracy": _ratio(int(node_correct[valid].sum()), true_count * target.shape[1]),
        "valid_exact_next_state_accuracy": _ratio(int(state_correct[valid].sum()), true_count),
        "valid_changed_node_accuracy": _ratio(int((node_correct & changed).sum()), int(changed.sum())),
        "validity_accuracy": float(validity_correct.float().mean()),
        "validity_true_recall": true_recall, "validity_false_recall": false_recall,
        "validity_balanced_accuracy": ((true_recall + false_recall) / 2
                                        if true_recall is not None and false_recall is not None else None),
        "full_transition_accuracy": float(full_correct.float().mean()),
        "heldout_state_action_seen_fraction": float((~unseen).float().mean()),
        "heldout_unseen_state_action_examples": int(unseen.sum()),
        "heldout_unique_state_actions": len(unique),
        "heldout_unseen_unique_state_actions": len(unique - seen),
        "unseen_valid_examples": int(unseen_valid.sum()),
        "unseen_valid_exact_next_state_accuracy": _ratio(int(state_correct[unseen_valid].sum()),
                                                         int(unseen_valid.sum())),
        "unseen_full_transition_accuracy": _ratio(int(full_correct[unseen].sum()), int(unseen.sum())),
        "input_sha256": object_digest({name: value.cpu().tolist() for name, value in batch.items()}),
        "state_label_policy": "invalid next-state storage labels excluded from all state accuracy metrics",
    }


def train_actions(
    run_dir, device="auto", steps=1000, max_seconds=180, seed=41022, batch_size=64,
):
    """Run one finite action-model training schedule and save its audit checkpoint.

    Time is checked between updates; the fixed initial/final 2048-example audits
    are bounded overhead. Validation seeds are disjoint from training seeds, while
    actual state/action overlap is measured rather than assumed absent.
    """
    for name, value in (("steps", steps), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if steps >= 1 << 20:
        raise ValueError("steps exceeds the reserved nonoverlapping seed domain")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    destination = Path(run_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("use a new output directory; existing runs are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    config = DynamicsConfig(entities=4, width=64, layers=2, heads=4)
    source = source_manifest()
    torch.manual_seed(seed)
    model = ActionWorldModel(config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    validation_seed = seed * (1 << 21) + (1 << 20)
    manifest = {
        "schema_version": 1, "model_config": asdict(config), "seed": seed,
        "steps": steps, "batch_size": batch_size, "max_seconds": max_seconds,
        "learning_rate": 1e-3, "optimizer": "AdamW", "device": selected_device,
        "validation_examples": 2048, "validation_seed": validation_seed,
        "training_seed_rule": "seed * 2**21 + zero_based_step",
        "supervision": "independent typed simulator next-state and applicability labels",
        "environment": environment(), "parameters": sum(p.numel() for p in model.parameters()),
        "scope": "synthetic action transitions only; no manuscript inputs or historical grounding",
    }
    save_json(destination / "manifest.json", manifest)
    start = time.monotonic()
    heldout = transition_batch(validation_seed, 2048, config.entities)
    initial = evaluate_actions(model, heldout)
    seen = set()
    history = []
    completed, status = 0, "completed"
    for step in range(steps):
        if time.monotonic() - start >= max_seconds:
            status = "time_budget"
            break
        batch = transition_batch(seed * (1 << 21) + step, batch_size, config.entities)
        moved = {name: value.to(selected_device) for name, value in batch.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, validity = model(moved["states"], moved["actions"], moved["kinds"])
        losses = dynamics_loss(logits, validity, moved)
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError("non-finite action training loss")
        losses["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        seen.update(transition_keys(batch))
        completed += 1
        if completed == 1 or completed % 100 == 0 or completed == steps:
            history.append({"step": completed, "loss": float(losses["loss"].detach()),
                            "state_loss": float(losses["state_loss"].detach()),
                            "validity_loss": float(losses["validity_loss"].detach()),
                            "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - start})
    final = evaluate_actions(model, heldout, seen_transitions=seen)
    summary = {
        "schema_version": 1, "status": status, "steps": completed, "seed": seed,
        "training_examples": completed * batch_size, "unique_training_state_actions": len(seen),
        "elapsed_seconds": time.monotonic() - start, "initial": initial, "final": final,
        "history": history, "parameters": manifest["parameters"],
        "limitations": ["No pass/fail threshold or historical interpretation is claimed",
                        "Seed-disjoint validation may repeat trained state/action pairs; overlap is reported",
                        "Initial/final fixed-size audits are overhead outside per-update time checks",
                        "One local model seed; label-supervised synthetic dynamics only"],
    }
    checkpoint = {"schema_version": 1, "model_config": asdict(config), "seed": seed,
                  "steps": completed, "device": selected_device, "source_sha256": source,
                  "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                  "summary": summary}
    temporary = destination / "model.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(destination / "model.pt")
    save_json(destination / "summary.json", summary)
    return summary
