"""Action evaluation masks and tiny trainer lifecycle tests; no scientific fit."""

import json

import pytest
import torch
from torch import nn

import voynich.communication.action_training as action_training
from voynich.communication.action_training import evaluate_actions, train_actions, transition_keys
from voynich.communication.dynamics import ActionWorldModel, DynamicsConfig, transition_batch


class IdentityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(()))

    def forward(self, state, action, kinds):
        logits = torch.nn.functional.one_hot(state, 3).float() * 4 + self.placeholder * 0
        return logits, torch.where(action[:, 0] == 1, 1.0, -1.0)


def _fixed_batch():
    return {"states": torch.tensor([[0, 1, 0], [0, 1, 0]]),
            "actions": torch.tensor([[1, 0, 1], [2, 0, 1]]),
            "kinds": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            "next_states": torch.tensor([[0, 2, 0], [0, 1, 0]]),
            "valid": torch.tensor([True, False])}


def test_invalid_storage_identity_does_not_receive_next_state_credit():
    batch = _fixed_batch()
    model = IdentityPredictor().train()
    report = evaluate_actions(model, batch, batch_size=1)
    assert model.training
    assert report["valid_node_accuracy"] == pytest.approx(2 / 3)
    assert report["valid_exact_next_state_accuracy"] == 0
    assert report["valid_changed_node_accuracy"] == 0
    assert report["validity_balanced_accuracy"] == 1
    assert report["full_transition_accuracy"] == 0.5
    changed = {**batch, "next_states": batch["next_states"].clone()}
    changed["next_states"][1] = torch.tensor([2, 2, 2])
    rechecked = evaluate_actions(model, changed)
    for key in ("valid_node_accuracy", "valid_exact_next_state_accuracy", "full_transition_accuracy"):
        assert rechecked[key] == report[key]


def test_unseen_state_action_metrics_use_actual_overlap_and_handle_empty_subgroups():
    batch = _fixed_batch()
    keys = transition_keys(batch)
    report = evaluate_actions(IdentityPredictor(), batch, seen_transitions={keys[0]})
    assert report["heldout_state_action_seen_fraction"] == 0.5
    assert report["heldout_unseen_unique_state_actions"] == 1
    assert report["unseen_valid_examples"] == 0
    assert report["unseen_valid_exact_next_state_accuracy"] is None
    assert report["unseen_full_transition_accuracy"] == 1
    all_seen = evaluate_actions(IdentityPredictor(), batch, seen_transitions=set(keys))
    assert all_seen["unseen_full_transition_accuracy"] is None
    changed = {**batch, "next_states": torch.zeros_like(batch["next_states"]), "valid": ~batch["valid"]}
    assert transition_keys(changed) == keys  # labels are not part of the overlap identity


def test_metric_masks_and_shapes_are_strict_not_integer_advanced_indexing():
    batch = _fixed_batch()
    with pytest.raises(ValueError, match="boolean"):
        evaluate_actions(IdentityPredictor(), {**batch, "valid": batch["valid"].long()})
    with pytest.raises(ValueError, match="shapes"):
        transition_keys({**batch, "actions": batch["actions"][:1]})
    with pytest.raises(ValueError, match="domain"):
        evaluate_actions(IdentityPredictor(), {**batch, "next_states": batch["next_states"] + 3})


def test_tiny_training_checkpoint_is_loadable_and_refuses_overwrite(tmp_path, monkeypatch):
    # Shrink the oracle batch only for this lifecycle test. The production trainer
    # requests 2048 heldout cases; this is not a reported research run.
    original = transition_batch
    monkeypatch.setattr(action_training, "transition_batch",
                        lambda seed, count, entities: original(seed, min(count, 8), entities))
    destination = tmp_path / "unit-run"
    summary = train_actions(destination, device="cpu", steps=2, max_seconds=10, seed=57, batch_size=4)
    assert summary["steps"] == 2
    assert summary["training_examples"] == 8
    checkpoint = torch.load(destination / "model.pt", map_location="cpu", weights_only=True)
    restored = ActionWorldModel(DynamicsConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    assert checkpoint["source_sha256"]["dynamics.py"]
    assert checkpoint["summary"] == summary
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["validation_seed"] > 57 * (1 << 21) + 2
    assert json.loads((destination / "summary.json").read_text()) == summary
    with pytest.raises(ValueError, match="overwritten"):
        train_actions(destination, device="cpu", steps=1)


@pytest.mark.parametrize("arguments", [{"steps": 0}, {"batch_size": 0}, {"max_seconds": 0},
                                      {"max_seconds": float("inf")}, {"seed": True}, {"steps": 1 << 20}])
def test_invalid_run_bounds_fail_before_writing(tmp_path, arguments):
    destination = tmp_path / "bad"
    with pytest.raises(ValueError):
        train_actions(destination, **arguments)
    assert not destination.exists()


def test_time_stop_retains_auditable_untrained_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(action_training, "transition_batch",
                        lambda seed, count, entities: transition_batch(seed, 2, entities))
    summary = train_actions(tmp_path / "stopped", device="cpu", steps=4, max_seconds=1e-12, seed=51)
    assert summary["status"] == "time_budget"
    assert summary["steps"] == 0
    assert summary["unique_training_state_actions"] == 0
    assert summary["initial"] == summary["final"]
