"""Controls for an actual learned state-transition model, separate from its simulator."""

import inspect

import pytest
import torch

from voynich.communication.dynamics import ActionWorldModel, DynamicsConfig, dynamics_loss, transition_batch
from voynich.communication.worlds import (
    Action,
    Entity,
    EntityState,
    PROCEDURE_OPERATORS,
    execute_action,
    state_projection,
)


def _model():
    with torch.random.fork_rng():
        torch.manual_seed(31)
        return ActionWorldModel(DynamicsConfig(width=16, layers=1, heads=4)).eval()


def _forward(model, batch, **kwargs):
    return model(batch["states"], batch["actions"], batch["kinds"], **kwargs)


def test_transition_dataset_matches_independent_executor_and_preserves_invalid_convention():
    batch = transition_batch(71, 64)
    assert set(batch) == {"states", "actions", "kinds", "next_states", "valid"}
    assert batch["valid"].sum().item() == 32
    assert torch.equal(batch["states"][~batch["valid"]], batch["next_states"][~batch["valid"]])
    assert len(set(batch["kinds"].argmin(1).tolist())) == 4
    for index in range(64):
        table = tuple(Entity(i, "agent" if kind == 0 else "vessel")
                      for i, kind in enumerate(batch["kinds"][index].tolist()))
        state = tuple(EntityState(i, value > 0, value == 2)
                      for i, value in enumerate(batch["states"][index].tolist()))
        operator, subject, obj = batch["actions"][index].tolist()
        action = Action(PROCEDURE_OPERATORS[operator], subject, obj)
        if batch["valid"][index]:
            result = execute_action(table, state, action)
            assert state_projection(result) == tuple(batch["next_states"][index].tolist())
        else:
            with pytest.raises(ValueError):
                execute_action(table, state, action)


def test_generation_is_reproducible_and_does_not_touch_torch_rng():
    before = torch.random.get_rng_state().clone()
    left = transition_batch(41, 19)
    right = transition_batch(41, 19)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert not torch.equal(left["actions"], transition_batch(42, 19)["actions"])


def test_entity_permutation_equivariance_includes_action_role_bindings():
    model = _model()
    batch = transition_batch(13, 10)
    permutation = torch.tensor([2, 0, 3, 1])  # new slot i contains old slot permutation[i]
    inverse = torch.argsort(permutation)
    permuted_actions = batch["actions"].clone()
    permuted_actions[:, 1:] = inverse[permuted_actions[:, 1:]]
    with torch.no_grad():
        logits, validity, hidden = _forward(model, batch, return_hidden=True)
        transformed, transformed_validity, transformed_hidden = model(
            batch["states"][:, permutation], permuted_actions, batch["kinds"][:, permutation],
            return_hidden=True,
        )
    torch.testing.assert_close(transformed, logits[:, permutation], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(transformed_hidden, hidden[:, permutation], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(transformed_validity, validity, atol=2e-6, rtol=2e-6)


def test_network_has_finite_gradients_and_no_future_state_input():
    model = _model().train()
    batch = transition_batch(9, 16)
    logits, validity = _forward(model, batch)
    assert logits.shape == (16, 4, 3)
    assert validity.shape == (16,)
    report = dynamics_loss(logits, validity, batch)
    assert report["valid_count"] == 8
    assert torch.isfinite(report["loss"])
    report["loss"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert sum(parameter.grad.abs().sum().item() for parameter in model.parameters()) > 0
    assert "next_states" not in inspect.signature(model.forward).parameters
    assert "valid" not in inspect.signature(model.forward).parameters


def test_invalid_next_state_labels_contribute_no_loss_or_gradient():
    batch = transition_batch(29, 8)
    logits = torch.zeros(8, 4, 3, requires_grad=True)
    validity = torch.zeros(8, requires_grad=True)
    report = dynamics_loss(logits, validity, batch)
    changed = {**batch, "next_states": batch["next_states"].clone()}
    changed["next_states"][~batch["valid"]] = (changed["next_states"][~batch["valid"]] + 1) % 3
    assert torch.equal(report["state_loss"], dynamics_loss(logits, validity, changed)["state_loss"])
    report["loss"].backward()
    assert torch.count_nonzero(logits.grad[~batch["valid"]]) == 0
    assert torch.count_nonzero(logits.grad[batch["valid"]]) > 0
    assert torch.count_nonzero(validity.grad) == 8
    all_invalid = {**batch, "valid": torch.zeros_like(batch["valid"])}
    assert dynamics_loss(logits, validity, all_invalid)["state_loss"] == 0


def test_identity_full_donor_and_selective_hidden_interventions():
    model = _model()
    recipient, donor = transition_batch(43, 6), transition_batch(44, 6)
    with torch.no_grad():
        baseline, baseline_valid, hidden = _forward(model, recipient, return_hidden=True)
        donor_logits, donor_valid, donor_hidden = _forward(model, donor, return_hidden=True)
        identical, identical_valid = _forward(model, recipient, hidden_patch=hidden)
        full, full_valid = _forward(model, recipient, hidden_patch=donor_hidden)
        mask = torch.zeros(6, 4, dtype=torch.bool)
        mask[:, 2] = True
        selective, _, selective_hidden = _forward(model, recipient, return_hidden=True,
                                                  hidden_patch={"values": donor_hidden, "mask": mask})
    assert torch.equal(identical, baseline)
    assert torch.equal(identical_valid, baseline_valid)
    assert torch.equal(full, donor_logits)
    assert torch.equal(full_valid, donor_valid)
    assert torch.equal(selective_hidden[~mask], hidden[~mask])
    assert torch.equal(selective_hidden[mask], donor_hidden[mask])
    assert torch.equal(selective[~mask], baseline[~mask])
    assert not torch.equal(selective[mask], baseline[mask])


@pytest.mark.parametrize("arguments", [
    {"entities": 2}, {"width": 15}, {"heads": 0}, {"layers": 0}, {"width": True},
])
def test_bad_configuration_fails_explicitly(arguments):
    with pytest.raises(ValueError):
        DynamicsConfig(**arguments)


def test_input_and_intervention_validation_rejects_silent_shape_or_domain_errors():
    model, batch = _model(), transition_batch(15, 4)
    with pytest.raises(ValueError, match="Long"):
        model(batch["states"].float(), batch["actions"], batch["kinds"])
    wrong_action = batch["actions"].clone()
    wrong_action[0, 1] = 4
    with pytest.raises(ValueError, match="domain"):
        model(batch["states"], wrong_action, batch["kinds"])
    with pytest.raises(ValueError, match="shape"):
        _forward(model, batch, hidden_patch=torch.zeros(4, 3, 16))
    with pytest.raises(ValueError, match="finite"):
        _forward(model, batch, hidden_patch=torch.full((4, 4, 16), float("nan")))
    with pytest.raises(ValueError, match="mask"):
        _forward(model, batch, hidden_patch={"values": torch.zeros(4, 4, 16), "mask": torch.zeros(4, 4)})
