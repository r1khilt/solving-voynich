"""Recipient-specific interchange targets and gradient path."""

import pytest
import torch

from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_objectives import (
    deranged_interchange_targets, interchange_loss,
)
from voynich.workspace.teacher14_tasks import causal_training_batch


def test_deranged_targets_preserve_each_recipient_multiset():
    targets = torch.tensor([[10, 20], [11, 21], [12, 22], [13, 23]])
    wrong = deranged_interchange_targets(targets)
    assert not torch.eq(targets, wrong).any()
    assert sorted(wrong[:, 0].tolist()) == sorted(targets[:, 0].tolist())
    assert sorted(wrong[:, 1].tolist()) == sorted(targets[:, 1].tolist())
    with pytest.raises(ValueError, match="at least two"):
        deranged_interchange_targets(targets[:1])


def test_correct_and_wrong_interchange_use_same_inputs_and_backpropagate():
    pairs = causal_training_batch(74611, 4, step=0)
    torch.manual_seed(1414)
    model = CandidateEdgeWorkspace()
    correct, correct_rows = interchange_loss(model, pairs, "cpu")
    wrong, wrong_rows = interchange_loss(model, pairs, "cpu", wrong_targets=True)
    assert torch.isfinite(correct) and torch.isfinite(wrong)
    assert correct_rows["patched_logits"].shape == (8, 2064)
    assert wrong_rows["patched_logits"].shape == (8, 2064)
    torch.testing.assert_close(
        correct_rows["patched_logits"], wrong_rows["patched_logits"],
        rtol=0, atol=0)
    torch.testing.assert_close(
        correct_rows["donor_first_state"], wrong_rows["donor_first_state"],
        rtol=0, atol=0)
    assert not torch.eq(correct_rows["targets"], wrong_rows["targets"]).any()
    correct.backward()
    assert model.query_proj.weight.grad is not None
    assert torch.isfinite(model.query_proj.weight.grad).all()
