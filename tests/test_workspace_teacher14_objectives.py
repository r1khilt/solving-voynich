"""Recipient-specific interchange targets and gradient path."""

import random

import pytest
import torch

from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_objectives import (
    deranged_interchange_targets, interchange_loss,
    matched_refinement_objectives, padded_tokens,
)
from voynich.workspace.teacher14_tasks import (
    RenderSpec, causal_training_batch, sample_episode,
)


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


def test_four_step_arms_share_corruption_objective_and_fifth_evaluation():
    episode = sample_episode(
        random.Random(1415), signal_hops=2, task="composed", distractors=1,
        spec=RenderSpec(.5, 1, ("prefix", "infix", "suffix")))
    torch.manual_seed(1415)
    recurrent = CandidateEdgeWorkspace(refinement="recurrent4")
    diffuse = CandidateEdgeWorkspace(refinement="diffusion4")
    diffuse.load_state_dict(recurrent.state_dict())
    ids = padded_tokens([episode], "cpu")
    results = []
    for model in (recurrent, diffuse):
        output = model(ids)
        uniform = torch.linspace(.02, .98, output.auxiliary["candidate_mask"].shape[1])[
            None, :]
        parts = matched_refinement_objectives(
            model, output, [episode.answer], time_step=2, uniform=uniform)
        assert set(parts) == {"answer", "edge_all_steps", "edge_denoise"}
        assert all(torch.isfinite(loss) for loss in parts.values())
        results.append(parts)
    torch.testing.assert_close(results[0]["edge_denoise"],
                               results[1]["edge_denoise"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="four-step"):
        matched_refinement_objectives(
            CandidateEdgeWorkspace(), recurrent(ids), [episode.answer],
            time_step=2, uniform=uniform)
