"""Random-weight numerical qualifications for the future finite patch assay."""

import pytest
import torch

from voynich.workspace.teacher14_train import Config, new_model
from voynich.workspace.teacher15_intervene import (
    capture, evaluate_surface, patch, recipient_vjp,
)
from voynich.workspace.teacher15_tasks import generate_split


def test_first_read_patch_identity_and_final_answer_control():
    group, other = generate_split("discovery", 2)
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    wrong_donor = next(cell.episode for cell in other.cells if (
        cell.f, cell.g, cell.distractor, cell.marked, cell.order, cell.task)
        == (1, 0, 0, True, 0, "composed"))
    rows = evaluate_surface(model, group, distractor=0, marked=True,
                            order=0, device="cpu", wrong_donor=wrong_donor)
    assert len(rows) == 3
    assert len({row["target_answer"] for row in rows}) == 3
    assert rows[0]["fixed_donor_answer"] == rows[0]["target_answer"]
    assert all(row["fixed_donor_answer"] != row["target_answer"]
               for row in rows[1:])
    for row in rows:
        assert row["identity_max_abs_logit_error"] < 1e-5
        assert row["final_donor_max_abs_logit_error"] < 1e-5
        assert row["final_donor_prediction"] == row["donor_prediction"]
        assert row["donor_delta_norm"] > 0
        assert len(row["transfer_logits"]) == 2064
        assert len(row["same_key_logits"]) == 2064
        assert len(row["wrong_key_logits"]) == 2064
        assert row["wrong_donor_render_id"] == wrong_donor.render_id


def test_copy_has_no_intermediate_key_hook():
    group = generate_split("discovery", 1)[0]
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    copy = next(cell.episode for cell in group.cells if cell.task == "copy")
    with pytest.raises(ValueError, match="Two-hop composed"):
        capture(model, copy, "cpu")
    oracle, _ = new_model(Config(), "oracle_rows_workspace", 0, "cpu")
    composed = next(cell.episode for cell in group.cells
                    if cell.task == "composed")
    with pytest.raises(ValueError, match="raw candidate-edge"):
        capture(oracle, composed, "cpu")


def test_recipient_vjp_matches_finite_patch_difference():
    group = generate_split("discovery", 1)[0]
    episode = next(cell.episode for cell in group.cells if (
        cell.f, cell.g, cell.distractor, cell.marked, cell.order, cell.task)
        == (0, 1, 0, True, 0, "composed"))
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    target, base = group.recipient_outputs[1][1], group.recipient_outputs[1][0]
    state, gradient, _ = recipient_vjp(
        model, episode, target_symbol=target, base_symbol=base, device="cpu")
    assert gradient.shape == state.shape == (1, 512)
    assert all(parameter.requires_grad for parameter in model.parameters())
    direction = torch.randn(state.shape, generator=torch.Generator().manual_seed(15))
    direction /= direction.norm()
    epsilon = 1e-2
    plus = patch(model, episode, state + epsilon * direction,
                 site="query.1", device="cpu")
    minus = patch(model, episode, state - epsilon * direction,
                  site="query.1", device="cpu")
    finite = float((plus[0, target] - plus[0, base] - minus[0, target] +
                    minus[0, base]) / (2 * epsilon))
    derivative = float((gradient * direction).sum().item())
    assert abs(finite - derivative) < 0.02
