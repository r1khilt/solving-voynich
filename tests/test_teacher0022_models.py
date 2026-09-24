"""Linked reader and visible-route objective invariants."""

import pytest
import torch

from voynich.workspace.teacher14_tasks import (
    RenderSpec, sample_episode,
)
from voynich.workspace.teacher22_models import (
    make_reader, reader_output, route_loss,
)


def _episodes():
    import random

    rng = random.Random(22001)
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    return [
        sample_episode(rng, signal_hops=2, task=task, distractors=4,
                       spec=spec, stage_partitions=("train", "train"))
        for task in ("copy", "first_hop", "direct", "composed")
    ]


def test_linked_reader_shares_address_map_and_carries_embedding():
    model = make_reader(linked=True, seed=84121, device="cpu")
    assert model.query_proj.weight is model.key_proj.weight
    assert isinstance(model.value_proj, torch.nn.Identity)
    assert model.update.log_scale.exp().item() == pytest.approx(0.1)
    assert model.parameter_count < make_reader(
        linked=False, seed=84121, device="cpu").parameter_count
    episodes = _episodes()
    with torch.no_grad():
        output, left, right, mask = reader_output(
            model, episodes, "cpu", capture=True)
    assert output.logits.shape == (4, 2064)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(route_loss(output, episodes, left, right, mask))


def test_route_loss_rejects_wrong_answer_metadata():
    from dataclasses import replace

    episodes = _episodes()
    model = make_reader(linked=False, seed=84121, device="cpu")
    output, left, right, mask = reader_output(
        model, episodes, "cpu", capture=True)
    altered = list(episodes)
    altered[-1] = replace(altered[-1], answer=altered[-1].query)
    with pytest.raises(ValueError, match="Visible paths disagree"):
        route_loss(output, altered, left, right, mask)


def test_route_objective_has_finite_gradients():
    episodes = _episodes()
    model = make_reader(linked=True, seed=84131, device="cpu")
    output, left, right, mask = reader_output(
        model, episodes, "cpu", capture=True)
    loss = route_loss(output, episodes, left, right, mask)
    loss.backward()
    assert model.key_proj.weight.grad is not None
    assert torch.isfinite(model.key_proj.weight.grad).all()
