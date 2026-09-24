"""Behavioral and causal-interface invariants for the TEACH-0014 model draft."""

import random

import pytest
import torch
import torch.nn.functional as F

from voynich.workspace.teacher14_models import (
    CandidateEdgeWorkspace, DenseEpisodeClassifier, public_edge_loss,
)
from voynich.workspace.teacher14_tasks import (
    COMPOSE, DIRECT, PAD, RenderSpec, sample_episode,
)


def _batch(episodes):
    width = max(len(episode.tokens) for episode in episodes)
    return torch.tensor(
        [episode.tokens + (PAD,) * (width - len(episode.tokens))
         for episode in episodes], dtype=torch.long)


def _rows(episodes):
    width = max(len(episode.serialized_rows) for episode in episodes)
    left = torch.zeros((len(episodes), width), dtype=torch.long)
    right = torch.zeros_like(left)
    mask = torch.zeros_like(left, dtype=torch.bool)
    for index, episode in enumerate(episodes):
        for position, (a, b) in enumerate(episode.serialized_rows):
            left[index, position] = a
            right[index, position] = b
            mask[index, position] = True
    return left, right, mask


def test_raw_and_oracle_interfaces_run_finite_forward_backward():
    rng = random.Random(140114)
    episodes = [
        sample_episode(rng, signal_hops=2, task="composed", distractors=count,
                       spec=RenderSpec(.5, 2, ("prefix", "infix", "suffix")))
        for count in (0, 4, 8)
    ]
    ids = _batch(episodes)
    targets = torch.tensor([episode.answer for episode in episodes])
    for oracle in (False, True):
        torch.manual_seed(14)
        model = CandidateEdgeWorkspace(oracle_rows=oracle)
        kwargs = {}
        if oracle:
            left, right, mask = _rows(episodes)
            kwargs = {"row_left": left, "row_right": right, "row_mask": mask}
        output = model(ids, capture=True, **kwargs)
        assert model.parameter_count > 20_000_000
        assert output.logits.shape == (3, 2064)
        assert torch.isfinite(output.logits).all()
        assert output.cache["read.0.attention"].shape[0] == 3
        assert output.cache["read.1.attention"].shape[0] == 3
        loss = F.cross_entropy(output.logits[:, 16:], targets - 16)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())


def test_oracle_rejects_hidden_stage_order_rows():
    rng = random.Random(140118)
    episode = sample_episode(
        rng, signal_hops=2, task="composed", distractors=4,
        spec=RenderSpec(.25, 2, ("prefix", "infix", "suffix")))
    assert episode.rows != episode.serialized_rows
    ids = _batch([episode])
    left = torch.tensor([[a for a, _ in episode.rows]])
    right = torch.tensor([[b for _, b in episode.rows]])
    mask = torch.ones_like(left, dtype=torch.bool)
    model = CandidateEdgeWorkspace(oracle_rows=True)
    with pytest.raises(ValueError, match="physical order"):
        model(ids, row_left=left, row_right=right, row_mask=mask)


def test_parser_state_cannot_see_task_or_query_suffix():
    rng = random.Random(140115)
    episode = sample_episode(
        rng, signal_hops=2, task="composed", distractors=4,
        spec=RenderSpec(.25, 2, ("prefix", "infix", "suffix")))
    ids = _batch([episode])
    other = ids.clone()
    other[0, -3] = DIRECT
    other[0, -2] = episode.signal_paths[1][1]
    torch.manual_seed(15)
    model = CandidateEdgeWorkspace().eval()
    with torch.no_grad():
        before = model(ids, capture=True)
        after = model(other, capture=True)
    for key in ("edge_gate_logits", "memory.keys", "memory.values",
                "encoder.5.residual", "parser.1.residual"):
        torch.testing.assert_close(before.cache[key], after.cache[key], rtol=0, atol=0)
    assert not torch.equal(before.cache["query.0"], after.cache["query.0"])


def test_memory_permutation_and_identity_patch_are_exact_noops():
    rng = random.Random(140116)
    episode = sample_episode(
        rng, signal_hops=2, task="composed", distractors=2,
        spec=RenderSpec(.5, 2, ("prefix", "infix", "suffix")))
    ids = _batch([episode])
    torch.manual_seed(16)
    model = CandidateEdgeWorkspace().eval()
    with torch.no_grad():
        base = model(ids, capture=True)
        patched = model(ids, interventions={"query.1": base.cache["query.1"]})
        torch.testing.assert_close(base.logits, patched.logits, rtol=0, atol=0)
        keys = base.cache["memory.keys"]
        values = base.cache["memory.values"]
        gates = base.cache["edge_gate_logits"]
        mask = base.cache["candidate_mask"]
        query = base.cache["query.0"]
        native = model.read(query, keys, values, gates, mask,
                            query_proj=model.query_proj)
        order = torch.arange(keys.shape[1] - 1, -1, -1)
        shuffled = model.read(query, keys[:, order], values[:, order],
                              gates[:, order], mask[:, order],
                              query_proj=model.query_proj)
        torch.testing.assert_close(native[0], shuffled[0], rtol=0, atol=1e-6)
        with pytest.raises(ValueError, match="shape/device mismatch"):
            model(ids, interventions={"query.1": torch.zeros(1, 2)})


def test_mean_address_control_does_not_receive_query_in_parser():
    rng = random.Random(140117)
    episode = sample_episode(
        rng, signal_hops=2, task="composed", distractors=1,
        spec=RenderSpec(1.0, 0, ("prefix",)))
    ids = _batch([episode])
    other = ids.clone()
    other[0, -3] = COMPOSE
    other[0, -2] = episode.signal_paths[1][0]
    torch.manual_seed(17)
    model = CandidateEdgeWorkspace(mean_address=True).eval()
    with torch.no_grad():
        before = model(ids, capture=True)
        after = model(other, capture=True)
    torch.testing.assert_close(before.cache["read.0.attention"],
                               after.cache["read.0.attention"], rtol=0, atol=0)


def test_four_hop_eight_distractor_case_uses_all_63_candidates():
    rng = random.Random(140119)
    episode = sample_episode(
        rng, signal_hops=4, task="composed", distractors=8,
        spec=RenderSpec(0.0, 3, ("prefix", "infix", "suffix")),
        graph_partition="confirm")
    ids = _batch([episode])
    torch.manual_seed(19)
    model = CandidateEdgeWorkspace().eval()
    with torch.no_grad():
        output = model(ids, capture=True)
    assert output.cache["edge_gate_logits"].shape == (1, 63)
    assert all(f"read.{step}.attention" in output.cache for step in range(4))
    assert torch.isfinite(output.logits).all()


def test_dense_control_matches_parameters_and_edge_aux_is_finite():
    rng = random.Random(140120)
    episode = sample_episode(
        rng, signal_hops=2, task="composed", distractors=3,
        spec=RenderSpec(.25, 2, ("prefix", "infix", "suffix")))
    ids = _batch([episode])
    torch.manual_seed(20)
    workspace = CandidateEdgeWorkspace()
    dense = DenseEpisodeClassifier()
    assert abs(dense.parameter_count / workspace.parameter_count - 1) < .001
    workspace_output = workspace(ids)
    dense_output = dense(ids)
    assert dense_output.logits.shape == workspace_output.logits.shape
    loss = public_edge_loss(workspace_output)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert workspace.edge_gate.weight.grad is not None
    with pytest.raises(ValueError, match="auxiliary tensors required"):
        public_edge_loss(dense_output)
