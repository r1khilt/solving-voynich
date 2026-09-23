"""CPU-only prelaunch checks for TEACH-0012 raw serialized binding."""

import random

import torch
import torch.nn.functional as F

from voynich.workspace.teacher12_models import (
    ARMS, collate_rows, model_for_arm,
)
from voynich.workspace.teacher12_train import Config, GROUP_WIDTHS, decision
from voynich.workspace.teacher12_tasks import (
    ANSWER, CONTEXT_LENGTH, SYMBOLS, RenderSpec, evaluation_suite, pad_batch,
    sample_episode, schedule, symbolic_oracle, training_batch,
)


PARAMETERS = {
    "parsed_memory": 3_160_576,
    "raw_shallow": 14_693_376,
    "raw_looped": 14_694_912,
    "raw_deep": 41_965_568,
    "raw_null": 14_693_376,
}


def test_generator_is_deterministic_functional_and_uses_shared_symbols():
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    episodes = [sample_episode(
        random.Random(1200 + index), f_partition="confirm", g_partition="confirm",
        task=task, distractor_chains=4, spec=spec)
        for index, task in enumerate(("first_hop", "direct", "composed", "copy"))]
    for episode in episodes:
        assert symbolic_oracle(episode) == episode.answer
        assert len(episode.tokens) <= CONTEXT_LENGTH
        assert episode.tokens[-1] == ANSWER
        assert len({left for left, _ in episode.rows}) == len(episode.rows)
        assert all(value in SYMBOLS for row in episode.rows for value in row)
    first = sample_episode(random.Random(91), f_partition="train", g_partition="train",
                           task="composed", distractor_chains=3, spec=spec)
    second = sample_episode(random.Random(91), f_partition="train", g_partition="train",
                            task="composed", distractor_chains=3, spec=spec)
    assert first == second


def test_evaluation_groups_preserve_or_change_exact_registered_factors():
    suite = evaluation_suite(72311, 4)
    assert evaluation_suite(72311, 4) == suite
    for episodes in suite.values():
        for episode in episodes:
            assert symbolic_oracle(episode) == episode.answer
            assert len(episode.tokens) <= CONTEXT_LENGTH
    for name in ("first_hop_query_groups", "direct_query_groups"):
        for offset in range(0, len(suite[name]), 4):
            group = suite[name][offset:offset + 4]
            assert len({ep.answer for ep in group}) == 4
            assert len({ep.query for ep in group}) == 4
    for offset in range(0, len(suite["factorial"]), 4):
        group = suite["factorial"][offset:offset + 4]
        assert len({ep.answer for ep in group}) == 4
    for name in ("order_groups", "distractor_groups", "boundary_groups"):
        for offset in range(0, len(suite[name]), 4):
            group = suite[name][offset:offset + 4]
            assert len({ep.answer for ep in group}) == 1
            assert len({(ep.f_rows, ep.g_rows, ep.task, ep.query) for ep in group}) == 1
            assert len({ep.tokens for ep in group}) == 4
    for offset in range(0, len(suite["distractor_groups"]), 4):
        assert [ep.distractor_chains for ep in suite["distractor_groups"][offset:offset + 4]] \
            == [0, 1, 2, 4]
    for offset in range(0, len(suite["boundary_groups"]), 4):
        assert [ep.marker_dropout for ep in suite["boundary_groups"][offset:offset + 4]] \
            == [0.0, .25, .50, 1.0]


def test_frozen_schedule_and_null_labels():
    assert schedule(0) == (("first_hop", .45), ("direct", .45), ("copy", .10))
    assert schedule(2000)[2] == ("composed", .40)
    assert schedule(7999)[2] == ("composed", .70)
    clean, clean_answers = training_batch(72221, 64, step=6000)
    null, null_answers = training_batch(72221, 64, step=6000, null_composed=True)
    assert clean == null
    for episode, clean_answer, null_answer in zip(
            clean, clean_answers, null_answers, strict=True):
        assert clean_answer == episode.answer
        if episode.task != "composed":
            assert null_answer == clean_answer
        else:
            assert null_answer in {right for _, right in episode.g_rows}


def test_parameter_counts_and_finite_forward_backward():
    episodes = [sample_episode(
        random.Random(400 + index), f_partition="train", g_partition="train",
        task="composed", distractor_chains=1,
        spec=RenderSpec(0.0, 0, ("prefix",))) for index in range(2)]
    padded, _ = pad_batch(episodes)
    ids = torch.tensor(padded)
    targets = torch.tensor([ep.answer for ep in episodes])
    for arm in ARMS:
        torch.manual_seed(17)
        net = model_for_arm(arm)
        assert net.parameter_count == PARAMETERS[arm]
        if arm == "parsed_memory":
            rows = collate_rows(episodes, "cpu")
            output = net(ids, *rows)
        elif arm == "raw_deep":
            # Parameter construction is the relevant prelaunch gate; the identical
            # shallow raw path below exercises token padding and tied output gradients.
            continue
        else:
            output = net(ids)
        assert output.logits.shape == (2, 2064)
        loss = F.cross_entropy(output.logits[:, 16:], targets - 16)
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                   for parameter in net.parameters())


def test_frozen_config_and_conjunctive_decision_logic():
    Config().validate()
    names = evaluation_suite(72311, 1)

    def cells(value):
        rows = {}
        for name in names:
            rows[name] = {"accuracy": value}
            if name in GROUP_WIDTHS:
                rows[name].update({"group_accuracy": value, "marker_free_accuracy": value})
        return rows

    scores = {}
    for replicate in ("0", "1"):
        scores[replicate] = {arm: cells(.99) for arm in ARMS}
        scores[replicate]["raw_shallow"] = cells(.70)
        scores[replicate]["raw_null"] = cells(0.0)
    result = decision(scores)
    assert result["verdict"] == "qualified"
    assert result["depth_helpful"] is True
    scores["1"]["raw_deep"]["factorial"]["group_accuracy"] = .74
    assert decision(scores)["verdict"] == "not_qualified"
    scores["1"]["raw_deep"]["factorial"]["group_accuracy"] = .99
    scores["0"]["raw_null"]["composed_confirm_confirm"]["accuracy"] = .50
    assert decision(scores)["verdict"] == "invalid"
