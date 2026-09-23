"""CPU-only qualification of TEACH-0003 frozen generator, heads and gates."""

from dataclasses import asdict
import json

import pytest
import torch

from voynich.workspace import teacher3_tasks as tasks
from voynich.workspace.teacher3_train import (
    ARMS, EXPECTED_PARAMETERS, Config, RowHeadTeacher, check_benchmark,
    decision, pointer_answers, row_targets, total_loss,
)


def test_fresh_family_partitions_and_oracle() -> None:
    suite = tasks.evaluation_suite(63311, 8)
    assert len(suite["factorial"]) == 16
    for name, episodes in suite.items():
        for ep in episodes:
            assert tasks.symbolic_oracle(ep.tokens) == ep.answer
            expected_f = "holdout" if name in (
                "first_hop_holdout", "first_hop_pairs_holdout", "copy", "factorial"
            ) or name.startswith("composed_holdout") else "train"
            if name.startswith("direct"):
                expected_f = "train"
            assert ep.f_partition == expected_f
    assert tasks.SPLIT_NAMESPACE != b"TEACH-0002-v1/"
    assert tasks.evaluation_suite(63311, 8) == suite


def test_paired_queries_and_four_way_factorial_are_nontrivial() -> None:
    suite = tasks.evaluation_suite(63311, 8)
    for name in ("first_hop_pairs_holdout", "direct_pairs_holdout"):
        for first, second in zip(suite[name][::2], suite[name][1::2], strict=True):
            assert first.tokens[:12] == second.tokens[:12]
            assert first.tokens[12] != second.tokens[12]
            assert first.answer != second.answer
    for offset in range(0, len(suite["factorial"]), 4):
        group = suite["factorial"][offset:offset + 4]
        assert len({ep.answer for ep in group}) == 4
        assert all(ep.f_partition == ep.g_partition == "holdout" for ep in group)


def test_true_auxiliary_targets_and_mapping_independent_null() -> None:
    first = tasks.make_episode((9, 10), (21, 22), (22, 21), (33, 34),
                               "first_hop", 10)
    direct = tasks.make_episode((9, 10), (21, 22), (22, 21), (33, 34),
                                "direct", 21)
    composed = tasks.make_episode((9, 10), (21, 22), (22, 21), (33, 34),
                                  "composed", 10)
    episodes = [first, direct, composed]
    assert pointer_answers(first) == (1, None)
    assert pointer_answers(direct) == (None, 1)
    assert pointer_answers(composed) == (1, 0)
    assert row_targets(episodes, "align_true", 42) == ([0, 2], [1, 1], [1, 2], [1, 0])
    null = row_targets(episodes, "align_random", 42)
    assert null[0] == [0, 2] and null[2] == [1, 2]
    assert null == row_targets(episodes, "align_random", 42)
    assert all(label in (0, 1) for label in null[1] + null[3])
    assert row_targets(episodes, "align_true", 42) == row_targets(
        episodes, "align_true", 999)


def test_cpu_forward_and_gradient_reach_learned_alignment_heads() -> None:
    torch.manual_seed(7)
    net = RowHeadTeacher("dense_small")
    episodes, answers = tasks.training_batch(63211, 16, arm="curriculum", step=1800)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long)
    answer_logits, f_logits, g_logits = net(ids)
    assert answer_logits.shape == (16, 45)
    assert f_logits.shape == g_logits.shape == (16, 2)
    loss = total_loss(answer_logits, f_logits, g_logits, episodes,
                      answers, "align_true", 63211, "cpu")
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert net.f_row.weight.grad is not None
    assert net.g_row.weight.grad is not None
    assert net.backbone.embedding.weight.grad is not None
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())


def test_parameter_scale_and_frozen_config() -> None:
    assert RowHeadTeacher("dense_small").parameter_count == 668544
    assert RowHeadTeacher("dense_medium").parameter_count == 14196864
    assert RowHeadTeacher("dense_large").parameter_count == 75582720
    assert 75582720 / 668032 > 100
    Config().validate()
    with pytest.raises(ValueError):
        Config(batch_size=128).validate()


def test_benchmark_gate_refuses_missing_or_stale_file(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="required"):
        check_benchmark(Config(), tmp_path, {"source_git_head": "x",
                                              "source_sha256": {}, "config_sha256": "y"})
    (tmp_path / "benchmark.json").write_text('{"status":"pass"}')
    with pytest.raises(RuntimeError, match="did not qualify"):
        check_benchmark(Config(), tmp_path, {"source_git_head": "x",
                                              "source_sha256": {}, "config_sha256": "y"})


def test_benchmark_clearance_requires_complete_timed_arm_set(tmp_path) -> None:
    config = Config()
    source = {"source_git_head": "frozen", "source_sha256": {"source": "hash"},
              "config_sha256": "config"}
    measured = {arm: {"parameters": EXPECTED_PARAMETERS[arm],
                      "median_timed_step_seconds": .1, "timed_steps": 20,
                      "timed_step_seconds": [.1] * 20,
                      "warmup_steps": 4} for arm in ARMS}
    report = {"experiment": "TEACH-0003", "mode": "benchmark", "status": "pass",
              "config": asdict(config), "source_worktree_status": [], **source,
              "numerical_qualification": {"finite_loss_and_gradients": True,
                                          "max_wrapped_native_logit_error": 0.0},
              "measured": measured, "projected_campaign_seconds": 5000.0,
              "conservative_projected_seconds": 7500.0,
              "peak_sampled_mps_allocated_bytes": 1000, "elapsed_seconds": 50.0}
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(report))
    assert check_benchmark(config, tmp_path, source)["status"] == "pass"
    report["measured"].pop("align_random")
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="did not qualify"):
        check_benchmark(config, tmp_path, source)


def test_both_seed_decision_and_auxiliary_controls() -> None:
    def cell(value: float) -> dict:
        return {"accuracy": value, "group_accuracy": value,
                "f_row_accuracy": value, "g_row_accuracy": value,
                "f_row_pair_accuracy": value, "g_row_pair_accuracy": value}

    names = tasks.evaluation_suite(63311, 8)
    scores = {str(rep): {
        arm: {name: cell(value) for name in names}
        for arm, value in (("dense_small", .2), ("dense_medium", .5),
                           ("dense_large", .95), ("align_true", .95),
                           ("align_random", .2))}
        for rep in range(2)}
    result = decision(scores)
    assert result["dense_scale_qualified"]
    assert not result["alignment_qualified"]  # true alignment ties dense large
    scores["1"]["dense_large"]["composed_holdout_holdout"]["accuracy"] = .4
    scores["1"]["dense_large"]["factorial"]["group_accuracy"] = .4
    scores["0"]["dense_large"]["composed_holdout_holdout"]["accuracy"] = .4
    scores["0"]["dense_large"]["factorial"]["group_accuracy"] = .4
    result = decision(scores)
    assert not result["dense_scale_qualified"]
    assert result["alignment_qualified"]
