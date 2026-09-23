"""CPU-only prelaunch checks for the frozen TEACH-0004 architecture study."""

from dataclasses import asdict
import json

import pytest
import torch

from voynich.workspace import teacher4_tasks as tasks
from voynich.workspace.teacher2_train import task_loss
from voynich.workspace.teacher4_models import model_for_arm
from voynich.workspace.teacher4_train import (
    ARMS, EXPECTED_PARAMETERS, Config, check_benchmark, decision, new_model, one_update,
)


def test_fresh_generator_oracle_pairs_and_factorial():
    suite = tasks.evaluation_suite(64311, 8)
    assert tasks.SPLIT_NAMESPACE == b"TEACH-0004-v1/"
    assert tasks.SPLIT_NAMESPACE != b"TEACH-0003-v1/"
    assert tasks.evaluation_suite(64311, 8) == suite
    for episodes in suite.values():
        for ep in episodes:
            assert tasks.symbolic_oracle(ep.tokens) == ep.answer
    for name in ("first_hop_pairs_holdout", "direct_pairs_holdout"):
        for first, second in zip(suite[name][::2], suite[name][1::2], strict=True):
            assert first.tokens[:12] == second.tokens[:12]
            assert first.tokens[12] != second.tokens[12]
            assert first.answer != second.answer
    for offset in range(0, len(suite["factorial"]), 4):
        quartet = suite["factorial"][offset:offset+4]
        assert len({ep.answer for ep in quartet}) == 4
        assert all(ep.f_partition == ep.g_partition == "holdout" for ep in quartet)


def test_shared_row_interface_and_learned_composed_path():
    torch.manual_seed(17)
    two = model_for_arm("two_read")
    one = model_for_arm("one_read")
    dense = model_for_arm("dense_row")
    assert sum(p.numel() for p in two.parameters()) == sum(p.numel() for p in one.parameters())
    episodes = [tasks.make_episode((9, 10), (21, 22), (22, 21), (33, 34),
                                   "composed", 9)]
    ids = torch.tensor([ep.tokens for ep in episodes])
    for net in (two, one, dense):
        logits, first, second = net(ids)
        assert logits.shape == (1, 45) and bool(torch.isfinite(logits).all())
        if net is dense:
            assert first is second is None
        else:
            assert first.shape == second.shape == (1, 2)
            torch.testing.assert_close(first.sum(-1), torch.ones(1))
            torch.testing.assert_close(second.sum(-1), torch.ones(1))
        loss = task_loss(logits, episodes, [episodes[0].answer], "cpu")
        loss.backward()
        assert net.interface.symbol.weight.grad is not None
        assert bool(torch.isfinite(net.interface.symbol.weight.grad).all())
    assert two.f_query.weight.grad.abs().sum() > 0
    assert one.f_query.weight.grad.abs().sum() == 0


def test_complete_row_swaps_are_invariant_without_symbolic_equality():
    torch.manual_seed(31)
    ep = tasks.make_episode((9, 10), (21, 22), (22, 21), (33, 34), "composed", 9)
    original = torch.tensor([ep.tokens])
    swapped = list(ep.tokens)
    swapped[2:6] = [*swapped[4:6], *swapped[2:4]]
    swapped[7:11] = [*swapped[9:11], *swapped[7:9]]
    changed = torch.tensor([swapped])
    assert tasks.symbolic_oracle(tuple(swapped)) == ep.answer
    for arm in ARMS:
        net = model_for_arm(arm).eval()
        with torch.no_grad():
            a = net(original)[0]
            b = net(changed)[0]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_one_cpu_update_and_frozen_benchmark_gate(tmp_path):
    config = Config()
    config.validate()
    with pytest.raises(ValueError):
        Config(batch_size=128).validate()
    net, optimizer = new_model(config, "two_read", 0, "cpu")
    loss = one_update(net, optimizer, config, 0, 0, "cpu")
    assert loss >= 0
    with pytest.raises(RuntimeError, match="required"):
        check_benchmark(config, tmp_path, {"source_git_head": "frozen",
                                           "source_sha256": {}, "config_sha256": "hash",
                                           "source_worktree_status": []})
    (tmp_path/"benchmark.json").write_text(json.dumps({"status": "pass",
                                                   "config": asdict(config)}))
    with pytest.raises(RuntimeError, match="did not qualify"):
        check_benchmark(config, tmp_path, {"source_git_head": "frozen",
                                           "source_sha256": {}, "config_sha256": "hash",
                                           "source_worktree_status": []})


def test_source_matched_projection_and_parameter_gate(tmp_path):
    config = Config()
    provenance = {"source_git_head": "frozen", "source_sha256": {"x": "hash"},
                  "config_sha256": "config", "source_worktree_status": []}
    measured = {arm: {"parameters": EXPECTED_PARAMETERS[arm],
                      "median_timed_step_seconds": .05,
                      "timed_step_seconds": [.05]*20, "timed_steps": 20,
                      "warmup_steps": 4} for arm in ARMS}
    report = {"experiment": "TEACH-0004", "mode": "benchmark", "status": "pass",
              "config": asdict(config), **provenance,
              "measured": measured,
              "numerical_qualification": {"finite_loss_and_gradients": True,
                                          "max_recompute_logit_error": 0.0},
              "projected_campaign_seconds": 1500.0,
              "conservative_projected_seconds": 2250.0,
              "peak_sampled_mps_allocated_bytes": 1000,
              "elapsed_seconds": 10.0}
    path = tmp_path/"benchmark.json"
    path.write_text(json.dumps(report))
    assert check_benchmark(config, tmp_path, provenance)["status"] == "pass"
    report["measured"]["one_read"]["parameters"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="did not qualify"):
        check_benchmark(config, tmp_path, provenance)


def test_both_seed_architecture_gain_is_required():
    def cell(value):
        return {name: {"accuracy": value, "group_accuracy": value}
                for name in tasks.evaluation_suite(64311, 8)}

    scores = {rep: {"two_read": cell(.95), "one_read": cell(.50),
                    "dense_row": cell(.50)} for rep in ("0", "1")}
    assert decision(scores)["verdict"] == "qualified"
    scores["1"]["dense_row"]["factorial"]["group_accuracy"] = .90
    result = decision(scores)
    assert result["verdict"] == "not_qualified"
    assert not result["clauses_by_seed"]["1"]["factorial_gain_over_dense_row_at_least_0.15"]
