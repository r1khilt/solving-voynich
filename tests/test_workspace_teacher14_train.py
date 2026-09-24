"""Draft trainer routing and closed launch gate; no optimizer update occurs."""

from dataclasses import replace
import random
import time

import pytest
import torch

from voynich.workspace import teacher14_train as campaign
from voynich.workspace.teacher14_tasks import RenderSpec, sample_episode


def _development_batch():
    rng = random.Random(84199)
    episodes = [sample_episode(
        rng, signal_hops=2, task="composed", distractors=count,
        spec=RenderSpec(.25, 1, ("prefix", "infix", "suffix")),
        stage_partitions=("train", "train")) for count in (0, 1)]
    return episodes, [episode.answer for episode in episodes]


def test_all_registered_arms_have_exact_parameter_expectations():
    assert len(campaign.ARMS) == 11
    for arm in campaign.ARMS:
        model, _ = campaign.new_model(campaign.Config(), arm, 0, "cpu")
        assert model.parameter_count == campaign.EXPECTED_PARAMETERS[arm]
        del model


def test_draft_branches_compute_finite_gradients_without_weight_update(monkeypatch):
    episodes, answers = _development_batch()
    monkeypatch.setattr(campaign, "training_batch",
                        lambda *args, **kwargs: (episodes, answers))
    records = {}
    for arm in ("oracle_rows_workspace", "latent_rows_edge_aux",
                "latent_rows_causal", "latent_rows_wrong_causal",
                "latent_rows_recurrent4", "latent_rows_diffuse"):
        model, optimizer = campaign.new_model(campaign.Config(), arm, 0, "cpu")
        optimizer.step = lambda: None
        parts = campaign.one_update(model, optimizer, campaign.Config(), 0,
                                    arm, 0, "cpu")
        assert parts["step"] == 0
        assert all(torch.isfinite(torch.tensor(value))
                   for value in parts["losses"].values())
        assert len(parts["answer_input_sha256"]) == 64
        assert torch.isfinite(torch.tensor(parts["gradient_norm"]))
        assert any(parameter.grad is not None for parameter in model.parameters())
        records[arm] = parts
        del model, optimizer
    assert len({row["answer_input_sha256"] for row in records.values()}) == 1
    assert len({row["answer_label_sha256"] for row in records.values()}) == 1
    assert records["latent_rows_causal"]["causal_input_sha256"] == (
        records["latent_rows_wrong_causal"]["causal_input_sha256"])


def test_benchmark_and_campaign_remain_closed_before_full_auditor(tmp_path):
    config = campaign.Config()
    assert not campaign.LAUNCH_ADMITTED
    with pytest.raises(RuntimeError, match="benchmark closed"):
        campaign.benchmark(config, tmp_path / "results", tmp_path / "outputs")
    with pytest.raises(RuntimeError, match="launch closed"):
        campaign.run(config, tmp_path / "results", tmp_path / "outputs")
    assert not (tmp_path / "results").exists()


def test_resource_cap_counts_checkpoint_and_result_directories(tmp_path):
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "outputs"
    result_dir.mkdir()
    output_dir.mkdir()
    (result_dir / "result.bin").write_bytes(b"123456")
    (output_dir / "checkpoint.bin").write_bytes(b"123456")
    config = replace(campaign.Config(), max_artifact_bytes=10)
    with pytest.raises(campaign.ResourceStop, match="artifact ceiling"):
        campaign.resource_check(
            config, time.monotonic(), {"peak_sampled_mps_allocated_bytes": 0},
            output_dir, result_dir, benchmark=False)
