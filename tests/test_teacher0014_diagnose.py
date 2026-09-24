"""Random-weight parser-gate intervention plumbing; no scientific outcome."""

import random

import pytest

from voynich.workspace.teacher14_diagnose import evaluate_batch, evaluate_episode
from voynich.workspace.teacher14_tasks import RenderSpec, sample_episode
from voynich.workspace.teacher14_train import Config, new_model


def _episode(task="composed"):
    return sample_episode(
        random.Random(140911), signal_hops=2, task=task, distractors=4,
        spec=RenderSpec(1.0, 2, ("prefix", "infix", "suffix")),
        stage_partitions=("train", "train"))


def test_gold_gate_and_native_identity_have_valid_occurrence_controls():
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    episode = _episode()
    result = evaluate_episode(model, episode, device="cpu")
    assert result["row_count"] == len(episode.serialized_rows)
    assert result["candidate_count"] == 2 * result["row_count"] - 1
    assert result["first_target_candidate_index"] % 2 == 0
    assert result["second_target_candidate_index"] % 2 == 0
    assert result["identity_max_abs_logit_error"] < 1e-5
    assert len(result["random_selected_indices"]) == result["row_count"]
    assert len(set(result["random_selected_indices"])) == result["row_count"]
    for condition in result["conditions"].values():
        assert len(condition["logits"]) == 2064
        assert 0 <= condition["first_target_attention"] <= 1
        assert 0 <= condition["second_target_attention"] <= 1
    assert result["conditions"]["gold"]["first_false_attention"] < 1e-4
    assert result["conditions"]["gold"]["second_false_attention"] < 1e-4
    repeated = evaluate_episode(model, episode, device="cpu")
    assert repeated["random_selected_indices"] == result["random_selected_indices"]


def test_diagnostic_rejects_unregistered_task_and_oracle_model():
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    with pytest.raises(ValueError, match="Two-hop composed"):
        evaluate_episode(model, _episode("copy"), device="cpu")
    oracle, _ = new_model(Config(), "oracle_rows_workspace", 0, "cpu")
    with pytest.raises(ValueError, match="Raw candidate-edge"):
        evaluate_episode(oracle, _episode(), device="cpu")


def test_batched_diagnostic_covers_all_read_lengths_and_matches_singleton():
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    long = sample_episode(
        random.Random(140919), signal_hops=4, task="composed",
        distractors=8, spec=RenderSpec(0.0, 3,
                                       ("prefix", "infix", "suffix")))
    episodes = [_episode("composed"), _episode("direct"),
                _episode("copy"), long]
    batched = evaluate_batch(model, episodes, device="cpu")
    assert [len(item["path_candidate_indices"]) for item in batched] == [2, 1, 0, 4]
    for item in batched:
        assert item["identity_max_abs_logit_error"] < 1e-5
        assert len(item["random_selected_indices"]) == item["row_count"]
        for condition in item["conditions"].values():
            assert len(condition["logits"]) == 2064
            assert len(condition["target_attention"]) == len(
                item["path_candidate_indices"])
            assert len(condition["target_rank"]) == len(
                item["path_candidate_indices"])
    assert batched[2]["conditions"]["native"]["first_state_norm"] is None
    singleton = evaluate_batch(model, [episodes[0]], device="cpu")[0]
    assert batched[0]["random_selected_indices"] == singleton["random_selected_indices"]
    for name in singleton["conditions"]:
        assert batched[0]["conditions"][name]["logits"] == pytest.approx(
            singleton["conditions"][name]["logits"], abs=1e-5)
