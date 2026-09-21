"""Evaluator controls: gold cannot choose a candidate or redefine metric denominators."""

from types import SimpleNamespace

import pytest

import voynich.communication.evaluation as evaluation
from voynich.communication.evaluation import (
    aggregate_scores,
    baseline_hypothesis,
    evaluate_joint,
    normalized_levenshtein,
    score_selected,
)
from voynich.communication.pipeline import make_episode
from voynich.communication.schema import Anchor, JointLayout, Observation
from voynich.communication.worlds import equivalent_entity_renaming


def _hypothesis(world):
    return {"inverse_key": world.inverse_key, "keep": world.keep, "boundaries": world.word_boundary,
            "grammar": world.config.grammar, "morphology": world.config.morphology, "family": world.config.family}


def test_rank_zero_is_selected_even_when_later_candidate_matches_gold():
    episode = make_episode(0, "validation", JointLayout(24, 80), seed=91, family="procedure")
    alternate = equivalent_entity_renaming(episode.world, (3, 2, 1, 0))
    result = {"status": "candidate_hypotheses", "candidates": [
        {"id": "rank-zero-alternative", "hypothesis": _hypothesis(alternate), "verification": {"valid": True}},
        {"id": "exact-gold-rank-one", "hypothesis": _hypothesis(episode.world), "verification": {"valid": True}},
    ]}
    score = score_selected(episode, result)
    assert score["selected_id"] == "rank-zero-alternative"
    assert score["selected_rank"] == 0
    assert score["observed_unanchored_key_accuracy"] < 1
    assert score["decoded_similarity"] < 1
    assert score["keep_f1"] == 1
    assert score["counts"]["valid_candidates"] == 2


def test_anchors_and_unobserved_keys_are_excluded_from_unanchored_key_denominator():
    episode = make_episode(8, "validation", JointLayout(24, 80), anchor_count=2, seed=79)
    observation = episode.observation
    anchored = {anchor.observed for anchor in observation.anchors}
    key = list(episode.world.inverse_key)
    for symbol in set(range(2, 26)) - set(observation.symbols):
        key[symbol - 2] = 0
    # An incorrect explicit anchor is a verifier error; it must not leak into the
    # separate metric that specifically excludes anchored positions.
    for symbol in anchored:
        key[symbol - 2] = 0
    hypothesis = {**_hypothesis(episode.world), "inverse_key": tuple(key)}
    result = {"status": "abstain", "candidates": [{"id": "candidate", "hypothesis": hypothesis,
                                                    "verification": {"valid": False}}]}
    score = score_selected(episode, result)
    assert score["counts"]["observed_unanchored_key_total"] == len(set(observation.symbols) - anchored)
    assert score["observed_unanchored_key_accuracy"] == 1
    assert score["abstained"]


def _score_row(key_correct, key_total, keep, boundary, valid_candidates, candidates, similarity):
    return {"counts": {"observed_unanchored_key_correct": key_correct,
                       "observed_unanchored_key_total": key_total, "nonnull_key_correct": key_correct,
                       "nonnull_key_total": key_total, "keep": keep, "boundary": boundary,
                       "valid_candidates": valid_candidates, "candidates": candidates},
            "decoded_similarity": similarity, "grammar_accuracy": 0, "morphology_accuracy": 1,
            "family_accuracy": 0, "selected_valid": bool(valid_candidates), "abstained": not valid_candidates}


def test_aggregate_uses_correct_micro_denominators_not_mean_of_ratios():
    rows = [_score_row(1, 1, [1, 0, 0], [1, 0, 0], 1, 2, 1),
            _score_row(0, 9, [1, 3, 1], [0, 1, 1], 0, 1, 0)]
    result = aggregate_scores(rows)
    assert result["observed_unanchored_key_accuracy"] == 0.1
    assert result["nonnull_key_accuracy"] == 0.1
    assert result["keep_f1"] == 0.5
    assert result["boundary_f1"] == 0.5
    assert result["valid_candidate_fraction"] == pytest.approx(1 / 3)
    assert result["decoded_similarity"] == 0.5
    assert result["abstained_fraction"] == 0.5
    empty_keys = aggregate_scores([_score_row(0, 0, [0, 0, 0], [0, 0, 0], 0, 0, 0)])
    assert empty_keys["nonnull_key_accuracy"] is None
    assert empty_keys["valid_candidate_fraction"] is None


def test_levenshtein_normalization_handles_insertions_substitutions_and_empty_sequences():
    assert normalized_levenshtein((), ()) == 1
    assert normalized_levenshtein((), (2, 3)) == 0
    assert normalized_levenshtein((2, 3, 4), (2, 4)) == pytest.approx(2 / 3)
    assert normalized_levenshtein((2, 3), (4, 5)) == 0


def test_baselines_use_only_observations_and_preserve_anchors_and_unknown_slots():
    observation = Observation("x", (2, 3, 4, 2), 24, (Anchor(2, 10, "declared-correspondence"),), "text")
    for kind in ("random_key", "all_null"):
        baseline = baseline_hypothesis(observation, seed=41, kind=kind)
        assert baseline == baseline_hypothesis(observation, seed=41, kind=kind)
        assert baseline["inverse_key"][0] == 10
        assert baseline["morphology"] in ("prefix", "suffix")
        assert set(baseline["unresolved_key_symbols"]) == set(range(5, 26))
        assert all(baseline["inverse_key"][i - 2] == 0 for i in range(5, 26))
        assert baseline["keep"] == tuple(baseline["inverse_key"][x - 2] != 0 for x in observation.symbols)
    assert baseline_hypothesis(observation, seed=41, kind="all_null")["keep"] == (True, False, False, True)


def test_end_to_end_evaluation_is_reproducible_and_validation_only(monkeypatch):
    received = []

    def fake_infer(model, observation, layout, **kwargs):
        assert not hasattr(observation, "world")
        assert not hasattr(observation, "seed")
        received.append(observation.split)
        hypothesis = baseline_hypothesis(observation, seed=kwargs["seed"], kind="all_null")
        return {"status": "abstain", "candidates": [{"id": "first", "hypothesis": hypothesis,
                                                       "verification": {"valid": False}}]}

    monkeypatch.setattr(evaluation, "infer", fake_infer)
    arguments = dict(seed=49, count=3, anchor_count=2, candidates=2, steps=2)
    left = evaluate_joint(SimpleNamespace(), JointLayout(24, 80), **arguments)
    right = evaluate_joint(SimpleNamespace(), JointLayout(24, 80), **arguments)
    assert left == right
    assert received == ["validation"] * 6
    assert set(left["metrics"]) == {"neural", "random_key", "all_null_except_anchors"}
    assert left["metrics"]["neural"]["abstained_fraction"] == 1
    assert all(row["anchor_count"] == 2 for row in left["worlds"])
    with pytest.raises(TypeError):
        evaluate_joint(SimpleNamespace(), JointLayout(24, 80), split="test")


def test_empty_candidate_set_is_abstention_not_a_correct_null_key():
    episode = make_episode(2, "validation", JointLayout(24, 80))
    score = score_selected(episode, {"status": "abstain", "candidates": []})
    assert score["selected_id"] is None
    assert score["observed_unanchored_key_accuracy"] == 0
    assert score["decoded_similarity"] == 0
    assert score["keep_f1"] == 0
