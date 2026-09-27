"""Toy-only integration checks for the independent final-block auditor."""

import copy
import itertools
from pathlib import Path

import pytest

from scripts import audit_naibbe003 as auditor
from scripts.audit_naibbe001 import IndependentLM
from scripts.audit_naibbe003 import (
    audit_prediction, compare_fit_freeze, decision, decoded_candidates,
    independent_metrics, model, self_test,
)


@pytest.mark.parametrize("config", [
    {"family": "legacy", "parameter": None},
    {"family": "dirichlet", "parameter": 16.},
    {"family": "absolute_discount", "parameter": .75},
])
def test_every_prior_replays_optimal_toy_and_rejects_corruption(config):
    data = {"alphabet": ["a", "b"], "class_ids": ["A", "B", "C", "D"],
            "groups": [["A", "B"], ["C", "D"]],
            "split": {"candidates": [[["A"], ["C"]], [["A", "D"]], [["B"]]]}}
    gold_key, key = [0, 1, 0, 1], [0, 1, 1, 0]
    gold = ["a", "ab", "b"]
    lm = model("aaaaabaabaaabbaa" * 7, "ab", config)
    candidates = decoded_candidates(data, key)
    choices = max(itertools.product(*(range(len(row)) for row in candidates)),
                  key=lambda path: lm.score("".join(row[i] for row, i in zip(candidates, path))))
    chunks = [row[i] for row, i in zip(candidates, choices)]
    prediction = {"choices": list(choices), "chunks": chunks}
    score = lm.score("".join(chunks))
    oracle_score = lm.best_lattice_score(decoded_candidates(data, gold_key))
    metrics, _ = independent_metrics(chunks, gold, data, key, gold_key)
    saved = {**metrics, "score": score, "gold_key_score": oracle_score,
             "gold_minus_learned_objective": oracle_score - score}
    actual, summary, delta = audit_prediction(prediction, data, key, gold_key, gold, lm,
                                              saved, "toy", oracle_score)
    assert actual == metrics and delta == 0.
    assert summary["unresolved_class_characters"] == 1
    assert "key_errors" not in summary
    for field in ("score", "gold_key_score", "gold_minus_learned_objective", "edit_distance"):
        wrong = {**saved, field: saved[field] + 1}
        with pytest.raises(AssertionError):
            audit_prediction(prediction, data, key, gold_key, gold, lm, wrong, "toy", oracle_score)
    corrupt = copy.deepcopy(prediction)
    corrupt["chunks"][0] = "b" if chunks[0] == "a" else "a"
    with pytest.raises(AssertionError, match="inconsistency"):
        audit_prediction(corrupt, data, key, gold_key, gold, lm, saved, "toy", oracle_score)
    # A legal but inferior parse must fail even if all its saved metrics agree.
    inferior = copy.deepcopy(prediction)
    inferior["choices"][0] = 1 - inferior["choices"][0]
    inferior["chunks"][0] = candidates[0][inferior["choices"][0]]
    assert lm.score("".join(inferior["chunks"])) < score
    with pytest.raises(AssertionError, match="Nonoptimal"):
        audit_prediction(inferior, data, key, gold_key, gold, lm, saved, "toy", oracle_score)
    oracle_candidates = decoded_candidates(data, gold_key)
    oracle_chunks = [row[0] for row in oracle_candidates]
    oracle_metrics, _ = independent_metrics(oracle_chunks, gold, data, gold_key, gold_key)
    oracle_saved = {**oracle_metrics, "score": oracle_score}
    audit_prediction({"choices": [0, 0, 0], "chunks": oracle_chunks}, data, gold_key, gold_key,
                     gold, lm, oracle_saved, "oracle_toy")


def test_legacy_adapter_matches_earlier_independent_model_and_resets_context():
    corpus, alphabet = "abacabadabacaba" * 5, "abcd"
    old = IndependentLM(corpus, alphabet)
    current = model(corpus, alphabet, {"family": "legacy", "parameter": None})
    for length in range(1, 5):
        for letters in itertools.product(alphabet, repeat=length):
            gram = "".join(letters)
            assert current.log_next(gram) == pytest.approx(old.log_next(gram), abs=1e-14)
    before = current.score("ddbac")
    current.score("abba" * 20)
    assert current.score("ddbac") == before
    assert current.score("ddbac") == pytest.approx(old.score("ddbac"), abs=1e-12)


def test_frozen_result_checks_keys_configuration_and_trace_incumbents():
    fit = {"arm": "latin_joint", "fit_score": -3., "seconds": 1., "budget_stop": False,
           "key": [0, 1], "input_sha256": "fit", "lm_sha256": "source", "source_commit": "revision",
           "seed": 920201, "source_hashes": {}, "environment": {},
           "trace": [{"score": -4., "best_score": -4.}, {"score": -3., "best_score": -3.}],
           "coordinated": True, "restarts": 8, "kicks": 6}
    freeze = {**fit, "result_sha256": "checked_elsewhere"}
    compare_fit_freeze(fit, freeze)
    wrong = {**freeze, "key": [1, 0]}
    with pytest.raises(AssertionError, match="disagreement"):
        compare_fit_freeze(fit, wrong)
    wrong = {key: value for key, value in freeze.items() if key != "input_sha256"}
    with pytest.raises(AssertionError, match="Incomplete"):
        compare_fit_freeze(fit, wrong)
    wrong = copy.deepcopy(fit)
    wrong["trace"][1]["best_score"] = -4.
    with pytest.raises(AssertionError, match="incumbent"):
        compare_fit_freeze(wrong, {**wrong, "result_sha256": "unused"})


def test_registered_gate_is_conjunctive_and_boundaries_are_inclusive():
    primary = {"cer": .02, "key_macro_accuracy": .95, "unique_class_weighted_key_accuracy": .99}
    assert decision(primary, {"cer": .015})[1] == "PASS"
    for field, value in (("cer", .020001), ("key_macro_accuracy", .94999),
                         ("unique_class_weighted_key_accuracy", .98999)):
        assert decision({**primary, field: value}, {"cer": .015})[1] == "FAIL"
    assert decision(primary, {"cer": .01499})[1] == "FAIL"
    assert decision({**primary, "cer": 0., "key_macro_accuracy": 130 / 138}, {"cer": 0.})[1] == "FAIL"


def test_self_test_never_opens_experiment_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Self-test attempted file access")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = self_test()
    assert result["self_test_pass"] and result["experiment_files_read"] is False
    assert result["source_configurations_checked"] == 11
    assert result["source_family_exhaustive_paths"] == 176


def test_post_evaluation_mode_refuses_early_answer_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Attempted answer or other file access before completed evaluation")

    monkeypatch.setattr(auditor, "ROOT", tmp_path)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(AssertionError, match="Complete evaluation"):
        auditor.audit()
