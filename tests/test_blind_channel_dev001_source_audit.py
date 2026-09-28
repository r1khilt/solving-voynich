import math

import pytest

from scripts.audit_blind_channel_dev001_source import (
    probabilities, sha, sufficient_statistics, validation_bits, verified_json,
)


def test_sufficient_statistics_never_join_source_bodies():
    unigram, starts, pairs = sufficient_statistics({"text": "abba", "body_boundaries": [2, 4]}, ("a", "b"))
    assert unigram == {"a": 2, "b": 2}
    assert starts == {"a": 1, "b": 1}
    assert pairs == {("a", "b"): 1, ("b", "a"): 1}


def test_source_probability_and_count_based_score_match_rational_fixture():
    training = sufficient_statistics({"text": "aab", "body_boundaries": [3]}, ("a", "b"))
    model = probabilities(training, ("a", "b"), 1, 2.)
    assert model == {"": {"a": 5 / 8, "b": 3 / 8}, "a": {"a": 9 / 16, "b": 7 / 16},
                     "b": {"a": 5 / 8, "b": 3 / 8}}
    validation = sufficient_statistics({"text": "aab", "body_boundaries": [2, 3]}, ("a", "b"))
    assert validation_bits(model, validation, 1) == pytest.approx(-math.log2(5 / 8 * 9 / 16 * 3 / 8))
    assert validation_bits(model, validation, 0) == pytest.approx(-math.log2((5 / 8)**2 * 3 / 8))


def test_source_bytes_are_hash_checked_before_json_decode(tmp_path):
    raw = b"deliberately not json"
    (tmp_path / "fixture.json").write_bytes(raw)
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_json(tmp_path, {"path": "fixture.json", "sha256": "0" * 64})
    assert sha(raw) != "0" * 64
