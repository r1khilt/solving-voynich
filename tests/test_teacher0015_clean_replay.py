"""TEACH-0015 clean-logit replay rejects numerical drift."""

import pytest

from scripts.teacher0015_clean_replay import _compare


def test_clean_replay_vector_matches_prediction_and_tolerance():
    reference = [0.0] * 2064
    reference[50] = 2.0
    actual = reference.copy()
    actual[50] = 2.001
    assert _compare(actual, reference, 50) == pytest.approx(.001)
    actual[50] = 3.0
    with pytest.raises(ValueError, match="checkpoint replay differs"):
        _compare(actual, reference, 50)
    with pytest.raises(ValueError, match="shape differs"):
        _compare(actual[:-1], reference, 50)
