"""Diagnostic replay tolerance rejects altered intervention logits."""

import pytest

from scripts.teacher0014_diagnostic_replay import _compare


def test_intervention_vector_replay_tolerance():
    reference = [0.0] * 2064
    observed = reference.copy()
    observed[27] = .001
    assert _compare(observed, reference) == pytest.approx(.001)
    observed[27] = .1
    with pytest.raises(ValueError, match="replay differs"):
        _compare(observed, reference)
    with pytest.raises(ValueError, match="shape mismatch"):
        _compare(observed[:-1], reference)
