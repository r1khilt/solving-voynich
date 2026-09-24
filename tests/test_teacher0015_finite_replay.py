"""Numerical integrity of sampled finite checkpoint replay."""

import pytest

from scripts.teacher0015_finite_replay import _compare


def test_sampled_replay_accepts_rounding_and_rejects_drift():
    assert _compare([1.0001, -2.0001], [1.0, -2.0],
                    label="fixture") == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="checkpoint replay differs"):
        _compare([1.1, -2.0], [1.0, -2.0], label="fixture")
    with pytest.raises(ValueError, match="shape/finite"):
        _compare([1.0], [1.0, 2.0], label="fixture")
