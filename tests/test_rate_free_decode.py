"""Unit tests for EXP-0021 rate-free threshold decode (no Voynich scores)."""

import numpy as np

from voynich.rate_free_decode import (
    REJECTED_RATE_MEDIAN_HI,
    REJECTED_RATE_MEDIAN_LO,
    SIGNAL_THRESHOLD,
    apply_pass_rule,
    implied_null_rates,
    rate_distribution,
    threshold_masks,
)


def test_threshold_masks_are_rate_free_not_exact_count():
    probs = [
        np.array([0.9, 0.1, 0.6, 0.4, 0.55]),
        np.array([0.49, 0.50, 0.51]),
    ]
    masks = threshold_masks(probs, SIGNAL_THRESHOLD)
    np.testing.assert_array_equal(masks[0], np.array([1, 0, 1, 0, 1]))
    np.testing.assert_array_equal(masks[1], np.array([0, 1, 1]))
    # Different windows imply different null rates (not a global exact count).
    rates = implied_null_rates(masks, [5, 3])
    assert abs(rates[0] - 0.4) < 1e-12
    assert abs(rates[1] - 1 / 3) < 1e-12


def test_pass_rule_requires_positive_gain_and_rate_spread():
    good = {
        "mean_gain": 0.02,
        "mean_random_gain": -0.01,
        "fraction_beats_random": 0.55,
        "implied_null_rate_distribution": {
            "min": 0.1,
            "median": 0.30,
            "mean": 0.31,
            "max": 0.5,
            "std": 0.12,
            "n_windows": 10,
        },
    }
    assert apply_pass_rule(good)["passed"] is True

    neg_gain = dict(good, mean_gain=-0.01)
    d = apply_pass_rule(neg_gain)
    assert d["passed"] is False
    assert d["criteria"]["mean_gain_gt_0"] is False

    collapse = dict(
        good,
        implied_null_rate_distribution={
            "min": 0.69,
            "median": 0.70,
            "mean": 0.70,
            "max": 0.71,
            "std": 0.01,
            "n_windows": 10,
        },
    )
    d2 = apply_pass_rule(collapse)
    assert d2["passed"] is False
    assert d2["criteria"]["implied_null_rate_std_gt_0_05"] is False
    assert d2["criteria"]["median_implied_null_rate_outside_0_65_0_75"] is False
    assert REJECTED_RATE_MEDIAN_LO <= 0.70 <= REJECTED_RATE_MEDIAN_HI


def test_rate_distribution_stats():
    rates = np.array([0.1, 0.2, 0.3, 0.4])
    dist = rate_distribution(rates)
    assert dist["n_windows"] == 4
    assert abs(dist["min"] - 0.1) < 1e-12
    assert abs(dist["max"] - 0.4) < 1e-12
    assert abs(dist["median"] - 0.25) < 1e-12
    assert abs(dist["mean"] - 0.25) < 1e-12
    assert dist["std"] > 0.0
