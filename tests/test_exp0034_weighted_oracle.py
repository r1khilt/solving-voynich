"""Exact local channel probabilities, padding and crop-boundary checks."""

from __future__ import annotations

import math

from scripts.audit_exp0034_weighted_oracle import explicit_posterior
from voynich.exp0034_weighted_oracle import emission_prob, weighted_posterior


def test_copy_mutate_emission_law() -> None:
    one_a, pair_aa, cross_a = emission_prob("a", "ab", "aa", 2)
    one_b, pair_ba, cross_b = emission_prob("a", "ab", "ba", 2)
    _one, pair_ab, _cross = emission_prob("a", "ab", "ab", 2)
    assert math.isclose(one_a, 0.5)
    assert math.isclose(one_b, 0.1)
    assert math.isclose(pair_aa, 0.2)
    assert math.isclose(pair_ab, 0.1)
    assert math.isclose(pair_ba, 0.1)
    assert math.isclose(cross_a, 0.3)
    assert math.isclose(cross_b, 0.1)
    assert math.isclose(one_a + one_b + pair_aa + pair_ab + pair_ba, 1.0)
    assert math.isclose(emission_prob("a", "ab", "a", 1)[0], 0.8)


def test_weighted_posterior_agrees_with_explicit_branch_auditor_on_short_padding() -> None:
    source = "ababa"
    observed = "aaababa" + " " * 121
    gold = [0, 0, 1, 1, 1, 1, 1] + [1] * 121
    result = weighted_posterior(observed, source, "ab", 126, gold)
    independent = explicit_posterior(observed, source, "ab", 126, gold)
    assert result["generated_length"] == 7
    assert result["posterior_mask_hex"] == independent["mask_hex"]
    assert abs(result["log_evidence"] - independent["log_evidence"]) < 1e-10
    assert abs(result["gold_mask_posterior"] - independent["gold_mask_posterior"]) < 1e-10
    assert max(abs(a - b) for a, b in zip(result["posterior_keep_probabilities"], independent["keep"], strict=True)) < 1e-10


def test_weighted_posterior_agrees_at_cropped_prefix() -> None:
    source = "a" * 100
    observed = "a" * 128
    gold = [1] * 100 + [0] * 28
    result = weighted_posterior(observed, source, "ab", 100, gold)
    independent = explicit_posterior(observed, source, "ab", 100, gold)
    assert result["generated_length"] == 143
    assert abs(result["log_evidence"] - independent["log_evidence"]) < 1e-9
    assert abs(result["gold_mask_posterior"] - independent["gold_mask_posterior"]) < 1e-9
    assert max(abs(a - b) for a, b in zip(result["posterior_keep_probabilities"], independent["keep"], strict=True)) < 1e-9
