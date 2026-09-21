"""Unit tests for EXP-0019 generator-adversary helpers (no full search)."""

from __future__ import annotations

from voynich.generator_adversary import (
    FLOORS,
    Genome,
    HELDOUT_METRICS,
    OPT_METRICS,
    decide_pass,
    metric_verdicts,
    word_h2_bits,
)


def test_genome_repair_caps_mutate_splice():
    g = Genome(window=100, p_cite=1.2, p_mutate_given_cite=0.8, p_splice_given_cite=0.8).repaired()
    assert 5 <= g.window <= 40
    assert 0.30 <= g.p_cite <= 0.95
    assert g.p_mutate_given_cite + g.p_splice_given_cite <= 0.9500001


def test_word_h2_bits_deterministic():
    words = ["a", "b", "a", "b", "a", "c"] * 10
    h = word_h2_bits(words)
    assert h == word_h2_bits(words)
    assert h > 0


def test_metric_verdicts_match_within_floor():
    train = {m: 0.5 for m in OPT_METRICS}
    val = {m: 0.5 for m in OPT_METRICS}
    gen = {m: 0.5 + FLOORS[m] * 0.5 for m in OPT_METRICS}
    out = metric_verdicts(train, val, gen, OPT_METRICS)
    assert out["n_match"] == 4


def test_decide_pass_modes():
    assert decide_pass(3, 2, 3)["passed"] is True
    assert decide_pass(2, 3, 3)["mode"] == "surface_unmatched"
    assert decide_pass(4, 1, 3)["mode"] == "heldout_also_matched"
    assert decide_pass(4, 3, 1)["mode"] == "iid_control_failed"


def test_heldout_metric_names_frozen():
    assert HELDOUT_METRICS == (
        "line_initial_type_share",
        "ab_top50_jaccard",
        "word_h2_bits",
    )
    assert len(OPT_METRICS) == 4
