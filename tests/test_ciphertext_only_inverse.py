"""Unit tests for EXP-0026 ciphertext-only inverse helpers."""

from __future__ import annotations

import numpy as np

from voynich.ciphertext_only_inverse import (
    EXPERIMENT_ID,
    OPS,
    apply_mask_ops_ct,
    enumerate_programs,
    repetition_splice_keep_scores,
)


def test_experiment_id_and_ops_include_splice_detector():
    assert EXPERIMENT_ID == "EXP-0026"
    assert "repetition_splice_detector" in OPS
    assert "exact_count_neural" in OPS


def test_repetition_splice_scores_prefer_novel_chars():
    text = "aaaa bbbb"
    scores = repetition_splice_keep_scores(text)
    assert scores.shape == (len(text),)
    # Later 'a's that copy recent should score lower than a fresh letter would.
    assert scores[3] < scores[0] or scores[1] < 0


def test_apply_mask_ops_ct_no_mask_arg_and_exact_rate():
    text = "abcabcabcabc" + ("x" * 116)
    assert len(text) == 128
    mask = apply_mask_ops_ct(text, ("repetition_splice_detector",), None)
    assert mask.shape == (128,)
    assert set(np.unique(mask)).issubset({0, 1})
    n_keep = int(round(0.70 * 128))
    assert int(mask.sum()) == n_keep


def test_enumerate_budget():
    progs = enumerate_programs(4026)
    assert len(progs) <= 200
    assert ("repetition_splice_detector",) in progs
