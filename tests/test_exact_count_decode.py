"""Decode-only exact-count / joint keep-run unit tests (no holdout scores)."""

import numpy as np

from voynich.exact_count_decode import (
    FROZEN_MATCHED_RANDOM_RECON,
    apply_pass_rule_v14,
    exact_count_keep_masks,
    joint_keep_run_masks,
)


def test_exact_count_keep_count_and_ties():
    probs = [np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])]
    masks = exact_count_keep_masks(probs, filler_rate=0.30)
    assert masks[0].sum() == 7
    ties = exact_count_keep_masks([np.array([0.5, 0.5, 0.5, 0.5])], filler_rate=0.50)
    assert ties[0].tolist() == [1, 1, 0, 0]


def test_joint_keep_run_exact_count():
    rng = np.random.default_rng(0)
    probs = [rng.random(32) for _ in range(5)]
    masks = joint_keep_run_masks(probs, filler_rate=0.30)
    for p, m in zip(probs, masks):
        n_keep = int(round(0.70 * len(p)))
        assert int(m.sum()) == n_keep
        assert m.shape == p.shape


def test_pass_rule_v14_uses_frozen_baseline():
    neural = {
        "null_recall": 0.6,
        "null_precision": 0.6,
        "pred_null_rate": 0.3,
        "recon_acc": 0.2044,
    }
    vocab = {"mask_f1": 0.4}
    decision = apply_pass_rule_v14(neural, vocab)
    assert decision["passed"] is True
    neural_fail = dict(neural, recon_acc=0.2043)
    assert apply_pass_rule_v14(neural_fail, vocab)["passed"] is False
    assert FROZEN_MATCHED_RANDOM_RECON == 0.2043264147237504
