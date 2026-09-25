"""Verify the new objective changes labels only, not baseline loss or batches."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from voynich.exp0035_model import expected_weighted_bce, selected_batch


def test_expected_loss_matches_hard_baseline_and_soft_expectation() -> None:
    logits = torch.tensor([[0.3, -1.1, 2.2, -0.4]], dtype=torch.float64)
    hard = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    old = F.binary_cross_entropy_with_logits(logits, hard, weight=torch.where(hard < 0.5, 2.0, 1.0))
    torch.testing.assert_close(expected_weighted_bce(logits, hard), old, rtol=0, atol=1e-12)
    soft = torch.tensor([[0.15, 0.45, 0.7, 0.95]], dtype=torch.float64)
    sampled_keep_loss = F.softplus(-logits)
    sampled_null_loss = 2.0 * F.softplus(logits)
    expectation = (soft * sampled_keep_loss + (1 - soft) * sampled_null_loss).mean()
    torch.testing.assert_close(expected_weighted_bce(logits, soft), expectation, rtol=0, atol=1e-12)


def test_selected_batch_replays_original_sampling_order() -> None:
    c_ids = np.arange(20)
    other_ids = np.arange(20, 40)
    seed = 320101
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    from_new = selected_batch(a, c_ids, other_ids)
    from_old = np.concatenate((b.choice(c_ids, size=22, replace=True), b.choice(other_ids, size=10, replace=True)))
    b.shuffle(from_old)
    np.testing.assert_array_equal(from_new, from_old)
