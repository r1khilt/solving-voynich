"""EXP-0024 oracle inverse helpers (no Finnish scoring in unit tests)."""

import numpy as np

from voynich.exact_count_decode import FROZEN_MATCHED_RANDOM_RECON
from voynich.oracle_copy_mutate_inverse import (
    EXPECTED_HOLDOUT_SHA,
    EXP0023_WINNER_RECON,
    ORACLE_NAME,
    decide_mode,
    oracle_inverse_masks,
)


def test_frozen_gate_and_parent_recon_constants():
    assert FROZEN_MATCHED_RANDOM_RECON == 0.2043264147237504
    assert EXP0023_WINNER_RECON == 0.17585403660739804
    assert EXPECTED_HOLDOUT_SHA.startswith("cd72bf0f")
    assert ORACLE_NAME == "oracle_gold_mask_delete"


def test_oracle_uses_stored_mask():
    samples = [
        {"text": "abcd", "mask": [1, 0, 1, 0]},
        {"text": "xy", "mask": [0, 1]},
    ]
    masks, err = oracle_inverse_masks(samples)
    assert err is None
    assert masks is not None
    assert np.array_equal(masks[0], np.array([1, 0, 1, 0]))
    assert np.array_equal(masks[1], np.array([0, 1]))


def test_missing_mask_is_not_invented():
    masks, err = oracle_inverse_masks([{"text": "ab"}])
    assert masks is None
    assert err is not None
    assert "missing stored mask" in err


def test_decision_modes_frozen():
    passed, mode, _ = decide_mode(True, invertible=True)
    assert passed is False
    assert mode == "search_missed_inverse"
    passed, mode, _ = decide_mode(False, invertible=True)
    assert passed is False
    assert mode == "gate_blind_to_copy_inverse"
    passed, mode, _ = decide_mode(False, invertible=False)
    assert passed is False
    assert mode == "channel_not_invertible"
