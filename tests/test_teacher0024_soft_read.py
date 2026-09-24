"""Finite soft-read diagnostic invariants."""

import json
import math
from pathlib import Path

import pytest
import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0024_soft_read_audit import _score, entropy, visible_path
from scripts.teacher0024_soft_read_run import CONDITIONS, evaluate_batch
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace


ROOT = Path(__file__).resolve().parents[1]


def test_entropy_and_attention_mass():
    normalized, effective = entropy([.25] * 4)
    assert normalized == pytest.approx(1)
    assert effective == pytest.approx(4)
    assert entropy([1, 0, 0, 0]) == pytest.approx((0, 1))
    with pytest.raises(ValueError, match="sum"):
        entropy([.2, .2])


def test_random_weight_identity_and_visible_hard_row():
    suite = json.loads((ROOT / "outputs/TEACH-0016/teach14-74111.json").read_text())
    raw = suite["panels"]["composed_confirm_confirm"][0]
    episode = episode_from_json(raw)
    model = CandidateEdgeWorkspace(oracle_rows=True).eval()
    with torch.no_grad():
        rows, logits = evaluate_batch(model, [episode])
    row = rows[0]
    visible, path = visible_path(raw)
    assert row["target_rows"] == path
    assert len(row["attention"]) == 2
    assert all(len(values) == len(visible) for values in row["attention"])
    assert set(row["predictions"]) == set(CONDITIONS)
    assert logits["clean"] == logits["identity"]


def test_paired_score_keeps_unfiltered_denominators():
    first = {"answer": 17, "target_rows": [0, 1],
             "native_top_rows": [0, 2],
             "attention": [[.6, .2, .2], [.1, .2, .7]],
             "predictions": {name: 18 for name in CONDITIONS}}
    first["predictions"]["hard_first"] = 17
    second = {"answer": 19, "target_rows": [0, 1],
              "native_top_rows": [2, 1],
              "attention": [[.2, .2, .6], [.2, .7, .1]],
              "predictions": {name: 19 for name in CONDITIONS}}
    second["predictions"]["hard_first"] = 18
    score = _score([first, second])
    assert score["clean_errors"] == 1
    assert score["first_correct_clean_errors"] == 1
    assert score["conditions"]["hard_first"]["rescued_clean_errors"] == 1
    assert score["conditions"]["hard_first"]["damaged_clean_correct"] == 1
    assert math.isfinite(score["attention_by_clean"]["clean_wrong"]["0"][
        "effective_rows_mean"])
