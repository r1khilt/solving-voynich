"""Visible-path and intervention invariants for TEACH-0021."""

import json
from pathlib import Path

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0021_reader_audit import _score, visible_path
from scripts.teacher0021_reader_run import evaluate_batch, target_rows
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace, public_row_tensors
from voynich.workspace.teacher14_train import padded_tokens


ROOT = Path(__file__).resolve().parents[1]


def _episode(panel: str):
    manifest = json.loads((ROOT / "outputs/TEACH-0016/teach14-74111.json").read_text())
    return episode_from_json(manifest["panels"][panel][0])


def test_visible_path_matches_independent_tensor_path():
    episode = _episode("composed_confirm_confirm")
    ids = padded_tokens([episode], "cpu")
    left, right, mask = public_row_tensors(ids)
    indices, symbols = target_rows(left, right, mask, [episode.query], 2,
                                   [episode.answer])
    assert (indices[0], symbols[0]) == visible_path(list(episode.tokens), 2)
    assert symbols[0][-1] == episode.answer


def test_random_weight_intervention_identity_and_one_hop_coincidence():
    torch.manual_seed(5)
    model = CandidateEdgeWorkspace(oracle_rows=True).eval()
    with torch.no_grad():
        rows, logits = evaluate_batch(model, [_episode("first_hop_confirm")], 1)
    row = rows[0]
    assert row["predictions"]["identity"] == row["predictions"]["clean"]
    assert row["predictions"]["gold_first"] == row["predictions"]["gold_last"]
    assert row["predictions"]["gold_last"] == row["predictions"]["gold_both"]
    assert logits["identity"] == logits["clean"]


def test_score_records_paired_rescue_and_damage():
    rows = [
        {"answer": 17, "target_rows": [0],
         "native_argmax_rows": [0],
         "predictions": {"clean": 18, "identity": 18,
                         "gold_first": 17, "gold_last": 17,
                         "gold_both": 17, "wrong_both": 18}},
        {"answer": 19, "target_rows": [1],
         "native_argmax_rows": [0],
         "predictions": {"clean": 19, "identity": 19,
                         "gold_first": 18, "gold_last": 18,
                         "gold_both": 18, "wrong_both": 18}},
    ]
    score = _score(rows, 1)
    assert score["address"]["0"]["hits"] == 1
    assert score["conditions"]["gold_both"] == {
        "correct": 1, "rescue_clean_wrong": 1,
        "damage_clean_correct": 1}
