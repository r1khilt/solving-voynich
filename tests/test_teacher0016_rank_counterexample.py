"""Visible relative-address rival, never a neural checkpoint test."""

import json
from pathlib import Path

from scripts.teacher0016_cross_score import score_split
from scripts.teacher0016_rank_audit import _rank as metadata_rank
from scripts.teacher0016_rank_counterexample import (
    _decode, _rank, evaluate_split,
)
from scripts.teacher0016_suite_audit import _episode


def test_visible_rank_state_is_order_invariant_and_not_token_identity():
    manifest = json.loads(Path("outputs/TEACH-0016/confirmation.json").read_text())
    group = manifest["groups"][0]
    ranks = []
    for order in (0, 1):
        for g in range(3):
            source = _episode(group, 1, g, 0, True, order)
            rank, lefts = _rank(source)
            assert rank == metadata_rank(source)
            assert lefts[rank] == group["key1"]
            ranks.append(rank)
            for recipient_g in range(3):
                recipient = _episode(group, 0, recipient_g, 0, True,
                                     1 - order)
                assert _decode(recipient, rank) == group[
                    "recipient_outputs"][recipient_g][1]
    assert len(set(ranks)) == 1
    assert ranks[0] < 64


def test_relative_rank_oracle_passes_behavioral_conjunction_without_key_state():
    manifest = json.loads(Path("outputs/TEACH-0016/confirmation.json").read_text())
    surfaces = evaluate_split(manifest)
    assert len(surfaces) == 1024
    assert sum(map(len, surfaces)) == 9216
    score = score_split(manifest, surfaces, ["test", "rank"])
    assert score["shifted_surface_pairs"] == 942
    assert score["scores"]["transfer"]["accuracy"] == 1.0
    assert score["scores"]["same_key"]["accuracy"] == 1.0
    assert score["scores"]["non_injection"]["accuracy"] == 1.0
    assert score["scores"]["reverse_eligible"]["accuracy"] == 1.0
    assert score["scores"]["pointer"]["correct"] == 0
    assert score["candidate_order_robust_state_pending_replay"]
    assert all(row["source_rank_state"] < 64 for surface in surfaces
               for row in surface)
