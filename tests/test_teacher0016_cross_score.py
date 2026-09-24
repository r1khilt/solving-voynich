"""Synthetic answer fixtures for the prospectively fixed score algebra."""

import json
from pathlib import Path

from scripts.teacher0016_cross_score import score_split
from scripts.teacher0016_suite_audit import SURFACES, _episode, _index


def _oracle_surfaces(manifest):
    surfaces = []
    for group in manifest["groups"]:
        for d, marked, source_order in SURFACES:
            rows = []
            for a in range(3):
                source = _episode(group, 1, a, d, marked, source_order)
                source_slot = _index(
                    source, group["key1"], group["recipient_outputs"][a][1])
                for b in range(3):
                    base = _episode(group, 0, b, d, marked, 1 - source_order)
                    target = _episode(group, 1, b, d, marked, 1 - source_order)
                    target_slot = _index(
                        base, group["key1"], group["recipient_outputs"][b][1])
                    rows.append({
                        "source_g": a, "recipient_g": b,
                        "source_slot": source_slot,
                        "shifted_slot": source_slot != target_slot,
                        "base_answer": base["answer"],
                        "target_answer": target["answer"],
                        "source_answer": source["answer"],
                        "base_prediction": base["answer"],
                        "target_prediction": target["answer"],
                        "transfer_prediction": target["answer"],
                        "same_key_prediction": target["answer"],
                        "wrong_key_prediction": base["answer"],
                        "distinct_prediction": base["answer"],
                        "random_prediction": base["answer"],
                        "reverse_prediction": base["answer"],
                    })
            surfaces.append(rows)
    return surfaces


def test_full_confirmation_score_exact_denominators_and_decision():
    manifest = json.loads(Path("outputs/TEACH-0016/confirmation.json").read_text())
    surfaces = _oracle_surfaces(manifest)
    scored = score_split(manifest, surfaces, ["fixture", "confirmation"])
    assert scored["shifted_surface_pairs"] == 942
    assert scored["shifted_offdiag_attempts"] == 5652
    assert scored["scores"]["full_transfer"]["total"] == 9216
    assert scored["scores"]["triples"]["total"] == 2826
    assert scored["scores"]["transfer"]["total"] == 5652
    assert scored["scores"]["pointer"]["correct"] == 0
    assert sum(cell["total"] for cell in scored[
        "full_per_source_recipient"].values()) == 9216
    assert sum(cell["total"] for cell in scored[
        "per_source_recipient"].values()) == 5652
    assert scored["candidate_order_robust_state_pending_replay"]
    assert scored["same_key_positive_control_pass"]
    assert all(item["accuracy"] == 1.0 for key, item in scored["scores"].items()
               if key in ("transfer", "triples", "same_key", "non_injection",
                          "reverse_eligible", "rich", "free",
                          "source_order_0", "source_order_1"))


def test_failed_order_change_positive_control_is_inconclusive():
    manifest = json.loads(Path("outputs/TEACH-0016/confirmation.json").read_text())
    one = {**manifest, "groups": manifest["groups"][:1]}
    surfaces = _oracle_surfaces(one)
    for rows in surfaces:
        for row in rows:
            row["same_key_prediction"] = row["base_answer"]
    scored = score_split(one, surfaces, ["fixture", "failed-positive"])
    assert not scored["same_key_positive_control_pass"]
    assert not scored["candidate_order_robust_state_pending_replay"]
    assert scored["interpretation"] == "inconclusive_cross_order_interface"
