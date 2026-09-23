"""CPU checks for TEACH-0013 clean scoring and residual-screen rows."""

from dataclasses import asdict

import torch

from voynich.workspace.teacher12_models import model_for_arm
from voynich.workspace.teacher13_discovery import (
    episode_from_record,
    fresh_panel_scores,
    residual_numerical_qualification,
    residual_screen,
    screen_conditions,
)
from voynich.workspace.teacher13_tasks import counterfactual_suite


def _records():
    group = counterfactual_suite(73151, discovery_groups=1, confirmation_groups=1)[
        "discovery"][0]
    return [asdict(group)]


def test_episode_record_roundtrip_and_clean_score_cells_are_bounded():
    records = _records()
    assert episode_from_record(records[0]["base"][0]).tokens == records[0]["base"][0]["tokens"]
    torch.manual_seed(29)
    scores = fresh_panel_scores(model_for_arm("raw_shallow").eval(), records)
    assert set(scores) == {
        "composed_items", "base_recipient_groups", "donor_recipient_groups",
        "marker_pairs", "order_pairs", "distractor_pairs", "first_hop", "direct", "copy",
    }
    assert all(0 <= value <= 1 for value in scores.values())


def test_residual_screen_emits_complete_three_recipient_cells_and_exact_metadata():
    records = _records()
    torch.manual_seed(30)
    rows = residual_screen(
        model_for_arm("raw_shallow").eval(), records, replicate=0,
        sites=("blocks.0.resid_post",), labels=("query", "queried_f.right"))
    assert len(rows) == 2 * 2 * 3
    assert {(row["render_stratum"], row["semantic_label"])
            for row in rows} == {
                ("marked", "query"), ("marked", "queried_f.right"),
                ("marker_free", "query"), ("marker_free", "queried_f.right"),
            }
    for render in ("marked", "marker_free"):
        for label in ("query", "queried_f.right"):
            cell = [row for row in rows if row["render_stratum"] == render
                    and row["semantic_label"] == label]
            assert {row["recipient"] for row in cell} == {0, 1, 2}
            assert all(row["site"] == "blocks.0.resid_post" for row in cell)
            assert all(0 <= row["target_probability"] <= 1 for row in cell)
            assert all(row["family"] == "f_content" for row in cell)
            assert all(row["wrong_destination"] in {
                "target", "base_answer", "fixed_donor_answer", "other_recipient_target",
                "other_legal_candidate", "outside_legal_candidates"} for row in cell)


def test_every_registered_secondary_family_forms_complete_recipient_groups():
    records = _records()
    torch.manual_seed(31)
    net = model_for_arm("raw_shallow").eval()
    for family in ("f_binding", "g_content", "g_binding", "order", "format", "distractor"):
        rows = residual_screen(
            net, records, replicate=0, sites=("blocks.0.resid_post",),
            labels=("query",), family=family)
        assert len(rows) == 3 * len(screen_conditions(family))
        assert {row["recipient"] for row in rows} == {0, 1, 2}
        assert all(row["family"] == family for row in rows)
        assert all(isinstance(row["target_logit"], float)
                   and isinstance(row["base_target_logit"], float) for row in rows)


def test_residual_identity_numerical_gate_covers_every_requested_cut():
    records = _records()
    torch.manual_seed(32)
    result = residual_numerical_qualification(
        model_for_arm("raw_shallow").eval(), records,
        sites=("embed", "blocks.0.resid_post"))
    assert result["qualified"]
    assert set(result["identity_errors"]) == {"embed", "blocks.0.resid_post"}
    assert result["maximum_identity_logit_error"] < 1e-6
