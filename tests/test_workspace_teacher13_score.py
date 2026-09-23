"""Tests for frozen TEACH-0013 scoring and deterministic site selection."""

import pytest

from voynich.workspace.teacher13_score import (
    effect_metrics,
    fresh_panel_decision,
    rank_secondary_residual_sites,
    recipient_transfer_decision,
    select_residual_site,
)


def _site_rows(*, replicate, cut, label, correct=True, injected=False, gain=.5):
    rows = []
    for render in ("marked", "marker_free"):
        for half in ("first_half", "second_half"):
            for group in range(5):
                for recipient in range(3):
                    target = 100 + group * 3 + recipient
                    fixed = 100 + group * 3
                    prediction = target if correct else 999
                    if injected and recipient != 0:
                        prediction = fixed
                    rows.append({
                        "replicate": replicate, "cut_index": cut,
                        "semantic_label": label, "group_id": f"{render}-{half}-{group}",
                        "recipient": recipient, "target": target, "prediction": prediction,
                        "fixed_donor_answer": fixed, "target_probability": .1 + gain,
                        "base_target_probability": .1, "render_stratum": render,
                        "position_stratum": half,
                    })
    return rows


def test_effect_metrics_require_complete_recipient_groups_and_reject_injection():
    rows = _site_rows(replicate=0, cut=1, label="query")
    metrics = effect_metrics(rows)
    assert metrics.item_accuracy == 1
    assert metrics.group_accuracy == 1
    assert metrics.non_injection == 1
    injected = effect_metrics(_site_rows(
        replicate=0, cut=1, label="query", injected=True))
    assert injected.item_accuracy == pytest.approx(1 / 3)
    assert injected.non_injection == 0
    with pytest.raises(ValueError, match="exactly three"):
        effect_metrics(rows[:-1])


def test_site_selection_uses_earliest_cut_then_minimum_effect_and_label_order():
    rows = []
    for replicate in (0, 1):
        rows += _site_rows(replicate=replicate, cut=1, label="query", gain=.40)
        rows += _site_rows(replicate=replicate, cut=1, label="answer", gain=.50)
        rows += _site_rows(replicate=replicate, cut=2, label="matched_g.right", gain=.60)
    result = select_residual_site(
        rows, ("query", "answer", "matched_g.right"), expected_groups_per_render=10)
    assert result["selection"]["cut_index"] == 1
    # Group/item rates tie at cut 1, so the fixed label order wins; probability is not a tie break.
    assert result["selection"]["semantic_label"] == "query"
    for replicate in (0, 1):
        rows += _site_rows(replicate=replicate, cut=0, label="query", correct=False)
    assert select_residual_site(
        rows, ("query", "answer", "matched_g.right"), expected_groups_per_render=10)[
        "selection"]["cut_index"] == 1


def test_candidate_missing_a_registered_stratum_cannot_qualify():
    rows = []
    for replicate in (0, 1):
        rows += [row for row in _site_rows(
            replicate=replicate, cut=1, label="queried_f.marker")
            if row["render_stratum"] == "marked"]
    result = select_residual_site(
        rows, ("queried_f.marker",), expected_groups_per_render=10)
    assert result["selection"] is None
    assert result["candidates"]["1:queried_f.marker"]["by_seed"]["0"][
        "marker_free"] is None


def test_secondary_ranking_maximizes_worst_seed_before_using_earlier_cut():
    rows = []
    for replicate in (0, 1):
        weak = _site_rows(replicate=replicate, cut=0, label="query", correct=False)
        strong = _site_rows(replicate=replicate, cut=2, label="answer", correct=True)
        for row in weak + strong:
            row["family"] = "g_content"
        rows.extend(weak + strong)
    result = rank_secondary_residual_sites(
        rows, ("query", "answer"), expected_groups=20)
    assert result["selection"]["cut_index"] == 2
    assert result["selection"]["semantic_label"] == "answer"


def test_stage_a_and_stage_c_decisions_are_conjunctive_across_seeds():
    fresh = {replicate: {
        "composed_items": .99, "base_recipient_groups": .99,
        "donor_recipient_groups": .99, "marker_pairs": .99, "order_pairs": .99,
        "distractor_pairs": .99, "first_hop": .99, "direct": .99, "copy": .99,
    } for replicate in ("0", "1")}
    assert fresh_panel_decision(fresh)["label"] == "pass"
    fresh["1"]["marker_pairs"] = .84
    assert fresh_panel_decision(fresh)["label"] == "inconclusive_fresh_panel_competence"

    transfer = {replicate: {
        "clean_base": 1., "clean_donor": 1., "input_f_replacement": 1.,
        "sufficiency_items_forward": .9, "sufficiency_groups_forward": .8,
        "sufficiency_items_reverse": .9, "sufficiency_groups_reverse": .8,
        "changed_non_injection_forward": 1., "changed_non_injection_reverse": 1.,
        "necessity_items": .9, "necessity_groups": .8, "rescue": 1.,
        "same_key_format": 1., "same_key_order": 1., "same_key_distractor": 1.,
        "minimum_control_advantage": .5, "mean_probability_gain": .5,
        "final_answer_injection": 1., "direct_loss": 0., "copy_loss": 0.,
    } for replicate in ("0", "1")}
    assert recipient_transfer_decision(
        transfer, discovery_selected=True, numerical_qualified=True)["label"] == \
        "recipient_key_transfer_supported"
    transfer["1"]["necessity_items"] = .79
    assert recipient_transfer_decision(
        transfer, discovery_selected=True, numerical_qualified=True)["label"] == "not_supported"
