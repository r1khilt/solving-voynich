"""Frozen physical-occurrence oracle tests for TEACH-0013 Stage D."""

from dataclasses import replace

import pytest

from voynich.workspace.teacher13_tasks import (
    counterfactual_suite,
    physical_order_shortcut_denominators,
    physical_order_shortcut_oracle,
    registered_physical_order_oracle_kind,
)


def _panel():
    suite = counterfactual_suite(73131, discovery_groups=5, confirmation_groups=5)
    rows, group_ids = [], []
    for groups in suite.values():
        for group in groups:
            for original, reordered in zip(
                    group.base, group.reordered_base, strict=True):
                rows.append(physical_order_shortcut_oracle(original, reordered))
                group_ids.append(group.group_id)
    return tuple(rows), tuple(group_ids)


def test_physical_order_oracle_reconstructs_slots_without_model_outputs():
    suite = counterfactual_suite(73131, discovery_groups=2, confirmation_groups=2)
    for groups in suite.values():
        for group in groups:
            for original, reordered in zip(
                    group.base, group.reordered_base, strict=True):
                oracle = physical_order_shortcut_oracle(original, reordered)
                assert oracle.semantic_target == original.answer
                assert oracle.queried_f_original_row[0] == original.query
                assert oracle.matched_g_original_row[0] \
                    == oracle.queried_f_original_row[1]
                assert oracle.queried_f_slot_occupant \
                    == reordered.serialized_rows[oracle.queried_f_slot]
                assert oracle.matched_g_slot_occupant \
                    == reordered.serialized_rows[oracle.matched_g_slot]

                g_mapping = dict(original.g_rows)
                physical_key = oracle.queried_f_slot_occupant[1]
                if physical_key in g_mapping:
                    assert oracle.f_slot.structurally_valid
                    assert oracle.f_slot.target == g_mapping[physical_key]
                else:
                    assert not oracle.f_slot.structurally_valid
                    assert oracle.f_slot.target is None
                    assert oracle.f_slot.invalid_reason \
                        == "queried_f_slot_rhs_is_not_a_g_key"
                assert oracle.g_slot.structurally_valid
                assert oracle.g_slot.target == oracle.matched_g_slot_occupant[1]

                coherent = physical_key == oracle.matched_g_slot_occupant[0]
                assert oracle.joint.structurally_valid == coherent
                if coherent:
                    assert oracle.joint.target == oracle.g_slot.target
                    expected = {"g_slot=joint"}
                    if oracle.f_slot.structurally_valid:
                        assert oracle.joint.target == oracle.f_slot.target
                        expected.update(("f_slot=g_slot", "f_slot=joint"))
                    assert set(oracle.target_agreements) == expected
                else:
                    assert oracle.joint.target is None
                    assert oracle.joint.invalid_reason \
                        == "physical_slot_rows_do_not_compose"


def test_physical_order_oracle_accepts_every_registered_order_surface():
    group = counterfactual_suite(
        73131, discovery_groups=1, confirmation_groups=1)["confirmation"][0]
    pairs = (
        (group.base, group.reordered_base),
        (group.donor, group.reordered_donor),
        (group.marker_free_base, group.marker_free_reordered_base),
        (group.marker_free_donor, group.marker_free_reordered_donor),
        (group.format_base, group.format_reordered_base),
        (group.format_donor, group.format_reordered_donor),
        (group.distractor_base, group.distractor_reordered_base),
        (group.distractor_donor, group.distractor_reordered_donor),
    )
    oracles = [
        physical_order_shortcut_oracle(original, reordered)
        for originals, reordereds in pairs
        for original, reordered in zip(originals, reordereds, strict=True)
    ]
    assert len(oracles) == 24
    assert all(oracle.g_slot.structurally_valid for oracle in oracles)


def test_shortcut_denominators_exclude_semantic_collisions_and_partial_groups():
    oracles, group_ids = _panel()
    fields = oracles[0].compact_fields("f_slot")
    assert set(fields) == {
        "oracle_kind", "oracle_target", "oracle_valid",
        "oracle_structurally_valid", "oracle_semantic_collision",
        "oracle_invalid_reason", "physical_f_slot_index",
        "physical_g_slot_index", "physical_f_slot_key", "physical_g_slot_key",
        "physical_g_slot_value", "physical_slots_compose",
    }
    assert fields["oracle_kind"] == "f_slot"
    assert fields["physical_f_slot_key"] \
        == oracles[0].queried_f_slot_occupant[1]
    assert fields["physical_g_slot_key"] \
        == oracles[0].matched_g_slot_occupant[0]
    assert fields["physical_g_slot_value"] \
        == oracles[0].matched_g_slot_occupant[1]
    collision = next(
        oracle.compact_fields("g_slot") for oracle in oracles
        if oracle.g_slot.semantic_collision)
    assert not collision["oracle_valid"]
    assert collision["oracle_structurally_valid"]
    assert collision["oracle_invalid_reason"] == "semantic_target_collision"
    assert physical_order_shortcut_denominators(
        oracles, kind="f_slot", group_ids=group_ids) == {
            "kind": "f_slot",
            "all_items": 30,
            "structurally_valid_items": 6,
            "semantic_collision_items": 0,
            "adversarially_eligible_items": 6,
            "item_denominator_rule": (
                "structurally_valid_and_target_differs_from_semantic_answer"),
            "all_groups": 10,
            "adversarially_eligible_groups": 2,
            "group_denominator_rule": (
                "exactly_three_recipients_and_every_item_adversarially_eligible"),
        }
    g_counts = physical_order_shortcut_denominators(
        oracles, kind="g_slot", group_ids=group_ids)
    assert (g_counts["structurally_valid_items"],
            g_counts["semantic_collision_items"],
            g_counts["adversarially_eligible_items"],
            g_counts["adversarially_eligible_groups"]) == (30, 2, 28, 8)
    joint_counts = physical_order_shortcut_denominators(
        oracles, kind="joint", group_ids=group_ids)
    assert (joint_counts["structurally_valid_items"],
            joint_counts["semantic_collision_items"],
            joint_counts["adversarially_eligible_items"],
            joint_counts["adversarially_eligible_groups"]) == (6, 1, 5, 1)

    with pytest.raises(ValueError, match="exactly three recipients"):
        physical_order_shortcut_denominators(
            oracles[:-1], kind="g_slot", group_ids=group_ids[:-1])


def test_primary_oracle_kind_is_frozen_from_mediator_identity():
    assert registered_physical_order_oracle_kind(
        mediator_kind="single", label="queried_f.right") == "f_slot"
    assert registered_physical_order_oracle_kind(
        mediator_kind="single", label="query") == "f_slot"
    assert registered_physical_order_oracle_kind(
        mediator_kind="path", source_label="other_f.0.left") == "f_slot"
    assert registered_physical_order_oracle_kind(
        mediator_kind="single", label="matched_g.right") == "g_slot"
    assert registered_physical_order_oracle_kind(
        mediator_kind="path", source_label="matched_g.left") == "g_slot"
    with pytest.raises(ValueError, match="requires only label"):
        registered_physical_order_oracle_kind(
            mediator_kind="single", label="query", source_label="queried_f.right")


def test_physical_order_oracle_rejects_nonregistered_pairs():
    group = counterfactual_suite(
        73131, discovery_groups=1, confirmation_groups=1)["discovery"][0]
    original, reordered = group.base[0], group.reordered_base[0]
    with pytest.raises(ValueError, match="one-slot cyclic reorder"):
        physical_order_shortcut_oracle(original, original)
    with pytest.raises(ValueError, match="one graph"):
        physical_order_shortcut_oracle(original, group.reordered_donor[0])
    with pytest.raises(ValueError, match="registered for composed"):
        physical_order_shortcut_oracle(
            group.direct_donor, group.direct_order_donor)
    with pytest.raises(ValueError, match="preserve the serialized surface"):
        physical_order_shortcut_oracle(
            original, replace(reordered, tokens=reordered.tokens[:-1]))
