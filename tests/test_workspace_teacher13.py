"""CPU-only tests for prospective TEACH-0013 counterfactual suites."""

from voynich.workspace.teacher12_tasks import symbolic_oracle, table_partitions
from voynich.workspace.teacher13_tasks import (
    counterfactual_suite,
    semantic_layout,
    surface_skeleton,
)


def test_counterfactual_suite_is_deterministic_disjoint_and_confirm_family():
    suite = counterfactual_suite(73131, discovery_groups=2, confirmation_groups=2)
    assert counterfactual_suite(73131, discovery_groups=2, confirmation_groups=2) == suite
    assert {name: len(groups) for name, groups in suite.items()} == {
        "discovery": 2, "confirmation": 2,
    }
    ids = [group.group_id for groups in suite.values() for group in groups]
    assert len(ids) == len(set(ids))
    assert all(group.split == split for split, groups in suite.items() for group in groups)
    for groups in suite.values():
        for group in groups:
            for episode in (group.base + group.donor + group.marker_free_base
                            + group.marker_free_donor + group.reordered_base
                            + group.reordered_donor):
                assert table_partitions(episode.f_rows, episode.g_rows) == ("confirm", "confirm")


def test_binding_swap_and_recipient_remap_hold_surface_structure_fixed():
    groups = counterfactual_suite(73132, discovery_groups=1, confirmation_groups=1)
    for group in groups["discovery"] + groups["confirmation"]:
        episodes = group.base + group.donor
        assert len({surface_skeleton(episode) for episode in episodes}) == 1
        assert len(set(group.base_answers)) == 3
        assert len(set(group.recipient_answers)) == 3
        assert group.fixed_donor_answer == group.recipient_answers[0]
        assert all(episode.marker_dropout == 0 for episode in group.base + group.donor)
        assert all(episode.marker_dropout == 1
                   for episode in group.marker_free_base + group.marker_free_donor)
        assert len({episode.tokens for episode in group.base + group.reordered_base}) == 6
        assert group.g_content_base.answer != group.g_content_donor.answer
        assert surface_skeleton(group.g_content_base) == surface_skeleton(group.g_content_donor)
        assert group.g_content_base.f_rows == group.g_content_donor.f_rows
        assert len([(row0, row1) for row0, row1 in zip(
            group.g_content_base.g_rows, group.g_content_donor.g_rows, strict=True)
            if row0 != row1]) == 2
        for base, donor, base_answer, recipient_answer in zip(
                group.base, group.donor, group.base_answers,
                group.recipient_answers, strict=True):
            assert base.query == donor.query
            assert base.g_rows == donor.g_rows
            assert base.distractor_rows == donor.distractor_rows
            assert base.answer == base_answer == symbolic_oracle(base)
            assert donor.answer == recipient_answer == symbolic_oracle(donor)
            assert base.answer != donor.answer
            changed = [(left0, right0, right1) for (left0, right0), (left1, right1)
                       in zip(base.f_rows, donor.f_rows, strict=True)
                       if right0 != right1 and left0 == left1]
            assert len(changed) == 2
            assert {right for _, right in base.f_rows} == {right for _, right in donor.f_rows}


def test_semantic_layout_exactly_identifies_dynamic_relation_positions():
    suite = counterfactual_suite(73133, discovery_groups=1, confirmation_groups=1)
    for groups in suite.values():
        for group in groups:
            for episode in (group.base + group.donor + group.marker_free_base
                            + group.marker_free_donor + group.reordered_base
                            + group.reordered_donor + (group.format_donor,
                            group.distractor_donor, group.g_content_base,
                            group.g_content_donor)):
                layout = semantic_layout(episode)
                assert len(layout.roles) == len(episode.tokens)
                assert len(layout.labels) == len(episode.tokens)
                assert len(set(layout.labels)) == len(layout.labels)
                assert layout.positions("query") == (len(episode.tokens) - 2,)
                assert layout.positions("answer") == (len(episode.tokens) - 1,)
                assert layout.label_position("query") == len(episode.tokens) - 2
                assert len(layout.positions("queried_f.left")) == 1
                assert len(layout.positions("queried_f.right")) == 1
                assert len(layout.positions("matched_g.left")) == 1
                assert len(layout.positions("matched_g.right")) == 1
                f_right = layout.positions("queried_f.right")[0]
                g_left = layout.positions("matched_g.left")[0]
                assert episode.tokens[f_right] == episode.tokens[g_left]
                assert len(layout.positions("false_path_first.left")) == 2
                assert len(layout.positions("false_path_terminal.right")) == 2


def test_task_controls_use_same_inventory_and_exact_oracles():
    suite = counterfactual_suite(73134, discovery_groups=1, confirmation_groups=1)
    for groups in suite.values():
        for group in groups:
            controls = (group.first_hop_base, group.first_hop_donor,
                        group.direct_base, group.direct_donor, group.copy_control)
            assert [episode.task for episode in controls] == [
                "first_hop", "first_hop", "direct", "direct", "copy"]
            assert group.first_hop_base.answer == group.key_base
            assert group.first_hop_donor.answer == group.key_donor
            assert group.direct_base.query == group.key_base
            assert group.direct_donor.query == group.key_donor
            assert group.copy_control.answer == group.copy_control.query
            assert all(symbolic_oracle(episode) == episode.answer for episode in controls)
