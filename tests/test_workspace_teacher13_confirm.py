"""CPU smoke and invariance checks for TEACH-0013 confirmation primitives."""

from dataclasses import asdict

import torch

from voynich.workspace.teacher12_models import WIDTH, model_for_arm
from voynich.workspace.teacher13_confirm import (
    MediatorSpec,
    bidirectional_sufficiency,
    corruption_rescue_logits,
    corruption_rescue_rows,
    cross_task_mediator_logits,
    fresh_panel_confirmation_rows,
    identity_error,
    mediator_logits,
    matched_sufficiency_controls,
    norm_matched_donor_logits,
    norm_matched_random_logits,
    nuisance_preservation,
    positive_control_rows,
    summarize_confirmation_rows,
    summarize_fresh_panel_rows,
    summarize_rescue_rows,
    task_specificity_rows,
    wrong_position_logits,
)
from voynich.workspace.teacher13_discovery import episode_from_record, fresh_panel_scores
from voynich.workspace.teacher13_intervene import raw_forward
from voynich.workspace.teacher13_tasks import counterfactual_suite


def _groups():
    suite = counterfactual_suite(74111, discovery_groups=1, confirmation_groups=1)
    return [asdict(suite["confirmation"][0])]


def test_single_and_path_self_donors_are_exact_instrumented_identities():
    groups = _groups()
    episodes = tuple(episode_from_record(row) for row in groups[0]["base"])
    torch.manual_seed(41)
    net = model_for_arm("raw_shallow").eval()
    single = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    path = MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                        late_site="blocks.0.resid_post", destination_label="query")
    for spec in (single, path):
        result = identity_error(net, episodes, spec)
        assert result["maximum_identity_logit_error"] < 1e-6
        assert result["qualified"] == (
            result["maximum_fused_instrumented_logit_error"] < 1e-6)
        assert mediator_logits(net, episodes, episodes, spec).shape[0] == 3


def test_bidirectional_sufficiency_rows_keep_full_recipient_denominators():
    groups = _groups()
    torch.manual_seed(42)
    rows = bidirectional_sufficiency(
        model_for_arm("raw_shallow").eval(), groups,
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
        episode_loader=episode_from_record)
    assert len(rows) == 2 * 3
    assert {(row["direction"], row["recipient"]) for row in rows} == {
        (direction, recipient) for direction in ("forward", "reverse")
        for recipient in range(3)}
    assert all(row["condition"] == "sufficiency_marked" for row in rows)
    assert all(row["fixed_donor_answer"] != row["target"]
               for row in rows if row["recipient"] != 0)
    for direction in ("forward", "reverse"):
        summary = summarize_confirmation_rows(rows, direction=direction)
        assert summary["items"] == 3 and summary["groups"] == 1


def test_nuisance_conditions_preserve_registered_targets_and_group_shape():
    groups = _groups()
    torch.manual_seed(43)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    for family in ("format", "order", "distractor"):
        rows = nuisance_preservation(
            net, groups, spec, family=family, episode_loader=episode_from_record)
        assert len(rows) == 3
        assert all(row["condition"] == f"same_key_{family}"
                   and row["target"] == row["base_answer"] for row in rows)
        assert summarize_confirmation_rows(rows)["groups"] == 1


def test_norm_matched_random_control_is_deterministic_for_single_and_path_protocols():
    groups = _groups()
    base = tuple(episode_from_record(row) for row in groups[0]["base"])
    donor0 = episode_from_record(groups[0]["donor"][0])
    donor = (donor0,) * 3
    torch.manual_seed(44)
    net = model_for_arm("raw_shallow").eval()
    specs = (
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
        MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                     late_site="blocks.0.resid_post", destination_label="query"),
    )
    for spec in specs:
        left = norm_matched_random_logits(net, base, donor, spec, seed=73211)
        right = norm_matched_random_logits(net, base, donor, spec, seed=73211)
        other = norm_matched_random_logits(net, base, donor, spec, seed=73212)
        assert torch.equal(left, right)
        assert left.shape == other.shape == (3, left.shape[1])
        assert not torch.equal(left, other)


def test_norm_matched_cyclic_donor_runs_through_single_and_path_protocols():
    suite = counterfactual_suite(74112, discovery_groups=1, confirmation_groups=2)
    groups = [asdict(group) for group in suite["confirmation"]]
    base = tuple(episode_from_record(row) for row in groups[0]["base"])
    true0 = episode_from_record(groups[0]["donor"][0])
    control0 = episode_from_record(groups[1]["donor"][0])
    true_donor, control_donor = (true0,) * 3, (control0,) * 3
    torch.manual_seed(45)
    net = model_for_arm("raw_shallow").eval()
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
            MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                         late_site="blocks.0.resid_post", destination_label="query")):
        logits = norm_matched_donor_logits(net, base, true_donor, control_donor, spec)
        assert logits.shape[0] == 3 and torch.isfinite(logits).all()


def test_matched_controls_cover_both_directions_conditions_and_recipients():
    suite = counterfactual_suite(74113, discovery_groups=1, confirmation_groups=2)
    groups = [asdict(group) for group in suite["confirmation"]]
    torch.manual_seed(46)
    rows = matched_sufficiency_controls(
        model_for_arm("raw_shallow").eval(), groups,
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
        episode_loader=episode_from_record)
    assert len(rows) == 2 * 4 * 2 * 3
    assert {row["condition"] for row in rows} == {
        "norm_matched_gaussian", "norm_matched_cyclic",
        "norm_matched_wrong_position", "norm_matched_wrong_key"}
    assert {row["direction"] for row in rows} == {"forward", "reverse"}
    for condition in ("norm_matched_gaussian", "norm_matched_cyclic",
                      "norm_matched_wrong_position", "norm_matched_wrong_key"):
        for direction in ("forward", "reverse"):
            selected = [row for row in rows if row["condition"] == condition
                        and row["direction"] == direction]
            assert summarize_confirmation_rows(selected)["groups"] == 2


def test_positive_controls_cover_clean_input_replacement_and_final_injection():
    groups = _groups()
    torch.manual_seed(47)
    rows = positive_control_rows(
        model_for_arm("raw_shallow").eval(), groups,
        episode_loader=episode_from_record, final_site="blocks.3.resid_post")
    assert len(rows) == 3 * 3
    assert {row["condition"] for row in rows} == {
        "clean_base", "clean_donor_input_f_replacement",
        "final_state_fixed_answer_injection"}
    injected = [row for row in rows
                if row["condition"] == "final_state_fixed_answer_injection"]
    assert all(row["target"] == row["fixed_donor_answer"] for row in injected)


def test_direct_copy_specificity_uses_frozen_same_key_nuisance_variants():
    groups = _groups()
    torch.manual_seed(48)
    net = model_for_arm("raw_shallow").eval()
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post",
                         label="queried_f.right"),
            MediatorSpec(kind="path", early_site="embed",
                         source_label="queried_f.right",
                         late_site="blocks.0.resid_post", destination_label="query")):
        rows = task_specificity_rows(
            net, groups, spec, episode_loader=episode_from_record)
        assert len(rows) == 2 * 3
        assert {row["condition"] for row in rows} == {
            f"{task}_same_key_{family}" for task in ("direct", "copy")
            for family in ("format", "order", "distractor")}
        assert all(row["direction"] == "specificity" for row in rows)


def test_cross_task_subspace_patch_full_width_matches_full_cross_task_edit():
    groups = _groups()
    group = groups[0]
    base = (episode_from_record(group["first_hop_base"]),)
    donor = (episode_from_record(group["first_hop_donor"]),)
    reference = (episode_from_record(group["donor"][0]),)
    torch.manual_seed(481)
    net = model_for_arm("raw_shallow").eval()
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post",
                         label="queried_f.right"),
            MediatorSpec(kind="path", early_site="embed",
                         source_label="queried_f.right",
                         late_site="blocks.0.resid_post", destination_label="query")):
        expected = cross_task_mediator_logits(net, base, donor, reference, spec)
        actual = cross_task_mediator_logits(
            net, base, donor, reference, spec, basis=torch.eye(WIDTH))
        assert torch.allclose(actual.float(), expected.float(), atol=1e-6, rtol=1e-6)


def test_retained_fresh_panel_rows_reproduce_existing_clean_gate_scores():
    groups = _groups()
    torch.manual_seed(481)
    net = model_for_arm("raw_shallow").eval()
    rows = fresh_panel_confirmation_rows(
        net, groups, episode_loader=episode_from_record, batch_size=5)
    assert summarize_fresh_panel_rows(rows) == fresh_panel_scores(net, groups)
    assert all("pair_id" in row and row["direction"] == "stage_a" for row in rows)


def test_wrong_position_and_corruption_rescue_run_for_single_and_path_protocols():
    groups = _groups()
    base = tuple(episode_from_record(row) for row in groups[0]["base"])
    donor0 = episode_from_record(groups[0]["donor"][0])
    donor = (donor0,) * 3
    torch.manual_seed(49)
    net = model_for_arm("raw_shallow").eval()
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
            MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                         late_site="blocks.0.resid_post", destination_label="query")):
        wrong = wrong_position_logits(net, base, donor, spec)
        corrupted, rescued = corruption_rescue_logits(net, base, donor, spec)
        assert wrong.shape == corrupted.shape == rescued.shape
        assert torch.isfinite(wrong).all() and torch.isfinite(corrupted).all()
        clean = raw_forward(net, base).logits
        assert torch.allclose(rescued.float(), clean.float(), atol=1e-5, rtol=1e-5)


def test_corruption_rescue_rows_are_paired_and_restore_instrumented_native_state():
    groups = _groups()
    torch.manual_seed(50)
    rows = corruption_rescue_rows(
        model_for_arm("raw_shallow").eval(), groups,
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
        episode_loader=episode_from_record)
    assert len(rows) == 2 * 2 * 3
    summary = summarize_rescue_rows(rows)
    assert summary["paired_items"] == 6
    if summary["changed_cases"]:
        assert summary["rescue_rate"] == 1.0
    rescue = [row for row in rows if row["condition"] == "native_state_rescue"]
    assert all(row["prediction"] == row["clean_prediction"] for row in rescue)
