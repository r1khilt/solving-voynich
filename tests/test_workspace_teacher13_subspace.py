"""Finite single-site and propagated-path causal subspace checks."""

from dataclasses import asdict

import torch

from voynich.workspace.teacher12_models import WIDTH, model_for_arm
from voynich.workspace.teacher13_confirm import MediatorSpec, mediator_logits
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_subspace import (
    equal_energy_offspace_states,
    equal_norm_component_corruption_states,
    factor_equal_energy_rows,
    factor_state_panel,
    factor_rank_screen,
    fit_registered_factor_geometries,
    factor_transfer_rows,
    mediator_state_pairs,
    mediator_endpoint_logits,
    mediator_subspace_logits,
    mean_ablation_states,
    summarize_factor_ranks,
)
from voynich.workspace.teacher13_tasks import counterfactual_suite


def _batch():
    group = asdict(counterfactual_suite(
        74211, discovery_groups=1, confirmation_groups=1)["discovery"][0])
    base = tuple(episode_from_record(row) for row in group["base"])
    donor0 = episode_from_record(group["donor"][0])
    return base, (donor0,) * 3


def test_full_width_basis_reproduces_single_and_path_mediator_logits():
    torch.manual_seed(61)
    net = model_for_arm("raw_shallow").eval()
    base, donor = _batch()
    basis = torch.eye(WIDTH)
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
            MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                         late_site="blocks.1.resid_post", destination_label="query")):
        expected = mediator_logits(net, base, donor, spec)
        actual = mediator_subspace_logits(net, base, donor, spec, basis)
        assert torch.allclose(actual.float(), expected.float(), atol=1e-6, rtol=1e-6)


def test_state_pairs_use_single_site_or_propagated_late_path_coordinates():
    torch.manual_seed(62)
    net = model_for_arm("raw_shallow").eval()
    base, donor = _batch()
    single = mediator_state_pairs(
        net, base, donor,
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"))
    path = mediator_state_pairs(
        net, base, donor,
        MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                     late_site="blocks.1.resid_post", destination_label="query"))
    assert single.base.shape == single.donor.shape == (3, WIDTH)
    assert path.base.shape == path.donor.shape == (3, WIDTH)
    assert single.endpoint == "single" and path.endpoint == "path_destination"
    rank_one = torch.eye(WIDTH)[:, :1]
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
            MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                         late_site="blocks.1.resid_post", destination_label="query")):
        subspace = mediator_subspace_logits(net, base, donor, spec, rank_one)
        complement = mediator_subspace_logits(
            net, base, donor, spec, rank_one, component="complement")
        assert subspace.shape == complement.shape and torch.isfinite(subspace).all()


def test_registered_factor_panels_are_complete_and_fit_blocked_geometries():
    suite = counterfactual_suite(74212, discovery_groups=2, confirmation_groups=1)
    groups = [asdict(group) for group in suite["discovery"]]
    torch.manual_seed(63)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    panels = {
        name: factor_state_panel(
            net, groups, spec, factor_name=name, episode_loader=episode_from_record,
            batch_size=7)
        for name in ("content", "binding", "order")
    }
    assert panels["content"].states.shape == (2 * 5 * 3 * 2, WIDTH)
    assert panels["binding"].states.shape == (2 * 5 * 3 * 2, WIDTH)
    assert panels["order"].states.shape == (2 * 8 * 3 * 2, WIDTH)
    fitted = fit_registered_factor_geometries(
        panels["content"], panels["binding"], panels["order"])
    assert fitted.content.basis.shape[0] == WIDTH
    assert set(fitted.orthogonal.forward) == {"content", "binding", "order"}
    assert all(torch.isfinite(geometry.eigenvalues).all()
               for geometry in (fitted.content, fitted.binding, fitted.order))


def test_factor_rank_screen_keeps_full_group_denominators_for_all_variables():
    suite = counterfactual_suite(74213, discovery_groups=1, confirmation_groups=1)
    groups = [asdict(group) for group in suite["discovery"]]
    torch.manual_seed(64)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    basis = torch.eye(WIDTH)[:, :2]
    expected = {"content": 15, "binding": 15, "order": 24}
    for factor in ("content", "binding", "order"):
        rows = factor_rank_screen(
            net, groups, spec, basis, factor_name=factor,
            episode_loader=episode_from_record, candidate_ranks=(1, 2))
        assert len(rows) == expected[factor] * 3
        summary = summarize_factor_ranks(rows)
        assert set(summary) == {"full", "1", "2"}
        assert all(cell["groups"] == ({"content": 5, "binding": 5, "order": 8}[factor])
                   for cell in summary.values())


def test_explicit_endpoint_states_reproduce_full_single_and_path_mediators():
    torch.manual_seed(65)
    net = model_for_arm("raw_shallow").eval()
    base, donor = _batch()
    for spec in (
            MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
            MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                         late_site="blocks.1.resid_post", destination_label="query")):
        pairs = mediator_state_pairs(net, base, donor, spec)
        expected = mediator_logits(net, base, donor, spec)
        actual = mediator_endpoint_logits(net, base, spec, pairs.donor)
        assert torch.allclose(actual.float(), expected.float(), atol=1e-6, rtol=1e-6)


def test_mean_ablation_and_equal_energy_control_have_registered_geometry():
    base = torch.tensor([[2., 3., 4.], [-1., 5., 2.]])
    donor = torch.tensor([[5., 7., 4.], [3., 1., 2.]])
    basis = torch.eye(3)[:, :2]
    mean = torch.tensor([1., 1., 1.])
    ablated = mean_ablation_states(base, mean, basis)
    assert torch.allclose(ablated[:, :2], mean[:2].expand(2, 2))
    assert torch.equal(ablated[:, 2], base[:, 2])
    controlled = equal_energy_offspace_states(base, donor, basis, seed=123)
    selected = ((donor - base) @ basis) @ basis.T
    off = controlled - base
    assert torch.allclose(off @ basis, torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(off.norm(dim=1), selected.norm(dim=1), atol=1e-6)
    assert torch.equal(controlled, equal_energy_offspace_states(
        base, donor, basis, seed=123))
    corrupted = equal_norm_component_corruption_states(base, mean, basis, seed=456)
    native_component = ((base - mean) @ basis) @ basis.T
    corrupt_component = corrupted - mean - ((base - mean) - native_component)
    assert torch.allclose(corrupt_component @ basis, torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(corrupt_component.norm(dim=1),
                          native_component.norm(dim=1), atol=1e-6)


def test_factor_confirmation_rows_are_bidirectional_complete_recipient_groups():
    suite = counterfactual_suite(74214, discovery_groups=1, confirmation_groups=1)
    groups = [asdict(suite["confirmation"][0])]
    torch.manual_seed(66)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    basis = torch.eye(WIDTH)[:, :2]
    for factor, surface in (("content", "marked"), ("binding", "marked"),
                            ("order", "f0_marked")):
        rows = factor_transfer_rows(
            net, groups, spec, basis, factor_name=factor,
            condition_prefix=f"{factor}_selected", surfaces=(surface,),
            episode_loader=episode_from_record)
        assert len(rows) == 6
        assert {row["direction"] for row in rows} == {"forward", "reverse"}
        assert {row["recipient"] for row in rows} == {0, 1, 2}
        assert len({row["item_id"] for row in rows}) == 6
    controlled = factor_equal_energy_rows(
        net, groups, spec, basis, factor_name="content",
        condition_prefix="content_equal_energy_00", surfaces=("marked",),
        episode_loader=episode_from_record, seed=74301)
    assert len(controlled) == 6
    assert all(row["condition"] == "content_equal_energy_00_marked"
               for row in controlled)
