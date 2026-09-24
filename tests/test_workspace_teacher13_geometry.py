"""Synthetic-ground-truth checks for TEACH-0013 gauge-aware geometry."""

import math

import torch

from voynich.workspace.teacher13_geometry import (
    balanced_contrast_geometry,
    blocked_contrast_geometry,
    deranged_blocked_bases,
    haar_random_bases,
    linear_cka,
    orthogonal_factor_geometry,
    orthonormal_union,
    orthogonal_procrustes,
    orthogonalize_basis,
    principal_angles,
    select_causal_rank,
    select_joint_causal_rank,
)


def test_balanced_factorial_contrast_recovers_factor_not_nuisance_direction():
    rows, factor, nuisance = [], [], []
    for content in (-2.0, 0.0, 2.0):
        for position in (-3.0, 3.0):
            for repeat in range(3):
                rows.append([content, position, .01 * repeat, 0.0])
                factor.append(content)
                nuisance.append(position)
    result = balanced_contrast_geometry(
        torch.tensor(rows), factor, nuisance, maximum_rank=2)
    assert result.basis.shape == (4, 1)
    assert abs(float(result.basis[:, 0] @ torch.tensor([1., 0., 0., 0.],
                                                       dtype=torch.double))) > .999
    assert result.participation_ratio == 1


def test_orthogonalization_angles_and_procrustes_are_gauge_aware():
    first = torch.eye(5, dtype=torch.double)[:, :2]
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.double)
    same = first @ rotation
    assert torch.allclose(principal_angles(first, same), torch.zeros(2, dtype=torch.double))
    candidate = torch.eye(5, dtype=torch.double)[:, 1:4]
    remainder = orthogonalize_basis(candidate, first)
    assert remainder.shape == (5, 2)
    assert torch.allclose(first.T @ remainder, torch.zeros(2, 2, dtype=torch.double), atol=1e-8)

    torch.manual_seed(4)
    source = torch.randn(30, 5, dtype=torch.double)
    q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.double))
    target = source @ q + 2
    mapping = orthogonal_procrustes(source, target)
    assert mapping.normalized_residual < 1e-12
    assert torch.allclose(mapping.transform(source), target, atol=1e-10)


def test_linear_cka_is_rotation_scale_invariant_but_not_causal_evidence():
    torch.manual_seed(5)
    left = torch.randn(40, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    assert linear_cka(left, 3 * left @ q) > .999999
    assert 0 <= linear_cka(left, torch.randn(40, 6)) <= 1
    assert math.isnan(linear_cka(torch.ones(4, 2), torch.ones(4, 2)))


def test_blocked_contrasts_preserve_heterogeneous_within_group_directions():
    rows, factor, nuisance, blocks = [], [], [], []
    directions = (torch.tensor([1., 0., 0., 0.]),
                  torch.tensor([0., 1., 0., 0.]),
                  torch.tensor([1., 1., 0., 0.]))
    for block, direction in enumerate(directions):
        for level in (-1., 1.):
            for position in (-2., 2.):
                rows.append(level * direction + torch.tensor([0., 0., position, 0.]))
                factor.append(level)
                nuisance.append(position)
                blocks.append(block)
    result = blocked_contrast_geometry(
        torch.stack(rows), factor, nuisance, blocks, maximum_rank=4)
    assert result.basis.shape == (4, 2)
    expected = torch.eye(4, dtype=torch.double)[:, :2]
    assert torch.allclose(principal_angles(result.basis, expected),
                          torch.zeros(2, dtype=torch.double), atol=1e-7)


def test_registered_orthogonalization_and_haar_controls_are_deterministic():
    content = torch.eye(6, dtype=torch.double)[:, :2]
    binding = torch.eye(6, dtype=torch.double)[:, 1:4]
    order = torch.eye(6, dtype=torch.double)[:, 3:5]
    result = orthogonal_factor_geometry(content, binding, order)
    assert result.forward["content"].shape[1] == 2
    assert result.forward["binding"].shape[1] == 2
    assert result.forward["order"].shape[1] == 1
    assert result.reverse["content"].shape[1] == 1
    assert torch.allclose(
        result.forward["content"].T @ result.forward["binding"],
        torch.zeros(2, 2, dtype=torch.double), atol=1e-8)
    left = haar_random_bases(6, 2, 32, seed=73221, orthogonal_to=content)
    right = haar_random_bases(6, 2, 32, seed=73221, orthogonal_to=content)
    assert len(left) == 32 and all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    assert all(torch.allclose(content.T @ basis, torch.zeros(2, 2, dtype=torch.double),
                              atol=1e-8) for basis in left)


def test_causal_rank_selection_takes_smallest_registered_finite_effect_match():
    measurements = [
        {"rank": 1, "item_accuracy": .7, "group_accuracy": .5,
         "mean_probability_gain": .7},
        {"rank": 2, "item_accuracy": .81, "group_accuracy": .61,
         "mean_probability_gain": .94},
        {"rank": 4, "item_accuracy": .9, "group_accuracy": .8,
         "mean_probability_gain": .96},
    ]
    result = select_causal_rank(
        measurements, full_probability_gain=1.0,
        minimum_item_accuracy=.8, minimum_group_accuracy=.6)
    assert result["selection"] == 4
    assert not result["measurements"]["2"]["qualified"]
    assert result["measurements"]["4"]["qualified"]


def test_joint_rank_selection_requires_the_same_rank_to_pass_both_seeds():
    seed0 = [
        {"rank": 1, "item_accuracy": .81, "group_accuracy": .61,
         "mean_probability_gain": .96},
        {"rank": 2, "item_accuracy": .9, "group_accuracy": .8,
         "mean_probability_gain": .98},
    ]
    seed1 = [
        {"rank": 1, "item_accuracy": .7, "group_accuracy": .5,
         "mean_probability_gain": .96},
        {"rank": 2, "item_accuracy": .85, "group_accuracy": .7,
         "mean_probability_gain": .97},
    ]
    result = select_joint_causal_rank(
        {"0": seed0, "1": seed1},
        full_probability_gain_by_seed={"0": 1.0, "1": 1.0},
        minimum_item_accuracy=.8, minimum_group_accuracy=.6,
        candidate_ranks=(1, 2))
    assert result["selection"] == 2
    assert result["by_seed"]["0"]["selection"] == 1
    assert result["by_seed"]["1"]["selection"] == 2


def test_deranged_blocked_bases_are_deterministic_complete_design_nulls():
    rows, factor, nuisance, blocks = [], [], [], []
    for block in range(8):
        direction = torch.randn(5)
        for nuisance_cell in range(4):
            offset = torch.randn(5)
            for level, sign in ((0, -1.), (1, 1.)):
                rows.append(offset + sign * direction)
                factor.append(level)
                nuisance.append(nuisance_cell)
                blocks.append(block)
    left = deranged_blocked_bases(
        torch.stack(rows), factor, nuisance, blocks, count=4, seed=991)
    right = deranged_blocked_bases(
        torch.stack(rows), factor, nuisance, blocks, count=4, seed=991)
    assert len(left) == 4 and all(
        item.basis.shape[0] == 5 and item.basis.shape[1] > 0 for item in left)
    assert all(torch.equal(a.basis, b.basis) and torch.equal(a.eigenvalues, b.eigenvalues)
               and a.donor_blocks == b.donor_blocks
               for a, b in zip(left, right, strict=True))
    assert all(all(index != donor for index, donor in enumerate(item.donor_blocks))
               for item in left)


def test_orthonormal_union_deduplicates_overlapping_component_directions():
    first = torch.eye(5, dtype=torch.double)[:, :2]
    second = torch.eye(5, dtype=torch.double)[:, 1:4]
    union = orthonormal_union(first, second)
    assert union.shape == (5, 4)
    assert torch.allclose(union.T @ union, torch.eye(4, dtype=torch.double))
    expected = torch.eye(5, dtype=torch.double)[:, :4]
    assert torch.allclose(union @ union.T, expected @ expected.T)
