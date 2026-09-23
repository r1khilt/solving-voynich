"""Synthetic-ground-truth checks for TEACH-0013 gauge-aware geometry."""

import math

import torch

from voynich.workspace.teacher13_geometry import (
    balanced_contrast_geometry,
    linear_cka,
    orthogonal_procrustes,
    orthogonalize_basis,
    principal_angles,
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
