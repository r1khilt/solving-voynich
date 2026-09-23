"""Gauge-aware factorial geometry for prospective TEACH-0013 subspace analysis."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ContrastGeometry:
    mean: Tensor
    basis: Tensor
    eigenvalues: Tensor
    factor_levels: tuple
    nuisance_cells: tuple
    participation_ratio: float


def _as_matrix(states: Tensor) -> Tensor:
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
        raise ValueError("States must have shape [at least two samples, width]")
    if not torch.isfinite(states).all():
        raise ValueError("States contain nonfinite values")
    return states.double()


def participation_ratio(values: Tensor) -> float:
    values = values.double().clamp_min(0)
    denominator = values.square().sum()
    return 0.0 if denominator == 0 else float(values.sum().square() / denominator)


def balanced_contrast_geometry(states: Tensor, factor, nuisance, *,
                               maximum_rank: int | None = None,
                               tolerance: float = 1e-10) -> ContrastGeometry:
    """Estimate a factor subspace after equal-weight marginalization over nuisance cells."""
    states = _as_matrix(states)
    factor, nuisance = tuple(factor), tuple(nuisance)
    if len(factor) != states.shape[0] or len(nuisance) != states.shape[0]:
        raise ValueError("Labels and states must have equal sample count")
    factors = tuple(sorted(set(factor), key=repr))
    nuisances = tuple(sorted(set(nuisance), key=repr))
    if len(factors) < 2 or not nuisances:
        raise ValueError("Need at least two factor levels and one nuisance cell")
    cell_means = {}
    for factor_level in factors:
        for nuisance_cell in nuisances:
            selected = [index for index, (left, right) in enumerate(zip(
                factor, nuisance, strict=True))
                if left == factor_level and right == nuisance_cell]
            if not selected:
                raise ValueError("Factorial design is not complete")
            cell_means[(factor_level, nuisance_cell)] = states[selected].mean(0)
    marginalized = torch.stack([
        torch.stack([cell_means[(level, cell)] for cell in nuisances]).mean(0)
        for level in factors
    ])
    mean = marginalized.mean(0)
    contrasts = marginalized - mean
    covariance = contrasts.T @ contrasts / len(factors)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    scale = max(float(eigenvalues[0].abs()), 1.0)
    rank = int((eigenvalues > tolerance * scale).sum())
    if maximum_rank is not None:
        if maximum_rank <= 0:
            raise ValueError("maximum_rank must be positive")
        rank = min(rank, maximum_rank)
    basis = eigenvectors[:, :rank]
    return ContrastGeometry(mean, basis, eigenvalues, factors, nuisances,
                            participation_ratio(eigenvalues))


def orthogonalize_basis(basis: Tensor, against: Tensor, *, tolerance=1e-8) -> Tensor:
    """Remove an already assigned subspace and return an orthonormal remainder."""
    basis, against = basis.double(), against.double()
    if basis.ndim != 2 or against.ndim != 2 or basis.shape[0] != against.shape[0]:
        raise ValueError("Bases must be width-by-rank matrices with matching width")
    if against.shape[1]:
        remainder = basis - against @ (against.T @ basis)
    else:
        remainder = basis
    if remainder.shape[1] == 0:
        return remainder
    u, singular, _ = torch.linalg.svd(remainder, full_matrices=False)
    if singular.numel() == 0:
        return u[:, :0]
    keep = singular > tolerance * max(float(singular[0]), 1.0)
    return u[:, keep]


def principal_angles(left: Tensor, right: Tensor) -> Tensor:
    """Principal angles in radians between two orthonormal subspaces."""
    left, right = left.double(), right.double()
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Subspaces must have matching ambient width")
    if left.shape[1] == 0 or right.shape[1] == 0:
        return torch.empty(0, dtype=torch.double)
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return singular.acos()


@dataclass(frozen=True)
class ProcrustesMap:
    source_mean: Tensor
    target_mean: Tensor
    rotation: Tensor
    normalized_residual: float

    def transform(self, states: Tensor) -> Tensor:
        return (states.double() - self.source_mean) @ self.rotation + self.target_mean


def orthogonal_procrustes(source: Tensor, target: Tensor) -> ProcrustesMap:
    """Fit an orthogonal paired-state map on discovery data only."""
    source, target = _as_matrix(source), _as_matrix(target)
    if source.shape != target.shape:
        raise ValueError("Paired source/target states must have identical shape")
    source_mean, target_mean = source.mean(0), target.mean(0)
    left, right = source - source_mean, target - target_mean
    u, _, vh = torch.linalg.svd(left.T @ right, full_matrices=False)
    rotation = u @ vh
    residual = torch.linalg.norm(left @ rotation - right)
    denominator = torch.linalg.norm(right)
    normalized = math.inf if denominator == 0 else float(residual / denominator)
    return ProcrustesMap(source_mean, target_mean, rotation, normalized)


def linear_cka(left: Tensor, right: Tensor) -> float:
    """Centered linear CKA; descriptive similarity, not a causal score."""
    left, right = _as_matrix(left), _as_matrix(right)
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must have equal sample count")
    left, right = left - left.mean(0), right - right.mean(0)
    cross = torch.linalg.norm(left.T @ right).square()
    denominator = torch.linalg.norm(left.T @ left) * torch.linalg.norm(right.T @ right)
    return math.nan if denominator == 0 else float(cross / denominator)
