"""Gauge-aware factorial geometry for prospective TEACH-0013 subspace analysis."""

from dataclasses import dataclass
import math
from collections.abc import Iterable

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


@dataclass(frozen=True)
class OrthogonalFactorGeometry:
    """Raw and order-dependent orthogonalized content/binding/order bases."""

    raw: dict[str, Tensor]
    forward: dict[str, Tensor]
    reverse: dict[str, Tensor]
    raw_principal_angles: dict[str, Tensor]


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


def blocked_contrast_geometry(states: Tensor, factor, nuisance, blocks, *,
                              maximum_rank: int | None = None,
                              tolerance: float = 1e-10) -> ContrastGeometry:
    """Estimate covariance over within-block factorial contrasts.

    Each block must contain every factor-by-nuisance cell. Marginalizing nuisance
    within a block and then pooling its centered factor contrasts preserves
    heterogeneous causal directions that a single global binary mean would collapse.
    """
    states = _as_matrix(states)
    factor, nuisance, blocks = tuple(factor), tuple(nuisance), tuple(blocks)
    if any(len(labels) != states.shape[0] for labels in (factor, nuisance, blocks)):
        raise ValueError("Labels, blocks and states must have equal sample count")
    factors = tuple(sorted(set(factor), key=repr))
    nuisances = tuple(sorted(set(nuisance), key=repr))
    block_levels = tuple(sorted(set(blocks), key=repr))
    if len(factors) < 2 or not nuisances or not block_levels:
        raise ValueError("Need factors, nuisance cells and blocks")
    contrast_rows, means = [], []
    for block in block_levels:
        marginalized = []
        for factor_level in factors:
            cell_means = []
            for nuisance_cell in nuisances:
                selected = [index for index, labels in enumerate(zip(
                    factor, nuisance, blocks, strict=True))
                            if labels == (factor_level, nuisance_cell, block)]
                if not selected:
                    raise ValueError("Blocked factorial design is not complete")
                cell_means.append(states[selected].mean(0))
            marginalized.append(torch.stack(cell_means).mean(0))
        marginalized = torch.stack(marginalized)
        block_mean = marginalized.mean(0)
        means.append(block_mean)
        contrast_rows.extend(marginalized - block_mean)
    contrasts = torch.stack(contrast_rows)
    covariance = contrasts.T @ contrasts / contrasts.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    scale = max(float(eigenvalues[0].abs()), 1.0)
    rank = int((eigenvalues > tolerance * scale).sum())
    if maximum_rank is not None:
        if maximum_rank <= 0:
            raise ValueError("maximum_rank must be positive")
        rank = min(rank, maximum_rank)
    return ContrastGeometry(
        torch.stack(means).mean(0), eigenvectors[:, :rank], eigenvalues,
        factors, nuisances, participation_ratio(eigenvalues))


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


def orthogonal_factor_geometry(content: Tensor, binding: Tensor,
                               order: Tensor) -> OrthogonalFactorGeometry:
    """Orthogonalize in registered forward order and diagnostic reverse order."""
    raw = {"content": content.double(), "binding": binding.double(),
           "order": order.double()}
    widths = {basis.shape[0] for basis in raw.values() if basis.ndim == 2}
    if len(widths) != 1 or any(basis.ndim != 2 for basis in raw.values()):
        raise ValueError("Factor bases must be width-by-rank matrices")

    def sequential(names):
        assigned = torch.empty(next(iter(widths)), 0, dtype=torch.double)
        result = {}
        for name in names:
            result[name] = orthogonalize_basis(raw[name], assigned)
            assigned = torch.cat((assigned, result[name]), dim=1)
        return result

    pairs = (("content", "binding"), ("content", "order"), ("binding", "order"))
    angles = {f"{left}:{right}": principal_angles(raw[left], raw[right])
              for left, right in pairs}
    return OrthogonalFactorGeometry(
        raw, sequential(("content", "binding", "order")),
        sequential(("order", "binding", "content")), angles)


def haar_random_bases(width: int, rank: int, count: int, *, seed: int,
                      orthogonal_to: Tensor | None = None) -> tuple[Tensor, ...]:
    """Generate deterministic Haar-random matched-rank controls."""
    if width <= 0 or rank <= 0 or count <= 0 or rank > width:
        raise ValueError("Invalid Haar basis dimensions")
    projector = None
    available = width
    if orthogonal_to is not None:
        orthogonal_to = orthogonal_to.double()
        if orthogonal_to.ndim != 2 or orthogonal_to.shape[0] != width:
            raise ValueError("Orthogonal exclusion basis has the wrong width")
        projector = torch.eye(width, dtype=torch.double) \
            - orthogonal_to @ orthogonal_to.T
        available = width - orthogonal_to.shape[1]
        if rank > available:
            raise ValueError("Requested random rank exceeds orthogonal complement")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for _ in range(count):
        sample = torch.randn(width, rank, generator=generator, dtype=torch.double)
        if projector is not None:
            sample = projector @ sample
        q, r = torch.linalg.qr(sample, mode="reduced")
        signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
        results.append(q * signs)
    return tuple(results)


def select_causal_rank(measurements: Iterable[dict], *, full_probability_gain: float,
                       minimum_item_accuracy: float, minimum_group_accuracy: float,
                       effect_fraction: float = .95,
                       candidate_ranks=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    """Freeze the smallest registered rank recovering the full finite effect."""
    if not math.isfinite(full_probability_gain) or full_probability_gain <= 0:
        raise ValueError("Full-state probability gain must be finite and positive")
    if not 0 < effect_fraction <= 1:
        raise ValueError("Effect fraction must lie in (0, 1]")
    allowed = tuple(candidate_ranks)
    if not allowed or any(type(rank) is not int or rank <= 0 for rank in allowed) \
            or tuple(sorted(set(allowed))) != allowed:
        raise ValueError("Candidate ranks must be sorted unique positive integers")
    table = {}
    for row in measurements:
        rank = row.get("rank")
        if rank not in allowed or rank in table:
            raise ValueError("Rank measurements must uniquely cover registered candidates")
        for name in ("item_accuracy", "group_accuracy", "mean_probability_gain"):
            if name not in row or not math.isfinite(row[name]):
                raise ValueError(f"Nonfinite or missing rank measurement: {name}")
        qualified = (row["mean_probability_gain"] >= effect_fraction * full_probability_gain
                     and row["item_accuracy"] >= minimum_item_accuracy
                     and row["group_accuracy"] >= minimum_group_accuracy)
        table[rank] = {**row, "qualified": qualified,
                       "effect_fraction": row["mean_probability_gain"]
                       / full_probability_gain}
    selection = next((table[rank] for rank in allowed
                      if rank in table and table[rank]["qualified"]), None)
    return {"candidate_ranks": list(allowed), "effect_fraction_threshold": effect_fraction,
            "minimum_item_accuracy": minimum_item_accuracy,
            "minimum_group_accuracy": minimum_group_accuracy,
            "full_probability_gain": full_probability_gain,
            "measurements": {str(rank): table[rank] for rank in sorted(table)},
            "selection": None if selection is None else selection["rank"]}


def select_joint_causal_rank(measurements_by_seed: dict[str, Iterable[dict]], *,
                             full_probability_gain_by_seed: dict[str, float],
                             minimum_item_accuracy: float,
                             minimum_group_accuracy: float,
                             effect_fraction: float = .95,
                             candidate_ranks=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    """Choose one smallest rank satisfying the finite-effect rule in every seed."""
    if set(measurements_by_seed) != set(full_probability_gain_by_seed) \
            or not measurements_by_seed:
        raise ValueError("Joint rank selection requires matching nonempty seed maps")
    by_seed = {seed: select_causal_rank(
        rows, full_probability_gain=full_probability_gain_by_seed[seed],
        minimum_item_accuracy=minimum_item_accuracy,
        minimum_group_accuracy=minimum_group_accuracy,
        effect_fraction=effect_fraction, candidate_ranks=candidate_ranks)
        for seed, rows in measurements_by_seed.items()}
    common = [rank for rank in candidate_ranks
              if all(str(rank) in result["measurements"]
                     and result["measurements"][str(rank)]["qualified"]
                     for result in by_seed.values())]
    return {"candidate_ranks": list(candidate_ranks), "by_seed": by_seed,
            "selection": common[0] if common else None,
            "rule": "smallest_rank_qualified_in_every_seed"}


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
