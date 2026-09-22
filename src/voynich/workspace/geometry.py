"""Small auditable geometric operations; no model- or label-based selection."""

import numpy as np


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Directions must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Zero directions cannot define coordinates")
    return matrix / norms


def coordinate_swap(hidden, directions, *, strength=1.0, max_condition=1e8):
    """Minimum Euclidean displacement exchanging two normalized readout coordinates.

    In a nonorthogonal frame, adding vector1-vector2 is NOT a coordinate swap.
    Solve the two-by-two Gram system, preserving the orthogonal complement.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    rows = normalize_rows(directions)
    if rows.shape != (2, hidden.size) or hidden.ndim != 1 or not np.isfinite(hidden).all():
        raise ValueError("Two directions and one compatible hidden vector required")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Invalid intervention strength")
    gram = rows @ rows.T
    condition = float(np.linalg.cond(gram))
    if condition > max_condition:
        raise ValueError("Readout coordinates are too collinear for a stable swap")
    coordinates = rows @ hidden
    delta = strength * (rows.T @ np.linalg.solve(gram, coordinates[::-1] - coordinates))
    return delta.astype(np.float32), {
        "gram_condition": condition, "before": coordinates.tolist(),
        "after": (rows @ (hidden + delta)).tolist(), "delta_norm": float(np.linalg.norm(delta)),
        "strength": float(strength),
    }


def matched_random_delta(reference, *, seed, orthogonal_to=None):
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("Reference displacement must be a finite vector")
    vector = np.random.default_rng(seed).normal(size=reference.size)
    if orthogonal_to is not None:
        rows = normalize_rows(orthogonal_to)
        vector -= rows.T @ (np.linalg.pinv(rows @ rows.T) @ (rows @ vector))
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("No random direction remains in requested complement")
    return (vector * np.linalg.norm(reference) / norm).astype(np.float32)


def sparse_nonnegative_projection(hidden, directions, *, max_features=5):
    """Greedy support selection + exact NNLS on support; a cone approximation.

    This is an explicit restricted-dictionary comparator, not the full J-space
    or Anthropic's gradient-pursuit implementation. Always retain residual error.
    """
    from scipy.optimize import nnls

    hidden = np.asarray(hidden, dtype=np.float64)
    rows = normalize_rows(directions)
    if hidden.shape != (rows.shape[1],) or not np.isfinite(hidden).all():
        raise ValueError("Invalid hidden vector")
    if type(max_features) is not int or not 1 <= max_features <= len(rows):
        raise ValueError("Invalid sparsity")
    residual, selected, coeff = hidden.copy(), [], np.zeros(len(rows))
    for _ in range(max_features):
        alignment = rows @ residual
        alignment[selected] = -np.inf
        index = int(alignment.argmax())
        if alignment[index] <= 1e-12:
            break
        selected.append(index)
        weights, _ = nnls(rows[selected].T, hidden)
        coeff[:] = 0
        coeff[selected] = weights
        residual = hidden - coeff @ rows
    energy = float(hidden @ hidden)
    return (hidden - residual).astype(np.float32), {
        "support": [i for i in selected if coeff[i] > 0],
        "coefficients": coeff.tolist(), "residual_fraction": float(residual @ residual / energy) if energy else 0.0,
        "method": "greedy restricted support with NNLS; not a globally optimal sparse decomposition",
    }
