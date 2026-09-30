"""Held-out predictive calibration of depth-specific interpolation masses.

Count statistics and the exact suffix automaton are unchanged. This optimizes
predictive loss, not the marginal evidence of a fully Bayesian language model.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voynich.sparse_suffix_source import SuffixSource


class DepthSource(SuffixSource):
    def __init__(self, alphabet, masses, counts):
        masses = tuple(masses)
        if any(isinstance(m, bool) or not math.isfinite(m) or m <= 0 for m in masses):
            raise ValueError('Masses must be positive and finite')
        super().__init__(alphabet, len(masses), 1., counts)
        self.masses = tuple(float(m) for m in masses)
        probabilities = np.empty_like(self.probabilities)
        probabilities[0] = self.probabilities[0]
        for index, context in enumerate(self.contexts[1:], 1):
            row = self.counts[context]
            mass = self.masses[len(context) - 1]
            values = np.array([row.get(c, 0) for c in self.alphabet], dtype=np.float64)
            lower = probabilities[self.indices[context[1:]]]
            probabilities[index] = (values + mass * lower) / (sum(row.values()) + mass)
        if (not np.all(np.isfinite(probabilities)) or not np.all(probabilities > 0)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-12)):
            raise ValueError('Invalid calibrated probability table')
        probabilities.setflags(write=False)
        self.probabilities = probabilities

    def to_dict(self):
        return {'schema_version': 1, 'kind': 'depth_dirichlet', 'alphabet': list(self.alphabet),
                'masses': list(self.masses), 'counts': {c: dict(self.counts[c]) for c in self.contexts}}

    @classmethod
    def from_dict(cls, raw):
        if (set(raw) != {'schema_version','kind','alphabet','masses','counts'}
                or type(raw['schema_version']) is not int or raw['schema_version'] != 1
                or raw['kind'] != 'depth_dirichlet'):
            raise ValueError('Invalid calibrated source schema')
        return cls(raw['alphabet'], raw['masses'], raw['counts'])


@dataclass(frozen=True)
class Features:
    root: np.ndarray
    next_counts: np.ndarray
    context_counts: np.ndarray

    def __post_init__(self):
        root = np.array(self.root, dtype=np.float64, copy=True)
        next_counts = np.array(self.next_counts, dtype=np.float64, copy=True)
        context_counts = np.array(self.context_counts, dtype=np.float64, copy=True)
        if (root.ndim != 1 or not root.size or next_counts.ndim != 2
                or next_counts.shape != context_counts.shape or next_counts.shape[1] != root.size
                or not np.all(np.isfinite(root)) or not np.all((0 < root) & (root <= 1))
                or not np.all(np.isfinite(next_counts)) or not np.all(np.isfinite(context_counts))
                or not np.all((next_counts >= 0) & (next_counts <= context_counts))):
            raise ValueError('Invalid calibration features')
        for name, array in (('root',root), ('next_counts',next_counts), ('context_counts',context_counts)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def extract_features(source: SuffixSource, records: Sequence[str]) -> Features:
    if not records or any(not r or set(r) - set(source.alphabet) for r in records):
        raise ValueError('Nonempty in-alphabet calibration records required')
    n = sum(map(len, records))
    root = np.empty(n)
    counts, totals = np.zeros((source.order, n)), np.zeros((source.order, n))
    offset = 0
    for record in records:
        for i, char in enumerate(record):
            root[offset] = source.probabilities[0, source.letters[char]]
            for depth in range(1, min(i, source.order) + 1):
                row = source.counts.get(record[i - depth:i])
                if row is not None:
                    counts[depth - 1, offset], totals[depth - 1, offset] = row.get(char, 0), sum(row.values())
            offset += 1
    return Features(root, counts, totals)


def loss_gradient(log_masses, features: Features):
    """Mean natural-log predictive loss and exact reverse-mode derivative."""
    theta = np.asarray(log_masses, dtype=np.float64)
    if theta.shape != (features.next_counts.shape[0],) or not np.all(np.isfinite(theta)):
        raise ValueError('Log-mass vector dimension or values invalid')
    masses = np.exp(theta)
    if not np.all(np.isfinite(masses)) or not np.all(masses > 0):
        raise ValueError('Mass overflow/underflow')
    probabilities = [features.root]
    for d, mass in enumerate(masses):
        probabilities.append((features.next_counts[d] + mass * probabilities[-1]) / (features.context_counts[d] + mass))
    terminal = probabilities[-1]
    if not np.all(np.isfinite(terminal)) or not np.all(terminal > 0):
        raise ValueError('Nonfinite calibration loss')
    loss = float(-np.log(terminal).mean())
    adjoint = -1 / (terminal * terminal.size)
    gradient = np.zeros_like(theta)
    for d in range(len(masses) - 1, -1, -1):
        factor = masses[d] / (features.context_counts[d] + masses[d])
        gradient[d] = np.sum(adjoint * factor * (probabilities[d] - probabilities[d + 1]))
        adjoint = adjoint * factor
    return loss, gradient


def calibrate(features: Features, initial_mass: float, *, steps=500, learning_rate=.05,
              lower=.25, upper=4096.):
    """Deterministic projected full-batch Adam; return every iterate, no early stop."""
    if (type(steps) is not int or steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0
            or not 0 < lower <= initial_mass <= upper or not math.isfinite(upper)):
        raise ValueError('Invalid bounded optimizer settings')
    theta = np.full(features.next_counts.shape[0], math.log(initial_mass))
    first, second, trace = np.zeros_like(theta), np.zeros_like(theta), []
    for step in range(steps + 1):
        loss, gradient = loss_gradient(theta, features)
        trace.append({'step':step, 'loss_nats_per_character':loss,
                      'log_masses':theta.tolist(), 'gradient_norm':float(np.linalg.norm(gradient))})
        if step == steps:
            break
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient ** 2
        direction = (first / (1 - .9 ** (step + 1))) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
        theta = np.clip(theta - learning_rate * direction, math.log(lower), math.log(upper))
    return trace
