"""Finite, normalized character backoff priors for source-only calibration.

The recursive Dirichlet mean uses a plug-in lower-order distribution; it is
not full Bayesian inference over a hierarchy. Absolute discount uses raw
counts at every order, not Kneser--Ney continuation counts.
"""
from __future__ import annotations

import numpy as np

from voynich.naibbe_key_search import CharacterLM


CONFIGS = ([{"family": "legacy", "parameter": None}]
           + [{"family": "dirichlet", "parameter": value}
              for value in (.25, 1., 4., 16., 64., 256.)]
           + [{"family": "absolute_discount", "parameter": value}
              for value in (.25, .5, .75, .9)])


def encode(text: str, alphabet: str) -> np.ndarray:
    if not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet must contain unique characters")
    indices = {letter: i for i, letter in enumerate(alphabet)}
    try:
        return np.array([indices[letter] for letter in text], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Text contains a character outside the alphabet") from error


def train(text: str, alphabet: str, config: dict) -> CharacterLM:
    values = encode(text, alphabet)
    if len(values) < 4:
        raise ValueError("Need at least four source characters")
    family, parameter = config["family"], config["parameter"]
    if family == "legacy":
        if parameter is not None:
            raise ValueError("Legacy prior has no variable parameter")
        return CharacterLM.train(text, alphabet)
    if family not in {"dirichlet", "absolute_discount"}:
        raise ValueError("Unknown source-prior family")
    if parameter is None or not np.isfinite(parameter) or parameter <= 0:
        raise ValueError("Parameter must be finite and positive")
    if family == "absolute_discount" and parameter >= 1:
        raise ValueError("Discount must lie strictly between zero and one")
    size, logs, lower = len(alphabet), [], None
    for order in range(1, 5):
        windows = np.lib.stride_tricks.sliding_window_view(values, order)
        codes = windows @ (size ** np.arange(order - 1, -1, -1))
        counts = np.bincount(codes, minlength=size**order).reshape(-1, size)
        total = counts.sum(axis=1, keepdims=True)
        if order == 1:
            probabilities = (counts + .1) / (total + .1 * size)
        else:
            # Full history's final order-2 characters select the lower row.
            backoff = lower[np.arange(size ** (order - 1)) % len(lower)]
            if family == "dirichlet":
                probabilities = (counts + parameter * backoff) / (total + parameter)
            else:
                seen = total[:, 0] > 0
                probabilities = backoff.copy()
                types = (counts[seen] > 0).sum(axis=1, keepdims=True)
                probabilities[seen] = (np.maximum(counts[seen] - parameter, 0)
                                       + parameter * types * backoff[seen]) / total[seen]
        if (not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0)
                or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0)):
            raise AssertionError("Invalid conditional probability table")
        logs.append(np.log(probabilities.ravel()))
        lower = probabilities
    return CharacterLM(alphabet, logs)


def log_terms(lm: CharacterLM, text: str) -> np.ndarray:
    """Each character's conditional log probability; reset at text boundary."""
    values = encode(text, lm.alphabet)
    terms = np.empty(len(values), dtype=float)
    size = len(lm.alphabet)
    for order in range(1, min(4, len(values) + 1)):
        code = int(values[:order] @ (size ** np.arange(order - 1, -1, -1)))
        terms[order - 1] = lm.logs[order - 1][code]
    if len(values) >= 4:
        windows = np.lib.stride_tricks.sliding_window_view(values, 4)
        terms[3:] = lm.logs[3][windows @ (size ** np.arange(3, -1, -1))]
    return terms
