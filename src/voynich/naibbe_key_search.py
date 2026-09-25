"""Explicit shared substitution key and exact finite-order lattice decoding.

The lattice and equivalence classes are GIVEN. This is a restricted recovery
instrument, not discovery of the codebook or cipher family. Scores are Viterbi
search objectives, not marginal evidence for a language or historical channel.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class CharacterLM:
    alphabet: str
    logs: list[np.ndarray]

    @classmethod
    def train(cls, text: str, alphabet: str) -> CharacterLM:
        """Normalized interpolated 1--4 gram probabilities; no word dictionary."""
        size = len(alphabet)
        index = {c: i for i, c in enumerate(alphabet)}
        values = np.array([index[c] for c in text], dtype=np.int64)
        if len(values) < 4:
            raise ValueError("Need at least four training characters")
        raw = []
        weights = np.array([.02, .08, .20, .70])
        logs = []
        for order in range(1, 5):
            windows = np.lib.stride_tricks.sliding_window_view(values, order)
            powers = size ** np.arange(order - 1, -1, -1)
            counts = np.bincount(windows @ powers, minlength=size**order).reshape(-1, size)
            probs = (counts + .1) / (counts.sum(axis=1, keepdims=True) + .1 * size)
            raw.append(probs.ravel())
            ids = np.arange(size**order)
            mixed = sum(weights[j] * raw[j][ids % (size ** (j + 1))] for j in range(order))
            mixed /= weights[:order].sum()
            logs.append(np.log(mixed))
        return cls(alphabet, logs)

    def advance(self, context: tuple[int, ...], emitted: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
        score = 0.
        size = len(self.alphabet)
        for value in emitted:
            sequence = (*context, value)
            code = 0
            for digit in sequence:
                code = code * size + digit
            score += float(self.logs[len(sequence) - 1][code])
            context = sequence[-3:]
        return context, score

    def score(self, sequence: list[int] | np.ndarray) -> float:
        return self.advance((), tuple(map(int, sequence)))[1]


def viterbi(lattice: list[list[tuple[int, ...]]], key: np.ndarray,
            lm: CharacterLM) -> tuple[float, list[int]]:
    """Exact best legal parse, merging only equal last-three-letter states."""
    states = {(): 0.}
    back = []
    for alternatives in lattice:
        next_states = {}
        pointers = {}
        for context, old_score in states.items():
            for choice, emitted in enumerate(alternatives):
                next_context, increment = lm.advance(context, tuple(int(key[c]) for c in emitted))
                score = old_score + increment
                if next_context not in next_states or score > next_states[next_context]:
                    next_states[next_context] = score
                    pointers[next_context] = (context, choice)
        if not next_states:
            raise ValueError("Empty token lattice")
        states = next_states
        back.append(pointers)
    if not back:
        return 0., []
    context = max(states, key=states.get)
    score = states[context]
    choices = []
    for pointers in reversed(back):
        context, choice = pointers[context]
        choices.append(choice)
    return score, choices[::-1]


def flatten(lattice: list[list[tuple[int, ...]]], choices: list[int]) -> np.ndarray:
    return np.array([c for paths, choice in zip(lattice, choices, strict=True)
                     for c in paths[choice]], dtype=np.int64)


class SequenceObjective:
    """Exact source objective compressed into ngram counts, including prefix."""

    def __init__(self, sequence: np.ndarray, lm: CharacterLM):
        self.lm = lm
        self.rows = []
        for order in range(1, 5):
            if order < 4:
                grams = [tuple(sequence[:order])] if len(sequence) >= order else []
            else:
                grams = map(tuple, np.lib.stride_tricks.sliding_window_view(sequence, order)) \
                    if len(sequence) >= order else []
            counts = Counter(grams)
            self.rows.append((np.array(list(counts), dtype=np.int64).reshape(-1, order),
                              np.array(list(counts.values())),
                              len(lm.alphabet) ** np.arange(order - 1, -1, -1)))

    def scores(self, keys: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(keys))
        for order, (grams, counts, powers) in enumerate(self.rows):
            if len(grams):
                ids = (keys[:, grams] * powers).sum(axis=-1)
                scores += (self.lm.logs[order][ids] * counts).sum(axis=-1)
        return scores


def climb(initial: np.ndarray, objective: SequenceObjective, max_steps: int = 200,
          deadline: float = float("inf")) -> tuple[np.ndarray, float, int]:
    """Steepest improving transposition search; fixed all-pairs enumeration."""
    key = initial.copy()
    swaps = np.array(list(itertools.combinations(range(len(key)), 2)))
    score = float(objective.scores(key[None])[0])
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            return key, score, step
        candidates = np.tile(key, (len(swaps), 1))
        row = np.arange(len(swaps))
        candidates[row, swaps[:, 0]] = key[swaps[:, 1]]
        candidates[row, swaps[:, 1]] = key[swaps[:, 0]]
        scores = np.concatenate([objective.scores(batch)
                                 for batch in np.array_split(candidates, 8)])
        winner = int(np.argmax(scores))
        if scores[winner] <= score + 1e-8:
            return key, score, step
        key, score = candidates[winner], float(scores[winner])
    return key, score, max_steps


def search(lattice: list[list[tuple[int, ...]]], lm: CharacterLM, seed: int,
           joint: bool = True, restarts: int = 8, kicks: int = 6,
           max_seconds: float = 3600.) -> dict:
    """Multi-start key search alternating with exact latent token parsing.

    Every restart begins at unigram frequency matching plus seed-dependent
    perturbations. Fixed-parse baseline never calls the lattice optimizer.
    """
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    size = len(lm.alphabet)
    first = [0] * len(lattice)
    sequence = flatten(lattice, first)
    frequencies = np.bincount(sequence, minlength=size)
    initial = np.empty(size, dtype=np.int64)
    initial[np.argsort(-frequencies, kind="stable")] = np.argsort(-lm.logs[0], kind="stable")
    best_score, best_key, best_choices = -np.inf, None, None
    trace = []
    stopped = False
    for restart in range(restarts):
        key = initial.copy()
        for _ in range(3 * restart):
            a, b = rng.choice(size, 2, replace=False)
            key[a], key[b] = key[b], key[a]
        choices = first.copy()
        for cycle in range(kicks + 1):
            if time.monotonic() - started > max_seconds:
                stopped = True
                break
            steps = 0
            for _ in range(3 if joint else 1):
                objective = SequenceObjective(flatten(lattice, choices), lm)
                key, score, moves = climb(key, objective, deadline=started + max_seconds)
                steps += moves
                if joint:
                    score, new_choices = viterbi(lattice, key, lm)
                    stable = new_choices == choices
                    choices = new_choices
                    if stable:
                        break
            if score > best_score:
                best_score, best_key, best_choices = score, key.copy(), choices.copy()
            trace.append({"restart": restart, "cycle": cycle, "score": score,
                          "best_score": best_score, "moves": steps,
                          "seconds": time.monotonic() - started})
            # Explore around the global incumbent; no development/answer feedback.
            key, choices = best_key.copy(), best_choices.copy()
            for _ in range(2 + cycle % 4):
                a, b = rng.choice(size, 2, replace=False)
                key[a], key[b] = key[b], key[a]
        if stopped:
            break
    if best_key is None:
        raise RuntimeError("Budget expired before any candidate")
    return {"key": best_key.tolist(), "choices": best_choices, "fit_score": float(best_score),
            "trace": trace, "seconds": time.monotonic() - started,
            "budget_stop": stopped or time.monotonic() - started > max_seconds,
            "seed": seed, "joint": joint, "restarts": restarts, "kicks": kicks}


def encode_lattice(split: dict, class_ids: list[str]) -> list[list[tuple[int, ...]]]:
    index = {value: i for i, value in enumerate(class_ids)}
    return [[tuple(index[c] for c in path) for path in alternatives] for alternatives in split["candidates"]]
