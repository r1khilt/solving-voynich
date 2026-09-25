"""Six independent within-table keys; cross-table letter groups are unknown."""
from __future__ import annotations

import itertools
import time

import numpy as np

from voynich.naibbe_key_search import CharacterLM, SequenceObjective, flatten, viterbi


def validate_groups(groups: list[list[int]], size: int) -> None:
    ids = [c for group in groups for c in group]
    if sorted(ids) != list(range(len(ids))) or any(len(g) != size for g in groups):
        raise ValueError("Groups must partition all classes into alphabet-sized tables")


def neighbors(key: np.ndarray, groups: list[list[int]], coordinated: bool) -> np.ndarray:
    """Within-table transpositions and optional shared plaintext transpositions."""
    size = len(groups[0])
    swaps = np.array([pair for group in groups for pair in itertools.combinations(group, 2)])
    candidates = np.tile(key, (len(swaps), 1))
    rows = np.arange(len(swaps))
    candidates[rows, swaps[:, 0]] = key[swaps[:, 1]]
    candidates[rows, swaps[:, 1]] = key[swaps[:, 0]]
    if not coordinated:
        return candidates
    relabels = []
    for a, b in itertools.combinations(range(size), 2):
        renamed = key.copy()
        renamed[key == a], renamed[key == b] = b, a
        relabels.append(renamed)
    return np.concatenate((candidates, np.array(relabels)))


def climb(initial, objective, groups, coordinated, deadline, max_steps=300):
    key = initial.copy()
    score = float(objective.scores(key[None])[0])
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            return key, score, step
        candidates = neighbors(key, groups, coordinated)
        scores = np.concatenate([objective.scores(candidates[i:i + 24])
                                 for i in range(0, len(candidates), 24)])
        winner = int(np.argmax(scores))
        if scores[winner] <= score + 1e-8:
            return key, score, step
        key, score = candidates[winner], float(scores[winner])
    return key, score, max_steps


def search(lattice, lm: CharacterLM, groups, seed=920201, coordinated=True,
           restarts=8, kicks=6, max_seconds=1200.):
    started = time.monotonic()
    size = len(lm.alphabet)
    validate_groups(groups, size)
    rng = np.random.default_rng(seed)
    first = [0] * len(lattice)
    frequencies = np.bincount(flatten(lattice, first), minlength=size * len(groups))
    initial = np.empty(size * len(groups), dtype=np.int64)
    language_rank = np.argsort(-lm.logs[0], kind="stable")
    for group in groups:
        ranked = np.array(group)[np.argsort(-frequencies[group], kind="stable")]
        initial[ranked] = language_rank
    best_score, best_key, best_choices = -np.inf, None, None
    trace = []
    stopped = False
    for restart in range(restarts):
        key = initial.copy()
        for group in groups:
            for _ in range(2 * restart):
                a, b = rng.choice(group, 2, replace=False)
                key[a], key[b] = key[b], key[a]
        choices = first.copy()
        for cycle in range(kicks + 1):
            if time.monotonic() - started >= max_seconds:
                stopped = True
                break
            moves = 0
            for _ in range(3):
                objective = SequenceObjective(flatten(lattice, choices), lm)
                key, _, steps = climb(key, objective, groups, coordinated,
                                      started + max_seconds)
                moves += steps
                score, updated = viterbi(lattice, key, lm)
                stable = choices == updated
                choices = updated
                if stable:
                    break
            if score > best_score:
                best_score, best_key, best_choices = score, key.copy(), choices.copy()
            trace.append({"restart": restart, "cycle": cycle, "score": score,
                          "best_score": best_score, "moves": moves,
                          "seconds": time.monotonic() - started})
            key, choices = best_key.copy(), best_choices.copy()
            # Spread perturbations across groups, retaining each table's bijection.
            for _ in range((2 + cycle % 4) * len(groups)):
                group = groups[int(rng.integers(len(groups)))]
                a, b = rng.choice(group, 2, replace=False)
                key[a], key[b] = key[b], key[a]
        if stopped:
            break
    if best_key is None:
        raise RuntimeError("No candidate before deadline")
    return {"key": best_key.tolist(), "choices": best_choices, "fit_score": float(best_score),
            "trace": trace, "seconds": time.monotonic() - started,
            "budget_stop": stopped or time.monotonic() - started >= max_seconds,
            "seed": seed, "coordinated": coordinated, "restarts": restarts, "kicks": kicks}
