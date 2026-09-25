"""Exact subsequence-count controls against exhaustive small cases."""

from itertools import combinations

import numpy as np

from voynich.exp0033_alignment import alignment_counts


def brute_force(observed: str, target: str) -> tuple[int, list[int]]:
    accepted = [choice for choice in combinations(range(len(observed)), len(target)) if "".join(observed[i] for i in choice) == target]
    return len(accepted), [sum(i in choice for choice in accepted) for i in range(len(observed))]


def test_exact_counts_and_position_marginals_match_enumeration() -> None:
    rng = np.random.default_rng(33)
    for n in range(1, 9):
        for _ in range(30):
            observed = "".join(rng.choice(list("ab "), size=n))
            length = int(rng.integers(n + 1))
            indices = sorted(map(int, rng.choice(n, size=length, replace=False)))
            target = "".join(observed[i] for i in indices)
            assert alignment_counts(observed, target) == brute_force(observed, target)


def test_repeated_glyphs_are_ambiguous_but_distinct_symbols_can_be_unique() -> None:
    assert alignment_counts("aaaa", "aa") == (6, [3, 3, 3, 3])
    assert alignment_counts("abcd", "bd") == (1, [0, 1, 0, 1])
    assert alignment_counts("abba", "xyz") == (0, [0, 0, 0, 0])
