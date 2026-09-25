import itertools
import time

import numpy as np

from voynich.naibbe_homophone_search import climb, neighbors, search, validate_groups
from voynich.naibbe_key_search import CharacterLM, SequenceObjective, flatten, viterbi


def test_every_move_preserves_table_bijections_and_has_expected_coverage():
    groups = [[0, 2, 4], [1, 3, 5]]
    key = np.array([0, 2, 1, 0, 2, 1])
    validate_groups(groups, 3)
    proposals = neighbors(key, groups, True)
    assert proposals.shape == (9, 6)
    assert len({tuple(row) for row in proposals}) == 9
    for candidate in proposals:
        for group in groups:
            assert sorted(candidate[group]) == [0, 1, 2]
    assert all(np.sum(row != key) == 2 for row in proposals[:6])
    assert all(np.sum(row != key) == 4 for row in proposals[6:])


def test_two_table_toy_climb_local_optimality_and_exhaustive_multistart():
    lm = CharacterLM.train("aaaabacababcbacbaababc" * 15, "abc")
    groups = [[0, 1, 2], [3, 4, 5]]
    sequence = np.array([0, 4, 2, 3, 1, 5, 0, 3, 1, 4, 2, 5] * 10)
    objective = SequenceObjective(sequence, lm)
    keys = np.array([(*a, *b) for a, b in itertools.product(itertools.permutations(range(3)), repeat=2)])
    global_best = objective.scores(keys).max()
    finals = []
    for initial in keys:
        key, score, _ = climb(initial, objective, groups, True, time.monotonic() + 60)
        assert objective.scores(neighbors(key, groups, True)).max() <= score + 1e-8
        finals.append(score)
    assert abs(max(finals) - global_best) < 1e-8


def test_ambiguous_search_returns_consistent_score_and_valid_keys():
    lm = CharacterLM.train("aaaabacababcbacbaababc" * 15, "abc")
    groups = [[0, 1, 2], [3, 4, 5]]
    lattice = [[(0,), (3,)], [(1, 5)], [(4,)], [(2,), (5,)]] * 10
    result = search(lattice, lm, groups, restarts=2, kicks=2, max_seconds=10)
    key = np.array(result["key"])
    score, _ = viterbi(lattice, key, lm)
    assert abs(score - result["fit_score"]) < 1e-9
    assert abs(lm.score(key[flatten(lattice, result["choices"])]) - score) < 1e-9
    for group in groups:
        assert sorted(key[group]) == [0, 1, 2]
