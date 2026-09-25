import itertools

import numpy as np

from voynich.naibbe_key_search import CharacterLM, SequenceObjective, climb, flatten, viterbi


def test_normalization_and_histogram_match_sequential_source_score():
    lm = CharacterLM.train("abbacadabacabadaba" * 5, "abcd")
    for probabilities in lm.logs:
        np.testing.assert_allclose(np.exp(probabilities).reshape(-1, 4).sum(axis=1), 1.)
    sequences = [np.array([0]), np.array([2, 1]), np.array([0, 3, 1, 2, 2, 1, 0, 0])]
    keys = np.array(list(itertools.permutations(range(4))))
    for sequence in sequences:
        actual = SequenceObjective(sequence, lm).scores(keys)
        expected = [lm.score(key[sequence]) for key in keys]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_exact_lattice_against_exhaustive_paths_for_every_key():
    lm = CharacterLM.train("abbacadabacabadaba" * 5, "abcd")
    lattice = [[(0,), (1, 2)], [(3, 2)], [(1,), (2, 0)], [(0, 1), (3,)]]
    for key_tuple in itertools.permutations(range(4)):
        key = np.array(key_tuple)
        exhaustive = [(lm.score(key[flatten(lattice, list(path))]), list(path))
                      for path in itertools.product(*(range(len(c)) for c in lattice))]
        expected = max(score for score, _ in exhaustive)
        actual, choices = viterbi(lattice, key, lm)
        assert abs(actual - expected) < 1e-10
        assert abs(lm.score(key[flatten(lattice, choices)]) - expected) < 1e-10


def test_class_renaming_preserves_viterbi_and_search_objective():
    lm = CharacterLM.train("abbacadabacabadaba" * 5, "abcd")
    lattice = [[(0,), (1,)], [(3, 2)], [(1,), (2, 0)]]
    key = np.array([2, 0, 3, 1])
    permutation = np.array([1, 3, 0, 2])
    renamed = [[tuple(int(permutation[c]) for c in path) for path in candidates] for candidates in lattice]
    renamed_key = np.empty(4, dtype=int)
    renamed_key[permutation] = key
    assert viterbi(lattice, key, lm) == viterbi(renamed, renamed_key, lm)
    objective = SequenceObjective(flatten(lattice, [0, 0, 0]), lm)
    renamed_objective = SequenceObjective(flatten(renamed, [0, 0, 0]), lm)
    np.testing.assert_allclose(objective.scores(key[None]), renamed_objective.scores(renamed_key[None]))


def test_climb_returns_local_optimum_and_multistart_reaches_exhaustive_best():
    alphabet = "abcd"
    text = "aaabacababadabacabaadabcababbac" * 30
    lm = CharacterLM.train(text, alphabet)
    source = np.array([alphabet.index(c) for c in text])
    hidden = np.array([2, 0, 3, 1])
    cipher = np.argsort(hidden)[source]
    objective = SequenceObjective(cipher, lm)
    keys = np.array(list(itertools.permutations(range(4))))
    global_score = objective.scores(keys).max()
    reached = []
    for initial in keys:
        key, score, _ = climb(initial, objective)
        assert sorted(key) == list(range(4))
        for a, b in itertools.combinations(range(4), 2):
            neighbor = key.copy()
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
            assert objective.scores(neighbor[None])[0] <= score + 1e-8
        reached.append(score)
    assert abs(max(reached) - global_score) < 1e-8
