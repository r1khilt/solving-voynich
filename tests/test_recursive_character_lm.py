import itertools

import numpy as np
import pytest

from voynich.naibbe_key_search import SequenceObjective, viterbi
from voynich.recursive_character_lm import CONFIGS, log_terms, train


@pytest.mark.parametrize("config", CONFIGS[1:])
def test_normalization_unseen_backoff_and_finite_unknown_letters(config):
    lm = train("aaaabbabbbabb", "abc", config)
    for order, logs in enumerate(lm.logs, 1):
        rows = np.exp(logs).reshape(-1, 3)
        np.testing.assert_allclose(rows.sum(axis=1), 1, rtol=0, atol=1e-14)
        assert np.all(rows > 0)
        if order > 1:
            # Every context beginning with unseen c is unseen.
            prior = np.exp(lm.logs[order - 2]).reshape(-1, 3)
            for history in itertools.product(range(3), repeat=order - 2):
                context = (2, *history)
                row = sum(value * 3**i for i, value in enumerate(reversed(context)))
                suffix = row % len(prior)
                np.testing.assert_allclose(rows[row], prior[suffix], rtol=0, atol=1e-14)


@pytest.mark.parametrize("config", [CONFIGS[3], CONFIGS[-2]])
def test_scalar_terms_count_objective_and_exhaustive_lattice_agree(config):
    lm = train("ababbacabbacaa", "abc", config)
    lattice = [[(0,), (1, 2)], [(2, 1), (1,)], [(1, 0)], [(2,), (0,)]]
    key = np.array([1, 2, 0])
    score, _ = viterbi(lattice, key, lm)
    possible = []
    for path in itertools.product(*lattice):
        sequence = np.array([value for part in path for value in part])
        text = "".join(lm.alphabet[key[v]] for v in sequence)
        scalar = lm.score(key[sequence])
        assert abs(log_terms(lm, text).sum() - scalar) < 1e-12
        assert abs(SequenceObjective(sequence, lm).scores(key[None])[0] - scalar) < 1e-12
        possible.append(scalar)
    assert abs(score - max(possible)) < 1e-12
    for text in ("", "a", "ab", "abc"):
        assert abs(log_terms(lm, text).sum() - lm.score([lm.alphabet.index(c) for c in text])) < 1e-12


@pytest.mark.parametrize("family,parameter", [("dirichlet", 0), ("dirichlet", float("nan")),
                                               ("absolute_discount", 1), ("bogus", 1)])
def test_rejects_invalid_parameters(family, parameter):
    with pytest.raises(ValueError):
        train("abab", "abc", {"family": family, "parameter": parameter})
