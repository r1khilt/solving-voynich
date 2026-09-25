import itertools
import math

import pytest

from scripts.audit_naibbe003_source import DISCOUNTS, TAUS, ScalarSourceLM, bootstrap_gain, self_test


@pytest.mark.parametrize("family,parameter", [("baseline", None),
                                            *[("dirichlet", tau) for tau in TAUS],
                                            *[("absolute_discount", d) for d in DISCOUNTS]])
def test_all_toy_rows_positive_normalized_and_unseen_histories_back_off(family, parameter):
    model = ScalarSourceLM("abacabaababbca", "abcd", family, parameter)
    result = model.normalization_audit()
    assert result["histories"] == sum(4 ** length for length in range(4))
    assert result["max_normalization_error"] < 1e-12
    assert result["minimum_probability"] > 0
    assert result["unseen_histories"] > 0
    if family != "baseline":
        for letter in model.alphabet:
            assert model.probability("ddd", letter) == model.probability("", letter)


def test_scalar_dirichlet_and_raw_absolute_discount_against_hand_counts():
    direct = ScalarSourceLM("ababa", "abc", "dirichlet", 4.)
    discounted = ScalarSourceLM("ababa", "abc", "absolute_discount", .5)
    assert direct.counts[1] == {"a": 3, "b": 2}
    assert direct.counts[2] == {"ab": 2, "ba": 2}
    assert direct.context_totals[2] == {"a": 2, "b": 2}
    assert direct.continuation_types[2] == {"a": 1, "b": 1}
    assert direct.probability("", "c") == pytest.approx(.1 / 5.3)
    assert direct.probability("a", "b") == pytest.approx((2. + 4. * 2.1 / 5.3) / 6.)
    assert direct.probability("a", "c") == pytest.approx(4. * .1 / 5.3 / 6.)
    assert discounted.probability("a", "b") == pytest.approx(1.5 / 2. + .5 / 2. * 2.1 / 5.3)
    assert discounted.probability("a", "c") == pytest.approx(.5 / 2. * .1 / 5.3)


def test_each_score_call_starts_with_empty_context_and_sequence_probabilities_normalize():
    model = ScalarSourceLM("abacabadabacaba", "abcd", "dirichlet", 1.)
    before = model.score("cab")
    model.score("ddd" * 20)
    assert model.score("cab") == before
    expected = (math.log(model.probability("", "c")) + math.log(model.probability("c", "a"))
                + math.log(model.probability("ca", "b")))
    assert before == pytest.approx(expected)
    total = math.fsum(math.exp(model.score("".join(chars)))
                      for chars in itertools.product("abcd", repeat=4))
    assert total == pytest.approx(1., abs=1e-12)
    assert model.score("") == 0.
    with pytest.raises(ValueError):
        model.nll_per_character("")


def test_no_counts_for_artificial_cross_split_ngrams():
    model = ScalarSourceLM("ababa", "abc", "dirichlet", 1.)
    before = {order: counts.copy() for order, counts in model.counts.items()}
    model.score("ccc")
    assert model.counts == before
    assert model.counts[2]["ac"] == 0
    assert model.probability("ccc", "a") == model.probability("", "a")


@pytest.mark.parametrize("family,parameter", [("dirichlet", 0.), ("dirichlet", -1.),
                                            ("absolute_discount", 0.), ("absolute_discount", 1.),
                                            ("dirichlet", math.inf), ("dirichlet", math.nan),
                                            ("baseline", .1), ("unknown", None)])
def test_invalid_source_model_parameters_fail(family, parameter):
    with pytest.raises(ValueError):
        ScalarSourceLM("ab", "abc", family, parameter)


def test_unsupported_characters_and_invalid_alphabets_fail():
    for text, alphabet in (("", "abc"), ("abc", "aba"), ("abc", ""), ("abd", "abc")):
        with pytest.raises(ValueError):
            ScalarSourceLM(text, alphabet, "dirichlet", 1.)
    model = ScalarSourceLM("ab", "abc", "absolute_discount", .5)
    for history, symbol in (("d", "a"), ("a", "d"), ("a", "ab")):
        with pytest.raises(ValueError):
            model.probability(history, symbol)


def test_self_test_has_no_source_or_cipher_reads():
    result = self_test()
    assert result["self_test_pass"]
    assert result["normalized_toy_histories"] == 11 * (1 + 3 + 9 + 27)
    assert result["source_files_read"] is False
    assert result["cipher_files_read"] is False


def test_bootstrap_keeps_partial_block_and_uses_character_weighting():
    gains = [1.] * 1000 + [3.] * 326
    blocks, interval = bootstrap_gain(gains)
    assert blocks == [{"sum": 1000., "count": 1000}, {"sum": 978., "count": 326}]
    assert interval == [1., 3.]
    assert sum(row["sum"] for row in blocks) / sum(row["count"] for row in blocks) == pytest.approx(1978 / 1326)
    assert bootstrap_gain([.25] * 2326)[1] == [.25, .25]
    assert bootstrap_gain(gains) == (blocks, interval)
    for invalid in ([], [math.nan], [math.inf]):
        with pytest.raises(ValueError):
            bootstrap_gain(invalid)
