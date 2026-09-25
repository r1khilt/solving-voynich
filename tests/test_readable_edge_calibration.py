"""Toy-only structural checks for EXP-0041 before comparator scoring."""

from voynich.readable_edge_calibration import (
    bijection,
    evaluate_view,
    fold_words,
    homophones,
    shuffle_seed,
    transpositions,
    wrap,
)


def _skeleton(n: int) -> list[dict]:
    return [{"leaf": "toy", "locus": "toy", "context": ("A", "1"),
             "words": [("x",)] * n}]


def test_reversible_channels_preserve_word_count_and_length() -> None:
    words = [tuple(word) for word in ("alpha", "beta", "gamma", "delta", "alpha")]
    for channel in (bijection(words), homophones(words, 410200),
                    transpositions(words, 410300)):
        assert len(channel) == len(words)
        assert [len(word) for word in channel] == [len(word) for word in words]


def test_accent_fold_does_not_split_words() -> None:
    assert fold_words("lähteäni vivía naïve") == ["lahteani", "vivia", "naive"]


def test_bijection_replays_exact_plaintext_shuffles() -> None:
    assert shuffle_seed(0) == shuffle_seed(4) == 410400
    assert len({shuffle_seed(index) for index in range(21)}) == 20


def test_wrap_requires_exact_length() -> None:
    words = [tuple(word) for word in ("alpha", "beta", "gamma", "delta")]
    assert wrap(words, _skeleton(4))[0]["words"] == words


def test_bijective_cipher_does_not_change_fitted_scores() -> None:
    train = [tuple(word) for word in ("alpha", "beta", "gamma", "delta", "theta",
                                     "alpha", "beta", "gamma")]
    valid = [tuple(word) for word in ("theta", "alpha", "gamma", "delta", "beta")]
    plain = evaluate_view(train, valid, _skeleton(8), _skeleton(5), 411000)
    enc = bijection(train + valid)
    coded = evaluate_view(enc[:8], enc[8:], _skeleton(8), _skeleton(5), 411000)
    for key in ("last_gain_bits_per_pair", "last_minus_shuffle_bits_per_pair",
                "last_minus_first_bits_per_pair", "identity_over_last_bits_per_pair"):
        assert abs(plain[key] - coded[key]) < 1e-10
