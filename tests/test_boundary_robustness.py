import math

from voynich.boundary_robustness import (
    SYMBOLS, encode, fit, probability, score, shuffle, tokenize,
)


def test_declared_tokenizations_are_reversible() -> None:
    assert tokenize("cthaiin", "codepoint") == tuple("cthaiin")
    assert tokenize("cthaiin", "basic_compound") == ("cth", "a", "i", "i", "n")
    assert tokenize("cthaiin", "cuva_like") == ("cth", "a", "iin")
    assert tokenize("shol", "cuva_like") == ("sh", "o", "l")
    assert tokenize("eeey", "cuva_like") == ("eee", "y")
    assert len(SYMBOLS) == len(set(SYMBOLS))


def test_hierarchical_predictor_is_normalized() -> None:
    groups = [
        {"leaf": "x", "context": ("H", "1"), "clean": True,
         "words": ["ab", "bc", "cd", "de", "ea"]},
        {"leaf": "y", "context": ("H", "2"), "clean": True,
         "words": ["bc", "cd", "de", "ea", "ab"]},
    ]
    encoded = encode(groups, "cuva_like")
    model = fit(encoded)
    conditional = [probability(model, ("H", "1"), "first", "last", "b", target)[0]
                   for target in SYMBOLS]
    baseline = [probability(model, ("H", "1"), "first", "last", "b", target)[1]
                for target in SYMBOLS]
    assert math.isclose(sum(conditional), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(baseline), 1.0, abs_tol=1e-12)
    assert all(math.isfinite(row["bits"][feature]) for row in score(encoded, model).values()
               for feature in ("last", "first", "length"))


def test_half_shuffle_preserves_endpoints_and_half_multisets() -> None:
    words = ["ab", "bc", "cd", "de", "ea", "af", "fg", "gh"]
    group = {"leaf": "x", "context": ("H", "1"), "clean": True, "words": words}
    original = encode([group], "codepoint")
    altered = shuffle(original, 380038)[0]["items"]
    before = original[0]["items"]
    assert altered[0] == before[0] and altered[-1] == before[-1]
    interior_split = (len(before) - 2) // 2
    assert sorted(altered[1:1 + interior_split]) == sorted(before[1:1 + interior_split])
    assert sorted(altered[1 + interior_split:-1]) == sorted(before[1 + interior_split:-1])
