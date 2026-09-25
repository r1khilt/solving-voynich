from voynich.boundary_order import aggregate, fit, gain, pairs, score, shuffle, toy


def test_boundary_counts_and_gain() -> None:
    groups = [{"leaf": "x", "section": "H", "words": ["ab", "bc", "ca"]}]
    model = fit(groups, tuple("abc"))
    assert pairs(groups[0]["words"]) == [("b", "b"), ("c", "c")]
    assert model["joint"][("b", "b")] == 1
    assert model["joint"][("c", "c")] == 1
    assert gain(("b", "b"), model) > gain(("b", "c"), model)
    assert aggregate(score(groups, model)) > 0


def test_shuffle_preserves_multisets_and_toy_signal() -> None:
    train = toy(360236, 100)
    valid = toy(360336, 30)
    shuffled = shuffle(valid, 360436)
    assert [sorted(group["words"]) for group in shuffled] == [
        sorted(group["words"]) for group in valid
    ]
    model = fit(train, tuple("abcdefgh"))
    assert aggregate(score(valid, model)) - aggregate(score(shuffled, model)) > 0.1
