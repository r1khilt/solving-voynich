from voynich.boundary_endpoint import aggregate, edges, eligible, fit, permute, score
from voynich.boundary_order import toy


def test_edge_roles_and_endpoint_permutations() -> None:
    words = ["ab", "bc", "cd", "de", "ea"]
    assert list(edges(words)) == [
        ("first", "b", "b"), ("middle", "c", "c"),
        ("middle", "d", "d"), ("last", "e", "e"),
    ]
    group = {"leaf": "x", "section": "H", "words": words}
    for mode in ("endpoint", "half"):
        result = permute([group], 370037, mode)[0]["words"]
        assert result[0] == words[0] and result[-1] == words[-1]
        assert sorted(result) == sorted(words)
    assert eligible([group, {**group, "words": words[:3]}]) == [group]


def test_position_aware_toy_positive() -> None:
    train = toy(360236, 100)
    valid = toy(360336, 30)
    model = fit(train, tuple("abcdefgh"))
    real = aggregate(score(valid, model))
    null = aggregate(score(permute(valid, 370836, "endpoint"), model))
    assert real - null > 0.1
