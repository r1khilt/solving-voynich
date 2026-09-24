import math

from voynich.word_copy_channel import CharModel, edit_parts, probability


def test_edit_components_normalize_with_duplicate_paths():
    source = "aa"
    alphabet = "ab"
    words = [""]
    for length in range(1, 4):
        words += ["".join("ab"[(n >> bit) & 1] for bit in range(length))
                  for n in range(2 ** length)]
    masses = [sum(edit_parts(source, word, len(alphabet))[k] for word in words)
              for k in range(4)]
    assert all(math.isclose(mass, 1.0) for mass in masses)
    assert edit_parts("aa", "aaa", 2)[2] == 0.5  # three insertion paths, not one
    assert edit_parts("aa", "a", 2)[3] == 1.0  # two deletion paths


def test_character_model_and_empty_cache_are_proper():
    char = CharModel("ab").fit(["a", "ab", "aa"])
    assert 0 < char.prob("ab") < 1
    assert char.prob("z") == 0
    row = {"count": 1, "char_p": char.prob("a"), "has_history": False,
           "features": {"4:4": (0, 0, 0, 0)}}
    base = (1 + 10 * char.prob("a")) / 13
    assert math.isclose(probability(row, 3, {"arm": "edit", "beta": 10,
                                               "lambda": 0.8, "window": 4,
                                               "half": 4, "q_exact": 0.5}), base)
