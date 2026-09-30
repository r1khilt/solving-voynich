"""Enumerate complete source strings before computing their edit distance."""
import itertools
from functools import lru_cache

import pytest

from voynich.unit_channel_edit_floor import minimum_unit_edit_distance


def words(alphabet, bound):
    return ("".join(chars) for length in range(bound + 1) for chars in itertools.product(alphabet, repeat=length))


@lru_cache(None)
def recursive_distance(left, right):
    if not left or not right:
        return len(left) + len(right)
    return min(1 + recursive_distance(left[1:], right), 1 + recursive_distance(left, right[1:]),
               (left[0] != right[0]) + recursive_distance(left[1:], right[1:]))


def test_all_small_dictionaries_observations_and_truths_match_complete_plaintext_enumeration():
    alphabet, vocabulary = ("a", "b"), ("x", "y", "xx", "xy", "yx", "yy")
    truths = (*words("ab", 3), "z", "aabaa")
    for units in itertools.product(vocabulary, repeat=2):
        parses = {}
        for text in words(alphabet, 4):
            observed = "".join(units[alphabet.index(char)] for char in text)
            if len(observed) <= 4:
                parses.setdefault(observed, []).append(text)
        for observed in words("xy", 4):
            candidates = parses.get(observed, ())
            for truth in truths:
                expected = min((recursive_distance(text, truth) for text in candidates), default=None)
                actual = minimum_unit_edit_distance(alphabet, units, observed, truth)
                assert actual == expected
                assert (actual == 0) == (truth in candidates)


def test_true_generating_dictionary_always_has_zero_floor_even_with_ambiguous_units():
    for units in (("x", "xx"), ("xy", "xy"), ("x\0", "雪")):
        for truth in words("ab", 5):
            observed = "".join(units["ab".index(char)] for char in truth)
            assert minimum_unit_edit_distance("ab", units, observed, truth) == 0


def test_empty_unsupported_long_unit_and_non_source_truth_characters_are_distinct():
    assert minimum_unit_edit_distance("ab", ("x", "xx"), "", "bbb") == 3
    assert minimum_unit_edit_distance("ab", ("x", "xx"), "xxxx", "") == 2
    assert minimum_unit_edit_distance("ab", ("xx", "yy"), "x", "a") is None
    assert minimum_unit_edit_distance("ab", ("xyz", "x"), "xyzx", "za") == 2
    assert minimum_unit_edit_distance(("α", "β"), ("雪\0", "雪"), "雪\0雪", "αβ") == 0


def test_floor_ignores_source_zero_constraints_and_cell_limit_never_prunes():
    # A source assigning b zero probability could only read aa. The dictionary
    # relaxation still admits bb, so zero floor is not a source-aware promise.
    assert minimum_unit_edit_distance("ab", ("x", "x"), "xx", "bb") == 0
    expected = minimum_unit_edit_distance("ab", ("x", "xx"), "xxx", "aba", max_cells=16)
    assert expected == 1
    with pytest.raises(ValueError, match="cell cap"):
        minimum_unit_edit_distance("ab", ("x", "xx"), "xxx", "aba", max_cells=15)
    with pytest.raises(ValueError):
        minimum_unit_edit_distance("a", ("",), "x", "a")
