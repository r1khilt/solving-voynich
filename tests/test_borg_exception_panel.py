from collections import Counter

from scripts.plan_borg_exception_review004 import greedy_cover


def test_cover_prioritizes_new_types_before_count_and_never_omits_singletons():
    counts = {0: Counter({"a": 1000}), 1: Counter({"a": 1, "b": 1}),
              2: Counter({"z": 1})}
    assert greedy_cover(counts) == [(1, ["a", "b"]), (2, ["z"])]


def test_cover_tie_break_is_count_then_ordinal_not_dictionary_order():
    counts = {9: Counter({"z": 5}), 7: Counter({"a": 5}), 0: Counter({"b": 1})}
    assert greedy_cover(counts) == [(7, ["a"]), (9, ["z"]), (0, ["b"])]


def test_empty_inventory_has_no_panel():
    assert greedy_cover({}) == []
