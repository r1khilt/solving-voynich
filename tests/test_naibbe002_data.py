import re
from collections import Counter

import pytest

from scripts.build_naibbe002_data import anonymous_classes, inverse_lattice, write_immutable_json


def records_fixture():
    return [{"table": table, "letter": letter, "role": role,
             "glyphs": f"{table}:{role}:{letter}"}
            for table in ("one", "two") for letter in "ab"
            for role in ("unigram", "prefix", "suffix")]


def test_same_glyph_in_different_tables_retains_both_classes():
    records = records_fixture()
    for row in records:
        if row["role"] == "unigram" and row["letter"] == "a":
            row["glyphs"] = "dar"
    mapping, _, _, _ = anonymous_classes(("one", "two"), "ab", seed=6069202)
    lattice = inverse_lattice(records, mapping)
    assert lattice["dar"] == {(mapping[("one", "a")],), (mapping[("two", "a")],)}
    assert len(lattice["dar"]) == 2


def test_within_table_role_linkage_and_unigram_precedence():
    records = records_fixture()
    for row in records:
        if (row["table"], row["letter"], row["role"]) == ("one", "a", "prefix"):
            row["glyphs"] = "x"
        if (row["table"], row["letter"], row["role"]) == ("two", "b", "suffix"):
            row["glyphs"] = "y"
        if (row["table"], row["letter"], row["role"]) == ("two", "a", "unigram"):
            row["glyphs"] = "xy"
    mapping, _, _, _ = anonymous_classes(("one", "two"), "ab", seed=13)
    lattice = inverse_lattice(records, mapping)
    assert lattice["xy"] == {(mapping[("two", "a")],)}
    assert lattice["one:unigram:a"] == {(mapping[("one", "a")],)}
    assert lattice["xtwo:suffix:a"] == {(mapping[("one", "a")], mapping[("two", "a")])}


def test_independent_group_permutations_and_presentation_are_anonymous():
    tables = tuple(f"table_{i}" for i in range(6))
    alphabet = "abcdefghilmnopqrstuvxyz"
    result = anonymous_classes(tables, alphabet, seed=6069202)
    mapping, class_ids, groups, group_ids = result
    assert result == anonymous_classes(tuple(reversed(tables)), alphabet, seed=6069202)
    assert result != anonymous_classes(tables, alphabet, seed=6069203)
    assert len(class_ids) == len(set(class_ids)) == 138
    assert len(groups) == len(group_ids) == len(set(group_ids)) == 6
    assert all(len(group) == len(set(group)) == 23 for group in groups)
    assert Counter(identifier for group in groups for identifier in group) == Counter(class_ids)
    assert all(re.fullmatch(r"c_[0-9a-f]{20}", identifier) for identifier in class_ids)
    assert all(re.fullmatch(r"t_[0-9a-f]{20}", identifier) for identifier in group_ids)
    reverse = {identifier: (table, letter) for (table, letter), identifier in mapping.items()}
    for group in groups:
        assert len({reverse[identifier][0] for identifier in group}) == 1
        assert {reverse[identifier][1] for identifier in group} == set(alphabet)
    # Neither group display slots nor global class order supply aligned letter rows.
    display_orders = {tuple(reverse[identifier][1] for identifier in group) for group in groups}
    assert len(display_orders) == 6
    assert list(mapping.values()) != class_ids


def test_record_order_does_not_change_lattice_and_other_table_classes_stay_distinct():
    records = records_fixture()
    mapping, _, _, _ = anonymous_classes(("one", "two"), "ab", seed=77)
    lattice = inverse_lattice(records, mapping)
    assert lattice == inverse_lattice(list(reversed(records)), mapping)
    assert lattice["one:prefix:atwo:suffix:a"] == {(mapping[("one", "a")], mapping[("two", "a")])}
    assert mapping[("one", "a")] != mapping[("two", "a")]


def test_prepared_data_can_replay_but_cannot_silently_change(tmp_path):
    path = tmp_path / "inputs" / "fit.json"
    write_immutable_json(path, {"value": [1, 2]})
    original = path.read_bytes()
    write_immutable_json(path, {"value": [1, 2]})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_immutable_json(path, {"value": [2, 1]})
    assert path.read_bytes() == original
