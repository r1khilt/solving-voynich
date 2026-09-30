import itertools

import pytest

from voynich.unit_channel_edit_floor import minimum_unit_edit_distance


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        updated = [i]
        for j, b in enumerate(right, 1):
            updated.append(min(row[j] + 1, updated[-1] + 1, row[j - 1] + (a != b)))
        row = updated
    return row[-1]


@pytest.mark.parametrize("units", [("X", "XX"), ("XY", "YX"), ("X", "X")])
def test_floor_matches_exhaustive_compatible_plaintexts(units):
    alphabet = ("a", "b")
    for n in range(5):
        for raw in itertools.product("XY", repeat=n):
            record = "".join(raw)
            compatible = []
            for m in range(n + 1):
                for chars in itertools.product(alphabet, repeat=m):
                    if "".join(units[alphabet.index(char)] for char in chars) == record:
                        compatible.append("".join(chars))
            for truth in ("", "a", "ba", "bbbb", "aabaa"):
                expected = min((distance(plain, truth) for plain in compatible), default=None)
                assert minimum_unit_edit_distance(alphabet, units, record, truth) == expected


def test_literal_unicode_empty_invalid_and_cap():
    assert minimum_unit_edit_distance(("α", "β"), ("雪", "\0"), "雪\0", "αβ") == 0
    assert minimum_unit_edit_distance(("a",), ("XX",), "", "abc") == 3
    assert minimum_unit_edit_distance(("a",), ("XX",), "X", "a") is None
    with pytest.raises(ValueError, match="cell cap"):
        minimum_unit_edit_distance(("a",), ("X",), "XX", "aa", max_cells=2)
    with pytest.raises(ValueError):
        minimum_unit_edit_distance(("a",), ("",), "X", "a")
