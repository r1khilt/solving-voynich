"""Independent length-assignment reference, duplicate units and cap honesty."""
import itertools
import math

import numpy as np
import pytest

from voynich.fragment_key_proposals import (
    FragmentMatcher, build_fragment_native, coherent_key, gram_table, window_offsets,
)
from voynich.compact_suffix_source import fit_compact


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    return build_fragment_native(tmp_path_factory.mktemp("fragment-build"))


def reference(grams, cipher, rows=2, glyphs=2):
    result = []
    for index, gram in enumerate(grams):
        used = sorted(set(gram))
        for lengths in itertools.product((1, 2), repeat=len(used)):
            sizes, offset, key, valid = dict(zip(used, lengths)), 0, [-1]*rows, True
            for row in gram:
                length = sizes[row]
                unit = cipher[offset:offset+length]
                if len(unit) != length:
                    valid = False
                    break
                code = unit[0] if length == 1 else glyphs+glyphs*unit[0]+unit[1]
                if key[row] not in (-1, code):
                    valid = False
                    break
                key[row], offset = code, offset+length
            if valid:
                result.append({"gram": index, "consumed": offset, "partial_key": key})
    return result


def test_all_tiny_cipher_prefixes_against_length_assignments(build):
    grams = np.array(list(itertools.product(range(2), repeat=4)), dtype=np.int32)
    matcher = FragmentMatcher(grams, np.arange(1, 17), build, rows=2, glyphs=2)
    for length in range(1, 7):
        for cipher in itertools.product(range(2), repeat=length):
            expected = reference(grams.tolist(), cipher)
            expected.sort(key=lambda m: (m["gram"], m["partial_key"], m["consumed"]))
            found = matcher.match(cipher, keep=4096)
            assert found["total_matches"] == len(expected)
            assert found["matches"] == expected
            assert not found["output_rank_truncated"]


def test_rank_union_and_same_unit_for_distinct_rows(build):
    grams = np.array(list(itertools.product(range(2), repeat=4)), dtype=np.int32)
    counts = np.arange(1, 17)
    matcher = FragmentMatcher(grams, counts, build, rows=2, glyphs=2)
    cipher = (0,)*8
    expected = reference(grams.tolist(), cipher)
    chosen = set()
    for arm in range(2):
        ordered = sorted(expected, key=lambda m: (-matcher.ranks[arm, m["gram"]], m["gram"], m["partial_key"], m["consumed"]))
        chosen.update((m["gram"], tuple(m["partial_key"]), m["consumed"]) for m in ordered[:2])
    actual = matcher.match(cipher, keep=2)
    assert {(m["gram"], tuple(m["partial_key"]), m["consumed"]) for m in actual["matches"]} == chosen
    assert actual["total_matches"] == len(expected)
    assert any(m["partial_key"] == [0, 0] for m in matcher.match(cipher, keep=4096)["matches"])
    assert actual["output_rank_truncated"]


def test_exact_cap_failure_and_invalid_inputs(build):
    matcher = FragmentMatcher(np.array([[0, 1, 0]], dtype=np.int32), np.array([1]), build, rows=2, glyphs=2)
    with pytest.raises(RuntimeError, match="cap hit"):
        matcher.match((0,)*6, node_cap=1)
    for observed in ((), (2,), (True,)):
        with pytest.raises(ValueError):
            matcher.match(observed)
    with pytest.raises(ValueError):
        FragmentMatcher(np.array([[0], [0]]), np.array([1, 2]), build, rows=2, glyphs=2)
    altered = dict(build, library_sha256="0"*64)
    with pytest.raises(ValueError, match="changed"):
        FragmentMatcher(np.array([[0]]), np.array([1]), altered, rows=2, glyphs=2)


def test_twelve_distinct_rows_enumerate_all4096_length_assignments(build):
    matcher = FragmentMatcher(np.array([list(range(12))]), np.array([1]), build, rows=12, glyphs=2)
    found = matcher.match((0, 1)*12, keep=4096)
    assert found["nodes"] == 8191 and found["total_matches"] == 4096
    assert len(found["matches"]) == 4096 and not found["output_rank_truncated"]
    expected = reference([list(range(12))], (0, 1)*12, rows=12)
    expected.sort(key=lambda m: (m["gram"], m["partial_key"], m["consumed"]))
    assert found["matches"] == expected


def test_context_table_is_original_prefix_code_and_counts():
    compact = fit_compact(["ababaabababa"], "ab", order=3, minimum=1)
    grams, counts = gram_table(compact, 3)
    codes = np.array([sum(r*2**(2-i) for i, r in enumerate(g)) for g in grams], dtype=np.uint64)
    assert np.array_equal(codes, compact.levels[3]["contexts"])
    assert np.array_equal(counts, compact.levels[3]["totals"])
    counts[0] += 1
    assert not np.array_equal(counts, compact.levels[3]["totals"])
    assert math.isfinite(float(np.log(counts[0])))


def test_offsets_and_partial_completion():
    assert window_offsets(["A"*11]) == ()
    assert window_offsets(["A"*12]) == ((0, 0),)
    offsets = window_offsets(["A"*64, "B"*224])
    assert len(offsets) <= 96 and len(offsets) == len(set(offsets))
    assert (0, 0) in offsets and (1, 212) in offsets
    assert coherent_key(("A", "B", "C"), [-1, 1, 1], ("A", "AA")) == ("A", "AA", "AA")
    with pytest.raises(ValueError):
        coherent_key(("A",), [True], ("A",))
