"""Brute-force count equality proves the numeric collector and fixed cutoff."""
import itertools

import numpy as np
import pytest

from voynich.sparse_suffix_source import SuffixSource, collect_counts
from voynich.threshold_suffix_counts import collect_threshold_counts


@pytest.mark.parametrize('order', [0, 1, 3, 8, 12])
@pytest.mark.parametrize('minimum', [1, 2, 4, 7])
def test_all_counts_equal_literal_brute_force_with_support_filter(order, minimum):
    records = ['aaabcaaabcaaabc', 'baaac', 'aa', 'cbabacbaba']
    full = collect_counts(records, 'abc', order)
    expected = {c: row for c, row in full.items() if not c or sum(row.values()) >= minimum}
    actual = collect_threshold_counts(records, 'abc', order, minimum)
    assert actual == expected
    source = SuffixSource('abc', order, 16., actual)
    assert np.allclose(source.probabilities.sum(axis=1), 1., atol=1e-14, rtol=0)
    for c in actual:
        if c:
            assert c[:-1] in actual and c[1:] in actual


def test_record_end_resets_leading_zeroes_and_rare_successors():
    result = collect_threshold_counts(['aaaab', 'aaaac', 'aaaaa', 'aaaaa'], 'abc', 4, 4)
    assert result['aaaa'] == {'a': 2, 'b': 1, 'c': 1}
    assert 'b' not in result and 'c' not in result
    assert result[''] == {'a': 18, 'b': 1, 'c': 1}


def test_uint64_code_boundary_is_exact():
    alphabet = 'abcdefghiklmnopqrstuxyz'
    text = alphabet[-1] * 14 + alphabet[0] * 14
    actual = collect_threshold_counts([text], alphabet, 12, 1)
    assert actual == collect_counts([text], alphabet, 12)
    with pytest.raises(ValueError):
        collect_threshold_counts(['a'], ''.join(chr(65 + i) for i in range(32)), 12, 1)


def test_caps_and_invalid_settings_never_change_cutoff():
    with pytest.raises(RuntimeError, match='do not change cutoff'):
        collect_threshold_counts(['ababa'], 'ab', 3, 1, max_contexts=1)
    for order, minimum in itertools.product([-1, 13, True], [0, 1, True]):
        with pytest.raises(ValueError):
            collect_threshold_counts(['ab'], 'ab', order, minimum)
    with pytest.raises(ValueError):
        collect_threshold_counts([], 'ab')
    with pytest.raises(ValueError):
        collect_threshold_counts(['z'], 'ab')
