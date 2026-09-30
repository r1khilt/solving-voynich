import itertools

import numpy as np
import pytest

from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.kbest_suffix import key_kbest
from voynich.sparse_suffix_source import decode


@pytest.mark.parametrize('order', [0, 1, 3, 8, 12])
@pytest.mark.parametrize('minimum', [1, 4, 100])
def test_every_row_and_transition_equals_lazy_source(order, minimum):
    counts = fit_compact(['aaabacaaabacaaabac', 'ccccbbcc', 'abc'], 'abc', order, minimum)
    lazy = CompactSuffixAdapter(counts, 64.)
    dense = DenseSuffixAdapter(counts, 64., block_rows=2)
    assert dense.array_bytes == len(dense.probabilities) * 3 * 12
    for state in range(len(dense.probabilities)):
        np.testing.assert_allclose(dense.row(state), lazy.row(state), rtol=0, atol=1e-15)
        for letter in range(3):
            assert dense.step(state, letter) == lazy.step(state, letter)
    for n in range(6):
        for chars in itertools.product('abc', repeat=n):
            text = ''.join(chars)
            state = dense.state(text)
            assert state == lazy.state(text)
            for letter, char in enumerate('abc'):
                assert dense.step(state, letter) == dense.state(text + char)
    assert not dense.probabilities.flags.writeable and not dense.transitions.flags.writeable


@pytest.mark.parametrize('units', [('x', 'x'), ('x', 'xx'), ('xy', 'y')])
def test_exact_decoding_and_kbest_scores_are_unchanged(units):
    counts = fit_compact(['aabbaabbabba', 'bbbbaabbbb'], 'ab', 8, 2)
    lazy, dense = CompactSuffixAdapter(counts, 16.), DenseSuffixAdapter(counts, 16.)
    for n in range(5):
        for chars in itertools.product('xy', repeat=n):
            text = ''.join(chars)
            a, b = decode(lazy, units, text, .2), decode(dense, units, text, .2)
            assert a == b
            assert key_kbest(lazy, units, [text, ''], .2, 4) == key_kbest(dense, units, [text, ''], .2, 4)


def test_allocation_cap_precedes_large_allocations_and_invalid_indices_fail():
    counts = fit_compact(['aaaaab'] * 4, 'ab', 4, 4)
    with pytest.raises(MemoryError, match='allocation'):
        DenseSuffixAdapter(counts, 16., max_array_bytes=1)
    dense = DenseSuffixAdapter(counts, 16.)
    assert len({dense.state('a' * n) for n in range(5)}) == 5
    for state, letter in [(-1, 0), (0, -1), (len(dense.transitions), 0), (0, 2), (True, 0)]:
        with pytest.raises(ValueError):
            dense.step(state, letter)
    with pytest.raises(ValueError):
        dense.row(-1)


@pytest.mark.parametrize('options', [{'tau': 0}, {'tau': True}, {'block_rows': 0}, {'max_array_bytes': False}])
def test_invalid_settings(options):
    settings = {'tau': 16., **options}
    with pytest.raises(ValueError):
        DenseSuffixAdapter(fit_compact(['ababa'], 'ab', 2, 1), **settings)
