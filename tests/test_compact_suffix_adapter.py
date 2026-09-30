import itertools

import numpy as np
import pytest

from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import fit_compact
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def words(alphabet, length):
    for n in range(length + 1):
        yield from (''.join(row) for row in itertools.product(alphabet, repeat=n))


@pytest.mark.parametrize('order', [0, 1, 3, 8, 12])
@pytest.mark.parametrize('minimum', [1, 4])
def test_every_reachable_row_and_step_matches_literal_full_history(order, minimum):
    training, alphabet = ['aaabacaaabacaaabac', 'ccccbbcc', 'abc'], 'abc'
    counts = {c: row for c, row in collect_counts(training, alphabet, order).items()
              if not c or sum(row.values()) >= minimum}
    explicit = SuffixSource(alphabet, order, 64., counts)
    compact = CompactSuffixAdapter(fit_compact(training, alphabet, order, minimum), 64., cache_size=3)
    for history in words(alphabet, 5):
        state = compact.state(history)
        expected = explicit.probabilities[explicit.state(history)]
        np.testing.assert_allclose(compact.row(state), expected, rtol=0, atol=1e-15)
        for letter, char in enumerate(alphabet):
            assert compact.step(state, letter) == compact.state(history + char)


@pytest.mark.parametrize('units', [('x', 'x'), ('x', 'xx'), ('xy', 'y')])
def test_exact_variable_unit_inference_reuses_unchanged_decoder(units):
    training = ['aabbaabbaabb', 'bbbbaabbbb']
    counts = {c: row for c, row in collect_counts(training, 'ab', 8).items()
              if not c or sum(row.values()) >= 2}
    explicit = SuffixSource('ab', 8, 16., counts)
    compact = CompactSuffixAdapter(fit_compact(training, 'ab', 8, 2), 16.)
    for observed in words('xy', 5):
        a, b = decode(explicit, units, observed, .2), decode(compact, units, observed, .2)
        assert a.plaintext == b.plaintext
        assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-12)
        assert a.joint_log_probability == pytest.approx(b.joint_log_probability, abs=1e-12)
        assert a.reachable_nodes == b.reachable_nodes


def test_leading_zero_contexts_have_distinct_ids_and_cache_limits():
    adapter = CompactSuffixAdapter(fit_compact(['aaaaab'] * 4, 'ab', 4, 4), 16., cache_size=2)
    assert len({adapter.state('a' * n) for n in range(5)}) == 5
    for n in range(5):
        adapter.row(adapter.state('a' * n))
    assert adapter.row.cache_info().currsize <= 2
    with pytest.raises(ValueError):
        adapter.state('z')
    with pytest.raises(ValueError):
        adapter.row(-1)
