import itertools
import math
import shutil
import copy

import numpy as np
import pytest

from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.native_partial_bound import NativePartialBound
from voynich.native_suffix_marginal import NativeMarginal, build_native
from voynich.partial_unit_bound import relaxed_record


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    if not (shutil.which('clang++') or shutil.which('c++')):
        pytest.skip('Optional local C++ compiler unavailable')
    return build_native(tmp_path_factory.mktemp('native-relaxation')/'build')


@pytest.mark.parametrize('order', [0, 1, 3, 12])
def test_all_partial_binary_dictionaries_match_python_nodes_edges_and_sums(build, order):
    source = DenseSuffixAdapter(fit_compact(['abbaabba', 'babbbbb', 'aabab'], 'ab', order, 1), 4.)
    native, ordinary = NativePartialBound(source, build), NativeMarginal(source, build)
    pool = ('x', 'y', 'xy')
    for partial in itertools.product((None, *pool), repeat=2):
        for length in range(5):
            for chars in itertools.product('xy', repeat=length):
                record = ''.join(chars)
                expected = relaxed_record(source, partial, pool, record, .25)
                actual = native.record(partial, pool, record, .25)
                assert actual['nodes'] == expected['nodes'] and actual['edges'] == expected['edges']
                assert actual['log_likelihood_upper_bound'] == pytest.approx(expected['log_likelihood_upper_bound'], abs=1e-12)
                if None not in partial:
                    value = ordinary.score(partial, record, .25).log_likelihood
                    assert actual['log_likelihood_upper_bound'] == pytest.approx(None if value == -math.inf else value, abs=1e-12)


def test_long_nonnormalized_relaxation_log_domain_and_caps(build):
    source = DenseSuffixAdapter(fit_compact(['abbababbbbba'], 'ab', 3, 1), 16.)
    native = NativePartialBound(source, build)
    partial, pool, observed = (None, None), ('x', 'xx', 'xxx'), 'x'*100
    actual = native.record(partial, pool, observed, .2)
    expected = relaxed_record(source, partial, pool, observed, .2)
    assert actual['log_likelihood_upper_bound'] > 0
    assert actual['log_likelihood_upper_bound'] == pytest.approx(expected['log_likelihood_upper_bound'], abs=1e-10)
    assert native.record(partial, pool, observed, .2, max_nodes=actual['nodes'], max_edges=actual['edges']) == actual
    with pytest.raises(RuntimeError, match='node cap; no pruning'):
        native.record(partial, pool, observed, .2, max_nodes=actual['nodes']-1)
    with pytest.raises(RuntimeError, match='edge cap; no pruning'):
        native.record(partial, pool, observed, .2, max_edges=actual['edges']-1)


@pytest.mark.parametrize('changes', [
    {'partial': (None,)}, {'pool': ()}, {'pool': ('x', 'x')}, {'rho': True}, {'max_nodes': 0},
    {'observed': None}, {'partial': ('z', None)},
])
def test_invalid_settings_rejected_before_native_call(build, changes):
    source = DenseSuffixAdapter(fit_compact(['abba'], 'ab', 1, 1), 4.)
    with pytest.raises(ValueError):
        NativePartialBound(source, build).record(**{
            'partial': (None, None), 'pool': ('x', 'y'), 'observed': 'xy', 'rho': .2, **changes})


def test_replaced_unnormalized_dense_arrays_are_rejected(build):
    source = DenseSuffixAdapter(fit_compact(['abba'], 'ab', 1, 1), 4.)
    for replacement in (.7, 0.):
        altered = copy.copy(source)
        altered.probabilities = np.full_like(source.probabilities, replacement)
        altered.probabilities.setflags(write=False)
        with pytest.raises(ValueError, match='normalized positive'):
            NativePartialBound(altered, build)
