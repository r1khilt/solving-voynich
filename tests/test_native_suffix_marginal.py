import copy
import itertools
import math
import shutil

import numpy as np
import pytest

from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.native_suffix_marginal import NativeMarginal, build_native, marginal_python
from voynich.sparse_suffix_source import decode


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    if not (shutil.which('clang++') or shutil.which('c++')):
        pytest.skip('Optional local C++ compiler unavailable')
    return build_native(tmp_path_factory.mktemp('native-marginal') / 'build')


def source(order=3, alphabet='ab'):
    return DenseSuffixAdapter(fit_compact([alphabet * 7, alphabet[::-1] * 3, alphabet[0] * 11], alphabet, order, 1), 16.)


@pytest.mark.parametrize('order', [0, 1, 3, 12])
@pytest.mark.parametrize('units', [('x', 'x'), ('x', 'xx'), ('xy', 'y')])
def test_every_binary_observation_to_length_five_matches_full_decoder(build, order, units):
    model = source(order)
    native = NativeMarginal(model, build)
    for size in range(6):
        for chars in itertools.product('xy', repeat=size):
            observed = ''.join(chars)
            baseline = decode(model, units, observed, .2)
            for result in [marginal_python(model, units, observed, .2), native.score(units, observed, .2)]:
                assert result.reachable_nodes == baseline.reachable_nodes
                assert result.edges == baseline.edges
                assert result.log_likelihood == pytest.approx(baseline.log_likelihood, abs=1e-12)


def test_native_marginal_matches_exhaustive_text_probabilities(build):
    model = source(4)
    native = NativeMarginal(model, build)
    units = ['x', 'xx']
    for rho in [.00001, .2, .9, 1-1e-12]:
        for length in range(7):
            mass = []
            for size in range(length + 1):
                for text in itertools.product('ab', repeat=size):
                    if ''.join(units[model.alphabet.index(c)] for c in text) != 'x' * length:
                        continue
                    state, probability = model.state(''), rho * (1-rho)**size
                    for char in text:
                        i = model.alphabet.index(char)
                        probability *= model.probabilities[state, i]
                        state = model.step(state, i)
                    mass.append(probability)
            assert native.score(units, 'x' * length, rho).log_likelihood == pytest.approx(math.log(math.fsum(mass)), abs=1e-12)


def test_unicode_and_embedded_nul_units_are_matched_as_python_codepoints(build):
    model = source(3, 'αβ')
    native = NativeMarginal(model, build)
    for record in ['', '雪\0雪', '雪雪\0雪雪', '\0']:
        result = native.score(('雪', '雪\0'), record, .01)
        baseline = decode(model, ('雪', '雪\0'), record, .01)
        assert result.log_likelihood == pytest.approx(baseline.log_likelihood, abs=1e-12)
        assert (result.reachable_nodes, result.edges) == (baseline.reachable_nodes, baseline.edges)


def test_long_ambiguous_input_stays_in_log_domain(build):
    model = source(6)
    result = NativeMarginal(model, build).score(('x', 'xx'), 'x' * 2000, .9)
    baseline = marginal_python(model, ('x', 'xx'), 'x' * 2000, .9)
    assert result.log_likelihood < -1000
    assert result.log_likelihood == pytest.approx(baseline.log_likelihood, abs=1e-8)
    assert (result.reachable_nodes, result.edges) == (baseline.reachable_nodes, baseline.edges)


def test_native_caps_raise_without_returning_partial_evidence(build):
    model = source()
    native = NativeMarginal(model, build)
    for scorer in [native.score, lambda *a, **k: marginal_python(model, *a, **k)]:
        with pytest.raises(RuntimeError, match='node cap'):
            scorer(('x', 'xx'), 'xxxx', .2, max_nodes=1)
        with pytest.raises(RuntimeError, match='edge cap'):
            scorer(('x', 'xx'), 'xxxx', .2, max_edges=1)
        assert scorer(('x', 'xx'), '', .2, max_nodes=1).log_likelihood == math.log(.2)


@pytest.mark.parametrize('override', [{'rho': 0}, {'rho': True}, {'max_nodes': False}, {'max_edges': 0},
                                     {'units': ('', 'x')}, {'units': ('x',)}, {'record': None}])
def test_invalid_settings_fail_before_native_access(build, override):
    native = NativeMarginal(source(), build)
    with pytest.raises(ValueError):
        native.score(**{'units': ('x', 'y'), 'record': 'xy', 'rho': .2, **override})


def test_library_source_and_memory_guards(build):
    model = source()
    for field in ['source_sha256', 'library_sha256', 'abi']:
        altered = dict(build, **{field: 'altered'})
        with pytest.raises(ValueError, match='identity'):
            NativeMarginal(model, altered)
    altered = copy.copy(model)
    altered.transitions = np.ones_like(model.transitions, dtype=np.int64)
    with pytest.raises(ValueError, match='layout'):
        NativeMarginal(altered, build)
    altered = copy.copy(model)
    altered.transitions = np.full_like(model.transitions, len(model.transitions))
    altered.transitions.setflags(write=False)
    with pytest.raises(ValueError, match='transition'):
        NativeMarginal(altered, build).score(('x', 'y'), 'x', .2)


def test_build_will_not_overwrite_existing_directory(build):
    from pathlib import Path
    with pytest.raises(FileExistsError):
        build_native(Path(build['library_path']).parent)


def test_scorer_pins_original_arrays_and_alphabet_when_source_attributes_change(build):
    model = source()
    native = NativeMarginal(model, build)
    original = native.score(('x', 'xx'), 'xxxx', .2)
    model.probabilities = np.zeros((1, 1))
    model.transitions = np.zeros((1, 1), dtype=np.uint32)
    model.alphabet = ('z',)
    model.state = lambda history: 99999
    assert native.score(('x', 'xx'), 'xxxx', .2) == original
