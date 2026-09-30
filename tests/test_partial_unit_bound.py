"""Independent full-history rational sums over relaxed and consistent paths."""
import itertools
import math
from fractions import Fraction as F

import pytest

from scripts.audit_blind_channel_dev004 import literal_model_bits
from voynich.partial_unit_bound import minimum_code_bits, partial_fit_bound, relaxed_record


class Source:
    alphabet = tuple('ab')
    probabilities = [[F(3, 5), F(2, 5)], [F(1, 4), F(3, 4)], [F(4, 5), F(1, 5)]]
    def state(self, history): return 0 if not history else self.alphabet.index(history[-1])+1
    def step(self, state, letter): return letter+1


CONTEXT = {'source_alphabet': list('ab'), 'glyph_alphabet': list('xy'),
           'max_emission_length': 2, 'max_states': 2, 'max_alternatives': 3, 'source_count': 1}
POOL = ('x', 'y', 'xy')


def enumerated(partial, pool, maximum, rho=F(1, 4)):
    groups = {}
    def visit(plain, emitted, mass):
        groups[emitted] = groups.get(emitted, F(0))+mass*rho
        # The reference reads full plaintext history, not source state/step.
        weights = (F(3, 5), F(2, 5)) if not plain else (
            (F(1, 4), F(3, 4)) if plain[-1] == 'a' else (F(4, 5), F(1, 5)))
        for letter, char in enumerate('ab'):
            for unit in (pool if partial[letter] is None else (partial[letter],)):
                if len(emitted)+len(unit) <= maximum:
                    visit(plain+char, emitted+unit, mass*(1-rho)*weights[letter])
    visit('', '', F(1))
    return groups


def records(maximum=4):
    return [''.join(chars) for n in range(maximum+1) for chars in itertools.product('xy', repeat=n)]


def test_every_partial_and_record_matches_independent_relaxed_enumeration():
    for partial in itertools.product((None, *POOL), repeat=2):
        expected = enumerated(partial, POOL, 4)
        for observed in records():
            value = relaxed_record(Source(), partial, POOL, observed, .25)
            probability = expected.get(observed, 0)
            if probability:
                assert value['log_likelihood_upper_bound'] == pytest.approx(math.log(float(probability)), abs=3e-15)
            else:
                assert value['log_likelihood_upper_bound'] is None
            assert not value['normalized_channel']


def test_all_completion_objectives_are_bounded_and_refinement_is_monotone():
    observed = ('xyx', 'xx', '')  # Separate BOS and stop for every record.
    completions = list(itertools.product(POOL, repeat=2))
    for partial in itertools.product((None, *POOL), repeat=2):
        value = partial_fit_bound(Source(), partial, POOL, observed, .25, CONTEXT)
        upper = -math.inf if value['log_objective_upper_bound'] is None else value['log_objective_upper_bound']
        bits = value['minimum_model_bits']
        for key in completions:
            if all(p is None or p == u for p, u in zip(partial, key)):
                groups = enumerated(key, POOL, 3)
                masses = [groups.get(r, 0) for r in observed]
                actual = -math.inf if not all(masses) else math.fsum(math.log(float(v)) for v in masses)
                actual -= literal_model_bits(key, CONTEXT)*math.log(2)
                assert actual <= upper+1e-14
                assert bits <= literal_model_bits(key, CONTEXT)
                if None not in partial:
                    assert actual == pytest.approx(upper, abs=1e-14)
        for row, unit in itertools.product(range(2), POOL):
            if partial[row] is None:
                child = list(partial)
                child[row] = unit
                child_bound = partial_fit_bound(Source(), child, POOL, observed, .25, CONTEXT)
                lower = -math.inf if child_bound['log_objective_upper_bound'] is None else child_bound['log_objective_upper_bound']
                assert lower <= upper+1e-14
        assert not value['interval_arithmetic_certified']


def test_relaxation_can_exceed_probability_one_and_is_not_viterbi():
    value = relaxed_record(Source(), (None, None), ('x', 'xx', 'xxx'), 'x'*8, .2)
    assert value['log_likelihood_upper_bound'] > 0
    complete = enumerated(('x', 'x'), POOL, 3)
    # Duplicate rows give two plaintext paths for one glyph; sum exceeds each.
    actual = relaxed_record(Source(), ('x', 'x'), POOL, 'x', .25)['log_likelihood_upper_bound']
    assert actual == pytest.approx(math.log(float(complete['x'])))
    assert actual > math.log(float(F(1, 4)*F(3, 4)*F(3, 5)))


def test_empty_and_impossible_records_and_zero_probability_letters():
    assert relaxed_record(Source(), (None, None), POOL, '', .25)['log_likelihood_upper_bound'] == math.log(.25)
    assert relaxed_record(Source(), ('x', 'x'), POOL, 'y', .25)['log_likelihood_upper_bound'] is None
    class Zero(Source):
        probabilities = [[1., 0.], [1., 0.], [1., 0.]]
    assert relaxed_record(Zero(), ('x', 'y'), POOL, 'y', .25)['log_likelihood_upper_bound'] is None
    assert relaxed_record(Zero(), ('x', 'y'), POOL, 'xx', .25)['log_likelihood_upper_bound'] == pytest.approx(math.log(.25*.75**2))


def test_node_and_edge_caps_fail_at_exact_boundary_without_pruning():
    actual = relaxed_record(Source(), (None, None), POOL, 'xyxy', .25)
    assert relaxed_record(Source(), (None, None), POOL, 'xyxy', .25,
                          max_nodes=actual['nodes'], max_edges=actual['edges']) == actual
    with pytest.raises(RuntimeError, match='node cap; no pruning'):
        relaxed_record(Source(), (None, None), POOL, 'xyxy', .25, max_nodes=actual['nodes']-1)
    with pytest.raises(RuntimeError, match='edge cap; no pruning'):
        relaxed_record(Source(), (None, None), POOL, 'xyxy', .25, max_edges=actual['edges']-1)


def test_shortest_unit_code_lower_bound_for_nonuniform_lengths():
    pool = ('xy', 'xyx')
    context = {**CONTEXT, 'max_emission_length': 3}
    assert minimum_code_bits((None, 'xyx'), pool, context) == literal_model_bits(('xy', 'xyx'), context)
    assert minimum_code_bits(('xyx', 'xyx'), pool, context) == literal_model_bits(('xyx', 'xyx'), context)


def test_unnormalized_source_fails_instead_of_producing_a_probabilistic_bound():
    class Bad(Source):
        probabilities = [[.6, .6], [.25, .75], [.8, .2]]
    with pytest.raises(ValueError, match='not normalized'):
        relaxed_record(Bad(), (None, None), POOL, 'x', .25)


@pytest.mark.parametrize('rho', [0., 1., True, math.nan])
def test_invalid_settings_fail(rho):
    with pytest.raises(ValueError):
        relaxed_record(Source(), (None, None), POOL, 'x', rho)
    with pytest.raises(ValueError):
        partial_fit_bound(Source(), (None, None), POOL, (), .25, CONTEXT)
    with pytest.raises(ValueError):
        minimum_code_bits((None, None), POOL, {**CONTEXT, 'glyph_alphabet': ['x']})
