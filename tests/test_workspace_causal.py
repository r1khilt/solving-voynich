import copy

import numpy as np

from voynich.workspace.causal_campaign import condition_seed, decide, make_delta, meets, rates, select_layer
from voynich.workspace.geometry import coordinate_swap, normalize_rows


def test_swap_is_reflection_involution_and_norm_preserving():
    h = np.array([2., -3., 7., 4.])
    rows = normalize_rows([[1., 2., 0, 0], [0., 2., 1., 0]])
    n = rows[0] - rows[1]
    n /= np.linalg.norm(n)
    delta, _ = coordinate_swap(h, rows)
    np.testing.assert_allclose(delta, -2 * (h @ n) * n, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(h+delta), np.linalg.norm(h), rtol=1e-6)
    inverse, _ = coordinate_swap(h+delta, rows)
    np.testing.assert_allclose(h+delta+inverse, h, atol=1e-6)


def test_controls_match_reference_norm_and_preserve_declared_complement():
    h, donor = np.arange(6.), -np.arange(6.)
    rows = np.eye(6)[:2]
    swap, _ = make_delta('swap', h, donor, rows, rows, seed=9)
    for condition in ('random', 'orthogonal'):
        delta, _ = make_delta(condition, h, donor, rows, rows, seed=9)
        np.testing.assert_allclose(np.linalg.norm(delta), np.linalg.norm(swap), rtol=1e-6)
        if condition == 'orthogonal':
            np.testing.assert_allclose(rows @ delta, 0, atol=1e-6)
    np.testing.assert_array_equal(make_delta('identity', h, donor, rows, rows, seed=9)[0], 0)
    np.testing.assert_array_equal(make_delta('donor', h, donor, rows, rows, seed=9)[0], donor-h)
    assert condition_seed(1, 'x', 2, 'random') != condition_seed(1, 'x', 2, 'orthogonal')


def test_copy_and_baseline_failure_do_not_inflate_counterfactual_rate():
    rows = [dict(expected_effect='change', baseline_correct=True, donor_correct=True,
                 counterfactual_correct=False, original_correct=True, valid=True),
            dict(expected_effect='change', baseline_correct=False, donor_correct=True,
                 counterfactual_correct=True, original_correct=False, valid=True),
            dict(expected_effect='preserve', baseline_correct=True, donor_correct=True,
                 counterfactual_correct=True, original_correct=True, valid=True)]
    score = rates(rows)
    assert score['counterfactual'] == {'successes': 0, 'count': 1, 'rate': 0.}
    assert score['counterfactual_all']['rate'] == .5
    assert score['copy_preserved']['rate'] == 1.


def _summary():
    base = {'counterfactual': {'rate': .7}, 'copy_preserved': {'rate': 1.}, 'invalid_count': 0,
            'by_relation': {k: {'counterfactual': {'rate': .6, 'count': 8}}
                            for k in ('capital', 'currency', 'language', 'capital_continent')}}
    conditions = {k: copy.deepcopy(base) for k in ('swap', 'random', 'orthogonal', 'raw_swap', 'donor', 'identity')}
    for c in ('random', 'orthogonal', 'raw_swap'):
        conditions[c]['counterfactual']['rate'] = .1
    return {'3': copy.deepcopy(conditions), '7': copy.deepcopy(conditions)}


CONFIG = {'layers': [3, 7], 'confirm_copy_preserve_min': .95, 'confirm_swap_min': .5,
          'confirm_control_advantage_min': .2, 'confirm_relation_min': .3}


def test_selection_tie_and_copy_gate():
    summary = _summary()
    assert select_layer(summary, CONFIG)['selected_layer'] == 3
    summary['3']['swap']['copy_preserved']['rate'] = .9
    assert select_layer(summary, CONFIG)['selected_layer'] == 7
    summary['7']['swap']['copy_preserved']['rate'] = .9
    assert select_layer(summary, CONFIG)['selected_layer'] is None


def test_confirmation_fails_on_missing_competence_or_relation():
    summary = _summary()
    assert decide(summary, 3, CONFIG)['supported']
    summary['3']['swap']['by_relation'].pop('currency')
    assert not decide(summary, 3, CONFIG)['supported']


def test_decimal_threshold_and_count_based_layer_ties():
    assert meets(.7-.5, .2)
    assert not meets(.19999, .2)
    summary = _summary()
    for layer, successes in [('3', 7), ('7', 6)]:
        for condition in ('swap', 'random', 'orthogonal', 'raw_swap'):
            n = successes if condition == 'swap' else successes-2
            summary[layer][condition]['counterfactual'] = {'successes': n, 'count': 10, 'rate': n/10}
    assert select_layer(summary, CONFIG)['selected_layer'] == 3
    summary['3']['swap']['counterfactual']['rate'] = None
    assert not decide(summary, 3, CONFIG)['supported']
