import math

import pytest

from scripts.diagnose_blind_channel_confirm002a import compare, endpoints, objective


def restart(i=0, score=10):
    return {'event': 'restart', 'restart': i, 'units': ['A', 'B'],
            'score': score, 'method': 'fixture'}


def sweep(accepted=True, complete=True):
    return {'event': 'sweep', 'restart': 0, 'parent_units': ['A', 'B'], 'parent_score': 10,
            'selected_units': ['B', 'A'], 'selected_score': 9, 'accepted': accepted,
            'complete': complete, 'stop_reason': 'improved' if accepted else 'local_optimum'}


@pytest.mark.parametrize('complete', [True, False])
def test_accepted_endpoint_preserved_even_in_partial_sweep(complete):
    result = endpoints([restart(), sweep(complete=complete)])
    assert result[0]['units'] == ['B', 'A'] and result[0]['score'] == 9


def test_unaccepted_candidate_never_changes_terminal_state():
    result = endpoints([restart(), sweep(accepted=False)])
    assert result[0]['units'] == ['A', 'B'] and result[0]['score'] == 10


def test_all_starts_preserved_including_unsupported_and_no_sweep():
    result = endpoints([restart(), restart(1, None)])
    assert len(result) == 2 and result[1]['score'] is None


@pytest.mark.parametrize('trace', [[sweep()], [restart(), restart()],
                                  [restart(), sweep(), sweep()], [{'event': 'unknown', 'restart': 0}]])
def test_bad_restart_trajectories_rejected(trace):
    with pytest.raises(ValueError):
        endpoints(trace)


def test_objective_preserves_code_cost_and_unsupported_value():
    assert objective(-3*math.log(2), 7) == pytest.approx(10)
    assert objective(None, 7) is None and objective(-math.inf, 7) is None
    for value, bits in [(math.nan, 7), (math.inf, 7), (0, -1), (0, True)]:
        with pytest.raises(ValueError):
            objective(value, bits)


def test_known_better_comparison_direction_and_tolerance():
    assert compare(12, 10)['true_key_is_known_better_candidate']
    assert not compare(9, 10)['true_key_is_known_better_candidate']
    assert not compare(10+1e-8, 10)['true_key_is_known_better_candidate']
    assert compare(None, 10)['true_key_is_known_better_candidate']
    with pytest.raises(ValueError):
        compare(10, None)
