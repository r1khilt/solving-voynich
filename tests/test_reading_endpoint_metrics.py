import copy

import pytest

from tests.test_stochastic_reading_metrics import outcomes
from voynich.reading_endpoint_metrics import competence_gate, reading_endpoint_metrics
from voynich.source_action_training import action_episode


def fixture():
    sampler, episode, correct, failed = outcomes()
    env, _ = action_episode(sampler, episode)
    result = []
    for old in (correct, failed):
        state = env.initial
        for a in old['actions']:
            state = env.advance(state, a)
        result.append({'actions': old['actions'], 'key': state.key, 'offsets': state.offsets,
                       'texts': state.texts, 'complete': old['status'] == 'complete_path'})
    return sampler, episode, *result


def test_chain_endpoint_has_no_q_and_penalizes_failures():
    sampler, episode, good, failed = fixture()
    perfect = reading_endpoint_metrics([good], [episode], sampler)
    assert competence_gate(perfect) and perfect['edits'] == 0 and perfect['exact_records'] == 2
    assert perfect['used_matches'] == perfect['used_rows'] == 2
    result = reading_endpoint_metrics([good, failed], [episode]*2, sampler)
    assert not competence_gate(result) and result['edits'] == 128
    assert result['cases'][1]['used_matches'] == 0 and result['complete'] == 1
    assert result['no_state_density_claim']


@pytest.mark.parametrize('field', ['actions', 'texts', 'offsets', 'key', 'complete'])
def test_literal_damage_rejected(field):
    sampler, episode, good, _ = fixture()
    bad = copy.deepcopy(good)
    if field == 'actions':
        bad[field] = bad[field][:-1]
    elif field == 'texts':
        bad[field] = ((), ())
    elif field == 'offsets':
        bad[field] = (0, 0)
    elif field == 'key':
        bad[field] = (-1,)*23
    else:
        bad[field] = 1
    with pytest.raises(ValueError):
        reading_endpoint_metrics([bad], [episode], sampler)


def test_gate_exact_thresholds_and_empty_panels():
    s = {'used_rows': 10, 'used_matches': 9, 'true_letters': 100, 'edits': 10,
         'record_attempts': 10, 'exact_records': 5, 'episodes': 10, 'complete_used_key': 5}
    assert competence_gate(s)
    for field in ('used_matches', 'exact_records', 'complete_used_key'):
        assert not competence_gate({**s, field: s[field]-1})
    assert not competence_gate({**s, 'edits': 11})
    assert not competence_gate({**s, 'episodes': 0})
