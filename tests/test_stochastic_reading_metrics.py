import copy

import pytest

from voynich.joint_key_training import EpisodeSampler
from voynich.source_action_training import action_episode
from voynich.stochastic_reading_metrics import stochastic_recovery_metrics


def outcomes():
    sampler = EpisodeSampler(['a'*24+'b'*400])
    episode = sampler.make([{'segment': 0, 'start': 0, 'length': 64}]*2, [0, 1]+[0]*21)
    env, truth = action_episode(sampler, episode)
    correct = {'status': 'complete_path', 'actions': truth[1], 'key': truth[2].key,
               'texts': truth[2].texts, 'path_log_probability': -1.}
    state, actions = env.initial, []
    for row in range(23):
        action = 2*row
        actions.append(action)
        state = env.advance(state, action)
    while env.legal_actions(state):
        actions.append(0)
        state = env.advance(state, 0)
    failed = {'status': 'dead_end', 'actions': actions, 'key': state.key,
              'texts': state.texts, 'path_log_probability': -2.}
    return sampler, episode, correct, failed


def test_stochastic_schema_complete_and_failed_end_to_end_without_greedy_field():
    sampler, episode, correct, failed = outcomes()
    result = stochastic_recovery_metrics([correct, failed], [episode, episode], sampler)
    assert result['density_kind'] == 'stochastic_action_path'
    assert result['exact_records'] == 2 and result['complete_used_keys'] == 1
    assert result['complete_readings'] == result['failed_readings'] == 1
    assert result['edits_with_failed_readings_full_length'] == 128
    assert result['cases'][1]['used_matches'] == 0 and result['used_rows'] == 4


@pytest.mark.parametrize('damage', ['greedy_only', 'positive_logq', 'truncated', 'altered_text'])
def test_bad_density_or_literal_record_is_rejected(damage):
    sampler, episode, correct, _ = outcomes()
    value = copy.deepcopy(correct)
    if damage == 'greedy_only':
        value['greedy_path_under_model_log_probability'] = value.pop('path_log_probability')
    elif damage == 'positive_logq':
        value['path_log_probability'] = .1
    elif damage == 'truncated':
        value['actions'] = value['actions'][:-1]
    else:
        value['texts'] = ((), ())
    with pytest.raises(ValueError):
        stochastic_recovery_metrics([value], [episode], sampler)
