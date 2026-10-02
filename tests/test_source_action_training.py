import copy

import numpy as np
import pytest

from voynich.joint_key_training import EpisodeSampler
from voynich.source_action_proposal import SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, recovery_metrics, validation


def fixture():
    sampler = EpisodeSampler(['abcdefghiklmnopqrstuxyz'*16])
    rng = np.random.default_rng(92623)
    episodes = [sampler.sample(rng) for _ in range(2)]
    return sampler, episodes


def test_real_episode_trace_targets_and_oracle_metric_ceiling():
    sampler, episodes = fixture()
    predictions = []
    for episode in episodes:
        env, trace = action_episode(sampler, episode)
        assert len(trace[1]) == sum(w['length'] for w in episode[2]['windows'])
        assert trace[0][0].key == (-1,)*23
        predictions.append({'status': 'complete_path', 'key': trace[2].key, 'texts': trace[2].texts,
            'actions': trace[1], 'greedy_path_under_model_log_probability': -1.})
    score = recovery_metrics(predictions, episodes, sampler)
    assert score['complete_used_keys'] == 2 and score['exact_records'] == 4
    assert score['edits_with_failed_readings_full_length'] == 0
    wrong = copy.deepcopy(episodes[0])
    wrong[2]['windows'][0]['start'] = 100_000
    with pytest.raises(ValueError):
        action_episode(sampler, wrong)


def test_real_finite_validation_whole_path_and_failure_accounting():
    sampler, episodes = fixture()
    model = SourceActionProposal(SourceActionConfig(width=8, heads=2, encoder_layers=1,
        decoder_layers=1, max_records=2)).double()
    value = validation(model, episodes, sampler, batch=2)
    assert value['mean_whole_path_nll'] > 0 and len(value['joint_target_path_logq']) == 2
    assert model.training
    assert sum(c['edits'] for c in value['recovery']['cases']) == value['recovery']['edits_with_failed_readings_full_length']
    assert sum(c['true_letters'] for c in value['recovery']['cases']) == value['recovery']['true_letters']
    for case in value['recovery']['cases']:
        if case['status'] == 'dead_end':
            assert case['edits'] == case['true_letters'] and case['used_matches'] == 0
    broken = copy.deepcopy(value['predictions'])
    broken[0]['actions'] = ()
    with pytest.raises(ValueError):
        recovery_metrics(broken, episodes, sampler)
