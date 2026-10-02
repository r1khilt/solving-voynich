"""Reporting/closure integration before the actual scientific namespace exists."""

import copy
import json

import numpy as np
import pytest
import torch

from scripts.audit_source_action_cipher001 import equivalent, integer_edits, path_arithmetic, teacher_arithmetic
from scripts.run_source_action_cipher001 import free_summary, public_free_cases, teacher_summary
from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.joint_key_training import EpisodeSampler
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_cipher_diagnosis import (
    NEURAL_MODES, SOURCE_MODES, measure, neural_greedy_checked, source_greedy, source_teacher_logits,
)
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, control_records


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fixture():
    sampler = EpisodeSampler([ALPHABET*20])
    episode = sampler.sample(np.random.default_rng(92831))
    env, truth = action_episode(sampler, episode)
    source = DenseSuffixAdapter(fit_compact([ALPHABET*20], ALPHABET, 2, 1), 4.)
    return sampler, episode, env, truth, source


def roundtrip(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def test_saved_teacher_report_and_independent_category_arithmetic():
    _, _, env, truth, source = fixture()
    scores = {}
    for mode in SOURCE_MODES:
        metrics, arrays = measure(source_teacher_logits(source, env, truth, mode), truth)
        teacher_arithmetic(env, truth, roundtrip(metrics), roundtrip(arrays))
        broken = copy.deepcopy(arrays)
        broken['category_logq'][0] -= 1.
        with pytest.raises(AssertionError):
            teacher_arithmetic(env, truth, metrics, broken)
        scores[mode] = metrics
    cases = [{'case': i, 'scores': scores} for i in range(64)]
    equivalent(teacher_summary(cases), teacher_summary(roundtrip(cases)))
    summary = teacher_summary(cases)
    for mode in SOURCE_MODES:
        assert summary[mode]['mean_whole_path_nll'] == pytest.approx(scores[mode]['whole_path_nll'])
    with pytest.raises(ValueError):
        teacher_summary([])


def test_all_seven_free_policy_schemas_include_all_97_allocations():
    sampler, episode, env, _, source = fixture()
    episodes = [episode]*64
    controls = control_records(episodes, seed=92833)
    actual = {}
    for mode in SOURCE_MODES:
        prediction, vectors = source_greedy(source, env, mode)
        path_arithmetic(env, roundtrip(prediction), vectors)
        actual[mode] = prediction
    torch.manual_seed(92837)
    model = SourceActionProposal(SourceActionConfig(width=8, heads=2, encoder_layers=1,
        decoder_layers=4, max_records=2)).double().eval()
    prediction, vectors, numeric = neural_greedy_checked(model, env, 'cross_bias_erase', seed=92839)
    _, check = path_arithmetic(env, roundtrip(prediction), **{
        'actual': vectors['actual_legal_logits'], 'reference': vectors['reference_legal_logits']})
    equivalent(check, numeric)
    # Actual tiny-model/source outputs exercise report schemas; no empirical claim
    # about the five real intervention modes is made by repeating this fixture.
    control_predictions = [source_greedy(source, ReadingEnvironment(c['records'], rows=23, glyphs=6),
                                         'row_only')[0] for c in controls]
    cases = []
    for mode in (*NEURAL_MODES, *SOURCE_MODES):
        values = [actual[mode] if mode in SOURCE_MODES else prediction]*64+control_predictions
        cases.extend({'case': i, 'mode': mode, 'prediction': p, 'archive': {'fixture': True}}
                     for i, p in enumerate(values))
    summary = free_summary(roundtrip(cases), episodes, sampler, controls)
    assert set(summary) == set((*NEURAL_MODES, *SOURCE_MODES))
    for item in summary.values():
        recovery = item['recovery']
        assert recovery['record_attempts'] == 128 and len(recovery['cases']) == 64
        assert recovery['complete_readings']+recovery['failed_readings'] == 64
        assert sum(g['cases'] for g in item['controls']['groups'].values()) == 33
        assert recovery['failure_penalty_is_full_length'] and recovery['unused_rows_not_scored']
    public = public_free_cases(cases)
    assert len(public) == 679 and all('prediction' not in c for c in public)
    assert all('prediction' in c for c in cases)  # no mutation of reporting inputs
    with pytest.raises(AssertionError):
        free_summary(cases[:-1], episodes, sampler, controls)
    unordered = copy.deepcopy(cases)
    unordered[0]['case'] = 1
    with pytest.raises(AssertionError):
        free_summary(unordered, episodes, sampler, controls)


def test_integer_edits_and_path_closure_reject_false_density():
    assert integer_edits((0, 1, 2), (0, 2)) == 1
    assert integer_edits((), (0, 1)) == 2
    _, _, env, _, source = fixture()
    prediction, vectors = source_greedy(source, env, 'row_only')
    prediction = roundtrip(prediction)
    prediction['greedy_path_under_model_log_probability'] += 1.
    with pytest.raises(AssertionError):
        path_arithmetic(env, prediction, vectors)
