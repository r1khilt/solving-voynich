import math

import numpy as np
import pytest
import torch

from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.joint_key_training import EpisodeSampler
from voynich.joint_reading_source import source_point_checked
from voynich.source_action_cache import cached_trace_logits
from voynich.source_action_cipher_diagnosis import (
    NEURAL_MODES, measure, neural_greedy_checked, neural_routes,
    permuted_glyph_input, source_action_scores, source_greedy, source_teacher_logits,
)
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, recovery_metrics
from voynich.recurrent_latin_source import ALPHABET


def small():
    torch.manual_seed(92801)
    model = SourceActionProposal(SourceActionConfig(rows=3, glyphs=2, width=16, heads=2,
        encoder_layers=2, decoder_layers=2, max_records=2, max_glyphs=32)).double().eval()
    env = ReadingEnvironment(((0, 1, 0, 0, 1, 1), (1, 0, 1, 1)), rows=3, glyphs=2)
    key = (0, 1, 3)
    texts = ((0, 1, 0, 2, 1), (1, 2, 1))
    return model, env, env.teaching_trace(texts, key)


@pytest.mark.parametrize('mode', NEURAL_MODES)
def test_intervention_hits_manual_cache_and_full_decoder_identically(mode):
    model, env, truth = small()
    with torch.no_grad():
        packet = model.pack([env], [truth])
        baseline = model(packet)[0]
        with neural_routes(model, mode, seed=92811) as counts:
            full = model(packet)[0]
            cached = cached_trace_logits(model, env, truth)
            torch.testing.assert_close(full, cached, rtol=0, atol=2e-12)
            assert counts['full_cross_calls'] == 2
            assert counts['cached_cross_calls'] == 2*len(truth[1])
            assert counts['glyph_calls'] == 2 and counts['binding_calls'] == len(truth[1])+1
        assert torch.equal(model(packet)[0], baseline)
        if mode in ('base', 'sham'):
            assert torch.equal(full, baseline)
        prediction, _, checked = neural_greedy_checked(model, env, mode, seed=92811)
        assert checked['maximum_legal_logit_delta'] < 2e-12
        assert checked['maximum_reference_argmax_deficit'] < 2e-12
        assert prediction['status'] in ('complete_path', 'dead_end')


def test_shuffled_lookup_retains_record_frames_and_padding():
    values = torch.tensor([[[0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 2, 2, 2]]])
    original = values.clone()
    changed = permuted_glyph_input(values, 2, 92813)
    assert torch.equal(values, original)
    assert torch.equal(changed, permuted_glyph_input(values, 2, 92813))
    for a, b in zip(values[0], changed[0], strict=True):
        assert sorted(a.tolist()) == sorted(b.tolist())
        assert torch.equal(a[:2], b[:2])
        assert torch.equal(a == 2, b == 2)


def test_bias_cut_removes_glyph_lookup_dependence_and_restores_on_error():
    model, env, truth = small()
    with torch.no_grad():
        packet = model.pack([env], [truth])
        baseline = model(packet)
        alternate = (permuted_glyph_input(packet[0], 2, 92817), *packet[1:])
        with neural_routes(model, 'cross_bias_erase'):
            torch.testing.assert_close(model(packet), model(alternate), rtol=0, atol=0)
        with pytest.raises(RuntimeError):
            with neural_routes(model, 'cross_bias'):
                raise RuntimeError('fixture interruption')
        assert torch.equal(model(packet), baseline)
    with pytest.raises(ValueError):
        with neural_routes(model, 'cross_bias'):
            pass


def test_source_increment_potential_is_exact_complete_joint_point_not_posterior():
    source = DenseSuffixAdapter(fit_compact(['aabacaabbaccabac'*4], 'abc', 3, 1), 4.)
    _, env, truth = small()
    logits = source_teacher_logits(source, env, truth, 'source_increment')
    potential = float(logits[torch.arange(len(truth[1])), torch.tensor(truth[1])].sum())
    point = source_point_checked(source, truth[2].texts)['log_probability']
    assert abs(potential-(point-3*math.log(len(env.pool)))) < 1e-12
    # A fully literal probability over legal actions for both declared policies.
    for mode in ('row_only', 'source_increment'):
        for snapshot in truth[0]:
            values = source_action_scores(source, env, snapshot, mode)
            assert torch.isfinite(values).sum() == len(env.legal_actions(snapshot))
            assert math.isclose(float(values.softmax(-1).sum()), 1., abs_tol=1e-12)
        prediction, arrays = source_greedy(source, env, mode)
        assert len(prediction['actions']) == len(arrays)


def test_category_decomposition_and_row_length_metrics_with_variable_binding():
    _, env, truth = small()
    source = DenseSuffixAdapter(fit_compact(['abcabac'*10], 'abc', 2, 1), 4.)
    logits = source_teacher_logits(source, env, truth, 'row_only')
    metrics, arrays = measure(logits, truth)
    assert len(arrays['target_logq']) == len(truth[1])
    assert sum(g['actions'] for g in metrics['groups'].values()) == len(truth[1])
    for group in metrics['groups'].values():
        assert abs(group['nll_sum']-group['category_nll_sum']-group['within_category_nll_sum']) < 1e-12
        assert group['correct_rows'] >= group['correct_actions']
        assert group['correct_lengths'] >= group['correct_actions']


def test_complete_reporting_pipeline_for_neural_and_source_greedy_schemas():
    sampler = EpisodeSampler([ALPHABET*20])
    episode = sampler.sample(np.random.default_rng(92819))
    env, truth = action_episode(sampler, episode)
    source = DenseSuffixAdapter(fit_compact([ALPHABET*20], ALPHABET, 2, 1), 4.)
    model = SourceActionProposal(SourceActionConfig(width=8, heads=2, encoder_layers=1,
        decoder_layers=1, max_records=2)).double().eval()
    policies = {}
    for mode in ('row_only', 'source_increment'):
        prediction, _ = source_greedy(source, env, mode)
        policies[mode] = recovery_metrics([prediction], [episode], sampler)
    prediction, _, _ = neural_greedy_checked(model, env, 'cross_bias_erase', seed=92823)
    policies['neural'] = recovery_metrics([prediction], [episode], sampler)
    assert all(p['record_attempts'] == 2 and len(p['cases']) == 1 for p in policies.values())
    # Whole-path/group report handles the same exact tuple schema as real run.
    metrics, _ = measure(source_teacher_logits(source, env, truth, 'source_increment'), truth)
    assert math.isfinite(metrics['whole_path_nll'])
