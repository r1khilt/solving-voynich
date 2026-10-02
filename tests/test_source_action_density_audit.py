import math

import pytest
import torch

from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.joint_reading_source import source_point_checked
from voynich.source_action_density_audit import replay_draws, sample_and_check
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal


@pytest.mark.parametrize('bindings', [False, True])
def test_original_stochastic_sampler_matches_full_reference_and_density(bindings):
    torch.manual_seed(92701)
    model = SourceActionProposal(SourceActionConfig(rows=3, glyphs=2, width=16, heads=2,
        encoder_layers=1, decoder_layers=2, max_glyphs=16, binding_input=bindings)).double()
    env = ReadingEnvironment(((0, 1, 0, 0), (1, 0)), rows=3, glyphs=2)
    result, logits, _, metrics = sample_and_check(model, env, seed=92703)
    other = model.propose(env, seed=92703)
    assert result['actions'] == other['actions'] and result['state'] == other['state']
    assert abs(result['path_log_probability']-other['path_log_probability']) < 2e-12
    assert metrics['maximum_legal_logit_delta'] < 2e-12 and model.training
    with pytest.raises(ValueError):
        replay_draws(env, result['actions'], logits[:-1], seed=92703)


def test_true_dead_end_is_retained_and_no_support_repair():
    model = SourceActionProposal(SourceActionConfig(rows=1, glyphs=2, width=16, heads=2,
        encoder_layers=1, decoder_layers=1, max_glyphs=8)).double()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([10., -10.]))
    env = ReadingEnvironment(((0, 1),), rows=1, glyphs=2)
    result, _, _, _ = sample_and_check(model, env, seed=0)
    assert result['status'] == 'dead_end' and result['actions'] == (0,)
    with pytest.raises(FloatingPointError, match='underflow'):
        replay_draws(env, (0,), [[1000., -1000.]], seed=0)
    with pytest.raises(ValueError, match='seed'):
        replay_draws(env, (1,), [[10., -10.]], seed=0)


def test_record_reset_geometric_law_and_independent_sparse_score():
    source = DenseSuffixAdapter(fit_compact(['aaaaababbabbb'], 'ab', 3, 1), 4.)
    a = source_point_checked(source, ((0, 1, 0),), rho=.2)
    pair = source_point_checked(source, ((0, 1, 0), (0, 1, 0)), rho=.2)
    assert math.isclose(pair['log_probability'], 2*a['log_probability'], abs_tol=1e-12)
    assert pair['maximum_total_log_delta'] < 1e-12
    for texts, rho in [((), .2), (((),), .2), (((True,),), .2), (((2,),), .2), (((0,),), True)]:
        with pytest.raises(ValueError):
            source_point_checked(source, texts, rho=rho)
