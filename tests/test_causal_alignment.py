import numpy as np
import pytest
import torch

from voynich.causal_alignment import make_pairs, orthonormalize, patch_residual, readout_basis, resumed_logits
from voynich.model import ModelConfig, VoynichTransformer


def model_and_inputs():
    torch.manual_seed(17)
    model = VoynichTransformer(ModelConfig(vocab_size=18, d_model=16, n_layers=2, n_heads=2,
                                           d_ff=32, context_length=32, dropout=0))
    model.eval().requires_grad_(False)
    return model, torch.randint(10, 18, (3, 9)), torch.randint(10, 18, (3, 9))


@pytest.mark.parametrize('site', [0, 1])
def test_cached_forward_and_intervention_gradient_match_full_hooks(site):
    model, x, donor_x = model_and_inputs()
    name = f'blocks.{site}.resid_post'
    clean = model(x, cache_names=[name])
    donor = model(donor_x, cache_names=[name]).cache[name]
    base = clean.cache[name]
    if site == 1:
        base, donor = base[:, -1:], donor[:, -1:]
    torch.testing.assert_close(resumed_logits(model, base, site), clean.logits[:, -1], atol=1e-6, rtol=1e-5)
    basis = orthonormalize(torch.randn(16, 3)).requires_grad_()
    changed, _ = patch_residual(base, donor, basis)
    resumed = resumed_logits(model, changed, site)
    weights = torch.randn_like(resumed)
    fast_grad, = torch.autograd.grad((resumed*weights).sum(), basis)
    full_basis = basis.detach().clone().requires_grad_()
    full = model(x, interventions={name: lambda v: patch_residual(v, donor, full_basis)[0]}).logits[:, -1]
    full_grad, = torch.autograd.grad((full*weights).sum(), full_basis)
    torch.testing.assert_close(resumed, full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fast_grad, full_grad, atol=1e-6, rtol=1e-5)
    assert fast_grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())


def test_late_complete_residual_matches_donor_and_identity_is_exact():
    model, x, donor_x = model_and_inputs()
    name = 'blocks.1.resid_post'
    clean, donor = model(x, cache_names=[name]), model(donor_x, cache_names=[name])
    value = clean.cache[name].clone()
    value[:, -1] = donor.cache[name][:, -1]
    torch.testing.assert_close(resumed_logits(model, value, 1), donor.logits[:, -1], rtol=1e-5, atol=1e-7)
    q = orthonormalize(torch.randn(16, 3))
    unchanged, delta = patch_residual(value, value, q)
    torch.testing.assert_close(unchanged, value, rtol=0, atol=0)
    assert delta.norm() == 0


def test_orthogonal_patch_and_norm_control_constraints():
    base, donor = torch.randn(4, 8, 16), torch.randn(4, 1, 16)
    q = orthonormalize(torch.randn(16, 3))
    torch.testing.assert_close(q.T@q, torch.eye(3), rtol=0, atol=5e-7)
    changed, displacement = patch_residual(base, donor, q)
    torch.testing.assert_close(changed[:, -1]@q, donor[:, -1]@q, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(changed[:, :-1], base[:, :-1], atol=0, rtol=0)
    random = orthonormalize(torch.randn(16, 3))
    _, controlled = patch_residual(base, donor, random, displacement.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(controlled.norm(dim=-1), displacement.norm(dim=-1), atol=1e-6, rtol=1e-5)


def test_pair_selection_depends_only_on_observations_and_generator_beliefs():
    records = [{'text': 'a', 'posterior': [.8, .1, .05, .05]},
               {'text': 'a', 'posterior': [.1, .8, .05, .05]},
               {'text': 'a', 'posterior': [.81, .09, .05, .05]},
               {'text': 'b', 'posterior': [.1, .1, .7, .1]},
               {'text': 'b', 'posterior': [.1, .7, .1, .1]},
               {'text': 'a', 'posterior': [.25]*4}]
    pairs = make_pairs(records, 10, 8)
    assert pairs == make_pairs(records, 10, 8)
    assert len(pairs) == 5
    for i, j in pairs:
        assert i != j and records[i]['text'][-1] == records[j]['text'][-1]
        assert np.argmax(records[i]['posterior']) != np.argmax(records[j]['posterior'])
        assert min(max(records[i]['posterior']), max(records[j]['posterior'])) >= .6
    preservation = make_pairs(records, 10, 8, preserve=True)
    assert preservation == [[0, 2], [2, 0]]
    assert make_pairs([records[-1]], 5, 8) == []


def test_readout_control_uses_rank_three_category_contrasts():
    model, _, _ = model_and_inputs()
    q = readout_basis(model, torch.arange(10, 18))
    assert q.shape == (16, 3)
    torch.testing.assert_close(q.T@q, torch.eye(3), rtol=0, atol=5e-7)
    effective = model.unembedding.weight[10:18]*model.final_norm.weight
    categories = effective.reshape(4, 2, 16).mean(1)
    contrasts = categories-categories.mean(0)
    torch.testing.assert_close((contrasts@q)@q.T, contrasts, rtol=1e-5, atol=1e-7)
