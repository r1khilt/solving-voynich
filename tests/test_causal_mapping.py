import numpy as np
import pytest
import torch

from voynich.causal_mapping import (
    HORIZONS, WIDTH, MappingSite, confirmation_decision, evaluate_mapping,
    head_basis, mapping_patch, mapping_sites, select_site, synthetic_pairs,
)
from voynich.model import ModelConfig, VoynichTransformer
from voynich.synthetic import filter_sequence, generate_page
from voynich.tokenizer import EVATokenizer


def tiny_model():
    tokenizer = EVATokenizer.fit(['abcdefgh'])
    model = VoynichTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16,
                                          n_layers=2, n_heads=2, d_ff=32,
                                          context_length=256, dropout=0.)).eval()
    return model, tokenizer


@pytest.mark.parametrize('kind,head', [('head', 1), ('mlp', None), ('residual', None)])
def test_patch_preserves_future_and_unselected_coordinates(kind, head):
    shape = (2, 135, 2, 16) if kind == 'head' else (2, 135, 16)
    base, donor = torch.randn(shape), torch.randn(shape)
    site = MappingSite(0, kind, 'recent8', head)
    patched = mapping_patch(donor, site)(base)
    assert torch.equal(patched[:, 128:], base[:, 128:])
    assert torch.equal(patched[:, :120], base[:, :120])
    if head is not None:
        assert torch.equal(patched[:, :, 0], base[:, :, 0])
        assert torch.equal(patched[:, 120:128, head], donor[:, 120:128, head])
    else:
        assert torch.equal(patched[:, 120:128], donor[:, 120:128])
    assert torch.equal(mapping_patch(base, site)(base), base)


def test_random_control_matches_each_position_norm_and_head_subspace():
    model, _ = tiny_model()
    site = MappingSite(0, 'head', 'all128', 0)
    basis = head_basis(model, site)
    base, donor = torch.randn(2, 135, 2, 16), torch.randn(2, 135, 2, 16)
    patched = mapping_patch(donor, site, mode='random', seed=11, basis=basis)(base)
    changed = patched[:, :128, 0]-base[:, :128, 0]
    desired = donor[:, :128, 0]-base[:, :128, 0]
    torch.testing.assert_close(changed.norm(dim=-1), desired.norm(dim=-1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close((changed@basis)@basis.T, changed, atol=3e-6, rtol=1e-5)
    assert not torch.allclose(changed, desired)


def test_fresh_pair_oracle_tracks_shared_continuation():
    rng = np.random.default_rng(91)
    texts = [generate_page(rng, WIDTH)[0] for _ in range(512)]
    pairs = synthetic_pairs(texts, 12, 92)
    for row in pairs:
        recipient, donor, wrong = [filter_sequence(row[key])[0][-1] for key in ('recipient', 'donor', 'wrong')]
        assert recipient.argmax() != donor.argmax()
        assert recipient.argmax() == wrong.argmax()
        assert min(recipient.max(), donor.max(), wrong.max()) >= .6
        assert row['recipient'][-1] == row['donor'][-1] == row['wrong'][-1]
        expected = filter_sequence(row['donor']+row['suffix'])[2][np.array(HORIZONS)+WIDTH-2]
        np.testing.assert_allclose(row['oracle'], expected)
        assert len(row['suffix']) == 7


def test_donor_pair_selection_is_independent_of_generated_suffix(monkeypatch):
    rng = np.random.default_rng(91)
    texts = [generate_page(rng, WIDTH)[0] for _ in range(512)]
    ordinary = synthetic_pairs(texts, 12, 92)
    monkeypatch.setattr('voynich.causal_mapping.continuation_from_belief', lambda *args: 'aaaaaaa')
    alternate = synthetic_pairs(texts, 12, 92)
    assert [row['pool_indexes'] for row in ordinary] == [row['pool_indexes'] for row in alternate]


def test_full_embedding_restore_and_late_output_cannot_change_future():
    model, tokenizer = tiny_model()
    rows = synthetic_pairs([generate_page(np.random.default_rng(seed), WIDTH)[0] for seed in range(256)], 3, 77)
    sites = [MappingSite(0, 'head', 'recent8', 0), MappingSite(1, 'residual', 'final')]
    report, arrays = evaluate_mapping(model, rows, tokenizer, sites, 'cpu', selected=sites[0])
    assert max(report['controls'].values()) < 1e-5
    assert report['controls']['late_prefix_future_max_logit_error'] == 0
    np.testing.assert_allclose(arrays['recipient'][:, 1:], arrays['late_readout'][:, 1:], atol=0, rtol=0)
    assert arrays[sites[0].name+'/donor'].shape == (3, 4, 2)


def test_donor_prefix_cache_cannot_see_shared_future_observations():
    model, tokenizer = tiny_model()
    prefix = tokenizer.encode('abcdefgh'*16, add_bos=False, add_eos=False)
    first = torch.tensor([prefix+tokenizer.encode('aaaaaaa', add_bos=False, add_eos=False)])
    second = torch.tensor([prefix+tokenizer.encode('hhhhhhh', add_bos=False, add_eos=False)])
    hooks = sorted({site.hook for site in mapping_sites(model)})
    with torch.no_grad():
        original = model(first, cache_names=hooks)
        alternate = model(second, cache_names=hooks)
    for hook in hooks:
        torch.testing.assert_close(original.cache[hook][:, :WIDTH], alternate.cache[hook][:, :WIDTH], atol=0, rtol=0)


def test_site_selection_ignores_next_answer_and_late_sites():
    model, _ = tiny_model()
    sites = mapping_sites(model)
    scores = {}
    wanted = next(site for site in sites if site.layer == 0 and site.kind == 'head' and site.region == 'final')
    for site in sites:
        scores[site.name+'/donor'] = {'equal_group_mean': [[-1000, 0], [5, 0], [5, 0], [-1000, 0]]}
    scores[wanted.name+'/donor']['equal_group_mean'] = [[1000, 0], [1, 0], [1, 0], [1000, 0]]
    late = next(site for site in sites if site.layer == 1)
    scores[late.name+'/donor']['equal_group_mean'] = [[-1000, 0]]*4
    assert select_site({'conditions': scores}, sites) == wanted


def test_confirmation_requires_two_future_horizons_and_both_controls():
    site = MappingSite(0, 'head', 'all128', 0)
    arrays = {'recipient': np.ones((4, 4, 2)), site.name+'/donor': np.full((4, 4, 2), .8),
              site.name+'/wrong': np.ones((4, 4, 2))}
    for i in range(4):
        arrays[site.name+f'/random{i}'] = np.ones((4, 4, 2))
    rows = [{'group': str(i//2)} for i in range(4)]
    assert confirmation_decision({}, site, rows, arrays)['meets_registered_practical_threshold']
    arrays[site.name+'/wrong'][:, 2, 0] = .8
    assert not confirmation_decision({}, site, rows, arrays)['meets_registered_practical_threshold']
