import pytest
import torch

from voynich.source_action_cache import _PathCache, cached_trace_logits, propose_cached
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal


@pytest.mark.parametrize('bindings', [True, False])
def test_all_cached_prefix_logits_match_independent_full_decoder(bindings):
    torch.manual_seed(9261)
    model = SourceActionProposal(SourceActionConfig(rows=3, glyphs=2, width=16, heads=2,
        encoder_layers=2, decoder_layers=3, max_glyphs=16, max_records=2, binding_input=bindings)).double()
    for key in ((0, 0, 5), (0, 2, 1), (3, 4, 5), (5, 0, 2)):
        texts = ((0, 1, 2, 0, 2), (2, 0))
        pool = ReadingEnvironment(((0,),), rows=3, glyphs=2).pool
        cipher = tuple(tuple(g for a in text for g in pool[key[a]]) for text in texts)
        env = ReadingEnvironment(cipher, rows=3, glyphs=2)
        trace = env.teaching_trace(texts, key)
        reference = model(model.pack([env], [trace]))[0]
        cached = cached_trace_logits(model, env, trace)
        torch.testing.assert_close(reference, cached, rtol=0, atol=2e-12)
        assert model.training
        # Same random draws and temperature; density agrees after legal masking.
        for seed in (921, 927):
            a, b = model.propose(env, seed=seed, temperature=.7), propose_cached(model, env, seed=seed, temperature=.7)
            assert a['state'] == b['state'] and a['actions'] == b['actions'] and a['status'] == b['status']
            assert abs(a['path_log_probability']-b['path_log_probability']) < 2e-12


def test_cache_cannot_be_training_or_skip_query_or_reuse_wrong_branch():
    model = SourceActionProposal(SourceActionConfig(rows=2, glyphs=2, width=16, heads=2,
        encoder_layers=1, decoder_layers=1, max_glyphs=8)).double()
    env = ReadingEnvironment(((0, 1, 0),), rows=2, glyphs=2)
    with pytest.raises(ValueError):
        _PathCache(model, env)
    model.eval()
    with torch.no_grad():
        cache = _PathCache(model, env)
        with pytest.raises(ValueError, match='query'):
            cache.advance(0)
        a, b = cache.scores(), cache.scores()
        assert a is b
        cache.advance(0)
        cache.scores()
        with pytest.raises(ValueError, match='legal'):
            cache.advance(0)
        model.decoder[0].norm_first = False
        with pytest.raises(ValueError, match='architecture'):
            _PathCache(model, env)


def test_cached_failure_is_preserved_without_refill():
    model = SourceActionProposal(SourceActionConfig(rows=1, glyphs=2, width=16, heads=2,
        encoder_layers=1, decoder_layers=1, max_glyphs=8)).double()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([1000., -1000.]))
    env = ReadingEnvironment(((0, 1),), rows=1, glyphs=2)
    result = propose_cached(model, env, seed=0)
    assert result['status'] == 'dead_end' and result['actions'] == (0,)
    assert result['state'].offsets == (1,) and model.training
