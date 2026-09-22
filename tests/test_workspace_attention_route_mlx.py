"""Live numerical checks on a tiny random Qwen3, with no downloaded weights."""

import numpy as np
import pytest

from test_workspace_mlx import model  # noqa: F401
from voynich.workspace.attention_route_backend import AttentionRouteWorkspace


def _route(toy):
    route = AttentionRouteWorkspace.__new__(AttentionRouteWorkspace)
    route.__dict__.update(toy.__dict__)
    return route


def test_head_capture_matches_native_attention_and_cached_identity(model):  # noqa: F811
    route = _route(model)
    ids = [2, 3, 5, 7, 11, 13, 17]
    clean, captures = route.capture_heads(ids, (0, 1, 2, 3))
    reference, _, _ = model.prefill(ids)
    np.testing.assert_allclose(clean, reference, atol=1e-5, rtol=1e-4)
    for layer, record in captures.items():
        assert record['native_error'] < 1e-5, layer
        rebuilt = route.heads_from_qkv(record['q'], record['k'], record['v'])
        np.testing.assert_allclose(rebuilt, record['heads'], atol=1e-6)
        zero, _ = route.prefill_attention_delta(ids, layer=layer, delta=np.zeros(model.width, dtype=np.float32))
        np.testing.assert_allclose(zero, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('layer', [0, 2])
def test_head_delta_matches_fixed_site_recompute_after_generation(model, layer):  # noqa: F811
    route = _route(model)
    source = [2, 3, 5, 7, 11, 13, 17]
    donor = [2, 3, 8, 7, 11, 13, 17]
    _, own = route.capture_heads(source, (layer,))
    _, other = route.capture_heads(donor, (layer,))
    delta = route.head_delta(layer, own[layer]['heads'], other[layer]['heads'], (0,))
    assert np.linalg.norm(delta) > 0
    logits, cache = route.prefill_attention_delta(source, layer=layer, delta=delta)
    mx = model.mx
    for step in range(3):
        h = model.model.model.embed_tokens(mx.array(source)[None]).astype(mx.float32)
        for i, block in enumerate(model.layers):
            if i == layer:
                write = block.self_attn(block.input_layernorm(h), mask='causal')
                middle = h+write
                middle = middle.at[0, 6].add(mx.array(delta))
                h = middle+block.mlp(block.post_attention_layernorm(middle))
            else:
                h = block(h, mask='causal')
        expected = model.model.lm_head(model.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(expected)
        np.testing.assert_allclose(logits, np.array(expected), atol=1e-5, rtol=1e-4)
        token = 20+step
        source.append(token)
        logits = route.cached_step(token, cache)
