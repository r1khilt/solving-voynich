"""Opt-in live architecture checks; no downloaded or pretrained model required."""

import os

import numpy as np
import pytest

if os.environ.get('VOYNICH_MLX_TEST') != '1':
    pytest.skip('Opt in to live Metal tests with VOYNICH_MLX_TEST=1', allow_module_level=True)

os.environ['MLX_ENABLE_TF32'] = '0'
import mlx.core as mx  # noqa: E402
from mlx_lm.models.qwen3 import Model, ModelArgs  # noqa: E402

from voynich.workspace.causal_backend import CausalWorkspace  # noqa: E402


@pytest.fixture(scope='module')
def model():
    mx.random.seed(51029)
    args = ModelArgs(model_type='qwen3', hidden_size=16, num_hidden_layers=4,
                     intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
                     vocab_size=32, num_key_value_heads=1, max_position_embeddings=512,
                     rope_theta=1000000, head_dim=8, tie_word_embeddings=False)
    wrapper = CausalWorkspace.__new__(CausalWorkspace)
    wrapper.model, wrapper.mx = Model(args), mx
    wrapper.model.eval()
    wrapper.model.freeze()
    wrapper.layers = wrapper.model.model.layers
    wrapper.width, wrapper.n_layers = 16, 4
    wrapper.config = {'eos_token_id': 31}
    return wrapper


@pytest.mark.parametrize('layer', [0, 1, 2, 3])
@pytest.mark.parametrize('position', [3, 11])
def test_real_qwen_cache_matches_fixed_prefix_intervention(model, layer, position):
    ids = [2, 5, 1, 9, 7, 3, 8, 4, 12, 11, 6, 13]
    delta = np.random.default_rng(90).normal(size=16).astype(np.float32)*.15
    patches = [{'layer': layer, 'position': position, 'delta': delta}]
    cached, cache, captures = model.prefill(ids, patches=patches, capture_layers=(0, 2, 3))
    for step in range(4):
        reference, hidden = model.forward(ids, patches=patches, capture_layers=(0, 2, 3))
        np.testing.assert_allclose(cached, reference[0], atol=1e-5, rtol=1e-4)
        assert cached.argmax() == reference[0].argmax()
        if step == 0:
            for captured_layer in captures:
                np.testing.assert_allclose(captures[captured_layer], hidden[captured_layer], atol=1e-6)
        token = 14+step
        ids.append(token)
        cached = model.cached_step(token, cache)


def test_real_qwen_neuron_trace_preserves_forward_and_gate_product(model):
    ids = [2, 3, 5, 7, 11, 13, 17]
    delta = np.ones(16, dtype=np.float32)*.1
    patches = [{'layer': 1, 'position': 6, 'delta': delta}]
    original, captures = model.forward(ids, patches=patches, capture_layers=tuple(range(4)))
    logits, trace = model.neuron_trace(ids, patches=patches)
    np.testing.assert_allclose(logits, original[0], atol=1e-6)
    np.testing.assert_allclose(trace['residual'], np.stack(list(captures.values())), atol=1e-6)
    np.testing.assert_allclose(trace['neurons'], trace['gate_activation']*trace['up_value'], atol=1e-6)
    for layer in range(4):
        reconstructed = model.layers[layer].mlp.down_proj(mx.array(trace['neurons'][layer]))
        mx.eval(reconstructed)
        np.testing.assert_allclose(np.array(reconstructed), trace['mlp_write'][layer], atol=1e-6)


def test_real_qwen_final_prefix_edit_does_not_change_future_hidden_state(model):
    # A final-block edit at an earlier position has no route to a later position
    # unless the generated token itself changes. Here continuation tokens are fixed.
    ids = [2, 3, 5, 7, 11, 13, 17]
    clean, _ = model.forward(ids)
    patched, _ = model.forward(ids, patches=[{'layer': 3, 'position': 4, 'delta': np.ones(16, dtype=np.float32)}])
    np.testing.assert_array_equal(clean, patched)


def test_real_qwen_selected_row_transport_matches_finite_differences(model):
    ids = [2, 3, 5, 7, 11, 13, 17]
    rows = np.eye(16, dtype=np.float32)[:3]
    transported = model.jacobian_rows(ids, rows, (0, 1, 2), skip_first=2)
    for index, layer in enumerate((0, 1, 2)):
        vector = transported[index, 0]/np.linalg.norm(transported[index, 0])
        epsilon = .01
        difference = (model.transport_scalar(ids, rows[0], layer, vector*epsilon, skip_first=2)
                      - model.transport_scalar(ids, rows[0], layer, -vector*epsilon, skip_first=2))/(2*epsilon)
        np.testing.assert_allclose(difference, transported[index, 0]@vector, rtol=2e-3, atol=1e-4)


def test_invalid_cached_edits_rejected(model):
    with pytest.raises(ValueError, match='Invalid cached'):
        model.prefill([1, 2], patches=[{'layer': 0, 'position': 1, 'delta': np.full(16, np.nan)}])
    with pytest.raises(ValueError, match='Invalid cached'):
        model.prefill([1, 2], patches=[{'layer': 0, 'position': 1.5, 'delta': np.zeros(16)}])
