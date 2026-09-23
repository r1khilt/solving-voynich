"""Numerical head mediation on a tiny random original Qwen architecture."""

import numpy as np

from test_workspace_mlx import model  # noqa: F401
from voynich.workspace.path8_backend import Path8Workspace


def test_head_capture_and_all_head_reversion_match_native(model):  # noqa: F811
    route = Path8Workspace.__new__(Path8Workspace)
    route.__dict__.update(model.__dict__)
    source = [2, 3, 5, 7, 11, 13, 17]
    donor = [2, 3, 8, 7, 11, 13, 17]
    own_logits, _, own_fields = route.prefill_field(source, capture_layers=(0,))
    _, _, donor_fields = route.prefill_field(donor, capture_layers=(0,))
    field = np.zeros_like(own_fields[0])
    field[2] = donor_fields[0][2]-own_fields[0][2]
    clean_logits, clean_heads = route.final_heads_after_field(
        source, upstream_layer=0, field=np.zeros_like(field), capture_layers=(1, 2, 3))
    edited_logits, edited_heads = route.final_heads_after_field(
        source, upstream_layer=0, field=field, capture_layers=(1, 2, 3))
    np.testing.assert_allclose(clean_logits, own_logits, atol=1e-5, rtol=1e-4)
    native_edited, _, _ = route.prefill_field(source, fields={0: field})
    np.testing.assert_allclose(edited_logits, native_edited, atol=1e-5, rtol=1e-4)
    for layer in (1, 2, 3):
        assert clean_heads[layer]['native_error'] < 1e-5
        assert edited_heads[layer]['native_error'] < 1e-5
    layer = 2
    delta = route.head_delta(layer, edited_heads[layer]['heads'],
                             clean_heads[layer]['heads'], (0, 1))
    expected_delta = clean_heads[layer]['write']-edited_heads[layer]['write']
    np.testing.assert_allclose(delta, expected_delta, atol=1e-5, rtol=1e-4)
    clean_write_logits, clean_writes = route.clean_final_writes(source, start=1)
    np.testing.assert_allclose(clean_write_logits, own_logits, atol=1e-5, rtol=1e-4)
    additive_logits, _, _ = route.prefill_cut(
        source, upstream_layer=0, upstream_field=field, clean_writes=clean_writes,
        cut_layers=(layer,), random_deltas={layer: delta})
    full_block_logits, _, _ = route.prefill_cut(
        source, upstream_layer=0, upstream_field=field, clean_writes=clean_writes,
        cut_layers=(layer,))
    np.testing.assert_allclose(additive_logits, full_block_logits, atol=1e-5, rtol=1e-4)
    all_cut, _, _ = route.prefill_cut(
        source, upstream_layer=0, upstream_field=field, clean_writes=clean_writes,
        cut_layers=(1, 2, 3))
    np.testing.assert_allclose(all_cut, own_logits, atol=1e-5, rtol=1e-4)
