"""Numerical mediation identity on a tiny Qwen architecture."""

import numpy as np

from test_workspace_mlx import model  # noqa: F401
from voynich.workspace.path3_backend import Path3Workspace


def test_full_final_attention_cut_restores_clean_logits(model):  # noqa: F811
    route = Path3Workspace.__new__(Path3Workspace)
    route.__dict__.update(model.__dict__)
    source = [2, 3, 5, 7, 11, 13, 17]
    donor = [2, 3, 8, 7, 11, 13, 17]
    clean, writes = route.clean_final_writes(source, start=1)
    own, _, own_capture = route.prefill_field(source, capture_layers=(0,))
    _, _, donor_capture = route.prefill_field(donor, capture_layers=(0,))
    np.testing.assert_allclose(clean, own, atol=1e-5, rtol=1e-4)
    field = np.zeros_like(own_capture[0])
    field[2] = donor_capture[0][2]-own_capture[0][2]
    changed, _, captured = route.prefill_cut(source, upstream_layer=0, upstream_field=field,
                                             clean_writes=writes, capture_writes=True)
    assert len(captured) == 3
    restored, _, _ = route.prefill_cut(source, upstream_layer=0, upstream_field=field,
                                       clean_writes=writes, cut_layers=(1, 2, 3))
    np.testing.assert_allclose(restored, clean, atol=1e-5, rtol=1e-4)
    assert np.isfinite(changed).all()
