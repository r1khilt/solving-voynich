"""Tiny-Qwen native recomputation for a factorial attention-write subset grid."""

import numpy as np

from test_workspace_mlx import model  # noqa: F401
from voynich.workspace.path3_backend import Path3Workspace


def test_subset_recomputation_and_all_later_restoration(model):  # noqa: F811
    route = Path3Workspace.__new__(Path3Workspace)
    route.__dict__.update(model.__dict__)
    source = [2, 3, 5, 7, 11, 13, 17]
    donor = [2, 3, 8, 7, 11, 13, 17]
    clean_logits, clean_writes = route.clean_final_writes(source, start=1)
    _, _, source_fields = route.prefill_field(source, capture_layers=(0,))
    _, _, donor_fields = route.prefill_field(donor, capture_layers=(0,))
    field = np.zeros_like(source_fields[0])
    field[2] = donor_fields[0][2]-source_fields[0][2]
    uncut_logits, _, edited_writes = route.prefill_cut(
        source, upstream_layer=0, upstream_field=field,
        clean_writes=clean_writes, capture_writes=True)
    assert set(edited_writes) == {1, 2, 3}
    grid = {}
    for mask in range(8):
        layers = tuple(1+bit for bit in range(3) if mask & (1 << bit))
        logits, _, _ = route.prefill_cut(source, upstream_layer=0,
                                         upstream_field=field, clean_writes=clean_writes,
                                         cut_layers=layers)
        grid[mask] = logits
        assert np.isfinite(logits).all()
    np.testing.assert_allclose(grid[0], uncut_logits, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(grid[7], clean_logits, atol=1e-5, rtol=1e-4)
    assert not np.allclose(grid[0], grid[7])
