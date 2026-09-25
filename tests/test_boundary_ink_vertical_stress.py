"""Controls for vertical mixing in the direct-ink gap estimator."""

import numpy as np

from scripts.boundary_ink_vertical_stress import vertical_metrics


def test_projection_can_create_a_gap_from_ink_on_different_rows():
    mask = np.zeros((8, 30), dtype=np.uint8)
    mask[2, 10] = 1
    mask[6, 20] = 1
    projected, same_row, supporting_rows = vertical_metrics(mask, 15, 0, 8)
    assert projected == 9
    assert same_row is None
    assert supporting_rows == 0


def test_same_row_gap_requires_both_strokes_on_same_scanline():
    mask = np.zeros((8, 30), dtype=np.uint8)
    mask[4, 10] = 1
    mask[4, 20] = 1
    projected, same_row, supporting_rows = vertical_metrics(mask, 15, 0, 8)
    assert projected == 9
    assert same_row == 9
    assert supporting_rows == 1
