"""Synthetic geometry controls for image-only box-to-scan registration."""

import cv2
import numpy as np
import pytest

from scripts.register_boundary_ink_pages import best_match, source_mask, target_map


def test_template_matching_recovers_known_translation() -> None:
    model = np.zeros((43, 70), dtype=np.float32)
    cv2.rectangle(model, (3, 5), (20, 14), 1, -1)
    cv2.rectangle(model, (35, 26), (65, 35), 1, -1)
    target = np.zeros((100, 140), dtype=np.float32)
    target[17:60, 29:99] = model
    score, tx, ty = best_match(target, model)
    assert (tx, ty) == (29, 17)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_source_layout_uses_box_positions_and_whitespace() -> None:
    boxes = [{"x": 4, "y": 3, "w": 8, "h": 5},
             {"x": 27, "y": 20, "w": 9, "h": 5}]
    mask = source_mask(boxes)
    assert mask.shape == (26, 37)
    assert mask[5, 7] == 1
    assert mask[12, 20] == 0
    assert mask[22, 30] == 1


def test_blank_scan_is_rejected_before_matching() -> None:
    with pytest.raises(ValueError, match="Blank or low-information"):
        target_map(np.full((900, 700, 3), 255, dtype=np.uint8))
