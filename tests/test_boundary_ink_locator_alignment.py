"""Known-geometry control for the exploratory scan-to-box line-pattern score."""

import numpy as np

from scripts.audit_boundary_ink_locator_alignment import (
    OFFSETS,
    line_score_map,
    profiles,
    target_profiles,
)


def test_line_pattern_score_recovers_injected_vertical_shift() -> None:
    boxes = [
        {"line": 0, "x": 10, "y": 20, "w": 28, "h": 8},
        {"line": 0, "x": 52, "y": 20, "w": 49, "h": 8},
        {"line": 1, "x": 15, "y": 58, "w": 58, "h": 8},
        {"line": 1, "x": 89, "y": 58, "w": 30, "h": 8},
    ]
    affine = {"sx": 1, "tx": 0, "sy": 1, "ty": 0}
    ink = np.zeros((120, 140), dtype=np.float32)
    injected_shift = 12
    for box in boxes:
        center = box["y"] + box["h"] // 2 + injected_shift
        ink[center - 2:center + 3, box["x"]:box["x"] + box["w"]] = 1
    lines = profiles(boxes, affine, ink.shape[1])
    scores = line_score_map(target_profiles(ink, lines), lines).mean(axis=0)
    # The 11-pixel integration band has about five pixels of offset ambiguity.
    assert abs(OFFSETS.start + int(np.argmax(scores)) - injected_shift) <= 5
    assert scores[injected_shift - OFFSETS.start] > scores[-OFFSETS.start]
