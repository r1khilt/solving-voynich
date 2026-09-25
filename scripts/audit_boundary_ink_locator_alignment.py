"""Exploratory six-page audit of the published direct-ink locator geometry.

This diagnostic uses page pixels and source word-box *shapes*, not separator
labels or reported gaps. It compares the archived locator position with nearby
vertical shifts. The archived affine is an input, so this is not an independent
new-folio registration method or a blinded manuscript result.
"""

from __future__ import annotations

import hashlib
import json
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.build_boundary_geometry_train import read_boxes
from scripts.replay_boundary_ink_pilot import (
    BOX_DIR, PILOT, PILOT_SHA, ROOT, SCAN_DIR, SCAN_SHA,
    author_measurement_module, geometry, pair,
)


OFFSETS = range(-30, 36)
RNG_SEED = 20260925
N_NULL = 200


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_centered(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values -= values.mean()
    norm = np.linalg.norm(values)
    return values / norm if norm > 1e-8 else np.zeros_like(values)


def dark_stroke_map(image: np.ndarray) -> np.ndarray:
    """Local dark contrast, with green illustration pigment suppressed."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), 8).astype(np.float32)
    ink = np.clip(local - gray.astype(np.float32) - 7, 0, 40)
    red, green = image[:, :, 2].astype(np.int16), image[:, :, 1].astype(np.int16)
    ink[green > red + 5] = 0
    return ink


def profiles(boxes: list[dict], affine: dict, image_width: int) -> list[dict]:
    lines: dict[int, list[dict]] = {}
    for box in boxes:
        lines.setdefault(box["line"], []).append(box)
    result = []
    for line_id, members in lines.items():
        x0 = max(0, round(affine["tx"] + affine["sx"] * min(b["x"] for b in members)))
        x1 = min(image_width, round(affine["tx"] + affine["sx"] * max(
            b["x"] + b["w"] for b in members
        )))
        if x1 - x0 < 80:
            continue
        model = np.zeros(x1 - x0, dtype=np.float32)
        for box in members:
            left = max(0, round(affine["tx"] + affine["sx"] * box["x"]) - x0)
            right = min(len(model), round(
                affine["tx"] + affine["sx"] * (box["x"] + box["w"])
            ) - x0)
            model[left:right] = 1
        model = cv2.GaussianBlur(model[:, None], (1, 0), 2)[:, 0]
        center_y = round(affine["ty"] + affine["sy"] * np.median([
            box["y"] + box["h"] / 2 for box in members
        ]))
        result.append({"line": line_id, "x0": x0, "x1": x1,
                       "center_y": center_y, "model": unit_centered(model)})
    return result


def target_profiles(ink: np.ndarray, lines: list[dict]) -> list[np.ndarray]:
    targets = []
    for line in lines:
        by_offset = []
        for offset in OFFSETS:
            center = line["center_y"] + offset
            top, bottom = max(0, center - 5), min(ink.shape[0], center + 6)
            if bottom <= top:
                by_offset.append(np.zeros(line["x1"] - line["x0"], dtype=np.float32))
                continue
            target = ink[top:bottom, line["x0"]:line["x1"]].sum(axis=0)
            target = cv2.GaussianBlur(target[:, None], (1, 0), 2)[:, 0]
            by_offset.append(unit_centered(target))
        targets.append(np.stack(by_offset))
    return targets


def line_score_map(targets: list[np.ndarray], lines: list[dict]) -> np.ndarray:
    return np.stack([target @ line["model"] for line, target in zip(lines, targets, strict=True)])


def shifted_x_null(targets: list[np.ndarray], lines: list[dict], observed: float) -> dict:
    """Max-over-offset null after moving each word pattern within its line span."""
    rng = np.random.default_rng(RNG_SEED)
    nulls = []
    for _ in range(N_NULL):
        values = []
        for line, target in zip(lines, targets, strict=True):
            length = len(line["model"])
            shift = int(rng.integers(20, length - 19))
            values.append(target @ np.roll(line["model"], shift))
        nulls.append(float(np.max(np.mean(values, axis=0))))
    return {"observed_max_over_offsets": observed,
            "null_max_over_offsets_mean": float(np.mean(nulls)),
            "null_max_over_offsets_max": max(nulls),
            "null_at_least_observed": sum(v >= observed for v in nulls),
            "null_draws": N_NULL}


def gap_displacement(outcome: dict) -> dict:
    """Only after pixel-only offset selection, compare archived and shifted gaps."""
    if sha(PILOT) != PILOT_SHA:
        raise AssertionError("Archived pilot changed")
    author = author_measurement_module()
    by_folio: dict[str, list[dict]] = {}
    for row in csv.DictReader(PILOT.open()):
        by_folio.setdefault(row["folio"], []).append(row)
    result = {}
    paired = {label: {"original": [], "shifted": []} for label in (".", ",")}
    for folio, rows in sorted(by_folio.items()):
        boxes = read_boxes(BOX_DIR / f"{folio}.js")
        a = outcome[folio]["affine_from_same_pilot_metadata"]
        offset = outcome[folio]["best_offset_px"]
        counts = {key: 0 for key in ("rows", "finite_archived", "finite_shifted",
                                   "finite_both", "same_gap_when_both_finite",
                                   "same_algorithmic_qc")}
        for row in rows:
            left, right = pair(row, boxes)
            geo = geometry(left, right, a["sx"], a["tx"], a["sy"], a["ty"] + offset)
            shifted = author.measure(SimpleNamespace(blind_id=row["blind_id"],
                                                      folio=folio, **geo), SCAN_DIR)
            old = float(row["gap_px"]) if row["gap_px"] else math.nan
            new = float(shifted["gap_px"])
            counts["rows"] += 1
            counts["finite_archived"] += math.isfinite(old)
            counts["finite_shifted"] += math.isfinite(new)
            counts["finite_both"] += math.isfinite(old) and math.isfinite(new)
            counts["same_gap_when_both_finite"] += (
                math.isfinite(old) and math.isfinite(new) and old == new
            )
            counts["same_algorithmic_qc"] += shifted["qc"] == row["qc"]
            if (row["include"] == "True" and math.isfinite(old) and
                    math.isfinite(new) and row["label"] in paired):
                paired[row["label"]]["original"].append(old)
                paired[row["label"]]["shifted"].append(new)
        result[folio] = counts
    return {"by_folio": result, "original_visual_qc_paired_rows_only": {
        label: {"rows": len(values["original"]),
                "mean_original_gap_px": float(np.mean(values["original"])),
                "mean_shifted_gap_px": float(np.mean(values["shifted"]))}
        for label, values in paired.items()},
        "warning": "Original blind visual-QC decisions are reused only for descriptive comparison and have NOT been repeated on shifted crops. The f7v offset is not x-null qualified. No corrected label effect is established."}


def run() -> dict:
    pilot_path = ROOT / "results/BOUNDARY-CHANNEL-0002/pilot_replay.json"
    pilot = json.loads(pilot_path.read_text())
    outcome = {}
    for folio, record in sorted(pilot["folio_results"].items()):
        scan = SCAN_DIR / f"{folio}.jpg"
        boxes_path = BOX_DIR / f"{folio}.js"
        if sha(scan) != SCAN_SHA[folio] or sha(boxes_path) != record["source_box_sha256"]:
            raise AssertionError(f"Pinned image or boxes changed: {folio}")
        image = cv2.imread(str(scan))
        ink = dark_stroke_map(image)
        lines = profiles(read_boxes(boxes_path), record["affine"], image.shape[1])
        targets = target_profiles(ink, lines)
        line_scores = line_score_map(targets, lines)
        scores = line_scores.mean(axis=0)
        best_index = int(np.argmax(scores))
        best_offset = OFFSETS[best_index]
        zero_index = 0 - OFFSETS.start
        outcome[folio] = {
            "scan_sha256": sha(scan), "box_sha256": sha(boxes_path),
            "affine_from_same_pilot_metadata": record["affine"],
            "source_lines_used": len(lines), "offset_search_px": [OFFSETS.start, OFFSETS.stop - 1],
            "archive_offset_score": float(scores[zero_index]),
            "best_offset_px": best_offset, "best_offset_score": float(scores[best_index]),
            "score_gain": float(scores[best_index] - scores[zero_index]),
            "line_scores_at_archive_offset": [float(v) for v in line_scores[:, zero_index]],
            "line_scores_at_best_offset": [float(v) for v in line_scores[:, best_index]],
            "x_shift_null_max_offset": shifted_x_null(targets, lines, float(scores[best_index])),
        }
    result = {"status": "exploratory_locator_alignment_diagnostic",
              "method": "mean linewise centered-cosine match of Gaussian-smoothed source word-box x occupancy to local dark-stroke x profile; 11-pixel row band; offset scan -30..35 px",
              "seed": RNG_SEED, "folios": outcome,
              "archived_results_sha256": PILOT_SHA,
              "gap_displacement_after_pixel_only_offset_selection": gap_displacement(outcome),
              "limit": "Same-pilot affine is an input and all offsets were inspected after seeing pilot results. Correlation and overlays are diagnostics, not proof of intended-word identity or a corrected direct-ink effect."}
    output = ROOT / "results/BOUNDARY-CHANNEL-0003/locator_alignment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({folio: {key: item[key] for key in (
        "source_lines_used", "best_offset_px", "archive_offset_score", "best_offset_score",
            "score_gain", "x_shift_null_max_offset")}
        for folio, item in outcome.items()}, indent=2))
    return result


if __name__ == "__main__":
    run()
