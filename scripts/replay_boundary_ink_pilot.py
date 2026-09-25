"""Reconstruct the published six-folio direct-ink pilot from public scans.

The archived blind locator manifest is not redistributed. Its affine geometry
is reconstructed from the publication CSV's locator midpoints, crop heights,
and grayscale thresholds. Labels and recorded ink gaps are never used to fit
registration. The pinned author measurement function is then run on Yale's
700-pixel IIIF derivatives and compared to the archived ink results.

This is source-code/pixel reproducibility, NOT an independent boundary study.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from scripts.build_boundary_geometry_train import read_boxes


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data/raw/external/voynich-units"
SCAN_DIR = ROOT / "data/raw/boundary_ink_0002/yale700"
PILOT = ARCHIVE / "data/direct_pixel/results_unblinded.csv"
AUTHOR_CODE = ARCHIVE / "analysis/direct_pixel/measure_direct_pixels.py"
BOX_DIR = ARCHIVE / "data/voynich-units/morphometry_voynichese/voynichese_boxes"
PILOT_SHA = "3c15627dc3ddfca4f331a1fed1e1a290a9b8412ebf9247fe4c9b2778412aedee"
CODE_SHA = "960ac72b7d174b639f0b3a36e5f83c664c492c3e0704f52d6b1b0768ff6e9f05"
SCAN_IDS = {"f7v": "1006089", "f14v": "1006101", "f23r": "1006118",
            "f39v": "1006151", "f42v": "1006157", "f56v": "1006185"}
SCAN_SHA = {
    "f14v": "a40c2025c835bde02423e3ba9766aa312b38aa6fb922a4512f4bcebd944aea65",
    "f23r": "3e78283b30ea1a92e29954c44e16ec453c01d7e4343775f281120c67bc7a1d48",
    "f39v": "5d76fa0112468c475ac3af4c699f9674a594091123e13f119e5a4ccc951cbd7a",
    "f42v": "6951b633415253940a073a7cd2bf6323523509261aa600f398798301d55436b4",
    "f56v": "ab14cbaa90dbb0e65772c2d2f456884cdb22beea5dc1ea41034d04d0530d1cf4",
    "f7v": "2188833e0996b3163b5e87ad713060d810db03bcd07aa4cd7bf4733fb6bd23f2",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(row: dict, boxes: list[dict]) -> tuple[dict, dict]:
    index = int(row["vi"])
    left, right = boxes[index], boxes[index + 1]
    if (left["word"], right["word"]) != (row["left_word"], row["right_word"]):
        raise AssertionError("Pilot-to-coordinate pair changed")
    return left, right


def geometry(left: dict, right: dict, sx: float, tx: float, sy: float, ty: int) -> dict:
    return {
        "lx0": tx + sx * left["x"], "lx1": tx + sx * (left["x"] + left["w"]),
        "rx0": tx + sx * right["x"], "rx1": tx + sx * (right["x"] + right["w"]),
        "ly0": ty + sy * left["y"], "ly1": ty + sy * (left["y"] + left["h"]),
        "ry0": ty + sy * right["y"], "ry1": ty + sy * (right["y"] + right["h"]),
    }


def crop_bounds(geo: dict, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x0 = max(0, math.floor(min(geo["lx0"], geo["rx0"]) - 5))
    x1 = min(image_width, math.ceil(max(geo["lx1"], geo["rx1"]) + 5))
    y0 = max(0, math.floor(min(geo["ly0"], geo["ry0"]) - 3))
    y1 = min(image_height, math.ceil(max(geo["ly1"], geo["ry1"]) + 3))
    return x0, x1, y0, y1


def fit_horizontal(rows: list[dict], boxes: list[dict]) -> tuple[float, int, float]:
    source, target = [], []
    for row in rows:
        left, right = pair(row, boxes)
        source.append((left["x"] + left["w"] + right["x"]) / 2)
        target.append(float(row["locator_mid"]))
    design = np.column_stack([source, np.ones(len(source))])
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    sx, tx = round(float(slope), 2), round(float(intercept))
    maximum = float(np.max(np.abs(design @ np.array([sx, tx]) - target)))
    if maximum > 1e-8:
        raise AssertionError("Horizontal locator transform is not exact")
    return sx, tx, maximum


def fit_vertical_scale(rows: list[dict], boxes: list[dict]) -> tuple[float, int]:
    """Use published crop heights only; an integer shift cancels out."""
    candidates = []
    for step in range(60, 111):
        sy = step / 100
        matches = 0
        for row in rows:
            left, right = pair(row, boxes)
            geo = geometry(left, right, 1, 0, sy, 0)
            predicted = math.ceil(max(geo["ly1"], geo["ry1"]) + 3) - math.floor(
                min(geo["ly0"], geo["ry0"]) - 3
            )
            matches += predicted == int(row["crop_h"])
        candidates.append((matches, sy))
    best = max(count for count, _ in candidates)
    winners = [scale for count, scale in candidates if count == best]
    if best != len(rows) or len(winners) != 1:
        raise AssertionError(f"Vertical scale not uniquely reconstructed: {best}/{len(rows)} {winners}")
    return winners[0], best


def otsu_threshold(crop: np.ndarray) -> int:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(80, min(200, min(170, max(105, int(otsu)))))


def fit_vertical_shift(rows: list[dict], boxes: list[dict], image: np.ndarray,
                       sx: float, tx: int, sy: float) -> tuple[int, int, int]:
    """Match threshold metadata on 18 evenly spaced blind IDs; never use gaps."""
    selected = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, min(18, len(rows)), dtype=int)]
    height, width = image.shape[:2]
    candidates = []
    for ty in range(151):
        exact = 0
        absolute_error = 0
        for row in selected:
            left, right = pair(row, boxes)
            geo = geometry(left, right, sx, tx, sy, ty)
            x0, x1, y0, y1 = crop_bounds(geo, width, height)
            if x1 <= x0 or y1 <= y0:
                raise AssertionError("Empty pilot crop")
            threshold = otsu_threshold(image[y0:y1, x0:x1])
            expected = int(row["threshold"])
            exact += threshold == expected
            absolute_error += abs(threshold - expected)
        candidates.append((exact, -absolute_error, -ty))
    best = max(candidates)
    return -best[2], best[0], -best[1]


def author_measurement_module():
    if sha(AUTHOR_CODE) != CODE_SHA:
        raise AssertionError("Published measurement code changed")
    spec = importlib.util.spec_from_file_location("pinned_direct_pixel", AUTHOR_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load published measurement code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_number(saved: str, replayed: float) -> bool:
    return (not saved and np.isnan(replayed)) or (bool(saved) and abs(float(saved) - replayed) < 1e-7)


def run() -> dict:
    if sha(PILOT) != PILOT_SHA:
        raise AssertionError("Archived blind-pilot result changed")
    rows = list(csv.DictReader(PILOT.open()))
    if len(rows) != 300:
        raise AssertionError("Pilot length changed")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["folio"]].append(row)
    if set(grouped) != set(SCAN_IDS):
        raise AssertionError("Pilot folios changed")
    author = author_measurement_module()
    summary = {}
    for folio, pilot_rows in sorted(grouped.items()):
        scan_path = SCAN_DIR / f"{folio}.jpg"
        if sha(scan_path) != SCAN_SHA[folio]:
            raise AssertionError(f"Yale scan changed: {folio}")
        image = np.asarray(Image.open(scan_path).convert("RGB"))
        boxes = read_boxes(BOX_DIR / f"{folio}.js")
        sx, tx, x_error = fit_horizontal(pilot_rows, boxes)
        sy, height_matches = fit_vertical_scale(pilot_rows, boxes)
        ty, fitted_thresholds, fit_error = fit_vertical_shift(
            pilot_rows, boxes, image, sx, tx, sy
        )
        counts = defaultdict(int)
        for row in pilot_rows:
            left, right = pair(row, boxes)
            geo = geometry(left, right, sx, tx, sy, ty)
            x0, x1, y0, y1 = crop_bounds(geo, image.shape[1], image.shape[0])
            counts["crop_width_exact"] += x1 - x0 == int(row["crop_w"])
            counts["crop_height_exact"] += y1 - y0 == int(row["crop_h"])
            replay = author.measure(SimpleNamespace(blind_id=row["blind_id"], folio=folio, **geo), SCAN_DIR)
            counts["threshold_exact"] += int(replay["threshold"]) == int(row["threshold"])
            counts["qc_exact"] += replay["qc"] == row["qc"]
            counts["gap_exact"] += same_number(row["gap_px"], replay["gap_px"])
            counts["left_edge_exact"] += same_number(row["left_edge_global"], replay["left_edge_global"])
            counts["right_edge_exact"] += same_number(row["right_edge_global"], replay["right_edge_global"])
        if counts["gap_exact"] != len(pilot_rows):
            raise AssertionError(f"Direct pixel gaps do not replay for {folio}: {counts}")
        summary[folio] = {
            "rows": len(pilot_rows), "source_box_sha256": sha(BOX_DIR / f"{folio}.js"),
            "yale_canvas_id": SCAN_IDS[folio], "scan_sha256": SCAN_SHA[folio],
            "image_size": list(Image.open(scan_path).size),
            "affine": {"sx": sx, "tx": tx, "sy": sy, "ty": ty},
            "x_max_locator_error_px": x_error, "crop_height_scale_matches": height_matches,
            "threshold_calibration_rows": min(18, len(pilot_rows)),
            "threshold_calibration_exact": fitted_thresholds,
            "threshold_calibration_abs_error": fit_error,
            "replay": {key: int(value) for key, value in counts.items()},
        }
    total = {key: sum(item["replay"].get(key, 0) for item in summary.values())
             for key in next(iter(summary.values()))["replay"]}
    total["finite_archived_gaps"] = sum(bool(row["gap_px"]) for row in rows)
    result = {
        "status": "published_pilot_pixel_replay",
        "archive_revision": "956a7c4fc39981f4d116fa3f4edfccce6d065571",
        "source_measurement_sha256": CODE_SHA, "archived_results_sha256": PILOT_SHA,
        "iiif_manifest_sha256": "317d58fd9ea90392a83d9858a91eada3d0b41416a3c835857dc0154bd123a309",
        "folio_results": summary, "totals": total,
        "limit": "Locator/crop/threshold metadata from this same pilot determine affine registration; this is exact method replay, not a fresh independent ink audit.",
    }
    output = ROOT / "results/BOUNDARY-CHANNEL-0002/pilot_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"folios": len(summary), "rows": len(rows), "totals": total,
                      "transforms": {f: z["affine"] for f, z in summary.items()}}, indent=2))
    return result


if __name__ == "__main__":
    run()
