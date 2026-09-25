"""Exploratory same-line and deliberate-misregistration stress on ink gaps.

Uses only the six archived direct-pixel pilot folios. The published blind QC
and page registration are retained; boundary labels are used only at summary.
No new manuscript folio or reserved text is inspected.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.boundary_ink_overlap import auc
from scripts.build_boundary_geometry_train import collapse, read_boxes
from scripts.replay_boundary_ink_pilot import (
    BOX_DIR, PILOT, PILOT_SHA, ROOT, SCAN_DIR, SCAN_SHA, author_measurement_module,
    crop_bounds, geometry, pair, sha,
)


REPLAY = ROOT / "results/BOUNDARY-CHANNEL-0002/pilot_replay.json"
REPLAY_SHA = "d46840fc5786e5ffcb30fd1046f28ceddda98a6ad3fb262b3fb3a1264fbc5992"
SEED = 430201
BOOTSTRAPS = 2000
SHIFTS = (-20, 20)


def centered_split(mask: np.ndarray, middle: float) -> int | None:
    height, width = mask.shape
    yy0, yy1 = round(.08 * height), round(.92 * height)
    col = mask[yy0:yy1, :].sum(axis=0)
    lo = max(1, math.floor(middle - 10))
    hi = min(width - 2, math.ceil(middle + 10))
    if lo > hi:
        return None
    return min(range(lo, hi + 1),
               key=lambda c: (int(col[max(0, c - 1):min(len(col), c + 2)].sum()),
                              abs(c - middle), c))


def vertical_metrics(mask: np.ndarray, split: int, ylo: int, yhi: int) -> tuple[float | None, float | None, int]:
    """Return projection gap and median same-scanline gap on common box height.

    Both sides must contain ink no more than 15 pixels from the fixed split.
    The second value requires the left and right strokes on the *same row*.
    """
    if yhi - ylo < 3:
        return None, None, 0
    band = mask[ylo:yhi, :]
    cols = band.sum(axis=0)
    tolerance = max(0, round(.02 * len(band)))
    active = cols > tolerance
    left = np.flatnonzero(active[:split + 1])
    right = np.flatnonzero(active[split + 1:])
    projected = None
    if len(left) and len(right):
        le, re = int(left[-1]), int(split + 1 + right[0])
        if split - le <= 15 and re - split <= 15:
            projected = float(max(0, re - le - 1))
    scanline_gaps = []
    for line in band:
        left = np.flatnonzero(line[:split + 1])
        right = np.flatnonzero(line[split + 1:])
        if not len(left) or not len(right):
            continue
        le, re = int(left[-1]), int(split + 1 + right[0])
        if split - le <= 15 and re - split <= 15:
            scanline_gaps.append(max(0, re - le - 1))
    same_row = float(np.median(scanline_gaps)) if scanline_gaps else None
    return projected, same_row, len(scanline_gaps)


def metric_summary(rows: list[dict], name: str, seed: int) -> dict:
    kept = [row for row in rows if row[name] is not None]
    values = np.array([row[name] for row in kept], dtype=float)
    uncertain = np.array([row["label"] == "," for row in kept])
    folios = np.array([row["folio"] for row in kept])
    if not np.any(uncertain) or np.all(uncertain):
        return {"n": len(kept), "certain": int((~uncertain).sum()),
                "uncertain": int(uncertain.sum()), "auc": None}
    means = {"certain": float(values[~uncertain].mean()),
             "uncertain": float(values[uncertain].mean())}
    by_folio = {}
    for folio in sorted(set(folios)):
        c = values[(folios == folio) & ~uncertain]
        u = values[(folios == folio) & uncertain]
        by_folio[folio] = {"certain": len(c), "uncertain": len(u),
                           "difference": float(c.mean() - u.mean()) if len(c) and len(u) else None}
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(folios)))
    members = {f: np.flatnonzero(folios == f) for f in unique}
    boot = []
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[f] for f in selected])
        if np.any(uncertain[indices]) and not np.all(uncertain[indices]):
            boot.append(auc(values[indices], uncertain[indices]))
    certain_by_pair = defaultdict(list)
    certain_by_folio_pair = defaultdict(list)
    for row in kept:
        if row["label"] == ".":
            certain_by_pair[row["edge_pair"]].append(row[name])
            certain_by_folio_pair[(row["folio"], row["edge_pair"])].append(row[name])
    matched = [float(np.mean(certain_by_pair[row["edge_pair"]]) - row[name])
               for row in kept if row["label"] == "," and certain_by_pair[row["edge_pair"]]]
    matched_local = [float(np.mean(certain_by_folio_pair[(row["folio"], row["edge_pair"])])
                           - row[name]) for row in kept
                     if row["label"] == "," and certain_by_folio_pair[
                         (row["folio"], row["edge_pair"])]]
    return {
        "n": len(kept), "certain": int((~uncertain).sum()),
        "uncertain": int(uncertain.sum()), "mean_px": means,
        "difference_px": means["certain"] - means["uncertain"],
        "auc": auc(values, uncertain),
        "folio_bootstrap95_auc": [float(np.quantile(boot, .025)),
                                  float(np.quantile(boot, .975))],
        "informative_folios": sum(z["difference"] is not None for z in by_folio.values()),
        "positive_folios": sum(z["difference"] is not None and z["difference"] > 0
                               for z in by_folio.values()),
        "by_folio": by_folio,
        "edge_pair_matched_uncertain": len(matched),
        "edge_pair_matched_mean_difference_px": float(np.mean(matched)) if matched else None,
        "within_folio_edge_pair_matched_uncertain": len(matched_local),
        "within_folio_edge_pair_matched_mean_difference_px": (
            float(np.mean(matched_local)) if matched_local else None),
    }


def run() -> dict:
    if sha(PILOT) != PILOT_SHA or sha(REPLAY) != REPLAY_SHA:
        raise AssertionError("Frozen pilot/replay input changed")
    transforms = {folio: item["affine"] for folio, item in json.loads(REPLAY.read_text())[
        "folio_results"].items()}
    archived = list(csv.DictReader(PILOT.open()))
    author = author_measurement_module()
    boxes = {folio: read_boxes(BOX_DIR / f"{folio}.js") for folio in transforms}
    scans = {}
    for folio in transforms:
        path = SCAN_DIR / f"{folio}.jpg"
        if sha(path) != SCAN_SHA[folio]:
            raise AssertionError("Public scan changed")
        scans[folio] = np.asarray(Image.open(path).convert("RGB"))
    measured = []
    for row in archived:
        if row["include"] != "True":
            continue
        folio = row["folio"]
        left, right = pair(row, boxes[folio])
        geo = geometry(left, right, **transforms[folio])
        image = scans[folio]
        x0, x1, y0, y1 = crop_bounds(geo, image.shape[1], image.shape[0])
        crop = image[y0:y1, x0:x1, :]
        mask, _ = author.local_ink_mask(crop)
        middle = (geo["lx1"] + geo["rx0"]) / 2 - x0
        split = centered_split(mask, middle)
        if split is None:
            raise AssertionError("Accepted source boundary has no split")
        lo = max(geo["ly0"], geo["ry0"])
        hi = min(geo["ly1"], geo["ry1"])
        band_lo = max(round(.08 * len(mask)), math.ceil(lo - y0))
        band_hi = min(round(.92 * len(mask)), math.floor(hi - y0))
        projected, same_row, scanlines = vertical_metrics(mask, split, band_lo, band_hi)
        entry = {"blind_id": row["blind_id"], "folio": folio, "label": row["label"],
                 "edge_pair": collapse(row["left_word"])[-1] + collapse(row["right_word"])[0],
                 "primary": float(row["gap_px"]), "overlap_projection": projected,
                 "same_scanline": same_row, "same_scanline_count": scanlines}
        for shift in SHIFTS:
            changed = dict(geo)
            for key in ("ly0", "ly1", "ry0", "ry1"):
                changed[key] += shift
            replay = author.measure(SimpleNamespace(blind_id=row["blind_id"],
                                                    folio=folio, **changed), SCAN_DIR)
            entry[f"shift_{shift:+d}"] = (float(replay["gap_px"])
                                               if replay["qc"] == "ok" and np.isfinite(replay["gap_px"])
                                               else None)
            changed_x = dict(geo)
            for key in ("lx0", "lx1", "rx0", "rx1"):
                changed_x[key] += shift
            replay_x = author.measure(SimpleNamespace(blind_id=row["blind_id"],
                                                       folio=folio, **changed_x), SCAN_DIR)
            entry[f"xshift_{shift:+d}"] = (float(replay_x["gap_px"])
                                                if replay_x["qc"] == "ok" and np.isfinite(replay_x["gap_px"])
                                                else None)
        measured.append(entry)
    names = (["primary", "overlap_projection", "same_scanline"]
             + [f"shift_{s:+d}" for s in SHIFTS]
             + [f"xshift_{s:+d}" for s in SHIFTS])
    summaries = {name: metric_summary(measured, name, SEED + i) for i, name in enumerate(names)}
    comparable_primary = {name: metric_summary(
        [row for row in measured if row[name] is not None], "primary", SEED + 100 + i)
        for i, name in enumerate(names) if name != "primary"}
    support = {}
    for label in (".", ","):
        values = [row["same_scanline_count"] for row in measured if row["label"] == label]
        support[label] = {"n": len(values), "zero": sum(value == 0 for value in values),
                          "median": float(np.median(values)), "at_least_three": sum(
                              value >= 3 for value in values)}
    row_path = ROOT / "data/processed/boundary_ink_0002/vertical_stress_rows.jsonl"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in measured))
    row_sha = hashlib.sha256(row_path.read_bytes()).hexdigest()
    result = {
        "status": "exploratory_six_folio_vertical_support_stress",
        "pilot_replay_sha256": REPLAY_SHA, "source_archived_pilot_sha256": PILOT_SHA,
        "accepted_rows": len(measured), "seed": SEED, "bootstrap_draws": BOOTSTRAPS,
        "rows_sha256": row_sha,
        "methods": {
            "overlap_projection": "author binary mask/split restricted to mapped left-right vertical box intersection; edges within 15 px",
            "same_scanline": "median rowwise nearest-ink gap on common vertical support, both edges within 15 px",
            "shift_controls": "author primary algorithm after shifting only the mapped vertical crop by plus/minus 20 image pixels; original QC plus new algorithmic QC",
            "xshift_controls": "author primary algorithm after shifting only horizontal locator/crop by plus/minus 20 image pixels; original QC plus new algorithmic QC",
        },
        "summaries": summaries,
        "primary_on_same_rows": comparable_primary,
        "same_scanline_support": support,
    }
    output = ROOT / "results/BOUNDARY-CHANNEL-0002/vertical_stress.json"
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({name: {k: report.get(k) for k in
                            ("n", "certain", "uncertain", "difference_px", "auc",
                             "positive_folios", "informative_folios")}
                      for name, report in summaries.items()}, indent=2))
    return result


if __name__ == "__main__":
    run()
