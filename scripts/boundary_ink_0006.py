"""Freeze and render a blinded word-box/ink-crop QC packet.

The packet deliberately shows no transcription, separator label, or gap score.
The key and scans stay ignored; the selection and protocol are tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.boundary_ink_overlap import auc
from scripts.build_boundary_geometry_train import read_boxes
from scripts.register_boundary_ink_pages import BOX_DIR, ROOT
from scripts.replay_boundary_ink_pilot import author_measurement_module, geometry


ROWS = ROOT / "data/processed/boundary_geometry/train_pairs.jsonl"
ROWS_MANIFEST = ROOT / "data/manifests/boundary_geometry_train.json"
REGISTRATION = ROOT / "results/BOUNDARY-CHANNEL-0005/prospective_registration.json"
CONFIG = ROOT / "data/manifests/boundary_registration_0005.json"
SELECTION = ROOT / "data/manifests/boundary_ink_0006.json"
PACKET = ROOT / "data/processed/boundary_ink_0006/blind_packet"
RATINGS = ROOT / "data/manifests/boundary_ink_0006_blind_ratings.json"
MEASURED = ROOT / "data/processed/boundary_ink_0006/measured_rows.jsonl"
RESULT = ROOT / "results/BOUNDARY-CHANNEL-0006/result.json"
SCAN_DIR = ROOT / "data/raw/boundary_ink_0005/yale700"
SEED = "BOUNDARY-CHANNEL-0006|2026-09-25"
N_PER_FOLIO_LABEL = 6
DECOYS_PER_FOLIO = 2
BOOTSTRAPS = 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def pair_id(row: dict) -> str:
    return f"{row['folio']}:{row['visual_left_index']}:{row['text_left_index']}"


def validated_sources() -> tuple[list[dict], dict, dict]:
    row_manifest = json.loads(ROWS_MANIFEST.read_text())
    registration = json.loads(REGISTRATION.read_text())
    config = json.loads(CONFIG.read_text())
    if sha(ROWS) != row_manifest["rows_sha256"]:
        raise AssertionError("Derived training rows changed")
    if registration["config_sha256"] != sha(CONFIG) or registration["gate"] != "PASS":
        raise AssertionError("Fresh layout registration is not frozen/passing")
    folios = {item["folio"] for item in config["folios"]}
    if set(registration["rankings"]) != folios:
        raise AssertionError("Registration set changed")
    for item in config["folios"]:
        folio = item["folio"]
        if (sha(BOX_DIR / f"{folio}.js") != item["box_sha256"] or
                sha(SCAN_DIR / f"{folio}.jpg") != registration["scan_sha256"][folio]):
            raise AssertionError(f"Source image/boxes changed: {folio}")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    rows = [row for row in rows if row["folio"] in folios]
    if len(rows) != 1024 or any(row["label"] not in (".", ",") for row in rows):
        raise AssertionError("Expected fixed 1,024 eligible training pairs")
    return rows, registration, config


def selection() -> dict:
    rows, registration, config = validated_sources()
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["folio"], row["label"])].append(row)
    chosen = []
    for item in config["folios"]:
        folio = item["folio"]
        for label in (".", ","):
            group = by_group[(folio, label)]
            group.sort(key=lambda row: tag("choose|" + pair_id(row)))
            chosen.extend(group[:N_PER_FOLIO_LABEL])
    if len(chosen) != 76:
        raise AssertionError(f"Selected count changed: {len(chosen)}")
    originals = [{"pair_id": pair_id(row), "folio": row["folio"]} for row in chosen]
    by_folio: dict[str, list[dict]] = defaultdict(list)
    for row in chosen:
        by_folio[row["folio"]].append(row)
    decoys = []
    for folio, group in sorted(by_folio.items()):
        group.sort(key=lambda row: tag("decoy|" + pair_id(row)))
        for offset, row in enumerate(group[:DECOYS_PER_FOLIO]):
            decoys.append({"pair_id": pair_id(row), "folio": folio,
                           "dy": 20 if offset == 0 else -20})
    if len(decoys) != 16:
        raise AssertionError("Expected 16 misregistered controls")
    return {
        "experiment": "BOUNDARY-CHANNEL-0006",
        "status": "frozen_blind_crop_selection",
        "selection_seed": SEED,
        "eligible_rows": len(rows),
        "selected_originals": originals,
        "selected_decoys": decoys,
        "sampling": "For each of 8 fixed train folios, hash-rank all eligible '.' and ',' pairs separately; take first 6 of each or all if fewer. Pick two selected pairs per folio by a separate hash and displace their mapped boxes +20/-20 image pixels vertically. Shuffle originals and decoys by a third hash for review. No gap or ink score is inspected for selection.",
        "source_rows_sha256": sha(ROWS),
        "source_rows_manifest_sha256": sha(ROWS_MANIFEST),
        "registration_result_sha256": sha(REGISTRATION),
        "registration_config_sha256": sha(CONFIG),
        "scan_sha256": registration["scan_sha256"],
        "box_sha256": {item["folio"]: item["box_sha256"] for item in config["folios"]},
    }


def freeze() -> dict:
    selected = selection()
    SELECTION.write_text(json.dumps(selected, sort_keys=True, indent=2) + "\n")
    return {"eligible": selected["eligible_rows"],
            "originals": len(selected["selected_originals"]),
            "decoys": len(selected["selected_decoys"]),
            "selection_sha256": sha(SELECTION)}


def transformed_box(box: dict, affine: dict, dy: int = 0) -> tuple[float, float, float, float]:
    return (affine["tx"] + affine["sx"] * box["x"],
            affine["ty"] + affine["sy"] * box["y"] + dy,
            affine["tx"] + affine["sx"] * (box["x"] + box["w"]),
            affine["ty"] + affine["sy"] * (box["y"] + box["h"]) + dy)


def render_crop(image: Image.Image, left: dict, right: dict, affine: dict,
                dy: int) -> Image.Image:
    a, b = transformed_box(left, affine, dy), transformed_box(right, affine, dy)
    width, height = image.size
    x0 = max(0, math.floor(min(a[0], b[0]) - 12))
    x1 = min(width, math.ceil(max(a[2], b[2]) + 12))
    y0 = max(0, math.floor(min(a[1], b[1]) - 22))
    y1 = min(height, math.ceil(max(a[3], b[3]) + 22))
    if x1 - x0 < 15 or y1 - y0 < 15:
        raise AssertionError("Crop is empty/too small")
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    for box, color in ((a, "#e52020"), (b, "#225ee8")):
        draw.rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0),
                       outline=color, width=1)
    return crop


def make_packet() -> dict:
    frozen = json.loads(SELECTION.read_text())
    if frozen != selection():
        raise AssertionError("Frozen crop selection does not replay")
    rows, registration, _ = validated_sources()
    by_id = {pair_id(row): row for row in rows}
    items = ([dict(item, kind="original", dy=0) for item in frozen["selected_originals"]]
             + [dict(item, kind="decoy") for item in frozen["selected_decoys"]])
    items.sort(key=lambda item: tag(f"packet|{item['kind']}|{item['pair_id']}|{item['dy']}"))
    PACKET.mkdir(parents=True, exist_ok=True)
    scans = {folio: Image.open(SCAN_DIR / f"{folio}.jpg").convert("RGB")
             for folio in registration["rankings"]}
    boxes = {folio: read_boxes(BOX_DIR / f"{folio}.js")
             for folio in registration["rankings"]}
    key = []
    thumbnails = []
    for index, item in enumerate(items):
        row = by_id[item["pair_id"]]
        folio = row["folio"]
        vi = row["visual_left_index"]
        left, right = boxes[folio][vi:vi + 2]
        if ((left["word"], right["word"]) !=
                (row["left_word"], row["right_word"]) or left["line"] != right["line"]):
            raise AssertionError(f"Word identity/source line changed: {item['pair_id']}")
        crop = render_crop(scans[folio], left, right,
                           registration["rankings"][folio]["raw_affine"], item["dy"])
        blind_id = f"B{index:03d}"
        crop.save(PACKET / f"{blind_id}.png")
        key.append({"blind_id": blind_id, "pair_id": item["pair_id"],
                    "kind": item["kind"], "folio": folio,
                    "label": row["label"], "dy": item["dy"],
                    "crop_sha256": sha(PACKET / f"{blind_id}.png")})
        tile = Image.new("RGB", (320, 116), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((7, 3), blind_id, fill="black", font=ImageFont.load_default())
        crop.thumbnail((310, 90))
        tile.paste(crop, ((320 - crop.width) // 2, 20 + (90 - crop.height) // 2))
        thumbnails.append(tile)
    for start in range(0, len(thumbnails), 24):
        sheet = Image.new("RGB", (4 * 320, 6 * 116), "#dddddd")
        for local_index, tile in enumerate(thumbnails[start:start + 24]):
            sheet.paste(tile, ((local_index % 4) * 320,
                               (local_index // 4) * 116))
        sheet.save(PACKET / f"sheet_{start // 24 + 1:02d}.jpg", quality=94)
    (PACKET / "key.json").write_text(json.dumps(key, sort_keys=True, indent=2) + "\n")
    (PACKET / "ratings_template.json").write_text(json.dumps(
        {item["blind_id"]: "UNRATED" for item in key}, indent=2) + "\n")
    return {"items": len(items), "sheets": math.ceil(len(items) / 24),
            "key_sha256": sha(PACKET / "key.json"),
            "selection_sha256": sha(SELECTION)}


def auc_summary(rows: list[dict], field: str) -> dict:
    kept = [row for row in rows if row[field] is not None]
    if not kept:
        return {"rows": 0, "certain": 0, "uncertain": 0, "auc": None}
    values = np.array([row[field] for row in kept], dtype=float)
    uncertain = np.array([row["label"] == "," for row in kept], dtype=bool)
    folios = sorted({row["folio"] for row in kept})
    if uncertain.all() or not uncertain.any():
        score = None
    else:
        score = auc(values, uncertain)
    rng = np.random.default_rng(6062026)
    members = {folio: np.array([index for index, row in enumerate(kept)
                                 if row["folio"] == folio]) for folio in folios}
    boot = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(folios, size=len(folios), replace=True)
        indices = np.concatenate([members[folio] for folio in sampled])
        if uncertain[indices].any() and not uncertain[indices].all():
            boot.append(auc(values[indices], uncertain[indices]))
    by_folio = {}
    for folio in folios:
        selected = members[folio]
        c = values[selected][~uncertain[selected]]
        u = values[selected][uncertain[selected]]
        by_folio[folio] = {"certain": len(c), "uncertain": len(u),
                           "mean_certain_minus_uncertain": (
                               float(c.mean() - u.mean()) if len(c) and len(u) else None)}
    return {
        "rows": len(kept), "certain": int((~uncertain).sum()),
        "uncertain": int(uncertain.sum()), "auc": score,
        "folio_bootstrap95": ([float(np.quantile(boot, .025)),
                               float(np.quantile(boot, .975))] if boot else None),
        "by_folio": by_folio,
    }


def score() -> dict:
    frozen = json.loads(SELECTION.read_text())
    if frozen != selection():
        raise AssertionError("Frozen sample no longer replays")
    key = json.loads((PACKET / "key.json").read_text())
    ratings = json.loads(RATINGS.read_text())
    expected = {item["blind_id"] for item in key}
    if set(ratings) != expected or any(rating not in ("pass", "fail", "unsure")
                                      for rating in ratings.values()):
        raise AssertionError("Incomplete or invalid blind ratings")
    if any(sha(PACKET / f"{item['blind_id']}.png") != item["crop_sha256"]
           for item in key):
        raise AssertionError("Reviewed crop changed")
    originals = [item for item in key if item["kind"] == "original"]
    decoys = [item for item in key if item["kind"] == "decoy"]
    if len(originals) != 76 or len(decoys) != 16:
        raise AssertionError("Blind packet count changed")
    accepted = [item for item in originals if ratings[item["blind_id"]] == "pass"]
    decoy_accepted = [item for item in decoys if ratings[item["blind_id"]] == "pass"]
    folio_accepted = Counter(item["folio"] for item in accepted)
    visual_gate = (len(accepted) >= 65 and len(decoy_accepted) <= 4 and
                   all(folio_accepted[folio] >= 3 for folio in frozen["scan_sha256"]))
    rows, registration, _ = validated_sources()
    boxes = {folio: read_boxes(BOX_DIR / f"{folio}.js") for folio in registration["rankings"]}
    author = author_measurement_module()
    blind_by_pair = {item["pair_id"]: item["blind_id"] for item in originals}
    measured = []
    for row in rows:
        folio = row["folio"]
        vi = row["visual_left_index"]
        left, right = boxes[folio][vi:vi + 2]
        if (left["word"], right["word"]) != (row["left_word"], row["right_word"]):
            raise AssertionError(f"Aligned pair changed: {pair_id(row)}")
        affine = registration["rankings"][folio]["raw_affine"]
        geo = geometry(left, right, **affine)
        measured_ink = author.measure(SimpleNamespace(blind_id=pair_id(row),
                                                       folio=folio, **geo), SCAN_DIR)
        visual_id = blind_by_pair.get(pair_id(row))
        chosen = visual_id is not None and ratings[visual_id] == "pass"
        norm = (geo["ly1"] - geo["ly0"] + geo["ry1"] - geo["ry0"]) / 2
        gap = (float(measured_ink["gap_px"]) / norm
               if measured_ink["qc"] == "ok" and math.isfinite(measured_ink["gap_px"])
               else None)
        entry = {"pair_id": pair_id(row), "folio": folio, "label": row["label"],
                 "blind_id": visual_id, "visual_pass": chosen,
                 "author_qc": measured_ink["qc"], "direct_ink_gap_over_box_height": gap,
                 "box_gap_over_median_word_width": row["gap_over_median_word_width"]}
        if chosen:
            shifted_geo = dict(geo)
            for coordinate in ("lx0", "lx1", "rx0", "rx1"):
                shifted_geo[coordinate] += 20
            shifted = author.measure(SimpleNamespace(blind_id=pair_id(row),
                                                     folio=folio, **shifted_geo), SCAN_DIR)
            entry["xshift_20_gap_over_box_height"] = (
                float(shifted["gap_px"]) / norm
                if shifted["qc"] == "ok" and math.isfinite(shifted["gap_px"]) else None)
        measured.append(entry)
    MEASURED.parent.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in measured))
    reviewed = [row for row in measured if row["visual_pass"]]
    primary = auc_summary(reviewed, "direct_ink_gap_over_box_height")
    box = auc_summary([row for row in reviewed if row["direct_ink_gap_over_box_height"]
                       is not None], "box_gap_over_median_word_width")
    paired = [row for row in reviewed if row["direct_ink_gap_over_box_height"] is not None
              and row.get("xshift_20_gap_over_box_height") is not None]
    paired_real = auc_summary(paired, "direct_ink_gap_over_box_height")
    paired_shift = auc_summary(paired, "xshift_20_gap_over_box_height")
    exploratory_all = auc_summary(measured, "direct_ink_gap_over_box_height")
    effect_gate = (visual_gate and primary["certain"] >= 30 and
                   primary["uncertain"] >= 20 and primary["auc"] is not None and
                   primary["auc"] >= .65 and primary["folio_bootstrap95"] is not None and
                   primary["folio_bootstrap95"][0] > .5 and
                   paired_real["auc"] is not None and paired_shift["auc"] is not None and
                   paired_real["auc"] - paired_shift["auc"] >= .05)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "BOUNDARY-CHANNEL-0006", "status": "completed",
        "selection_sha256": sha(SELECTION), "packet_key_sha256": sha(PACKET / "key.json"),
        "blind_ratings_sha256": sha(RATINGS), "source_rows_sha256": sha(ROWS),
        "registration_result_sha256": sha(REGISTRATION),
        "author_measurement_code_sha256": sha(ROOT / "data/raw/external/voynich-units/analysis/direct_pixel/measure_direct_pixels.py"),
        "measured_rows_sha256": sha(MEASURED),
        "visual_qc": {"original_pass": len(accepted), "original_total": len(originals),
                      "decoy_pass": len(decoy_accepted), "decoy_total": len(decoys),
                      "by_folio_pass": dict(folio_accepted),
                      "gate_pass": visual_gate},
        "reviewed_primary": primary, "reviewed_box_proxy_same_rows": box,
        "paired_real_xshift": paired_real, "paired_xshift_20": paired_shift,
        "unreviewed_all_eligible_exploratory": exploratory_all,
        "registered_effect_gate": "PASS" if effect_gate else "FAIL",
        "limit": "Same-agent crop review is blind to label/score but not independent. Only selected visually accepted pairs enter the primary effect. Passing cannot establish intended lexical words or decipherment.",
    }
    RESULT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"visual_qc": result["visual_qc"], "primary": primary,
            "paired_xshift_auc": paired_shift["auc"],
            "effect_gate": result["registered_effect_gate"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "packet", "score"), required=True)
    mode = parser.parse_args().mode
    print(json.dumps(freeze() if mode == "freeze" else
                     make_packet() if mode == "packet" else score(), indent=2))
