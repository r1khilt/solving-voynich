"""Register word-box layouts to Yale scans without pilot locator metadata.

The search sees only source boxes and page pixels. Pilot metadata, labels,
reported gaps, and transcription content are not imported or inspected.
This is a global affine *candidate* locator, not word-identity verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_boundary_ink_locator_alignment import dark_stroke_map
from scripts.build_boundary_geometry_train import leaf_id, read_boxes


ROOT = Path(__file__).resolve().parent.parent
BOX_DIR = ROOT / "data/raw/external/voynich-units/data/voynich-units/morphometry_voynichese/voynichese_boxes"
ARCHIVE = ROOT / "data/raw/external/voynich-units"
REGISTRATION = ROOT / "data/manifests/boundary_registration_0004.json"
SPLIT = ROOT / "data/manifests/zl3b_split.json"
YALE_MANIFEST = ROOT / "data/raw/yale_ms408_iiif_manifest.json"
SOURCE_REV = "956a7c4fc39981f4d116fa3f4edfccce6d065571"
SPLIT_SHA = "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e"
YALE_SHA = "317d58fd9ea90392a83d9858a91eada3d0b41416a3c835857dc0154bd123a309"
SCALES = np.arange(60, 101, 2, dtype=np.int32) / 100
MAX_TX = 260
MAX_TY = 150


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_mask(boxes: list[dict]) -> np.ndarray:
    """A binary source-coordinate layout, including whitespace between words."""
    height = max(item["y"] + item["h"] for item in boxes) + 1
    width = max(item["x"] + item["w"] for item in boxes) + 1
    mask = np.zeros((height, width), dtype=np.float32)
    for item in boxes:
        cv2.rectangle(mask, (item["x"], item["y"]),
                      (item["x"] + item["w"], item["y"] + item["h"]), 1, -1)
    if mask.std() < 1e-6:
        raise ValueError("Degenerate source layout")
    return mask


def target_map(image: np.ndarray) -> np.ndarray:
    """Dark local strokes after excluding green pigment; reject blank scans."""
    if image is None or image.ndim != 3 or image.shape[1] != 700:
        raise ValueError("Expected a readable 700-pixel-wide color Yale scan")
    target = cv2.GaussianBlur(dark_stroke_map(image), (0, 0), 1.2)
    if np.count_nonzero(target) < 100 or target.std() < 1e-5:
        raise ValueError("Blank or low-information target scan")
    return target


def best_match(target: np.ndarray, model: np.ndarray) -> tuple[float, int, int] | None:
    if model.shape[0] > target.shape[0] or model.shape[1] > target.shape[1]:
        return None
    response = cv2.matchTemplate(target, model, cv2.TM_CCOEFF_NORMED)
    response = response[:min(MAX_TY + 1, response.shape[0]),
                        :min(MAX_TX + 1, response.shape[1])]
    _, score, _, (tx, ty) = cv2.minMaxLoc(response)
    if not np.isfinite(score):
        raise ValueError("Nonfinite template match")
    return float(score), tx, ty


def all_matches(targets: dict[str, np.ndarray], sources: dict[str, np.ndarray]) -> dict:
    """Apply the identical scale/translation search to every source-target pair."""
    best: dict[str, dict[str, dict | None]] = {
        target: {source: None for source in sources} for target in targets
    }
    for source_name, source in sources.items():
        for sx in SCALES:
            for sy in SCALES:
                model = cv2.resize(source, (round(source.shape[1] * float(sx)),
                                            round(source.shape[0] * float(sy))),
                                   interpolation=cv2.INTER_AREA)
                for target_name, target in targets.items():
                    candidate = best_match(target, model)
                    if candidate is None:
                        continue
                    score, tx, ty = candidate
                    previous = best[target_name][source_name]
                    if previous is None or score > previous["score"]:
                        best[target_name][source_name] = {
                            "score": score, "sx": float(sx), "sy": float(sy),
                            "tx": tx, "ty": ty,
                        }
    return best


def validate_inputs(config: dict, group: str) -> tuple[list[dict], Path]:
    if (sha(SPLIT) != SPLIT_SHA or sha(YALE_MANIFEST) != YALE_SHA or
            subprocess.check_output(["git", "-C", str(ARCHIVE), "rev-parse", "HEAD"],
                                    text=True).strip() != SOURCE_REV):
        raise AssertionError("Frozen split, IIIF manifest, or box archive changed")
    selected = config["groups"][group]
    folios = selected["folios"]
    if len({item["folio"] for item in folios}) != len(folios):
        raise AssertionError("Duplicate selected folio")
    assignments = json.loads(SPLIT.read_text())["leaf_assignments"]
    yale = json.loads(YALE_MANIFEST.read_text())
    ids = {f"f{item['label']['none'][0]}": item["id"].split("/")[-1]
           for item in yale["items"] if item.get("label", {}).get("none")}
    for item in folios:
        folio = item["folio"]
        if assignments.get(leaf_id(folio)) != "train":
            raise AssertionError(f"Non-train physical leaf: {folio}")
        if ids.get(folio) != item["canvas_id"]:
            raise AssertionError(f"IIIF canvas changed: {folio}")
        if sha(BOX_DIR / f"{folio}.js") != item["box_sha256"]:
            raise AssertionError(f"Source boxes changed: {folio}")
    if group == "prospective_train":
        pilot_leaves = {leaf_id(item["folio"]) for item in config["groups"]["pilot_dev"]["folios"]}
        eligible = []
        for path in BOX_DIR.glob("*.js"):
            folio = path.stem
            if (leaf_id(folio) in pilot_leaves or assignments.get(leaf_id(folio)) != "train"
                    or folio not in ids):
                continue
            boxes = read_boxes(path)
            if len(boxes) < 70 or len({box["line"] for box in boxes}) < 8:
                continue
            rank = hashlib.sha256(f"BOUNDARY-CHANNEL-0004|{folio}".encode()).hexdigest()
            eligible.append((rank, folio))
        expected = [folio for _, folio in sorted(eligible)[:8]]
        if [item["folio"] for item in folios] != expected:
            raise AssertionError(f"Prospective selection does not replay: {expected}")
    scan_dir = ROOT / selected["scan_dir"]
    return folios, scan_dir


def run(group: str) -> dict:
    config = json.loads(REGISTRATION.read_text())
    folios, scan_dir = validate_inputs(config, group)
    images = {}
    scan_shas = {}
    sources = {}
    for item in folios:
        folio = item["folio"]
        path = scan_dir / f"{folio}.jpg"
        digest = sha(path)
        if "scan_sha256" in item and digest != item["scan_sha256"]:
            raise AssertionError(f"Pinned scan changed: {folio}")
        image = cv2.imread(str(path))
        images[folio] = target_map(image)
        scan_shas[folio] = digest
        sources[folio] = source_mask(read_boxes(BOX_DIR / f"{folio}.js"))
    start = time.monotonic()
    matches = all_matches(images, sources)
    elapsed = time.monotonic() - start
    rankings = {}
    for folio, row in matches.items():
        if any(value is None for value in row.values()):
            raise ValueError(f"At least one template cannot fit in scan {folio}")
        ordered = sorted(row, key=lambda source: row[source]["score"], reverse=True)
        own = row[folio]["score"]
        best_other = max(row[source]["score"] for source in row if source != folio)
        rankings[folio] = {"rank_of_correct_folio": ordered.index(folio) + 1,
                           "own_score": own, "best_impostor_score": best_other,
                           "own_minus_best_impostor": own - best_other,
                           "affine": {key: row[folio][key] for key in ("sx", "tx", "sy", "ty")}}
    result = {"status": "image_only_box_scan_registration", "group": group,
              "config_sha256": sha(REGISTRATION), "source_archive_revision": SOURCE_REV,
              "yale_manifest_sha256": YALE_SHA, "split_sha256": SPLIT_SHA,
              "method": "700px target local dark-stroke map; source filled-box layout; OpenCV TM_CCOEFF_NORMED; sx/sy .60..1.00 step .02; tx 0..260; ty 0..150",
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "scan_sha256": scan_shas, "seconds": elapsed,
              "matches": matches, "rankings": rankings,
              "caution": "Same-folio identity and global box overlap do not prove that every named word or adjacent ink pair is correctly localized."}
    output = ROOT / f"results/BOUNDARY-CHANNEL-0004/{group}_registration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"group": group, "seconds": round(elapsed, 2), "rankings": rankings},
                     indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("pilot_dev", "prospective_train"), required=True)
    run(parser.parse_args().group)
