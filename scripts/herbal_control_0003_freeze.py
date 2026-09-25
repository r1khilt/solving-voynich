"""Hash-verify every HERBAL-CONTROL-0003 source and crop before feature scoring.

This can only produce the full-input freeze after all 216 JPEGs and all 216
manually reviewed development/evaluation crops exist. It does not extract
features or inspect any retrieval score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
DEVELOPMENT = ROOT / "data/manifests/herbal_control_0003_development_images.json"
EVALUATION = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
DEVELOPMENT_BOXES = ROOT / "data/manifests/herbal_control_0003_development_boxes.json"
EVALUATION_BOXES = ROOT / "data/manifests/herbal_control_0003_evaluation_boxes.json"
OUT = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
ROLES = ("development_known", "development_unknown", "evaluation_known", "evaluation_unknown")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_crop_png(path: Path, expected_hash: str) -> None:
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_hash
            or not raw.startswith(b"\x89PNG\r\n\x1a\n")
            or raw[12:16] != b"IHDR"
            or int.from_bytes(raw[16:20], "big") != 512
            or int.from_bytes(raw[20:24], "big") != 512):
        raise ValueError(f"crop is not the registered 512-square PNG: {path}")


def verify_images(path: Path, boxes_path: Path, roles: tuple[str, str],
                  expected: dict[tuple[str, str, str], int],
                  sources: dict[int, dict]) -> int:
    images = json.loads(path.read_text())
    boxes_manifest = json.loads(boxes_path.read_text())
    expected_id = "HERBAL-CONTROL-0003-" + roles[0].split("_")[0]
    if (boxes_manifest.get("id") != expected_id + "-boxes"
            or boxes_manifest.get("status") != "complete-pre-score-manual-review"
            or boxes_manifest.get("panel_manifest_sha256") != PANEL_SHA256
            or len(boxes_manifest.get("boxes", [])) != 108):
        raise ValueError(f"box review is incomplete or changed: {boxes_path}")
    boxes = {}
    for box in boxes_manifest["boxes"]:
        key = (box["role"], box["chapter"], box["manuscript"])
        if key in boxes:
            raise ValueError(f"duplicate reviewed box: {key}")
        boxes[key] = box
    if (images.get("id") != expected_id
            or images.get("status") != "complete-pre-score-" + roles[0].split("_")[0] + "-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108
            or images.get("target_crops") != 108
            or images.get("boxes_manifest_sha256") != digest(boxes_path)):
        raise ValueError(f"crop manifest is incomplete or changed: {path}")
    rows = images.get("rows", [])
    if len(rows) != 108:
        raise ValueError(f"crop row count differs from 108: {path}")
    observed = set()
    for row in rows:
        key = (row["role"], row["chapter_class"], row["manuscript"])
        if (row["role"] not in roles or key not in expected or key in observed
                or row["source_pageid"] != expected[key]):
            raise ValueError(f"crop row differs from frozen panel: {key}")
        observed.add(key)
        source = sources[row["source_pageid"]]
        box = boxes.get(key)
        if (box is None or box["pageid"] != row["source_pageid"]
                or box["source_sha256"] != row["source_sha256"]
                or box["crop_xywh"] != row["crop_xywh"]
                or box["depiction_note"] != row["depiction_note"]):
            raise ValueError(f"crop differs from reviewed box: {key}")
        if row["source_sha256"] != source["sha256"] or row["source_file"] != source["file"]:
            raise ValueError(f"crop/source hash differs from freeze: {key}")
        verify_crop_png(ROOT / row["crop_file"], row["crop_sha256"])
    if observed != {key for key in expected if key[0] in roles} or set(boxes) != observed:
        raise ValueError(f"missing frozen panel crop: {path}")
    return len(rows)


def run() -> dict:
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("frozen panel hash changed")
    panel = json.loads(PANEL.read_text())
    expected = {(role, item["chapter"], manuscript): item["pages"][manuscript]["pageid"]
                for role in ROLES for item in panel["roles"][role]
                for manuscript in MANUSCRIPTS}
    if len(expected) != 216 or len(set(expected.values())) != 216:
        raise ValueError("frozen panel does not contain 216 distinct source pages")
    source_manifest = json.loads(SOURCES.read_text())
    if (source_manifest.get("status") != "all-fixed-source-pages-downloaded"
            or source_manifest.get("panel_manifest_sha256") != PANEL_SHA256
            or source_manifest.get("requested_source_pages") != 216
            or source_manifest.get("recorded_source_pages") != 216):
        raise ValueError("full 216-page source manifest missing")
    source_rows = source_manifest["sources"]
    if len(source_rows) != 216:
        raise ValueError("source row count differs from 216")
    sources = {}
    source_bytes = 0
    for row in source_rows:
        key = (row["role"], row["chapter"], row["manuscript"])
        pageid = row["pageid"]
        if key not in expected or expected[key] != pageid or pageid in sources:
            raise ValueError(f"source row differs from frozen panel: {key}")
        raw = (ROOT / row["file"]).read_bytes()
        if (not raw.startswith(b"\xff\xd8\xff") or len(raw) > 5 * 1024 * 1024
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
                or len(raw) != row["bytes"]):
            raise ValueError(f"source file hash/byte count differs: {pageid}")
        source_bytes += len(raw)
        sources[pageid] = row
    if (len(sources) != 216 or source_bytes > 150 * 1024 * 1024
            or source_bytes != source_manifest.get("recorded_total_bytes")):
        raise ValueError("some frozen sources are missing")
    crop_pages = 0
    crop_pages += verify_images(DEVELOPMENT, DEVELOPMENT_BOXES,
                                ("development_known", "development_unknown"), expected, sources)
    crop_pages += verify_images(EVALUATION, EVALUATION_BOXES,
                                ("evaluation_known", "evaluation_unknown"), expected, sources)
    result = {
        "id": "HERBAL-CONTROL-0003-input-freeze",
        "status": "complete-pre-score-inputs",
        "panel_manifest_sha256": PANEL_SHA256,
        "sources_manifest_sha256": digest(SOURCES),
        "development_images_sha256": digest(DEVELOPMENT),
        "evaluation_images_sha256": digest(EVALUATION),
        "source_pages": len(sources),
        "crop_pages": crop_pages,
        "claim_limit": "Pre-score source and crop integrity only; no feature or Voynich result.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
