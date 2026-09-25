"""Render manually reviewed HERBAL-CONTROL-0003 drawing boxes to 512-square PNGs.

This accepts a partial box manifest for incremental source review. Evaluation
crop review is per page/manuscript, independent of any model score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
CROP_DIR = ROOT / "data/raw/herbal_control_0003/crops"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
TARGET_COUNT = 108


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    run = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path),
    ], capture_output=True, text=True, check=True)
    width, height = map(int, run.stdout.strip().split("x"))
    return width, height


def run(split: str = "development") -> dict:
    if split not in ("development", "evaluation"):
        raise ValueError("split must be development or evaluation")
    boxes_path = ROOT / f"data/manifests/herbal_control_0003_{split}_boxes.json"
    output_path = ROOT / f"data/manifests/herbal_control_0003_{split}_images.json"
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("frozen panel hash changed")
    panel = json.loads(PANEL.read_text())
    sources = json.loads(SOURCES.read_text())
    if sources["panel_manifest_sha256"] != PANEL_SHA256:
        raise ValueError("source manifest is not for frozen panel")
    by_pageid = {row["pageid"]: row for row in sources["sources"]}
    boxes = json.loads(boxes_path.read_text())
    if boxes["panel_manifest_sha256"] != PANEL_SHA256:
        raise ValueError("box manifest is not for frozen panel")
    expected = {(role, item["chapter"], manuscript): item["pages"][manuscript]["pageid"]
                for role in (f"{split}_known", f"{split}_unknown")
                for item in panel["roles"][role]
                for manuscript in ("bnf", "egerton", "casanatense")}
    if len(expected) != TARGET_COUNT:
        raise ValueError("unexpected development split size")
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for box in boxes["boxes"]:
        key = (box["role"], box["chapter"], box["manuscript"])
        if key in seen or key not in expected or expected[key] != box["pageid"]:
            raise ValueError(f"duplicate or non-panel review box: {key}")
        seen.add(key)
        source = by_pageid[box["pageid"]]
        if "sha256" not in source or source["sha256"] != box["source_sha256"]:
            raise ValueError(f"source not downloaded or changed: {key}")
        source_path = ROOT / source["file"]
        if digest(source_path) != source["sha256"]:
            raise ValueError(f"source hash mismatch: {key}")
        width, height = image_size(source_path)
        x, y, crop_width, crop_height = box["crop_xywh"]
        if (not all(isinstance(v, int) for v in (x, y, crop_width, crop_height))
                or x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0
                or x + crop_width > width or y + crop_height > height):
            raise ValueError(f"invalid crop rectangle for {key}: {box['crop_xywh']} in {width}x{height}")
        crop_path = CROP_DIR / f"{box['manuscript']}_{box['pageid']}.png"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source_path),
            "-vf", f"crop={crop_width}:{crop_height}:{x}:{y},"
                   "scale=512:512:force_original_aspect_ratio=decrease,"
                   "pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
            "-frames:v", "1", str(crop_path),
        ], check=True)
        if image_size(crop_path) != (512, 512):
            raise ValueError(f"unexpected crop output size: {key}")
        rows.append({
            "role": box["role"], "chapter_class": box["chapter"],
            "manuscript": box["manuscript"], "source_pageid": box["pageid"],
            "source_file": source["file"], "source_sha256": source["sha256"],
            "crop_xywh": box["crop_xywh"], "crop_file": str(crop_path.relative_to(ROOT)),
            "crop_sha256": digest(crop_path), "depiction_note": box["depiction_note"],
        })
    result = {
        "id": f"HERBAL-CONTROL-0003-{split}",
        "status": f"complete-pre-score-{split}-crops" if len(rows) == TARGET_COUNT
                  else f"partial-pre-score-{split}-crops",
        "panel_manifest_sha256": PANEL_SHA256,
        "boxes_manifest_sha256": digest(boxes_path),
        "target_crops": TARGET_COUNT,
        "rendered_crops": len(rows),
        "rows": rows,
        "limitations": "One researcher's metadata-guided crop rectangles; some drawings overlap writing and source chapter labels are not species truth.",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    result = run(parser.parse_args().split)
    print(json.dumps({"status": result["status"], "rendered_crops": result["rendered_crops"]},
                     sort_keys=True))
