"""Frozen DINOv2 secondary features for HERBAL-CONTROL-0003.

Development and evaluation are separate invocations. Evaluation refuses to
open any crop until a hash-linked, independently audited DINO development
selection is supplied. Remote push verification remains a manual protocol gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
MODEL_DIR = ROOT / "data/raw/herbal_control_0003/models/dinov2-with-registers-base"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MODEL_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
MODEL_FILES = {
    "model.safetensors": "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
    "config.json": "6af60aa760138fc90db0ba37b7701730b140b9bf1742514412d03050e72bf7e0",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}
FEATURE_METHOD = "DINOv2-with-registers-base final CLS 224 direct resize L2 cosine"
MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 8 * 1024**3
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_manifest(split: str) -> Path:
    return ROOT / f"data/manifests/herbal_control_0003_{split}_images.json"


def preflight(split: str, selection_path: Path | None = None,
              audit_path: Path | None = None) -> tuple[dict, str, str]:
    """Validate provenance before opening any image pixels or loading weights."""
    if split not in ("development", "evaluation"):
        raise ValueError("split must be development or evaluation")
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("active panel changed")
    panel = json.loads(PANEL.read_text())
    image_path = image_manifest(split)
    other_path = image_manifest("evaluation" if split == "development" else "development")
    if not FREEZE.is_file() or not image_path.is_file() or not other_path.is_file():
        raise ValueError("complete 216/216 input freeze or image manifests missing")
    image_sha = digest(image_path)
    freeze_sha = digest(FREEZE)
    freeze = json.loads(FREEZE.read_text())
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216
            or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES)
            or freeze.get(f"{split}_images_sha256") != image_sha
            or freeze.get(f"{'evaluation' if split == 'development' else 'development'}_images_sha256")
            != digest(other_path)):
        raise ValueError("complete 216/216 input freeze missing or changed")
    if split == "evaluation":
        if selection_path is None or audit_path is None:
            raise ValueError("evaluation requires audited DINO development selection")
        selection_sha = digest(selection_path)
        selection = json.loads(selection_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (selection.get("id") != "HERBAL-CONTROL-0003-DINO-development-selection"
                or selection.get("status") != "development-score-produced-awaiting-audit-and-push"
                or selection.get("panel_manifest_sha256") != PANEL_SHA256
                or selection.get("full_input_freeze_sha256") != freeze_sha
                or selection.get("feature_method") != FEATURE_METHOD
                or audit.get("id") != "HERBAL-CONTROL-0003-DINO-independent-development-score-audit"
                or audit.get("status") != "pass"
                or audit.get("score_sha256") != selection_sha
                or audit.get("full_input_freeze_sha256") != freeze_sha
                or audit.get("feature_sha256") != selection.get("development_features_sha256")
                or audit.get("primary_method") != selection.get("primary_method")):
            raise ValueError("DINO development selection/audit missing or changed")
    images = json.loads(image_path.read_text())
    if (images.get("id") != f"HERBAL-CONTROL-0003-{split}"
            or images.get("status") != f"complete-pre-score-{split}-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108
            or len(images.get("rows", [])) != 108):
        raise ValueError(f"{split} crop manifest incomplete or changed")
    expected = {(role, chapter["chapter"], manuscript): chapter["pages"][manuscript]["pageid"]
                for role in (f"{split}_known", f"{split}_unknown")
                for chapter in panel["roles"][role]
                for manuscript in MANUSCRIPTS}
    rows = images["rows"]
    actual = {(row["role"], row["chapter_class"], row["manuscript"]): row["source_pageid"]
              for row in rows}
    if len(expected) != 108 or len(actual) != 108 or actual != expected:
        raise ValueError("crop rows do not exactly match frozen panel")
    if (sum(row["role"] == f"{split}_known" for row in rows) != 72
            or sum(row["role"] == f"{split}_unknown" for row in rows) != 36):
        raise ValueError("known/unknown counts changed")
    for row in rows:
        crop = ROOT / row["crop_file"]
        if not crop.is_file() or digest(crop) != row["crop_sha256"]:
            raise ValueError(f"crop changed: {row['crop_file']}")
    for name, sha in MODEL_FILES.items():
        if digest(MODEL_DIR / name) != sha:
            raise ValueError(f"local model file changed: {name}")
    return images, image_sha, freeze_sha


def extract(split: str, selection_path: Path | None = None,
            audit_path: Path | None = None) -> dict:
    images, image_sha, freeze_sha = preflight(split, selection_path, audit_path)
    started = time.monotonic()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=False, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=False, local_files_only=True,
        use_safetensors=True).eval().to("cpu")
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("DINO model load exceeded 15-minute cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_BYTES:
        raise MemoryError("DINO model load exceeded 8-GiB RSS cap")
    vectors = []
    with torch.inference_mode():
        for row in images["rows"]:
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("DINO feature extraction exceeded 15-minute cap")
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_BYTES:
                raise MemoryError("DINO feature extraction exceeded 8-GiB RSS cap")
            picture = Image.open(ROOT / row["crop_file"]).convert("RGB")
            tensor = processor(images=picture, return_tensors="pt",
                               size={"height": 224, "width": 224},
                               do_center_crop=False)
            vector = model(**tensor).last_hidden_state[0, 0].float()
            vector = torch.nn.functional.normalize(vector, dim=0)
            vectors.append(vector.numpy())
    features = np.stack(vectors).astype(np.float64)
    if features.shape != (108, 768) or not np.isfinite(features).all():
        raise ValueError("unexpected or non-finite DINO features")
    cosine = np.clip(features @ features.T, -1.0, 1.0)
    matrix = np.maximum(0.0, 1.0 - cosine)
    np.fill_diagonal(matrix, 0.0)
    row_order = [{key: row[key] for key in
                  ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256")}
                 for row in images["rows"]]
    return {
        "id": f"HERBAL-CONTROL-0003-{split}",
        "input_manifest_sha256": image_sha,
        "full_input_freeze_sha256": freeze_sha,
        "feature_method": FEATURE_METHOD,
        "model_revision": MODEL_REVISION,
        "model_files_sha256": MODEL_FILES,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": "cpu", "runtime_seconds": round(time.monotonic() - started, 3),
        "row_order": row_order,
        "distance_matrix": matrix.tolist(),
        **({"development_selection_sha256": digest(selection_path),
            "development_audit_sha256": digest(audit_path)}
           if split == "evaluation" and selection_path is not None
           and audit_path is not None else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = extract(args.split, args.selection, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split": args.split, "rows": len(result["row_order"]),
                      "feature_method": FEATURE_METHOD}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, TimeoutError, MemoryError) as error:
        print(f"HERBAL-CONTROL-0003 DINO feature refusal: {error}", file=sys.stderr)
        raise SystemExit(1) from None
