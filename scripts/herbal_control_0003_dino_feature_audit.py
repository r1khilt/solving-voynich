"""Recompute frozen DINO distances with manual preprocessing and direct model class.

This audit shares the source/crop preflight gate with the extractor but imports
neither its image processor nor its feature extraction routine. It checks the
entire 108x108 distance matrix, not a sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from scripts import herbal_control_0003_dino_features as source


def manual_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    normalized = (array - mean) / std
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


def audit(split: str, feature_path: Path, selection_path: Path | None = None,
          development_audit_path: Path | None = None) -> dict:
    images, images_sha, freeze_sha = source.preflight(
        split, selection_path, development_audit_path)
    features = json.loads(feature_path.read_text())
    if (features.get("id") != f"HERBAL-CONTROL-0003-{split}"
            or features.get("feature_method") != source.FEATURE_METHOD
            or features.get("input_manifest_sha256") != images_sha
            or features.get("full_input_freeze_sha256") != freeze_sha
            or features.get("model_revision") != source.MODEL_REVISION
            or features.get("model_files_sha256") != source.MODEL_FILES
            or features.get("device") != "cpu"):
        raise ValueError("DINO feature provenance differs")
    if split == "evaluation" and (
            features.get("development_selection_sha256") != source.digest(selection_path)
            or features.get("development_audit_sha256") != source.digest(development_audit_path)):
        raise ValueError("evaluation feature development gate differs")
    rows = features.get("row_order", [])
    if len(rows) != 108:
        raise ValueError("DINO feature row count differs")
    for actual, expected in zip(rows, images["rows"], strict=True):
        if any(actual.get(key) != expected[key] for key in
               ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256")):
            raise ValueError("DINO feature row/crop order differs")
    stored = np.asarray(features.get("distance_matrix"), dtype=np.float64)
    if stored.shape != (108, 108) or not np.isfinite(stored).all():
        raise ValueError("DINO feature matrix invalid")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    config = Dinov2WithRegistersConfig.from_pretrained(
        source.MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = Dinov2WithRegistersModel.from_pretrained(
        source.MODEL_DIR, config=config, local_files_only=True,
        trust_remote_code=False, use_safetensors=True).eval().to("cpu")
    vectors = []
    with torch.inference_mode():
        for row in images["rows"]:
            vector = model(pixel_values=manual_tensor(source.ROOT / row["crop_file"]))\
                .last_hidden_state[0, 0].double().numpy()
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
    embeddings = np.stack(vectors)
    replay = np.maximum(0.0, 1.0 - np.clip(embeddings @ embeddings.T, -1.0, 1.0))
    np.fill_diagonal(replay, 0.0)
    max_difference = float(np.max(np.abs(replay - stored)))
    if not np.isfinite(max_difference) or max_difference > 1e-5:
        raise ValueError(f"manual DINO matrix replay differs: {max_difference}")
    return {
        "id": f"HERBAL-CONTROL-0003-DINO-{split}-feature-audit",
        "status": "pass", "feature_sha256": source.digest(feature_path),
        "input_manifest_sha256": images_sha,
        "full_input_freeze_sha256": freeze_sha,
        "model_sha256": source.MODEL_FILES["model.safetensors"],
        "compared_distance_cells": 108 * 108,
        "max_abs_difference": max_difference,
        "method": "manual PIL bicubic/NumPy normalization and direct Dinov2WithRegistersModel CLS cosine",
        "limit": "Numerical feature replay only; chapter labels and botanical truth are separate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    result = audit(args.split, args.features, args.selection, args.development_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "max_abs_difference": result["max_abs_difference"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
