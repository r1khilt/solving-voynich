"""Post-result DINOv2 feasibility probe on the exposed HERBAL-CONTROL-0002 panel.

This must never be reported as a prospective or independent historical test.
It loads only pinned local safetensors and built-in Transformers code; no remote
model code or network access is allowed at inference time.
"""

from __future__ import annotations

import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from herbal_control_0002_match import (
    MANUSCRIPTS, best_triplets, induced_retrieval, raw_retrieval, validate,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/raw/herbal_control_0003/models/dinov2-with-registers-base"
MODEL_SHA256 = "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362"
CONFIG_SHA256 = "6af60aa760138fc90db0ba37b7701730b140b9bf1742514412d03050e72bf7e0"
PROCESSOR_SHA256 = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
IMAGES = ROOT / "data/manifests/herbal_control_0002_images.json"
SOURCES = ROOT / "data/manifests/herbal_control_0002_sources.json"
OUTPUT = ROOT / "results/HERBAL-CONTROL-0002/dinov2_exploratory.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    expected = {"model.safetensors": MODEL_SHA256, "config.json": CONFIG_SHA256,
                "preprocessor_config.json": PROCESSOR_SHA256}
    for name, sha in expected.items():
        if digest(MODEL_DIR / name) != sha:
            raise ValueError(f"local model file changed: {name}")
    images = json.loads(IMAGES.read_text())
    if images["source_manifest_sha256"] != digest(SOURCES):
        raise ValueError("source manifest changed")
    rows = images["rows"]
    for row in rows:
        if digest(ROOT / row["source_file"]) != row["source_file_sha256"]:
            raise ValueError(f"source changed: {row['source_file']}")
        if digest(ROOT / row["crop_file"]) != row["crop_sha256"]:
            raise ValueError(f"crop changed: {row['crop_file']}")
    # CPU makes the probe reproducible in the current shell, where MPS is unavailable.
    torch.set_num_threads(min(8, torch.get_num_threads()))
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=False, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=False, local_files_only=True,
        use_safetensors=True).eval().to("cpu")
    start = time.monotonic()
    features = []
    with torch.inference_mode():
        for row in rows:
            picture = Image.open(ROOT / row["crop_file"]).convert("RGB")
            tensor = processor(images=picture, return_tensors="pt",
                               size={"height": 224, "width": 224},
                               do_center_crop=False)
            vec = model(**tensor).last_hidden_state[0, 0].float()
            vec = torch.nn.functional.normalize(vec, dim=0)
            features.append(vec.numpy())
    feature_array = np.stack(features).astype(np.float64)
    similarity = np.clip(feature_array @ feature_array.T, -1.0, 1.0)
    matrix = np.maximum(0.0, 1.0 - similarity)
    np.fill_diagonal(matrix, 0.0)
    distances = matrix.tolist()
    groups = validate(rows, distances)
    raw = raw_retrieval(rows, distances, groups)
    triplets, best, second, medians = best_triplets(distances, groups)
    induced = induced_retrieval(rows, triplets)
    pairwise_correct = 0
    pairwise_maps = {}
    for i, a in enumerate(MANUSCRIPTS):
        for b in MANUSCRIPTS[i + 1:]:
            ia, ib = groups[a], groups[b]
            costs = matrix[np.ix_(ia, ib)]
            best_perm = min(permutations(range(6)),
                            key=lambda perm: sum(costs[x, perm[x]] for x in range(6)))
            pairs = [(ia[x], ib[best_perm[x]]) for x in range(6)]
            correct = sum(rows[x]["chapter_class"] == rows[y]["chapter_class"]
                          for x, y in pairs)
            pairwise_correct += 2 * correct
            pairwise_maps[f"{a}-{b}"] = [{"a": rows[x]["chapter_class"],
                                           "b": rows[y]["chapter_class"]}
                                          for x, y in pairs]
    return {
        "id": "HERBAL-CONTROL-0002-DINO-EXPLORATORY",
        "status": "post-result-feasibility-only-not-confirmatory",
        "input_images_sha256": digest(IMAGES),
        "model_repository": "facebook/dinov2-with-registers-base",
        "model_revision": "a1d738ccfa7ae170945f210395d99dde8adb1805",
        "model_files_sha256": expected,
        "preprocessing": "white-padded source crop resized directly to 224x224; no center crop; pinned BitImageProcessor rescale and ImageNet normalization",
        "feature": "final CLS token, float32 L2 normalization; pair distance=max(0,1-cosine)",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
        "device": "cpu", "runtime_seconds": round(time.monotonic() - start, 3),
        "raw_nearest_correct_out_of_36": sum(x["correct"] for x in raw),
        "three_way_correct_out_of_36": sum(x["correct"] for x in induced),
        "complete_triplets_out_of_6": sum(len({rows[x]["chapter_class"] for x in t}) == 1 for t in triplets),
        "pairwise_one_to_one_correct_out_of_36": pairwise_correct,
        "best_three_way_cost": best, "runner_up_cost": second,
        "ordered_pair_medians": medians,
        "triplets": [[{"manuscript": rows[x]["manuscript"],
                       "chapter": rows[x]["chapter_class"]} for x in t] for t in triplets],
        "pairwise_maps": pairwise_maps,
        "distance_matrix": distances,
    }


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("no arguments")
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "raw_nearest_correct_out_of_36", "three_way_correct_out_of_36",
        "pairwise_one_to_one_correct_out_of_36", "complete_triplets_out_of_6",
        "runtime_seconds")}, sort_keys=True))
