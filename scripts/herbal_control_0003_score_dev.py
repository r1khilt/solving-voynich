"""Select HERBAL-CONTROL-0003 methods from the frozen development half only.

The command refuses partial or substituted crops and malformed feature output.
No evaluation feature matrix is an input; the held-out half is never scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path

from voynich.herbal_open_set import (
    METHODS,
    MANUSCRIPTS,
    alternate_prediction_margin,
    batch_predictions,
    choose_primary,
    choose_threshold,
    directional_balanced_accuracy,
    fuse_costs,
    independent_predictions,
    prepare_direction_panels,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
DEV_IMAGES = ROOT / "data/manifests/herbal_control_0003_development_images.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
EVAL_IMAGES = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _development_classes(panel: dict) -> tuple[list[str], list[str], set[tuple[str, str, str]]]:
    known = [row["chapter"] for row in panel["roles"]["development_known"]]
    unknown = [row["chapter"] for row in panel["roles"]["development_unknown"]]
    if len(known) != 24 or len(unknown) != 12 or len(set(known + unknown)) != 36:
        raise ValueError("frozen development class counts changed")
    keys = {(role, row["chapter"], manuscript)
            for role in ("development_known", "development_unknown")
            for row in panel["roles"][role] for manuscript in MANUSCRIPTS}
    if len(keys) != 108:
        raise ValueError("development image keys are not unique")
    return known, unknown, keys


def _validate_inputs(features: dict, images: dict, freeze: dict,
                     freeze_raw: bytes, expected_keys: set[tuple[str, str, str]]) -> None:
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES.read_bytes())
            or freeze.get("development_images_sha256") != digest(DEV_IMAGES.read_bytes())
            or freeze.get("evaluation_images_sha256") != digest(EVAL_IMAGES.read_bytes())):
        raise ValueError("complete pre-score input freeze is missing or changed")
    if (images.get("id") != "HERBAL-CONTROL-0003-development"
            or images.get("status") != "complete-pre-score-development-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108
            or len(images.get("rows", [])) != 108):
        raise ValueError("development crop manifest is not complete and frozen")
    image_order = images["rows"]
    keys = {(row["role"], row["chapter_class"], row["manuscript"]) for row in image_order}
    if keys != expected_keys or len(keys) != len(image_order):
        raise ValueError("development crop rows do not match panel")
    if (features.get("id") != "HERBAL-CONTROL-0003-development"
            or features.get("feature_method") !=
            "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"
            or features.get("input_manifest_sha256") != digest(DEV_IMAGES.read_bytes())
            or features.get("full_input_freeze_sha256") != digest(freeze_raw)):
        raise ValueError("feature matrix source or method does not match crops")
    feature_order = features.get("row_order", [])
    if len(feature_order) != 108:
        raise ValueError("feature matrix has wrong row count")
    for expected, actual in zip(image_order, feature_order, strict=True):
        for field in ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256"):
            if expected[field] != actual.get(field):
                raise ValueError(f"feature row differs from crop manifest: {field}")
    matrix = features.get("distance_matrix")
    if not isinstance(matrix, list) or len(matrix) != 108:
        raise ValueError("feature matrix is not 108×108")
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 108:
            raise ValueError(f"feature matrix row {i} has wrong width")
        if any(not isinstance(x, (int, float)) or not isfinite(x) or x < 0 for x in row):
            raise ValueError(f"feature matrix row {i} has invalid distance")
        if abs(row[i]) > 1e-5:
            raise ValueError(f"feature matrix diagonal not zero at {i}")
        for j in range(i):
            if abs(row[j] - matrix[j][i]) > 1e-5:
                raise ValueError(f"feature matrix is asymmetric at {i},{j}")


def score(feature_path: Path) -> dict:
    panel_raw = PANEL.read_bytes()
    if digest(panel_raw) != PANEL_SHA256:
        raise ValueError("panel hash mismatch")
    panel = json.loads(panel_raw)
    known, unknown, expected_keys = _development_classes(panel)
    image_raw = DEV_IMAGES.read_bytes()
    images = json.loads(image_raw)
    freeze_raw = FREEZE.read_bytes()
    freeze = json.loads(freeze_raw)
    feature_raw = feature_path.read_bytes()
    features = json.loads(feature_raw)
    _validate_inputs(features, images, freeze, freeze_raw, expected_keys)
    first, second, truths, medians = prepare_direction_panels(
        features["distance_matrix"], features["row_order"], known, unknown, "development")
    selection = choose_primary(first, second, truths, 24, 12)
    methods = {}
    for method in METHODS:
        panels = {direction: fuse_costs(first[direction], second[direction], method)
                  for direction in MANUSCRIPTS}
        batch_tau = selection["methods"][method]["threshold"]
        independent_tau, independent_dev_ba = choose_threshold(
            panels, truths, 24, 12, batch=False)
        batch = {}
        independent = {}
        for direction in MANUSCRIPTS:
            predictions, _ = batch_predictions(panels[direction], batch_tau)
            control = independent_predictions(panels[direction], independent_tau)
            batch[direction] = {
                "predictions": predictions,
                "known_correct": sum(predictions[i] == i for i in range(24)),
                "unknown_rejected": sum(p is None for p in predictions[24:]),
                "balanced_accuracy": directional_balanced_accuracy(predictions, truths[direction], 24, 12),
                "assignment_margin": alternate_prediction_margin(panels[direction], batch_tau),
            }
            independent[direction] = {
                "predictions": control,
                "known_correct": sum(control[i] == i for i in range(24)),
                "unknown_rejected": sum(p is None for p in control[24:]),
                "balanced_accuracy": directional_balanced_accuracy(control, truths[direction], 24, 12),
            }
        methods[method] = {
            "batch_threshold": batch_tau,
            "batch_macro_ba": selection["methods"][method]["development_macro_ba"],
            "batch_by_direction": batch,
            "independent_threshold": independent_tau,
            "independent_macro_ba": independent_dev_ba,
            "independent_by_direction": independent,
        }
    return {
        "id": "HERBAL-CONTROL-0003-development-selection",
        "status": "development-score-produced-awaiting-audit-and-push",
        "panel_manifest_sha256": PANEL_SHA256,
        "full_input_freeze_sha256": digest(freeze_raw),
        "development_images_sha256": digest(image_raw),
        "development_features_sha256": digest(feature_raw),
        "known_classes": known,
        "unknown_classes": unknown,
        "normalization_medians": medians,
        "primary_method": selection["primary_method"],
        "methods": methods,
        "claim_limit": "Development-only chapter retrieval; no evaluation or Voynich meaning claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"primary_method": result["primary_method"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
