"""Score HERBAL-CONTROL-0003 evaluation once with frozen development choices.

No feature extraction, threshold selection or method choice occurs here. The
complete source/crop freeze, audited development selection and evaluation
feature matrix are mandatory inputs. The result remains subject to a separate
independent audit and the registered historical-image claim limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path

from voynich.herbal_open_set import (
    MANUSCRIPTS,
    METHODS,
    alternate_prediction_margin,
    batch_predictions,
    class_bootstrap,
    directional_balanced_accuracy,
    fuse_costs,
    independent_predictions,
    prepare_direction_panels,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
DEVELOPMENT_IMAGES = ROOT / "data/manifests/herbal_control_0003_development_images.json"
EVALUATION_IMAGES = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
FEATURE_METHOD = "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classes(panel: dict) -> tuple[list[str], list[str], set[tuple[str, str, str]]]:
    known = [item["chapter"] for item in panel["roles"]["evaluation_known"]]
    unknown = [item["chapter"] for item in panel["roles"]["evaluation_unknown"]]
    if len(known) != 24 or len(unknown) != 12 or len(set(known + unknown)) != 36:
        raise ValueError("evaluation class counts/order changed")
    keys = {(f"evaluation_{kind}", chapter, manuscript)
            for kind, names in (("known", known), ("unknown", unknown))
            for chapter in names for manuscript in MANUSCRIPTS}
    if len(keys) != 108:
        raise ValueError("evaluation crop keys are not unique")
    return known, unknown, keys


def _validate_matrix(features: dict, images: dict,
                     expected_keys: set[tuple[str, str, str]],
                     selection_path: Path, audit_path: Path,
                     feature_method: str = FEATURE_METHOD) -> None:
    if (features.get("id") != "HERBAL-CONTROL-0003-evaluation"
            or features.get("feature_method") != feature_method
            or features.get("input_manifest_sha256") != digest(EVALUATION_IMAGES)
            or features.get("full_input_freeze_sha256") != digest(FREEZE)
            or features.get("development_selection_sha256") != digest(selection_path)
            or features.get("development_audit_sha256") != digest(audit_path)):
        raise ValueError("evaluation feature matrix provenance differs")
    feature_rows = features.get("row_order", [])
    image_rows = images["rows"]
    if len(feature_rows) != 108:
        raise ValueError("evaluation feature matrix row count differs")
    keys = set()
    for feature, image in zip(feature_rows, image_rows, strict=True):
        for field in ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256"):
            if feature.get(field) != image[field]:
                raise ValueError(f"evaluation feature row/crop differs: {field}")
        keys.add((feature["role"], feature["chapter_class"], feature["manuscript"]))
    if keys != expected_keys or len(keys) != 108:
        raise ValueError("evaluation feature rows differ from frozen panel")
    matrix = features.get("distance_matrix", [])
    if len(matrix) != 108:
        raise ValueError("evaluation matrix is not 108 square")
    for i, row in enumerate(matrix):
        if len(row) != 108:
            raise ValueError("evaluation matrix row has wrong width")
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError("invalid evaluation feature distance")
            if i == j and abs(value) > 1e-5:
                raise ValueError("nonzero evaluation feature diagonal")
            if j < i and abs(value - matrix[j][i]) > 1e-5:
                raise ValueError("asymmetric evaluation feature matrix")


def _distance(left: str, right: str) -> int:
    old = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(old[j] + 1, new[j - 1] + 1,
                           old[j - 1] + (a != b)))
        old = new
    return old[-1]


def near_name_unknowns(known: list[str], unknown: list[str]) -> list[str]:
    result = []
    for name in unknown:
        token = name.casefold()
        if any(1 - _distance(token, other.casefold()) /
               max(len(token), len(other.casefold())) > 0.5 for other in known):
            result.append(name)
    return result


def _load_gate(selection_path: Path, audit_path: Path, *,
               selection_id: str = "HERBAL-CONTROL-0003-development-selection",
               audit_id: str = "HERBAL-CONTROL-0003-independent-development-score-audit",
               feature_method: str = FEATURE_METHOD) -> dict:
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("frozen panel hash changed")
    freeze = json.loads(FREEZE.read_text())
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES)
            or freeze.get("development_images_sha256") != digest(DEVELOPMENT_IMAGES)
            or freeze.get("evaluation_images_sha256") != digest(EVALUATION_IMAGES)):
        raise ValueError("complete source/crop freeze is missing or changed")
    selection = json.loads(selection_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (selection.get("id") != selection_id
            or selection.get("status") != "development-score-produced-awaiting-audit-and-push"
            or selection.get("panel_manifest_sha256") != PANEL_SHA256
            or selection.get("full_input_freeze_sha256") != digest(FREEZE)
            or audit.get("id") != audit_id
            or audit.get("status") != "pass"
            or audit.get("score_sha256") != digest(selection_path)
            or audit.get("full_input_freeze_sha256") != digest(FREEZE)
            or audit.get("feature_sha256") != selection.get("development_features_sha256")
            or audit.get("primary_method") != selection.get("primary_method")
            or set(selection.get("methods", {})) != set(METHODS)
            or (feature_method != FEATURE_METHOD
                and selection.get("feature_method") != feature_method)):
        raise ValueError("audited development selection is missing or changed")
    for method in METHODS:
        if (audit.get("batch_macro_ba", {}).get(method) !=
                selection["methods"][method].get("batch_macro_ba")):
            raise ValueError("audit/method development score differs")
        for key in ("batch_threshold", "independent_threshold"):
            if not isfinite(selection["methods"][method][key]):
                raise ValueError("nonfinite frozen development threshold")
    return selection


def score(feature_path: Path, selection_path: Path, audit_path: Path, *,
          feature_method: str = FEATURE_METHOD,
          selection_id: str = "HERBAL-CONTROL-0003-development-selection",
          audit_id: str = "HERBAL-CONTROL-0003-independent-development-score-audit",
          result_id: str = "HERBAL-CONTROL-0003-evaluation-score") -> dict:
    selection = _load_gate(selection_path, audit_path, selection_id=selection_id,
                           audit_id=audit_id, feature_method=feature_method)
    panel = json.loads(PANEL.read_text())
    known, unknown, expected_keys = _classes(panel)
    images = json.loads(EVALUATION_IMAGES.read_text())
    if (images.get("id") != "HERBAL-CONTROL-0003-evaluation"
            or images.get("status") != "complete-pre-score-evaluation-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108 or len(images.get("rows", [])) != 108):
        raise ValueError("evaluation crop manifest incomplete or changed")
    for row in images["rows"]:
        if digest(ROOT / row["crop_file"]) != row["crop_sha256"]:
            raise ValueError(f"evaluation crop changed: {row['crop_file']}")
    features = json.loads(feature_path.read_text())
    _validate_matrix(features, images, expected_keys, selection_path, audit_path,
                     feature_method)
    first, second, truths, medians = prepare_direction_panels(
        features["distance_matrix"], features["row_order"], known, unknown,
        "evaluation", selection["normalization_medians"])
    if medians != selection["normalization_medians"]:
        raise ValueError("evaluation replaced development distance medians")
    methods = {}
    batch_predictions_by_method = {}
    independent_predictions_by_method = {}
    for method in METHODS:
        costs = {direction: fuse_costs(first[direction], second[direction], method)
                 for direction in MANUSCRIPTS}
        selected = selection["methods"][method]
        batch_tau = selected["batch_threshold"]
        independent_tau = selected["independent_threshold"]
        batch_by_direction = {}
        independent_by_direction = {}
        batch_predictions_by_method[method] = {}
        independent_predictions_by_method[method] = {}
        for direction in MANUSCRIPTS:
            predicted, _ = batch_predictions(costs[direction], batch_tau)
            control = independent_predictions(costs[direction], independent_tau)
            batch_predictions_by_method[method][direction] = predicted
            independent_predictions_by_method[method][direction] = control
            batch_by_direction[direction] = {
                "predictions": predicted,
                "known_correct": sum(predicted[i] == i for i in range(24)),
                "unknown_rejected": sum(value is None for value in predicted[24:]),
                "unknown_false_accepts": [
                    {"query_chapter": unknown[i - 24], "predicted_chapter": known[value]}
                    for i, value in enumerate(predicted) if i >= 24 and value is not None
                ],
                "balanced_accuracy": directional_balanced_accuracy(
                    predicted, truths[direction], 24, 12),
                "assignment_margin": alternate_prediction_margin(costs[direction], batch_tau),
            }
            independent_by_direction[direction] = {
                "predictions": control,
                "known_correct": sum(control[i] == i for i in range(24)),
                "unknown_rejected": sum(value is None for value in control[24:]),
                "balanced_accuracy": directional_balanced_accuracy(
                    control, truths[direction], 24, 12),
            }
        methods[method] = {
            "batch_threshold": batch_tau,
            "batch_by_direction": batch_by_direction,
            "batch_macro_ba": sum(row["balanced_accuracy"] for row in
                                  batch_by_direction.values()) / 3,
            "independent_threshold": independent_tau,
            "independent_by_direction": independent_by_direction,
            "independent_macro_ba": sum(row["balanced_accuracy"] for row in
                                        independent_by_direction.values()) / 3,
        }
    primary = selection["primary_method"]
    primary_predictions = batch_predictions_by_method[primary]
    ci = class_bootstrap(primary_predictions, truths, 24, 12)
    two_source_ci = class_bootstrap(primary_predictions, truths, 24, 12, [
        batch_predictions_by_method["single_first"],
        batch_predictions_by_method["single_second"],
    ]) if primary in ("two_mean", "two_min") else None
    one_to_one_ci = class_bootstrap(primary_predictions, truths, 24, 12, [
        independent_predictions_by_method[primary],
    ])
    primary_result = methods[primary]
    feasibility = (primary_result["batch_macro_ba"] >= 0.80 and all(
        primary_result["batch_by_direction"][direction]["known_correct"] >= 18
        and primary_result["batch_by_direction"][direction]["unknown_rejected"] >= 10
        for direction in MANUSCRIPTS))
    two_source = (two_source_ci is not None
                  and primary_result["batch_macro_ba"] >=
                  max(methods["single_first"]["batch_macro_ba"],
                      methods["single_second"]["batch_macro_ba"]) + 0.05
                  and two_source_ci["paired_improvement_ci_95"][0] > 0)
    one_to_one = (primary_result["batch_macro_ba"] >=
                  primary_result["independent_macro_ba"] + 0.05
                  and one_to_one_ci["paired_improvement_ci_95"][0] > 0)
    return {
        "id": result_id,
        "status": "evaluation-score-produced-awaiting-independent-audit",
        "panel_manifest_sha256": PANEL_SHA256,
        "full_input_freeze_sha256": digest(FREEZE),
        "development_selection_sha256": digest(selection_path),
        "development_audit_sha256": digest(audit_path),
        "evaluation_images_sha256": digest(EVALUATION_IMAGES),
        "evaluation_features_sha256": digest(feature_path),
        "known_classes": known,
        "unknown_classes": unknown,
        "near_name_unknown_classes": near_name_unknowns(known, unknown),
        "normalization_medians": medians,
        "primary_method": primary,
        "methods": methods,
        "primary_macro_ba_ci_95": ci["primary_ci_95"],
        "two_source_comparison": two_source_ci,
        "one_to_one_comparison": one_to_one_ci,
        "gates": {
            "historical_image_open_set_feasibility": feasibility,
            "two_source_benefit": two_source,
            "one_to_one_benefit": one_to_one,
        },
        "claim_limit": "Historical chapter-metadata evaluation only; no Voynich image/text inference.",
        **({"feature_method": feature_method} if feature_method != FEATURE_METHOD else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("development_selection", type=Path)
    parser.add_argument("development_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(args.features, args.development_selection, args.development_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
