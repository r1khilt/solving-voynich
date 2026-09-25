"""Freeze the metadata-only folio challenger on the development split.

Refuses scoring until the same full 216-source/216-crop freeze needed by the
image method exists. It parses no evaluation image rows or feature scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from voynich.herbal_folio_order import fit_development_maps, folio_distance_matrix
from voynich.herbal_open_set import (
    METHODS,
    MANUSCRIPTS,
    batch_predictions,
    choose_primary,
    choose_threshold,
    directional_balanced_accuracy,
    fuse_costs,
    independent_predictions,
    prepare_direction_panels,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
DEVELOPMENT = ROOT / "data/manifests/herbal_control_0003_development_images.json"
EVALUATION = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
PANEL_SHA256 = "fc2bd19a8eb25d16d8f8387a26b551e42c1f221a048d8f8b07a3f20f02c4b725"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_development() -> dict:
    if sha(PANEL) != PANEL_SHA256:
        raise ValueError("frozen panel hash differs")
    freeze = json.loads(FREEZE.read_text())
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != sha(SOURCES)
            or freeze.get("development_images_sha256") != sha(DEVELOPMENT)
            or freeze.get("evaluation_images_sha256") != sha(EVALUATION)):
        raise ValueError("complete source/crop input freeze missing or changed")
    panel = json.loads(PANEL.read_text())
    known = [item["chapter"] for item in panel["roles"]["development_known"]]
    unknown = [item["chapter"] for item in panel["roles"]["development_unknown"]]
    if len(known) != 24 or len(unknown) != 12 or len(set(known + unknown)) != 36:
        raise ValueError("development class lists changed")
    maps = fit_development_maps(panel)
    matrix, rows = folio_distance_matrix(panel, "development", maps)
    first, second, truths, medians = prepare_direction_panels(
        matrix, rows, known, unknown, "development")
    selection = choose_primary(first, second, truths, 24, 12)
    methods = {}
    for method in METHODS:
        panels = {direction: fuse_costs(first[direction], second[direction], method)
                  for direction in MANUSCRIPTS}
        tau = selection["methods"][method]["threshold"]
        independent_tau, independent_ba = choose_threshold(
            panels, truths, 24, 12, batch=False)
        by_direction = {}
        for direction in MANUSCRIPTS:
            batch, objective = batch_predictions(panels[direction], tau)
            independent = independent_predictions(panels[direction], independent_tau)
            by_direction[direction] = {
                "batch_predictions": batch,
                "batch_objective": objective,
                "batch_balanced_accuracy": directional_balanced_accuracy(
                    batch, truths[direction], 24, 12),
                "independent_predictions": independent,
                "independent_balanced_accuracy": directional_balanced_accuracy(
                    independent, truths[direction], 24, 12),
            }
        methods[method] = {
            "batch_threshold": tau,
            "batch_macro_ba": selection["methods"][method]["development_macro_ba"],
            "independent_threshold": independent_tau,
            "independent_macro_ba": independent_ba,
            "by_direction": by_direction,
        }
    return {
        "id": "HERBAL-CONTROL-0003-folio-development-selection",
        "status": "development-folio-score-produced-awaiting-audit-and-push",
        "panel_manifest_sha256": PANEL_SHA256,
        "full_input_freeze_sha256": sha(FREEZE),
        "known_classes": known,
        "unknown_classes": unknown,
        "ordered_manuscript_maps": maps,
        "normalization_medians": medians,
        "primary_method": selection["primary_method"],
        "methods": methods,
        "claim_limit": "Folio-title metadata only, with evaluation metadata exposure disclosed; no image or Voynich meaning claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score_development()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"primary_method": result["primary_method"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
