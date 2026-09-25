"""Frozen folio-only challenger features for HERBAL-CONTROL-0003.

Development fits six maps from its 24 known classes. Evaluation reads only
the remote-frozen development maps and requires audited folio and primary
image development selections before scoring any evaluation folio positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from voynich.herbal_folio import FEATURE_METHOD, directed_matrix, fit_development_maps


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def images_path(split: str) -> Path:
    return ROOT / f"data/manifests/herbal_control_0003_{split}_images.json"


def _check_selection(selection_path: Path, audit_path: Path, *,
                     selection_id: str, audit_id: str,
                     freeze_sha: str, feature_method: str | None = None) -> dict:
    selection_sha = digest(selection_path)
    selection = json.loads(selection_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (selection.get("id") != selection_id
            or selection.get("status") != "development-score-produced-awaiting-audit-and-push"
            or selection.get("panel_manifest_sha256") != PANEL_SHA256
            or selection.get("full_input_freeze_sha256") != freeze_sha
            or audit.get("id") != audit_id or audit.get("status") != "pass"
            or audit.get("score_sha256") != selection_sha
            or audit.get("full_input_freeze_sha256") != freeze_sha
            or audit.get("feature_sha256") != selection.get("development_features_sha256")
            or audit.get("primary_method") != selection.get("primary_method")
            or (feature_method is not None
                and selection.get("feature_method") != feature_method)):
        raise ValueError(f"development selection/audit missing or changed: {selection_id}")
    return selection


def preflight(split: str, *, folio_features_path: Path | None = None,
              folio_feature_audit_path: Path | None = None,
              folio_selection_path: Path | None = None,
              folio_audit_path: Path | None = None,
              image_selection_path: Path | None = None,
              image_audit_path: Path | None = None) -> tuple[dict, dict, str, str]:
    if split not in ("development", "evaluation"):
        raise ValueError("split must be development or evaluation")
    panel_raw = PANEL.read_bytes()
    if hashlib.sha256(panel_raw).hexdigest() != PANEL_SHA256:
        raise ValueError("active panel changed")
    if not FREEZE.is_file() or not all(images_path(name).is_file()
                                       for name in ("development", "evaluation")):
        raise ValueError("complete 216/216 input freeze or image manifests missing")
    freeze_sha = digest(FREEZE)
    freeze = json.loads(FREEZE.read_text())
    image_sha = digest(images_path(split))
    other_split = "evaluation" if split == "development" else "development"
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES)
            or freeze.get(f"{split}_images_sha256") != image_sha
            or freeze.get(f"{other_split}_images_sha256") != digest(images_path(other_split))):
        raise ValueError("complete 216/216 input freeze missing or changed")
    if split == "evaluation":
        required = (folio_features_path, folio_feature_audit_path,
                    folio_selection_path, folio_audit_path,
                    image_selection_path, image_audit_path)
        if any(path is None for path in required):
            raise ValueError("evaluation requires audited folio features and both development selections")
        folio_selection = _check_selection(
            folio_selection_path, folio_audit_path,
            selection_id="HERBAL-CONTROL-0003-FOLIO-development-selection",
            audit_id="HERBAL-CONTROL-0003-FOLIO-independent-development-score-audit",
            freeze_sha=freeze_sha, feature_method=FEATURE_METHOD)
        _check_selection(
            image_selection_path, image_audit_path,
            selection_id="HERBAL-CONTROL-0003-development-selection",
            audit_id="HERBAL-CONTROL-0003-independent-development-score-audit",
            freeze_sha=freeze_sha)
        if folio_selection.get("development_features_sha256") != digest(folio_features_path):
            raise ValueError("frozen folio development features changed")
        feature_audit = json.loads(folio_feature_audit_path.read_text())
        if (feature_audit.get("id") != "HERBAL-CONTROL-0003-FOLIO-development-feature-audit"
                or feature_audit.get("status") != "pass"
                or feature_audit.get("feature_sha256") != digest(folio_features_path)
                or feature_audit.get("full_input_freeze_sha256") != freeze_sha
                or feature_audit.get("input_manifest_sha256") != digest(images_path("development"))
                or feature_audit.get("compared_distance_cells") != 108 * 108
                or feature_audit.get("compared_map_parameters") != 12):
            raise ValueError("folio development feature audit missing or changed")
    panel = json.loads(panel_raw)
    images = json.loads(images_path(split).read_text())
    if (images.get("id") != f"HERBAL-CONTROL-0003-{split}"
            or images.get("status") != f"complete-pre-score-{split}-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108
            or len(images.get("rows", [])) != 108):
        raise ValueError(f"{split} images are incomplete or changed")
    expected = {(role, item["chapter"], manuscript): item["pages"][manuscript]["pageid"]
                for role in (f"{split}_known", f"{split}_unknown")
                for item in panel["roles"][role]
                for manuscript in MANUSCRIPTS}
    rows = images["rows"]
    actual = {(row["role"], row["chapter_class"], row["manuscript"]): row["source_pageid"]
              for row in rows}
    if len(expected) != 108 or len(actual) != 108 or actual != expected:
        raise ValueError("image rows do not exactly match frozen panel")
    for row in rows:
        crop = ROOT / row["crop_file"]
        if not crop.is_file() or digest(crop) != row["crop_sha256"]:
            raise ValueError(f"crop changed: {row['crop_file']}")
    return panel, images, image_sha, freeze_sha


def extract(split: str, *, folio_features_path: Path | None = None,
            folio_feature_audit_path: Path | None = None,
            folio_selection_path: Path | None = None,
            folio_audit_path: Path | None = None,
            image_selection_path: Path | None = None,
            image_audit_path: Path | None = None) -> dict:
    panel, images, image_sha, freeze_sha = preflight(
        split, folio_features_path=folio_features_path,
        folio_feature_audit_path=folio_feature_audit_path,
        folio_selection_path=folio_selection_path,
        folio_audit_path=folio_audit_path,
        image_selection_path=image_selection_path,
        image_audit_path=image_audit_path)
    if split == "development":
        fits = fit_development_maps(panel)
    else:
        development = json.loads(folio_features_path.read_text())
        if (development.get("id") != "HERBAL-CONTROL-0003-development"
                or development.get("feature_method") != FEATURE_METHOD
                or development.get("full_input_freeze_sha256") != freeze_sha
                or development.get("input_manifest_sha256") != digest(images_path("development"))):
            raise ValueError("frozen folio fit provenance differs")
        fits = development["folio_maps"]
    row_order = [{key: row[key] for key in
                  ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256")}
                 for row in images["rows"]]
    matrix = directed_matrix(panel, split, row_order, fits)
    return {
        "id": f"HERBAL-CONTROL-0003-{split}",
        "feature_method": FEATURE_METHOD,
        "input_manifest_sha256": image_sha,
        "full_input_freeze_sha256": freeze_sha,
        "panel_manifest_sha256": PANEL_SHA256,
        "row_order": row_order,
        "distance_matrix": matrix,
        **({"folio_maps": fits} if split == "development" else {
            "folio_development_features_sha256": digest(folio_features_path),
            "folio_development_feature_audit_sha256": digest(folio_feature_audit_path),
            "development_selection_sha256": digest(folio_selection_path),
            "development_audit_sha256": digest(folio_audit_path),
            "image_development_selection_sha256": digest(image_selection_path),
            "image_development_audit_sha256": digest(image_audit_path),
        }),
    }


def verify_evaluation_feature_gate(feature_path: Path, *,
                                   folio_features_path: Path,
                                   folio_feature_audit_path: Path,
                                   folio_selection_path: Path,
                                   folio_audit_path: Path,
                                   image_selection_path: Path,
                                   image_audit_path: Path) -> None:
    """Recheck both development locks against a saved evaluation matrix."""
    _, _, image_sha, freeze_sha = preflight(
        "evaluation", folio_features_path=folio_features_path,
        folio_feature_audit_path=folio_feature_audit_path,
        folio_selection_path=folio_selection_path,
        folio_audit_path=folio_audit_path,
        image_selection_path=image_selection_path,
        image_audit_path=image_audit_path)
    feature = json.loads(feature_path.read_text())
    if (feature.get("id") != "HERBAL-CONTROL-0003-evaluation"
            or feature.get("feature_method") != FEATURE_METHOD
            or feature.get("input_manifest_sha256") != image_sha
            or feature.get("full_input_freeze_sha256") != freeze_sha
            or feature.get("folio_development_features_sha256") != digest(folio_features_path)
            or feature.get("folio_development_feature_audit_sha256") != digest(folio_feature_audit_path)
            or feature.get("development_selection_sha256") != digest(folio_selection_path)
            or feature.get("development_audit_sha256") != digest(folio_audit_path)
            or feature.get("image_development_selection_sha256") != digest(image_selection_path)
            or feature.get("image_development_audit_sha256") != digest(image_audit_path)):
        raise ValueError("evaluation folio feature development gates differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--folio-development-features", type=Path)
    parser.add_argument("--folio-feature-audit", type=Path)
    parser.add_argument("--folio-selection", type=Path)
    parser.add_argument("--folio-audit", type=Path)
    parser.add_argument("--image-selection", type=Path)
    parser.add_argument("--image-audit", type=Path)
    args = parser.parse_args()
    result = extract(
        args.split, folio_features_path=args.folio_development_features,
        folio_feature_audit_path=args.folio_feature_audit,
        folio_selection_path=args.folio_selection,
        folio_audit_path=args.folio_audit,
        image_selection_path=args.image_selection,
        image_audit_path=args.image_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split": args.split, "rows": len(result["row_order"])}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"HERBAL-CONTROL-0003 folio feature refusal: {error}", file=sys.stderr)
        raise SystemExit(1) from None
