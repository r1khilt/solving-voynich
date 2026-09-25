"""Independent folio parser, Theil–Sen fit and full directed-matrix replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from scripts import herbal_control_0003_folio_features as source


FOLIO_RE = re.compile(r"\bf\.(\d+)(r|v)(?:\.|\b)")
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def parse_position(title: str) -> int:
    found = list(FOLIO_RE.finditer(title))
    if len(found) != 1:
        raise ValueError("audit cannot identify exactly one folio coordinate")
    return int(found[0].group(1)) * 2 + int(found[0].group(2) == "v")


def replay_maps(panel: dict) -> dict[str, dict[str, float]]:
    known = panel["roles"]["development_known"]
    if len(known) != 24:
        raise ValueError("audit expected 24 development-known chapters")
    values = {manuscript: np.array([
        parse_position(row["pages"][manuscript]["title"]) for row in known], dtype=float)
        for manuscript in MANUSCRIPTS}
    if any(len(np.unique(value)) != 24 for value in values.values()):
        raise ValueError("audit found repeated development folio positions")
    left, right = np.triu_indices(24, k=1)
    maps = {}
    for query in MANUSCRIPTS:
        for reference in MANUSCRIPTS:
            if query == reference:
                continue
            x, y = values[query], values[reference]
            slopes = (y[right] - y[left]) / (x[right] - x[left])
            slope = float(np.median(slopes))
            intercept = float(np.median(y - slope * x))
            if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
                raise ValueError("audit found invalid folio map")
            maps[f"{query}->{reference}"] = {"slope": slope, "intercept": intercept}
    return maps


def replay_matrix(panel: dict, split: str, rows: list[dict],
                  maps: dict[str, dict[str, float]]) -> np.ndarray:
    positions = {}
    for role in (f"{split}_known", f"{split}_unknown"):
        for item in panel["roles"][role]:
            for manuscript in MANUSCRIPTS:
                positions[(role, item["chapter"], manuscript)] = parse_position(
                    item["pages"][manuscript]["title"])
    keys = [(row["role"], row["chapter_class"], row["manuscript"]) for row in rows]
    if len(keys) != 108 or len(set(keys)) != 108 or set(keys) != set(positions):
        raise ValueError("audit found wrong folio row order")
    values = np.zeros((108, 108), dtype=float)
    for i, query in enumerate(keys):
        for j, reference in enumerate(keys):
            if query[2] != reference[2]:
                fit = maps[f"{query[2]}->{reference[2]}"]
                values[i, j] = abs(fit["slope"] * positions[query] +
                                   fit["intercept"] - positions[reference])
    return values


def audit(split: str, feature_path: Path, *,
          folio_features_path: Path | None = None,
          folio_feature_audit_path: Path | None = None,
          folio_selection_path: Path | None = None,
          folio_audit_path: Path | None = None,
          image_selection_path: Path | None = None,
          image_audit_path: Path | None = None) -> dict:
    panel, images, image_sha, freeze_sha = source.preflight(
        split, folio_features_path=folio_features_path,
        folio_feature_audit_path=folio_feature_audit_path,
        folio_selection_path=folio_selection_path,
        folio_audit_path=folio_audit_path,
        image_selection_path=image_selection_path,
        image_audit_path=image_audit_path)
    features = json.loads(feature_path.read_text())
    if (features.get("id") != f"HERBAL-CONTROL-0003-{split}"
            or features.get("feature_method") != source.FEATURE_METHOD
            or features.get("panel_manifest_sha256") != source.PANEL_SHA256
            or features.get("input_manifest_sha256") != image_sha
            or features.get("full_input_freeze_sha256") != freeze_sha):
        raise ValueError("folio feature provenance differs")
    rows = features.get("row_order", [])
    if len(rows) != 108 or any(
            any(actual.get(key) != expected[key] for key in
                ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256"))
            for actual, expected in zip(rows, images["rows"], strict=True)):
        raise ValueError("folio feature row order differs")
    expected_maps = replay_maps(panel)
    if split == "development":
        maps = features.get("folio_maps", {})
    else:
        if (features.get("folio_development_features_sha256") != source.digest(folio_features_path)
                or features.get("folio_development_feature_audit_sha256") !=
                source.digest(folio_feature_audit_path)
                or features.get("development_selection_sha256") != source.digest(folio_selection_path)
                or features.get("development_audit_sha256") != source.digest(folio_audit_path)
                or features.get("image_development_selection_sha256") != source.digest(image_selection_path)
                or features.get("image_development_audit_sha256") != source.digest(image_audit_path)):
            raise ValueError("evaluation folio development gate differs")
        maps = json.loads(folio_features_path.read_text()).get("folio_maps", {})
    if set(maps) != set(expected_maps):
        raise ValueError("folio map keys differ")
    max_map_difference = max(abs(maps[key][field] - expected_maps[key][field])
                             for key in maps for field in ("slope", "intercept"))
    if max_map_difference > 1e-9:
        raise ValueError(f"folio Theil-Sen maps differ: {max_map_difference}")
    saved = np.asarray(features.get("distance_matrix"), dtype=float)
    expected = replay_matrix(panel, split, rows, expected_maps)
    if saved.shape != (108, 108) or not np.isfinite(saved).all():
        raise ValueError("folio matrix shape or values differ")
    max_difference = float(np.max(np.abs(saved - expected)))
    if max_difference > 1e-8:
        raise ValueError(f"folio directed matrix differs: {max_difference}")
    return {
        "id": f"HERBAL-CONTROL-0003-FOLIO-{split}-feature-audit",
        "status": "pass", "feature_sha256": source.digest(feature_path),
        "full_input_freeze_sha256": freeze_sha,
        "input_manifest_sha256": image_sha,
        "compared_distance_cells": 108 * 108,
        "compared_map_parameters": 12,
        "max_abs_map_difference": max_map_difference,
        "max_abs_matrix_difference": max_difference,
        "method": "independent regex/NumPy upper-triangle median map and all-cell matrix replay",
        "limit": "Numerical metadata replay only; image-specific value and historical identities are separate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--folio-development-features", type=Path)
    parser.add_argument("--folio-feature-audit", type=Path)
    parser.add_argument("--folio-selection", type=Path)
    parser.add_argument("--folio-audit", type=Path)
    parser.add_argument("--image-selection", type=Path)
    parser.add_argument("--image-audit", type=Path)
    args = parser.parse_args()
    result = audit(
        args.split, args.features,
        folio_features_path=args.folio_development_features,
        folio_feature_audit_path=args.folio_feature_audit,
        folio_selection_path=args.folio_selection,
        folio_audit_path=args.folio_audit,
        image_selection_path=args.image_selection,
        image_audit_path=args.image_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "max_abs_matrix_difference": result["max_abs_matrix_difference"]}))


if __name__ == "__main__":
    main()
