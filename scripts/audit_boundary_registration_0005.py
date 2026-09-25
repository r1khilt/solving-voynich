"""Independently replay the frozen cohort decision and saved affine scores.

This audits the 64 saved matches at their chosen scales/positions, not the
original 441-scale exhaustive search. It makes no word-identity claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scripts.build_boundary_geometry_train import read_boxes
from scripts.register_boundary_ink_pages import BOX_DIR, ROOT, source_mask, target_map


CONFIG = ROOT / "data/manifests/boundary_registration_0005.json"
RESULT = ROOT / "results/BOUNDARY-CHANNEL-0005/prospective_registration.json"
OUTPUT = ROOT / "results/BOUNDARY-CHANNEL-0005/prospective_audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(matrix: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Use a column-total formula, distinct from the production row loop."""
    names = sorted(matrix)
    count = len(names)
    totals = {source: math.fsum(matrix[target][source]["score"] for target in names)
              for source in names}
    checks = {}
    for target in names:
        centered = {source: (count * matrix[target][source]["score"] - totals[source])
                    / (count - 1) for source in names}
        own = centered[target]
        other = max(centered[source] for source in names if source != target)
        raw_order = sorted(names, key=lambda source: matrix[target][source]["score"],
                           reverse=True)
        centered_order = sorted(names, key=centered.__getitem__, reverse=True)
        checks[target] = {
            "centered": centered,
            "raw_rank": raw_order.index(target) + 1,
            "cohort_rank": centered_order.index(target) + 1,
            "own": own,
            "best_other": other,
            "margin": own - other,
            "best_impostor": next(source for source in centered_order if source != target),
        }
    return checks


def run() -> dict:
    config = json.loads(CONFIG.read_text())
    result = json.loads(RESULT.read_text())
    names = [item["folio"] for item in config["folios"]]
    expected = set(names)
    matrix = result["raw_matches"]
    if len(names) != 8 or len(expected) != 8 or set(matrix) != expected:
        raise AssertionError("Wrong number or identity of target pages")
    if (result["config_sha256"] != digest(CONFIG) or
            result["search_code_sha256"] != config["search_code_sha256"] or
            result["split_sha256"] != config["split_sha256"] or
            result["yale_manifest_sha256"] != config["yale_iiif_manifest_sha256"] or
            result["source_archive_revision"] != config["source_archive_revision"]):
        raise AssertionError("Registered inputs differ from result")
    source_archive = ROOT / "data/raw/external/voynich-units"
    revision = subprocess.check_output(["git", "-C", str(source_archive),
                                        "rev-parse", "HEAD"], text=True).strip()
    if revision != config["source_archive_revision"]:
        raise AssertionError("Source archive revision changed")
    if digest(ROOT / "scripts/register_boundary_ink_pages.py") != config["search_code_sha256"]:
        raise AssertionError("Frozen search changed")
    if (digest(ROOT / "data/manifests/zl3b_split.json") != config["split_sha256"] or
            digest(ROOT / "data/raw/yale_ms408_iiif_manifest.json") !=
            config["yale_iiif_manifest_sha256"]):
        raise AssertionError("Frozen split or Yale manifest changed")

    scans = {}
    sources = {}
    sizes = {}
    for item in config["folios"]:
        folio = item["folio"]
        path = ROOT / config["scan_dir"] / f"{folio}.jpg"
        if digest(path) != result["scan_sha256"][folio]:
            raise AssertionError(f"Scan changed: {folio}")
        image = cv2.imread(str(path))
        if image is None or image.shape[1] != 700:
            raise AssertionError(f"Invalid scan: {folio}")
        sizes[folio] = [int(image.shape[1]), int(image.shape[0])]
        scans[folio] = target_map(image)
        source_path = BOX_DIR / f"{folio}.js"
        if digest(source_path) != item["box_sha256"]:
            raise AssertionError(f"Source box changed: {folio}")
        sources[folio] = source_mask(read_boxes(source_path))
    try:
        target_map(np.full((900, 700, 3), 255, dtype=np.uint8))
    except ValueError:
        blank_rejected = True
    else:
        raise AssertionError("Blank target accepted")

    fixed_affine_checks = 0
    for target, row in matrix.items():
        if set(row) != expected:
            raise AssertionError(f"Incomplete wrong-page grid: {target}")
        for source, match in row.items():
            sx, sy = match["sx"], match["sy"]
            if (not any(abs(sx - value / 100) < 1e-12 for value in range(60, 101, 2)) or
                    not any(abs(sy - value / 100) < 1e-12 for value in range(60, 101, 2)) or
                    not 0 <= match["tx"] <= 260 or not 0 <= match["ty"] <= 150):
                raise AssertionError(f"Affine outside frozen grid: {target}/{source}")
            mask = sources[source]
            resized = cv2.resize(mask, (round(mask.shape[1] * sx),
                                        round(mask.shape[0] * sy)),
                                 interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(scans[target], resized, cv2.TM_CCOEFF_NORMED)
            tx, ty = match["tx"], match["ty"]
            replay = float(response[ty, tx])
            if not math.isclose(replay, match["score"], abs_tol=1e-6, rel_tol=0):
                raise AssertionError(f"Saved score does not replay: {target}/{source}")
            cropped = response[:min(151, response.shape[0]), :min(261, response.shape[1])]
            if replay < float(np.max(cropped)) - 1e-6:
                raise AssertionError(f"Saved translation is not scale-local best: {target}/{source}")
            fixed_affine_checks += 1

    checks = decision(matrix)
    if set(result["cohort_scores"]) != expected or set(result["rankings"]) != expected:
        raise AssertionError("Saved cohort output has wrong keys")
    for target, item in checks.items():
        if set(result["cohort_scores"][target]) != expected:
            raise AssertionError(f"Saved cohort row incomplete: {target}")
        for source, value in item["centered"].items():
            if not math.isclose(value, result["cohort_scores"][target][source],
                                abs_tol=1e-12, rel_tol=0):
                raise AssertionError(f"Cohort score mismatch: {target}/{source}")
        saved = result["rankings"][target]
        for key, value in (("raw_rank", item["raw_rank"]),
                           ("cohort_rank", item["cohort_rank"]),
                           ("cohort_own_score", item["own"]),
                           ("cohort_best_impostor_score", item["best_other"]),
                           ("cohort_margin", item["margin"])):
            if not math.isclose(saved[key], value, abs_tol=1e-12, rel_tol=0):
                raise AssertionError(f"Saved decision mismatch: {target}/{key}")
        if saved["raw_affine"] != {key: matrix[target][target][key] for key in
                                   ("sx", "tx", "sy", "ty")}:
            raise AssertionError(f"Saved own affine mismatch: {target}")
    pass_count = sum(item["cohort_rank"] == 1 and item["margin"] >= .03
                     for item in checks.values())
    gate = "PASS" if pass_count == len(names) else "FAIL"
    if gate != result["gate"]:
        raise AssertionError("Registered gate mismatch")

    counterfeit = {target: {source: dict(match) for source, match in row.items()}
                   for target, row in matrix.items()}
    counterfeit[names[0]] = {source: dict(match) for source, match in matrix[names[1]].items()}
    counterfeit_checks = decision(counterfeit)
    counterfeit_pass = sum(item["cohort_rank"] == 1 and item["margin"] >= .03
                           for item in counterfeit_checks.values())
    control = result["duplicate_target_control"]
    if (control["missing_folio"] != names[0] or control["duplicate_of"] != names[1] or
            control["passing_pages_after_duplicate"] != counterfeit_pass or
            not control["rejects_complete_claim"] or counterfeit_pass >= len(names)):
        raise AssertionError("Duplicate-target control mismatch")

    audit = {
        "status": "audited_registered_batch_identity",
        "result_sha256": digest(RESULT),
        "config_sha256": digest(CONFIG),
        "frozen_search_sha256": config["search_code_sha256"],
        "fixed_affine_raw_score_replays": fixed_affine_checks,
        "blank_rejected": blank_rejected,
        "duplicate_target_passing_pages": counterfeit_pass,
        "registered_primary_gate": gate,
        "passing_pages": pass_count,
        "checks": {name: {"raw_rank": item["raw_rank"],
                          "cohort_rank": item["cohort_rank"],
                          "margin": item["margin"],
                          "best_impostor": item["best_impostor"],
                          "scan_size": sizes[name]}
                   for name, item in checks.items()},
        "limit": "Independently replayed centered-score arithmetic and every saved score at its selected affine; did not rerun the exhaustive 441-scale search or verify exact word identity.",
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    audit = run()
    print(json.dumps({key: audit[key] for key in
                      ("registered_primary_gate", "passing_pages",
                       "fixed_affine_raw_score_replays", "blank_rejected",
                       "duplicate_target_passing_pages")}, indent=2))
