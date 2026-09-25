"""Check the frozen registration result and its nonlinguistic controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.build_boundary_geometry_train import read_boxes
from scripts.register_boundary_ink_pages import (
    BOX_DIR, REGISTRATION, ROOT, all_matches, source_mask, target_map,
    validate_inputs,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    result_path = ROOT / "results/BOUNDARY-CHANNEL-0004/prospective_train_registration.json"
    result = json.loads(result_path.read_text())
    config = json.loads(REGISTRATION.read_text())
    selected, scan_dir = validate_inputs(config, "prospective_train")
    folios = {item["folio"] for item in selected}
    if result["config_sha256"] != sha(REGISTRATION):
        raise AssertionError("Registration result does not use frozen configuration")
    if set(result["matches"]) != folios:
        raise AssertionError("Target folios changed")
    checks = {}
    for folio, row in sorted(result["matches"].items()):
        if set(row) != folios:
            raise AssertionError("Wrong-page impostor grid is incomplete")
        scan = scan_dir / f"{folio}.jpg"
        if sha(scan) != result["scan_sha256"][folio]:
            raise AssertionError(f"Scan changed: {folio}")
        image = cv2.imread(str(scan))
        if image is None or image.shape[1] != 700:
            raise AssertionError(f"Invalid scan size: {folio}")
        ordered = sorted(row, key=lambda source: row[source]["score"], reverse=True)
        rank = ordered.index(folio) + 1
        own = row[folio]["score"]
        next_best = max(value["score"] for source, value in row.items() if source != folio)
        saved = result["rankings"][folio]
        if (rank != saved["rank_of_correct_folio"] or
                not np.isclose(own, saved["own_score"], rtol=0, atol=1e-12) or
                not np.isclose(next_best, saved["best_impostor_score"], rtol=0, atol=1e-12) or
                not np.isclose(own - next_best, saved["own_minus_best_impostor"],
                               rtol=0, atol=1e-12)):
            raise AssertionError(f"Ranking arithmetic does not replay: {folio}")
        checks[folio] = {"rank": rank, "margin": own - next_best,
                         "image_size": [int(image.shape[1]), int(image.shape[0])],
                         "best_impostor": ordered[0] if rank != 1 else ordered[1]}
    if set(result["rankings"]) != folios:
        raise AssertionError("Ranking folios changed")

    pilot_scan = ROOT / "data/raw/boundary_ink_0002/yale700/f42v.jpg"
    image = cv2.imread(str(pilot_scan))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 12]]),
                             (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
    source = source_mask(read_boxes(BOX_DIR / "f42v.js"))
    before = all_matches({"f42v": target_map(image)}, {"f42v": source})["f42v"]["f42v"]
    after = all_matches({"f42v": target_map(shifted)}, {"f42v": source})["f42v"]["f42v"]
    if (after["tx"] - before["tx"], after["ty"] - before["ty"]) != (12, 12):
        raise AssertionError("Known image translation not recovered")
    if (after["sx"], after["sy"], after["score"]) != (
            before["sx"], before["sy"], before["score"]):
        raise AssertionError("Known shift changed source scale or score")
    try:
        target_map(np.full_like(image, 255))
    except ValueError:
        blank_rejected = True
    else:
        raise AssertionError("Blank scan accepted")

    ranked_first = sum(item["rank"] == 1 for item in checks.values())
    margins_pass = sum(item["rank"] == 1 and item["margin"] >= 0.02
                       for item in checks.values())
    audit = {"status": "audited_registered_registration_failure",
             "registration_result_sha256": sha(result_path),
             "config_sha256": sha(REGISTRATION), "target_checks": checks,
             "correct_top_one": ranked_first, "correct_with_margin_at_least_point_02": margins_pass,
             "registered_primary_gate": "FAIL" if margins_pass != len(folios) else "PASS",
             "blank_rejected": blank_rejected,
             "f42v_known_image_shift": {"dx": after["tx"] - before["tx"],
                                         "dy": after["ty"] - before["ty"],
                                         "scale_unchanged": True, "score_unchanged": True},
             "limit": "Ranks and margin arithmetic independently reconstructed from saved matrix; matching code is reused only for f42v perturbation. Not an independent re-search of all 64 pairs."}
    output = ROOT / "results/BOUNDARY-CHANNEL-0004/prospective_train_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: audit[key] for key in ("correct_top_one",
                        "correct_with_margin_at_least_point_02", "registered_primary_gate",
                        "blank_rejected", "f42v_known_image_shift")}, indent=2))
    return audit


if __name__ == "__main__":
    run()
