"""Prospective batch box/scan registration with source-cohort score centering.

The raw image search is exactly the frozen BOUNDARY-CHANNEL-0004 search. The
new score subtracts each source template's mean score on the other target
pages in this fixed batch. It uses no pilot geometry, text or gap labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2

from scripts.build_boundary_geometry_train import leaf_id, read_boxes
from scripts.register_boundary_ink_pages import (
    ARCHIVE, BOX_DIR, ROOT, SPLIT, SPLIT_SHA, SOURCE_REV, YALE_MANIFEST, YALE_SHA,
    all_matches, source_mask, target_map,
)


CONFIG = ROOT / "data/manifests/boundary_registration_0005.json"
OLD_CONFIG = ROOT / "data/manifests/boundary_registration_0004.json"
OLD_SEARCH = ROOT / "scripts/register_boundary_ink_pages.py"
PILOT_RESULT = ROOT / "results/BOUNDARY-CHANNEL-0004/pilot_dev_registration.json"
EXPOSED_RESULT = ROOT / "results/BOUNDARY-CHANNEL-0004/prospective_train_registration.json"
SEARCH_SHA = "d47269490a1c5e493585f5e6b45217753993709fa49e222939de42121de714a9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_scores(matches: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    """Subtract each source's mean score against the seven other targets."""
    targets = sorted(matches)
    if len(targets) < 3 or any(set(matches[target]) != set(targets) for target in targets):
        raise ValueError("Need a complete square source-target score matrix")
    centered = {}
    for target in targets:
        centered[target] = {}
        for source in targets:
            own = matches[target][source]["score"]
            background = sum(matches[other][source]["score"] for other in targets
                             if other != target) / (len(targets) - 1)
            centered[target][source] = own - background
    return centered


def rankings(matches: dict[str, dict[str, dict]], centered: dict[str, dict[str, float]]) -> dict:
    output = {}
    for target, row in centered.items():
        ranked = sorted(row, key=row.get, reverse=True)
        raw_ranked = sorted(matches[target], key=lambda source: matches[target][source]["score"],
                            reverse=True)
        own = row[target]
        best_other = max(value for source, value in row.items() if source != target)
        output[target] = {
            "raw_rank": raw_ranked.index(target) + 1,
            "cohort_rank": ranked.index(target) + 1,
            "cohort_own_score": own,
            "cohort_best_impostor_score": best_other,
            "cohort_margin": own - best_other,
            "raw_affine": {key: matches[target][target][key] for key in
                           ("sx", "tx", "sy", "ty")},
        }
    return output


def duplicate_target_control(matches: dict[str, dict[str, dict]], order: list[str]) -> dict:
    """Remove the first page by copying the second's score row into its slot."""
    first, second = order[:2]
    counterfeit = {target: {source: dict(value) for source, value in row.items()}
                   for target, row in matches.items()}
    counterfeit[first] = {source: dict(value) for source, value in matches[second].items()}
    scored = rankings(counterfeit, cohort_scores(counterfeit))
    passed = sum(item["cohort_rank"] == 1 and item["cohort_margin"] >= 0.03
                 for item in scored.values())
    return {"missing_folio": first, "duplicate_of": second,
            "passing_pages_after_duplicate": passed, "expected_less_than": len(order),
            "rejects_complete_claim": passed < len(order)}


def validate_selection(config: dict) -> tuple[list[dict], Path]:
    if (sha(OLD_SEARCH) != SEARCH_SHA or config["search_code_sha256"] != SEARCH_SHA or
            sha(SPLIT) != SPLIT_SHA or sha(YALE_MANIFEST) != YALE_SHA or
            subprocess.check_output(["git", "-C", str(ARCHIVE), "rev-parse", "HEAD"],
                                    text=True).strip() != SOURCE_REV):
        raise AssertionError("Frozen search or source version changed")
    old = json.loads(OLD_CONFIG.read_text())
    previous = old["groups"]["pilot_dev"]["folios"] + old["groups"]["prospective_train"]["folios"]
    prior_leaves = {leaf_id(item["folio"]) for item in previous}
    assignments = json.loads(SPLIT.read_text())["leaf_assignments"]
    yale = json.loads(YALE_MANIFEST.read_text())
    ids = {f"f{item['label']['none'][0]}": item["id"].split("/")[-1]
           for item in yale["items"] if item.get("label", {}).get("none")}
    eligible = []
    for path in BOX_DIR.glob("*.js"):
        folio = path.stem
        if (leaf_id(folio) in prior_leaves or assignments.get(leaf_id(folio)) != "train"
                or folio not in ids):
            continue
        boxes = read_boxes(path)
        if len(boxes) < 70 or len({box["line"] for box in boxes}) < 8:
            continue
        eligible.append((hashlib.sha256(f"BOUNDARY-CHANNEL-0005|{folio}".encode()).hexdigest(),
                         folio))
    expected = [folio for _, folio in sorted(eligible)[:8]]
    selected = config["folios"]
    if [item["folio"] for item in selected] != expected:
        raise AssertionError(f"Selection changed: expected {expected}")
    for item in selected:
        folio = item["folio"]
        if ids.get(folio) != item["canvas_id"] or sha(BOX_DIR / f"{folio}.js") != item["box_sha256"]:
            raise AssertionError(f"Canvas or boxes changed: {folio}")
    return selected, ROOT / config["scan_dir"]


def development() -> dict:
    records = {}
    for name, path in (("pilot_exposed", PILOT_RESULT), ("fresh_0004_now_exposed", EXPOSED_RESULT)):
        saved = json.loads(path.read_text())
        centered = cohort_scores(saved["matches"])
        records[name] = {"input_sha256": sha(path),
                         "rankings": rankings(saved["matches"], centered)}
    output = ROOT / "results/BOUNDARY-CHANNEL-0005/development_cohort_scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {folio: item["cohort_margin"] for folio, item in
        record["rankings"].items()} for name, record in records.items()}, indent=2))
    return records


def prospective() -> dict:
    config = json.loads(CONFIG.read_text())
    selected, scan_dir = validate_selection(config)
    targets = {}
    sources = {}
    scans = {}
    for item in selected:
        folio = item["folio"]
        path = scan_dir / f"{folio}.jpg"
        scans[folio] = sha(path)
        targets[folio] = target_map(cv2.imread(str(path)))
        sources[folio] = source_mask(read_boxes(BOX_DIR / f"{folio}.js"))
    start = time.monotonic()
    matches = all_matches(targets, sources)
    elapsed = time.monotonic() - start
    if any(value is None for row in matches.values() for value in row.values()):
        raise ValueError("At least one source template could not fit a target")
    centered = cohort_scores(matches)
    ranked = rankings(matches, centered)
    control = duplicate_target_control(matches, [item["folio"] for item in selected])
    if not control["rejects_complete_claim"]:
        raise AssertionError("Counterfeit duplicated target passes complete gate")
    complete = all(item["cohort_rank"] == 1 and item["cohort_margin"] >= .03
                   for item in ranked.values())
    result = {"status": "prospective_batch_registration",
              "gate": "PASS" if complete else "FAIL",
              "config_sha256": sha(CONFIG), "search_code_sha256": SEARCH_SHA,
              "source_archive_revision": SOURCE_REV, "split_sha256": SPLIT_SHA,
              "yale_manifest_sha256": YALE_SHA, "scan_sha256": scans,
              "seconds": elapsed, "opencv_version": cv2.__version__,
              "raw_matches": matches, "cohort_scores": centered, "rankings": ranked,
              "duplicate_target_control": control,
              "limit": "Cohort score uses the complete fixed batch of target scans; source/target identity and global layout do not certify named words or ink-gap labels."}
    output = ROOT / "results/BOUNDARY-CHANNEL-0005/prospective_registration.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": result["gate"], "seconds": round(elapsed, 2),
                      "rankings": ranked, "duplicate_target_control": control},
                     indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "prospective"), required=True)
    mode = parser.parse_args().mode
    development() if mode == "development" else prospective()
