"""Audit the complete fixed panel's bytes and crops, not glyph recognition."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.borg_control_parser import PINNED_SHA256, RAW, parse_bytes
from scripts.plan_borg_exception_review004 import make_plan

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def target_identity(target):
    return tuple(target[k] for k in ("codepoint", "source_line", "start_byte", "end_byte", "slice_sha256"))


def audit(root=ROOT):
    plan_path = root / "data/manifests/borg_exception_review004_plan.json"
    acquisition_path = root / "data/manifests/borg_exception_review004_acquisition.json"
    ledger_path = root / "results/BORG-EXCEPTION-REVIEW-004/visual_ledger.json"
    plan, acquisition, ledger = (json.loads(p.read_text()) for p in (plan_path, acquisition_path, ledger_path))
    raw = (root / RAW.relative_to(ROOT)).read_bytes()
    parsed = parse_bytes(raw, expected_sha256=PINNED_SHA256)
    coverage_path = root / "results/BORG-GLYPH-PREP-002/coverage.json"
    catalogue_path = root / "data/raw/borg-resource-review/vatican-manifest.json"
    coverage = json.loads(coverage_path.read_text())
    if digest(coverage_path.read_bytes()) != plan["coverage_sha256"] or digest(catalogue_path.read_bytes()) != plan["catalogue_sha256"]:
        raise ValueError("Selection input hash differs")
    replay = make_plan(raw, coverage, json.loads(catalogue_path.read_text()))
    if any(plan[k] != v for k, v in replay.items()):
        raise ValueError("Deterministic selection differs")
    if (acquisition["plan_sha256"] != digest(plan_path.read_bytes()) or
            ledger["plan_sha256"] != digest(plan_path.read_bytes()) or
            ledger["acquisition_sha256"] != digest(acquisition_path.read_bytes())):
        raise ValueError("Plan/acquisition binding differs")
    if len(ledger["pages"]) != len(plan["panel"]) or len(acquisition["images"]) != len(plan["panel"]):
        raise ValueError("Panel size differs")
    known = set(coverage["atlas_single_codepoint_labels"])
    allowed = {"localized_shape_observation_semantics_unresolved",
               "localized_but_shape_or_segmentation_uncertain", "unresolved_individual_correspondence"}
    crops, checks, targets, statuses = set(), 0, 0, Counter()
    for selected, acquired, page in zip(plan["panel"], acquisition["images"], ledger["pages"], strict=True):
        for field in ("block", "canvas"):
            if selected[field] != acquired[field] or selected[field] != page[field]:
                raise ValueError("Page identity differs")
        if selected["catalogue_label"] != acquired["label"] or acquired["label"] != page["label"] or selected["image_url"] != acquired["url"]:
            raise ValueError("Catalogue or URL differs")
        image_path = root / acquired["path"]
        image_raw = image_path.read_bytes()
        if (acquired["returncode"] != 0 or len(image_raw) != acquired["bytes"] or
                digest(image_raw) != acquired["sha256"] or page["image_sha256"] != acquired["sha256"] or
                page["image_path"] != acquired["path"]):
            raise ValueError("Acquired image differs")
        im = Image.open(image_path)
        if im.width != 1600 or len(image_raw) > 2500000:
            raise ValueError("Image outside frozen bounds")

        def crop(item):
            name, box = item["path"], item["box_xyxy"]
            if not (0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height):
                raise ValueError("Crop outside image")
            path = root / name
            if digest(path.read_bytes()) != item["sha256"]:
                raise ValueError("Crop hash differs")
            saved = Image.open(path)
            actual = im.crop(box)
            if saved.mode != actual.mode or saved.size != actual.size or saved.tobytes() != actual.tobytes():
                raise ValueError("Crop pixel replay differs")
            crops.add(name)

        proof = page["page_identity_checks"]
        if len(proof) < 2 or len({p["source_line"] for p in proof}) < 2:
            raise ValueError("Two distinct source-line checks required")
        for item in proof:
            a, b = item["start_byte"], item["end_byte"]
            text = raw[a:b].decode()
            symbols = [c for c in text if not c.isspace()]
            if digest(raw[a:b]) != item["slice_sha256"] or len(symbols) < 8 or len(symbols) != item["glyph_occurrences"] or not set(symbols) <= known:
                raise ValueError("Identity-check source interval differs")
            covered = [s for s in parsed.spans if s.start_byte < b and s.end_byte > a]
            if not covered or any(s.kind not in ("unresolved_body", "whitespace") or s.block_index != page["block"] or s.start_line != item["source_line"] for s in covered):
                raise ValueError("Identity check leaves declared source line/body")
            crop(item["crop"])
            checks += 1
        expected = [target_identity({"codepoint": code, **occ}) for code, occurrences in selected["target_occurrences"].items() for occ in occurrences]
        actual = [target_identity(t) for t in page["targets"]]
        if Counter(actual) != Counter(expected):
            raise ValueError("Targets omitted, duplicated or moved")
        for item in page["targets"]:
            a, b = item["start_byte"], item["end_byte"]
            if digest(raw[a:b]) != item["slice_sha256"] or raw[a:b].decode() != chr(int(item["codepoint"][2:], 16)):
                raise ValueError("Target codepoint differs")
            if item["status"] not in allowed or not item["observation"].strip():
                raise ValueError("Missing review or unsupported certainty claim")
            crop(item["crop"])
            statuses[item["status"]] += 1
            targets += 1
    if ledger["solver_records"] != 0:
        raise ValueError("Review cannot create solver records")
    return {"status": "artifact_audit_pass", "scope": "Root artifact checks, not independent visual recognition",
            "ledger_sha256": digest(ledger_path.read_bytes()), "images": len(ledger["pages"]),
            "page_identity_line_checks": checks, "target_occurrences": targets,
            "review_status_counts": dict(statuses), "unique_crops_replayed": len(crops),
            "solver_records": 0}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
