"""Fixed key-free atlas/legacy/image review; never creates solver records."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.audit_borg_glyph_resources import atlas_labels, verify_tree_inventory
from scripts.borg_control_parser import PINNED_SHA256, RAW, parse_bytes

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "data/manifests/borg_glyph_resource_inventory.json",
    "results/BORG-PAGE-REVIEW-003/visual_ledger.json",
    "results/BORG-EXCEPTION-REVIEW-004/visual_ledger.json",
)
PLAN = "data/manifests/borg_glyph_bridge005_plan.json"
LEDGER = "results/BORG-GLYPH-BRIDGE-005/visual_ledger.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path leaves repository")
    return path


def make_plan(blob, labels, image_ranges):
    """First lexical occurrence per distinct canvas; retain missing targets."""
    parsed = parse_bytes(blob)
    single = sorted(label for label in labels if len(label) == 1)
    if not single or set(single) & set(".,:"):
        raise ValueError("Expected literal atlas names separate from punctuation")
    ranges = sorted(image_ranges, key=lambda row: (row["start_line"], row["canvas"]))
    for row in ranges:
        if not 0 <= row["block"] < len(parsed.pages):
            raise ValueError("Image has unknown source block")
        page = parsed.pages[row["block"]]
        if not page.start_line <= row["start_line"] <= row["end_line"] <= page.end_line:
            raise ValueError("Image interval leaves source block")
    for i, row in enumerate(ranges):
        for other in ranges[:i]:
            if (row["block"] == other["block"] and
                    max(row["start_line"], other["start_line"]) <= min(row["end_line"], other["end_line"])):
                raise ValueError("Overlapping reviewed image intervals")
    targets = []
    for symbol in sorted(set(single) | set(".,:")):
        candidates = []
        seen = set()
        for span in parsed.spans:
            if span.kind != "unresolved_body":
                continue
            matches = [r for r in ranges if r["block"] == span.block_index
                       and r["start_line"] <= span.start_line <= r["end_line"]]
            if not matches or matches[0]["canvas"] in seen:
                continue
            row = matches[0]
            offset = span.start_byte
            for char in blob[span.start_byte:span.end_byte].decode():
                end = offset + len(char.encode())
                if char == symbol:
                    candidates.append({"symbol": symbol, "source_line": span.start_line,
                        "start_byte": offset, "end_byte": end,
                        "slice_sha256": digest(blob[offset:end]), "image": row})
                    seen.add(row["canvas"])
                    break
                offset = end
        targets.append({"symbol": symbol,
            "role": "literal_atlas_name" if symbol in single else "punctuation_candidate_no_alias",
            "occurrences": candidates[:2], "available_distinct_canvases": len(candidates),
            "missing_slots": max(0, 2 - len(candidates))})
    return {"experiment": "BORG-GLYPH-BRIDGE-005", "source_sha256": parsed.source_sha256,
        "selection": "First unresolved-body occurrence on each canvas, source-byte order; first two distinct canvases per literal single atlas name and comma/period/colon. No target replacement.",
        "image_ranges": ranges, "targets": targets, "solver_records": 0,
        "limits": {"new_downloads": 0, "model_runs": 0, "paid_cost_usd": 0}}


def plan_from_root(root):
    raw = local_file(root, str(RAW.relative_to(ROOT))).read_bytes()
    if digest(raw) != PINNED_SHA256:
        raise ValueError("Pinned transcription changed")
    atlas, pages, exceptions = [json.loads(local_file(root, p).read_text()) for p in INPUTS]
    verify_tree_inventory(root, atlas)
    labels = atlas_labels(root, atlas)
    parsed = parse_bytes(raw)
    ranges = []
    pilot = atlas["pilot"]
    matches = [p for p in parsed.pages if p.normalized_label_only == pilot["folio"]]
    if len(matches) != 1:
        raise ValueError("Pilot block ambiguous")
    p = matches[0]
    ranges.append({"block": p.block_index, "start_line": p.start_line, "end_line": p.end_line,
        "canvas": pilot["canvas"], "image_path": pilot["path"], "image_sha256": pilot["sha256"]})
    for row in pages["mappings"]:
        ranges.append({"block": row["original_block_index"], "start_line": row["source_start_line"],
            "end_line": row["source_end_line"], "canvas": row["canvas"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"]})
    for row in exceptions["pages"]:
        p = parsed.pages[row["block"]]
        ranges.append({"block": p.block_index, "start_line": p.start_line, "end_line": p.end_line,
            "canvas": row["canvas"], "image_path": row["image_path"], "image_sha256": row["image_sha256"]})
    for row in ranges:
        if digest(local_file(root, row["image_path"]).read_bytes()) != row["image_sha256"]:
            raise ValueError("Previously reviewed image changed")
    plan = make_plan(raw, labels, ranges)
    plan["input_bindings"] = {name: digest(local_file(root, name).read_bytes()) for name in INPUTS}
    return plan


def audit(root=ROOT):
    plan_path = local_file(root, PLAN)
    plan = json.loads(plan_path.read_text())
    if plan != plan_from_root(root):
        raise ValueError("Fixed selection replay differs")
    ledger = json.loads(local_file(root, LEDGER).read_text())
    if ledger["plan_sha256"] != digest(plan_path.read_bytes()) or ledger["solver_records"] != 0:
        raise ValueError("Ledger scope or binding differs")
    expected = [o for row in plan["targets"] for o in row["occurrences"]]
    if len(expected) != len(ledger["observations"]):
        raise ValueError("Target count differs")
    statuses, contexts, crops = Counter(), Counter(), set()
    for target, obs in zip(expected, ledger["observations"], strict=True):
        if obs["target"] != target:
            raise ValueError("Target omitted, replaced or reordered")
        if obs["shape_status"] not in {"local_atlas_resemblance", "uncertain_correspondence", "local_difference"}:
            raise ValueError("Unsupported shape certainty")
        if obs["context_status"] not in {"main_symbol_run", "auxiliary_or_cursive_context", "uncertain_context"}:
            raise ValueError("Unsupported context certainty")
        if not isinstance(obs["observation"], str) or not obs["observation"].strip():
            raise ValueError("Missing visual observation")
        image = Image.open(local_file(root, target["image"]["image_path"]))
        crop = obs["crop"]
        x1, y1, x2, y2 = crop["box_xyxy"]
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("Crop outside image")
        crop_path = local_file(root, crop["path"])
        if digest(crop_path.read_bytes()) != crop["sha256"]:
            raise ValueError("Crop hash differs")
        saved = Image.open(crop_path)
        actual = image.crop(crop["box_xyxy"])
        if (saved.mode, saved.size, saved.tobytes()) != (actual.mode, actual.size, actual.tobytes()):
            raise ValueError("Crop pixels differ")
        statuses[obs["shape_status"]] += 1
        contexts[obs["context_status"]] += 1
        crops.add(crop["path"])
    return {"status": "PASS_provenance_and_accounting_not_independent_shape_validation",
        "experiment": plan["experiment"], "plan_sha256": digest(plan_path.read_bytes()),
        "ledger_sha256": digest(local_file(root, LEDGER).read_bytes()),
        "targets": len(expected), "literal_classes": sum(t["role"] == "literal_atlas_name" for t in plan["targets"]),
        "distinct_target_canvases": len({t["image"]["canvas"] for t in expected}),
        "missing_slots": sum(t["missing_slots"] for t in plan["targets"]),
        "shape_statuses": dict(statuses), "context_statuses": dict(contexts),
        "replayed_crops": len(crops), "solver_records": 0, "paid_cost_usd": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "audit"))
    args = parser.parse_args()
    result = plan_from_root(ROOT) if args.action == "plan" else audit()
    name = PLAN if args.action == "plan" else "results/BORG-GLYPH-BRIDGE-005/audit.json"
    path = local_file(ROOT, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"targets", "image_ranges", "input_bindings"}}, sort_keys=True))
