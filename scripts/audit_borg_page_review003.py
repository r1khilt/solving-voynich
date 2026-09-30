"""Replay the source/image bindings in a manual Borg collation ledger.

This checks artifact identity, crop construction and complete split coverage.
It cannot independently establish that a reviewer recognized a glyph correctly.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.borg_control_parser import RAW, PINNED_SHA256, parse_bytes

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = ROOT / "data/manifests/borg_page_review003_acquisition.json"
LEDGER = ROOT / "results/BORG-PAGE-REVIEW-003/visual_ledger.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def complete_intervals(intervals, start, end):
    """A reviewed split may neither discard nor double-count source bytes."""
    cursor = start
    for left, right in sorted(intervals):
        if left != cursor or right <= left or right > end:
            raise ValueError("Source intervals gap, overlap or exceed original block")
        cursor = right
    if cursor != end:
        raise ValueError("Source intervals omit original block ending")


def checked_path(relative, expected):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Artifact leaves repository")
    if digest(path.read_bytes()) != expected:
        raise ValueError("Artifact hash mismatch")
    return path


def main():
    raw = RAW.read_bytes()
    parsed = parse_bytes(raw, expected_sha256=PINNED_SHA256)
    ledger = json.loads(LEDGER.read_text())
    if ledger["source_sha256"] != PINNED_SHA256 or ledger["solver_records"] != 0:
        raise ValueError("Review source/scope mismatch")
    checked_path(str(ACQUISITION.relative_to(ROOT)), ledger["acquisition_manifest_sha256"])
    acquisition = json.loads(ACQUISITION.read_text())
    if acquisition["status"] != "acquired":
        raise ValueError("Image acquisition not complete")
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    images = {}
    for item in acquisition["images"]:
        path = checked_path(item["path"], item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError("Image size differs")
        images[item["canvas"]] = (item, path)
    if len(images) != 8 or len(ledger["mappings"]) != 8:
        raise ValueError("Fixed eight-image panel differs")

    def check_evidence(evidence, mapping):
        source_line = evidence["source_line"]
        start, end = evidence["start_byte"], evidence["end_byte"]
        if not (mapping["source_start_line"] <= source_line <= mapping["source_end_line"]
                and offsets[source_line - 1] <= start < end <= offsets[source_line]):
            raise ValueError("Review slice crosses its declared source line")
        part = raw[start:end]
        if digest(part) != evidence["source_slice_sha256"]:
            raise ValueError("Review source slice differs")
        glyphs = "".join(part.decode("utf-8").split())
        if (len(glyphs) < 8 or len(glyphs) != evidence["glyph_occurrences"]
                or digest(glyphs.encode()) != evidence["reviewed_glyph_sequence_sha256"]):
            raise ValueError("Review glyph-count or sequence binding differs")
        crop = checked_path(evidence["crop_path"], evidence["crop_sha256"])
        with Image.open(images[mapping["canvas"]][1]) as image, Image.open(crop) as saved:
            x0, y0, x1, y1 = evidence["crop_box_xyxy"]
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError("Review crop exceeds image")
            replay = image.crop((x0, y0, x1, y1))
            if replay.mode != saved.mode or replay.size != saved.size or replay.tobytes() != saved.tobytes():
                raise ValueError("Review crop pixels differ")

    intervals = defaultdict(list)
    seen_canvases = set()
    for mapping in ledger["mappings"]:
        page = parsed.pages[mapping["original_block_index"]]
        if mapping["original_label"] != page.raw_label:
            raise ValueError("Original page identity differs")
        first, last = mapping["source_start_line"], mapping["source_end_line"]
        start, end = mapping["start_byte"], mapping["end_byte"]
        if (not page.start_line <= first <= last <= page.end_line
                or (start, end) != (offsets[first - 1], offsets[last])
                or digest(raw[start:end]) != mapping["slice_sha256"]):
            raise ValueError("Reviewed interval differs from raw page")
        intervals[page.block_index].append((start, end))
        image, _ = images[mapping["canvas"]]
        if (mapping["canvas"] in seen_canvases or mapping["image_path"] != image["path"]
                or mapping["image_sha256"] != image["sha256"]
                or mapping["proposed_canvas_label"] != image["label"]):
            raise ValueError("Wrong or reused image identity")
        seen_canvases.add(mapping["canvas"])
        if len({e["source_line"] for e in mapping["checks"]}) < 2:
            raise ValueError("Fewer than two distinct source-line checks")
        for evidence in mapping["checks"]:
            check_evidence(evidence, mapping)
    for block, parts in intervals.items():
        page = parsed.pages[block]
        complete_intervals(parts, offsets[page.start_line - 1], offsets[page.end_line])
    boundary = ledger["boundary_evidence"]
    marker = lines[boundary["source_line"] - 1]
    if marker.decode().strip() != boundary["literal_marker"] or digest(marker) != boundary["marker_sha256"]:
        raise ValueError("Original nonstandard page marker differs")
    recto = next(m for m in ledger["mappings"] if m["proposed_canvas_label"] == "99r")
    verso = next(m for m in ledger["mappings"] if m["proposed_canvas_label"] == "99v")
    if recto["end_byte"] != verso["start_byte"] or verso["source_start_line"] != boundary["source_line"]:
        raise ValueError("Boundary is not the preserved original marker")
    reference = boundary["starting_99v_check_reference"]
    if (reference != {"canvas_label": "99v", "check_index": 0}
            or verso["checks"][0]["source_line"] != boundary["source_line"] + 1
            or any(line.strip() for line in lines[boundary["ending_99r"]["source_line"]:
                                                 boundary["source_line"] - 1])):
        raise ValueError("Boundary evidence does not adjoin the original marker")
    check_evidence(boundary["ending_99r"], recto)
    checks = [e for m in ledger["mappings"] for e in m["checks"]] + [boundary["ending_99r"]]
    print(json.dumps({"status": "artifact_bindings_and_crop_replay_pass",
        "ledger_sha256": digest(LEDGER.read_bytes()), "images": len(images),
        "original_blocks": len(intervals), "reviewed_segments": len(ledger["mappings"]),
        "source_crop_checks": len(checks), "glyph_occurrences": sum(e["glyph_occurrences"] for e in checks),
        "complete_reviewed_block_byte_coverage": True,
        "independent_visual_validation": False, "solver_records": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
