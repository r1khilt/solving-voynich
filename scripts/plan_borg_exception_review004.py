"""Select a key-free image panel covering all unresolved non-atlas types.

Selection uses transcription counts only. No language score, glyph merging or
claim of visual identity is made by this program.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.borg_control_parser import PINNED_SHA256, RAW, parse_bytes

ROOT = Path(__file__).resolve().parents[1]


def greedy_cover(counts):
    remaining, seen, selected = set(counts), set(), []
    target = set().union(*(set(c) for c in counts.values()))
    while seen != target:
        page = max(remaining, key=lambda n: (len(set(counts[n]) - seen),
                                             sum(counts[n].values()), -n))
        new = set(counts[page]) - seen
        if not new:
            raise ValueError("Cannot cover target types")
        selected.append((page, sorted(new)))
        seen.update(new)
        remaining.remove(page)
    return selected


def make_plan(blob, coverage, catalogue):
    parsed = parse_bytes(blob, expected_sha256=PINNED_SHA256)
    known = set(coverage["atlas_single_codepoint_labels"]) | set(".,:")
    counts, occurrences = defaultdict(Counter), defaultdict(list)
    for span in parsed.spans:
        if span.kind != "unresolved_body":
            continue
        offset = span.start_byte
        for char in blob[span.start_byte:span.end_byte].decode():
            end = offset + len(char.encode())
            if char not in known:
                counts[span.block_index][char] += 1
                occurrences[span.block_index, char].append({"start_byte": offset,
                    "end_byte": end, "source_line": span.start_line,
                    "slice_sha256": hashlib.sha256(blob[offset:end]).hexdigest()})
            offset = end
    totals = sum(counts.values(), Counter())
    if {f"U+{ord(c):04X}": n for c, n in totals.items()} != coverage["remaining_codepoints"]:
        raise ValueError("Exception coverage differs from prior audit")
    panel = []
    for block, new in greedy_cover(counts):
        page = parsed.pages[block]
        label = page.normalized_label_only.lstrip("0")
        candidates = [c for c in catalogue["sequences"][0]["canvases"] if c["label"] == label]
        if len(candidates) != 1:
            raise ValueError("Selected page needs a separately reviewed canvas identity")
        c = candidates[0]
        panel.append({"block": block, "raw_label": page.raw_label,
            "source_start_line": page.start_line, "source_end_line": page.end_line,
            "canvas": c["@id"], "catalogue_label": c["label"],
            "image_url": c["images"][0]["resource"]["service"]["@id"] + "/full/1600,/0/default.jpg",
            "new_codepoints": [f"U+{ord(char):04X}" for char in new],
            "all_exception_counts": {f"U+{ord(char):04X}": n for char, n in sorted(counts[block].items())},
            "target_occurrences": {f"U+{ord(char):04X}": occurrences[block, char][:2] for char in new}})
    return {"experiment": "BORG-EXCEPTION-REVIEW-004", "status": "fixed_before_image_acquisition",
        "source_sha256": parsed.source_sha256, "exception_types": len(totals),
        "exception_occurrences": sum(totals.values()), "panel": panel,
        "selection": "Greedy uncovered type count, then total exception occurrences, then lowest block index; first two occurrences per newly covered type",
        "limits": {"images": len(panel), "concurrent_requests": 2, "request_seconds": 45,
                   "bytes_per_image": 2500000, "width_pixels": 1600, "paid_spend_usd": 0},
        "solver_records": 0}


if __name__ == "__main__":
    coverage_path = ROOT / "results/BORG-GLYPH-PREP-002/coverage.json"
    catalogue_path = ROOT / "data/raw/borg-resource-review/vatican-manifest.json"
    result = make_plan(RAW.read_bytes(), json.loads(coverage_path.read_text()),
                       json.loads(catalogue_path.read_text()))
    result["coverage_sha256"] = hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    result["catalogue_sha256"] = hashlib.sha256(catalogue_path.read_bytes()).hexdigest()
    output = ROOT / "data/manifests/borg_exception_review004_plan.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"pages": len(result["panel"]), "types": result["exception_types"],
                      "targets": sum(len(v) for p in result["panel"] for v in p["target_occurrences"].values())}))
