"""Verify key-free Borg image resources and report coverage, never tokenize.

An atlas label is not automatically a legacy transcription token. In particular,
multi-character example names must not trigger longest-match substitutions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.borg_control_parser import RAW, PINNED_SHA256, parse_bytes, validate_partition

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/borg_glyph_resource_inventory.json"
OUTPUT = ROOT / "results/BORG-GLYPH-PREP-002/coverage.json"
PREFIX = "gpu/few_shot_train/alphabet/borg/"


def checked_bytes(root: Path, entry: dict) -> bytes:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Resource path leaves repository")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Resource bytes differ")
    if "git_blob" in entry:
        git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_hash != entry["git_blob"]:
            raise ValueError("Resource differs from Git blob")
    return raw


def atlas_labels(root: Path, manifest: dict) -> set[str]:
    labels = set()
    for entry in manifest["artifacts"]:
        checked_bytes(root, entry)
        source = entry["source_path"]
        if source.startswith(PREFIX) and source.endswith("/1.jpg"):
            label = source[len(PREFIX):-len("/1.jpg")]
            if not label or "/" in label or label in labels:
                raise ValueError("Invalid or duplicate atlas label")
            labels.add(label)
    if not labels:
        raise ValueError("No image labels")
    return labels


def verify_tree_inventory(root: Path, manifest: dict) -> None:
    tree = json.loads(checked_bytes(root, manifest["source_tree"]))
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    entries = {row["path"]: row for row in tree["tree"]}
    seen = set()
    for resource in manifest["artifacts"]:
        path = resource["source_path"]
        if path in seen or path not in entries:
            raise ValueError("Missing or duplicate source resource")
        seen.add(path)
        original = entries[path]
        if (original.get("type") != "blob" or original.get("sha") != resource["git_blob"]
                or original.get("size") != resource["bytes"]):
            raise ValueError("Source path does not match pinned tree blob")
    expected_images = {path for path in entries if path.startswith(PREFIX) and path.endswith("/1.jpg")}
    if {path for path in seen if path.startswith(PREFIX)} != expected_images:
        raise ValueError("Atlas first-example inventory is incomplete")


def coverage(blob: bytes, labels: set[str]) -> dict:
    parsed = parse_bytes(blob)
    validate_partition(blob, parsed)
    single = {label for label in labels if len(label) == 1}
    multi = labels - single
    counts, adjacent = Counter(), Counter()
    for span in parsed.spans:
        if span.kind == "unresolved_body":
            text = blob[span.start_byte:span.end_byte].decode("utf-8")
            counts.update(text)
            for label in multi:
                adjacent[label] += sum(text.startswith(label, i) for i in range(len(text)))
    punctuation = set(":,.") - single
    remainder = {char: count for char, count in counts.items()
                 if char not in single and char not in punctuation}
    literal_count = sum(counts[label] for label in single)
    punctuation_count = sum(counts[label] for label in punctuation)
    assert literal_count + punctuation_count + sum(remainder.values()) == sum(counts.values())
    return {
        "status": "lexical_coverage_only_no_automatic_aliases_or_solver_records",
        "source_sha256": parsed.source_sha256,
        "atlas_labels": sorted(labels),
        "atlas_single_codepoint_labels": sorted(single),
        "atlas_multi_codepoint_names": sorted(multi),
        "unresolved_codepoints": sum(counts.values()),
        "unresolved_types": len(counts),
        "literal_single_label_occurrences": literal_count,
        "literal_single_label_types_present": len(single & counts.keys()),
        "punctuation_candidates_not_an_alias_map": {c: counts[c] for c in sorted(punctuation)},
        "remaining_codepoints": {f"U+{ord(c):04X}": n for c, n in sorted(remainder.items())},
        "remaining_occurrences": sum(remainder.values()),
        "literal_multi_name_substrings_not_glyph_tokens": {s: adjacent[s] for s in sorted(multi)},
        "solver_records": 0,
    }


def main():
    manifest = json.loads(MANIFEST.read_text())
    verify_tree_inventory(ROOT, manifest)
    labels = atlas_labels(ROOT, manifest)
    for entry in manifest["auxiliary_metadata"]:
        checked_bytes(ROOT, entry)
    checked_bytes(ROOT, manifest["pilot"])
    raw = RAW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_SHA256:
        raise ValueError("Legacy transcription differs")
    parsed = parse_bytes(raw, expected_sha256=PINNED_SHA256)
    review = manifest["pilot"]
    slices = []
    for start, end in review["visually_checked_raw_byte_spans"]:
        if not any(s.kind == "unresolved_body" and s.start_byte <= start < end <= s.end_byte
                   and s.start_line in review["raw_source_lines"] for s in parsed.spans):
            raise ValueError("Visual review interval leaves declared unresolved source lines")
        slices.append(raw[start:end].decode("utf-8"))
    glyphs = "".join(slices)
    if (len(glyphs) != review["checked_glyph_occurrences"]
            or len(set(glyphs)) != review["matched_label_types"]
            or not set(glyphs) <= {label for label in labels if len(label) == 1}):
        raise ValueError("Visual review metadata counts differ")
    result = coverage(raw, labels)
    result.update(resource_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  resources_verified=len(manifest["artifacts"]),
                  visual_review_scope="Single reviewer, 32 glyph instances on one preparation-exposed page; not full legend validation",
                  paid_cost_usd=0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
