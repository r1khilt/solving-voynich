"""Audit image-row-based fragment/label alignment without reading label text."""

import hashlib
import json
from pathlib import Path
import re

import anchor0002_source_audit as source


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/zl3b/validation.jsonl"
IMAGE_DIR = ROOT / "data/raw/anchor0002/previews"

# Registered from the Yale and Zandbergen page views before any label-form read.
# f89r2 row 3 has six drawings but only five Lf loci and is deliberately omitted.
ROW_LOCUS_IDS = {
    "f89r1": {
        1: ["f89r1.2", "f89r1.3", "f89r1.4", "f89r1.5"],
        2: ["f89r1.12", "f89r1.13", "f89r1.14"],
        3: ["f89r1.25", "f89r1.26", "f89r1.27"],
    },
    "f89r2": {
        1: ["f89r2.2", "f89r2.3", "f89r2.4", "f89r2.5"],
        2: ["f89r2.10", "f89r2.11"],
        4: ["f89r2.31", "f89r2.32", "f89r2.33", "f89r2.34"],
    },
    "f89v2": {
        1: ["f89v2.2", "f89v2.3", "f89v2.4", "f89v2.5", "f89v2.6"],
        2: ["f89v2.12", "f89v2.13", "f89v2.14"],
        3: ["f89v2.24", "f89v2.25", "f89v2.27", "f89v2.28"],
    },
}
PREVIEW_NAMES = {
    "f89r1": "f089r1_crd.jpg",
    "f89r2": "f089r2_crd.jpg",
    "f89v2": "f089v2_crd.jpg",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_structure() -> dict[str, dict[str, list[int]]]:
    """Return only locus IDs and editorial numbers, never transcription strings."""
    pages = {}
    for line in CORPUS.read_text().splitlines():
        page = json.loads(line)
        page_id = page["page_id"]
        if page_id not in ROW_LOCUS_IDS:
            continue
        labels = {}
        for locus in page["loci"]:
            if locus["locus_type"] != "Lf":
                continue
            numbers = [int(value) for annotation in locus["annotations"]
                       if annotation["kind"] == "editorial_comment"
                       for value in re.findall(r"<!\s*(\d+)\s*>", annotation["raw"])]
            labels[locus["locus_id"]] = numbers
        pages[page_id] = labels
    if set(pages) != set(ROW_LOCUS_IDS):
        raise ValueError("Missing target validation page")
    return pages


def run() -> dict:
    htmls, sources = source.load_sources()
    rows = source.numbered_fragments(htmls)
    direct, duplicates = source.numeric_locators()
    candidates = json.loads((ROOT / "data/manifests/anchor0002_candidates.json").read_text())
    if candidates["fully_direct_group_ids"] != [5, 16, 26]:
        raise ValueError("Base candidate inventory drifted")
    if candidates["duplicate_numeric_locators"] != {str(k): v for k, v in duplicates.items()}:
        raise ValueError("Duplicate locator inventory drifted")
    labels = validation_structure()
    inferred = {}
    row_details = []
    for page_id, row_map in ROW_LOCUS_IDS.items():
        used = set()
        for row_number, locus_ids in row_map.items():
            fragments = rows[page_id, row_number]
            if len(fragments) != len(locus_ids):
                raise ValueError(f"Row count mismatch: {page_id} row {row_number}")
            if len(set(locus_ids)) != len(locus_ids) or not set(locus_ids) <= set(labels[page_id]):
                raise ValueError(f"Missing/duplicate Lf locus: {page_id} row {row_number}")
            used.update(locus_ids)
            for fragment, locus_id in zip(fragments, locus_ids, strict=True):
                if labels[page_id][locus_id] and labels[page_id][locus_id] != [fragment]:
                    raise ValueError(f"Numbered anchor contradicts row order: {locus_id}")
                if fragment in direct and direct[fragment] != locus_id:
                    raise ValueError(f"Direct anchor contradicts row order: {fragment}")
                inferred[fragment] = locus_id
            row_details.append({"page": page_id, "row": row_number,
                                "fragment_numbers": fragments, "lf_locus_ids": locus_ids})
        unused = set(labels[page_id]) - used
        expected_unused = set() if page_id != "f89r2" else {
            "f89r2.17", "f89r2.18", "f89r2.19", "f89r2.20", "f89r2.21"
        }
        if unused != expected_unused:
            raise ValueError(f"Unexpected unaligned labels: {page_id}")

    eligible = []
    for group in candidates["groups"]:
        if len(group["members"]) != 2:
            continue
        members = []
        for member in group["members"]:
            fragment = member["fragment"]
            locus_id = None
            method = None
            if member["split"] != "test" and fragment in direct:
                locus_id, method = direct[fragment], "direct_editorial_locator"
            elif member["split"] != "test" and fragment in inferred:
                locus_id, method = inferred[fragment], "single_reviewer_image_row_order"
            members.append({"source_reference": member["source_reference"],
                            "fragment": fragment, "split": member["split"],
                            "lf_locus_id": locus_id, "alignment_method": method})
        if all(member["lf_locus_id"] for member in members):
            eligible.append({"source_group": group["source_group"], "members": members})
    group_ids = [row["source_group"] for row in eligible]
    if group_ids != [5, 7, 8, 10, 12, 16, 19, 23, 26]:
        raise ValueError(f"Unexpected eligible group set: {group_ids}")
    images = {}
    for page_id, filename in PREVIEW_NAMES.items():
        path = IMAGE_DIR / filename.replace("_crd", "")
        if not path.exists():
            raise FileNotFoundError(path)
        images[page_id] = {"url": f"https://www.voynich.nu/q15/{filename}",
                           "sha256": sha(path), "ignored_raw_path": str(path.relative_to(ROOT))}
    result = {
        "id": "ANCHOR-0002", "status": "label-geometry-feasibility-only",
        "base_candidate_manifest_sha256": sha(ROOT / "data/manifests/anchor0002_candidates.json"),
        "validation_corpus_sha256": sha(CORPUS), "source_files": sources,
        "image_previews": images, "row_alignments": row_details,
        "unaligned_f89r2_row3_reason": "six numbered fragments, five Lf loci",
        "new_provisional_group_ids": [7, 8, 10, 12, 19, 23],
        "all_potentially_alignable_group_ids": group_ids,
        "potentially_alignable_groups": eligible,
        "limitations": "No label strings read; no drawing-pair similarity rerating; f89 alignments are single-reviewer positional inferences; test text unopened.",
    }
    output = ROOT / "data/manifests/anchor0002_label_alignment.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"groups": result["all_potentially_alignable_group_ids"],
                      "new_provisional": result["new_provisional_group_ids"]}))
