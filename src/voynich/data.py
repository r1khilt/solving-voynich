"""Strict, dependency-free ingestion of the pinned ZL IVTFF 2.0 EVA stream.

This parser supports the documented subset used by ZL3b, not every possible
IVTFF alphabet or alternative locator convention. Unsupported input fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .tokenizer import ALTERNATIVE, UNREADABLE, UNKNOWN_SPAN, UNCERTAIN_SPACE, RARE_BASE, EVATokenizer

PAGE_PATTERN = r"(?:f\d+[rv]\d*|fRos)"
PAGE_RE = re.compile(rf"^<({PAGE_PATTERN})>\s*(?:<!(.*?)>)?\s*$")
LOCUS_RE = re.compile(rf"^<({PAGE_PATTERN})\.(\d+),([@+*=&~/!])([PLCR][a-z0-9])(?:;([A-Za-z0-9]))?>\s*(.*)$")
HEADER_RE = re.compile(r"^#=IVTFF Eva[-T] 2\.0 [MDA](?: \d+)?$")
PIPELINE_VERSION = "zl-eva-codepoint-v1"
DEFAULT_SEED = "voynich-section-leaf-split-20260921-v2"
UNCERTAIN_CHARS = {ALTERNATIVE, UNREADABLE, UNKNOWN_SPAN, UNCERTAIN_SPACE}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_locus(raw: str) -> tuple[str, list[dict]]:
    """Remove editorial markup, preserving decisions and uncertainty as metadata.

    Each annotation's start/end are offsets into the returned codepoint string.
    Uncertain alternatives are collapsed to one explicit unknown-length event,
    never silently resolved to their first reading. Raw text is kept in the locus.
    """
    output: list[str] = []
    annotations: list[dict] = []
    i = 0

    def record(kind: str, source: str, value: str = "", **extra: object) -> None:
        start = len(output)
        output.extend(value)
        annotations.append({"kind": kind, "raw": source, "start": start,
                            "end": len(output), **extra})

    while i < len(raw):
        char = raw[i]
        if char.isspace():
            i += 1  # IVTFF formatting whitespace is not manuscript whitespace.
            continue
        if char == "<":
            j = raw.find(">", i + 1)
            if j < 0:
                raise ValueError("Unclosed inline comment")
            source = raw[i:j + 1]
            if source.startswith("<!"):
                record("editorial_comment", source)
            elif source in {"<->", "<~>"}:
                record("drawing_interruption", source, " ")
            elif source in {"<%>", "<$>"}:
                record("paragraph_start" if source == "<%>" else "paragraph_end", source)
            elif re.fullmatch(r"<@[A-Z]=[A-Za-z0-9@]>", source):
                record("text_tag", source, variable=source[2], tag_value=source[4])
            else:
                raise ValueError(f"Unsupported inline annotation {source!r}")
            i = j + 1
        elif char == "[":
            j = raw.find("]", i + 1)
            if j < 0:
                raise ValueError("Unclosed alternative reading")
            alternatives = raw[i + 1:j].split(":")
            if len(alternatives) not in {2, 3} or not any(alternatives):
                raise ValueError("Alternative must contain two or three options, at least one nonempty")
            for alternative in alternatives:
                if any(c in alternative for c in "[]<>., "):
                    raise ValueError("Unsupported character inside alternative reading")
                normalize_locus(alternative)  # Validate even unselected alternatives.
            record("alternative", raw[i:j + 1], ALTERNATIVE, alternatives=alternatives)
            i = j + 1
        elif char == "{":
            j = raw.find("}", i + 1)
            if j < 0:
                raise ValueError("Unclosed ligature")
            inner = raw[i + 1:j]
            if not inner or any(c in inner for c in "{}[]<>.,? "):
                raise ValueError("Invalid ligature contents")
            value, nested = normalize_locus(inner)
            start = len(output)
            record("ligature", raw[i:j + 1], value)
            annotations.extend({**a, "start": a["start"] + start,
                                "end": a["end"] + start} for a in nested)
            i = j + 1
        elif char == "@":
            match = re.match(r"@(\d{3});", raw[i:])
            if not match or not 128 <= int(match[1]) <= 255:
                raise ValueError("Invalid high-ASCII EVA escape")
            record("rare_eva", match[0], chr(RARE_BASE + int(match[1])), eva_code=int(match[1]))
            i += len(match[0])
        elif char == "?":
            j = i + 1
            while j < len(raw) and raw[j] == "?":
                j += 1
            count = j - i
            if count == 3:
                record("unknown_span", raw[i:j], UNKNOWN_SPAN)
            elif count in {1, 2}:
                record("unreadable", raw[i:j], UNREADABLE * count)
            else:
                raise ValueError("Unsupported unreadable run: use ? or ???")
            i = j
        elif char == ".":
            output.append(" ")
            i += 1
        elif char == ",":
            record("uncertain_space", char, UNCERTAIN_SPACE)
            i += 1
        elif char.isascii() and (char.isalpha() or char == "'"):
            output.append(char)
            i += 1
        else:
            raise ValueError(f"Unexpected EVA/markup character {char!r}")
    return "".join(output), annotations


def leaf_id(page_id: str) -> str:
    """Keep all foldout panels together; f85/f86 share the Rosettes bifolio."""
    if page_id == "fRos" or re.fullmatch(r"f(?:85|86)[rv]\d*", page_id):
        return "f85-f86-Ros"
    match = re.fullmatch(r"(f\d+)[rv]\d*", page_id)
    if not match:
        raise ValueError(f"Unknown page identifier {page_id!r}")
    return match[1]


def parse_ivtff(content: str, transcriber: str | None = None) -> tuple[list[dict], dict]:
    physical_lines = content.splitlines()
    if not physical_lines or not HEADER_RE.fullmatch(physical_lines[0]):
        raise ValueError("Expected IVTFF 2.0 EVA file header")
    pages: list[dict] = []
    current: dict | None = None
    page_names: set[str] = set()
    seen_loci: set[str] = set()
    excluded: list[dict] = []
    source_loci = 0
    tags: dict[str, str] = {}
    index = 1
    while index < len(physical_lines):
        line_no = index + 1
        line = physical_lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        page_match = PAGE_RE.fullmatch(line)
        if page_match:
            page_id = page_match[1]
            if page_id in page_names:
                raise ValueError(f"Repeated page header: {page_id}")
            page_names.add(page_id)
            metadata = page_match[2] or ""
            variables = dict(re.findall(r"\$([A-Z])=([A-Za-z0-9@])", metadata))
            if re.sub(r"\$[A-Z]=[A-Za-z0-9@]", "", metadata).strip():
                raise ValueError(f"Unsupported page metadata at line {line_no}")
            current = {"page_id": page_id, "leaf_id": leaf_id(page_id),
                       "metadata": {"page_variables": variables}, "loci": []}
            pages.append(current)
            tags = {}
            continue
        locus_match = LOCUS_RE.fullmatch(line)
        if not locus_match or current is None:
            raise ValueError(f"Malformed or misplaced IVTFF record at line {line_no}")
        page_id, number, locator, kind, source, raw = locus_match.groups()
        if page_id != current["page_id"]:
            raise ValueError(f"Locus/page mismatch at line {line_no}")
        while raw.rstrip().endswith("/"):
            raw = raw.rstrip()[:-1]
            if index >= len(physical_lines) or not physical_lines[index].startswith("/ "):
                raise ValueError(f"Malformed continuation after line {index}")
            raw += physical_lines[index][2:]
            index += 1
        if source is not None and transcriber is None:
            raise ValueError("Interlinear data requires an explicit single transcriber selection")
        if source is not None and source != transcriber:
            continue
        source_loci += 1
        locus_id = f"{page_id}.{number}"
        if locus_id in seen_loci:
            raise ValueError(f"Duplicate locus {locus_id}; never stack alternate transcriptions")
        seen_loci.add(locus_id)
        text, annotations = normalize_locus(raw)
        for annotation in annotations:
            if annotation["kind"] == "text_tag":
                tags[annotation["variable"]] = annotation["tag_value"]
        locus = {"locus_id": locus_id, "source_line": line_no, "locator": locator,
                 "locus_type": kind, "raw": raw, "text": text,
                 "annotations": annotations, "text_tags": dict(tags)}
        if kind == "Lx" or locator == "!":
            excluded.append({"locus_id": locus_id, "reason": "extraneous_writing" if kind == "Lx"
                             else "invalid_locus", "source_line": line_no})
        elif text:
            current["loci"].append(locus)
    retained = []
    for page in pages:
        offset = 0
        for locus in page["loci"]:
            locus["page_start"] = offset
            offset += len(locus["text"]) + 1
        page["text"] = "\n".join(locus["text"] for locus in page["loci"])
        if page["text"]:
            retained.append(page)
    if not retained:
        raise ValueError("No manuscript text retained")
    return retained, {"source_pages": len(pages), "source_loci": source_loci,
                      "excluded_loci": excluded,
                      "empty_pages_excluded": [p["page_id"] for p in pages if not p["text"]]}


def duplicate_projection(text: str) -> str:
    """Ignore formatting/boundaries for duplicate controls; keep literal EVA case."""
    return "".join(c for c in text if not c.isspace() and c not in UNCERTAIN_CHARS)


def _stratified_group_assignment(groups: dict[str, list[str]], pages: list[dict],
                                 ordered_keys: list[str], capacities: dict[str, int]) -> tuple[dict, dict]:
    """Reserve feasible section coverage, then balance metadata-only deficits.

    Multi-label leaf components are indivisible. A bounded deterministic search
    reserves one component per required section/split without exhausting another
    section's choices; a greedy completion targets 80/10/10 section proportions.
    No text likelihood, model prediction, or learned label enters this procedure.
    """
    leaf_to_group = {leaf: key for key, leaves in groups.items() for leaf in leaves}
    labels: dict[str, set[str]] = {key: set() for key in groups}
    for page in pages:
        label = page.get("metadata", {}).get("page_variables", {}).get("I", "unset")
        labels[leaf_to_group[page["leaf_id"]]].add(label)
    support: dict[str, set[str]] = defaultdict(set)
    for key, section_labels in labels.items():
        for section in section_labels:
            support[section].add(key)
    section_targets = {}
    for section, members in sorted(support.items()):
        size = len(members)
        validation = max(1, round(size * .1)) if size >= 2 else 0
        test = max(1, round(size * .1)) if size >= 3 else 0
        section_targets[section] = {"train": size - validation - test,
                                    "validation": validation, "test": test}
    required = {(section, split) for section, counts in section_targets.items()
                for split, count in counts.items() if count}
    assignments: dict[str, str] = {}
    used = Counter()
    rank = {key: index for index, key in enumerate(ordered_keys)}
    split_rank = {split: index for index, split in enumerate(capacities)}
    search_nodes = 0

    def reserve_coverage() -> bool:
        nonlocal search_nodes
        search_nodes += 1
        if search_nodes > 100_000:
            raise ValueError("Section-coverage search limit reached; explicitly review split design")
        covered = {(section, split) for key, split in assignments.items() for section in labels[key]}
        missing = required - covered
        if not missing:
            return True
        for section in support:
            needed = sum(s == section for s, _ in missing)
            available = sum(key not in assignments for key in support[section])
            if needed > available:
                return False
        choices = {(section, split): [key for key in support[section] if key not in assignments]
                   if used[split] < capacities[split] else [] for section, split in missing}
        if any(not candidates for candidates in choices.values()):
            return False
        requirement = min(choices, key=lambda item: (len(choices[item]), item[0], split_rank[item[1]]))
        _, split = requirement
        candidates = sorted(choices[requirement], key=lambda key: (
            -sum((section, split) in missing for section in labels[key]), rank[key]))
        for key in candidates:
            assignments[key] = split
            used[split] += 1
            if reserve_coverage():
                return True
            used[split] -= 1
            del assignments[key]
        return False

    if not reserve_coverage():
        raise ValueError("Section coverage is infeasible at these group quotas; explicitly revise split design")
    achieved = {section: Counter(assignments[key] for key in members if key in assignments)
                for section, members in support.items()}
    remaining = sorted((key for key in groups if key not in assignments),
                       key=lambda key: (min(len(support[s]) for s in labels[key]), rank[key]))
    for key in remaining:
        def cost(split: str) -> tuple[float, int]:
            change = 0.0
            for section in labels[key]:
                target = section_targets[section][split]
                current = achieved[section][split]
                change += ((current + 1 - target) ** 2 - (current - target) ** 2) / max(1, target)
            capacity = capacities[split]
            change += .1 * ((used[split] + 1 - capacity) ** 2 - (used[split] - capacity) ** 2) / capacity
            return change, split_rank[split]

        split = min((s for s in capacities if used[s] < capacities[s]), key=cost)
        assignments[key] = split
        used[split] += 1
        for section in labels[key]:
            achieved[section][split] += 1
    report = {
        "field": "page_variables.I (illustration type)", "strategy": "multi_label_coverage_then_squared_deficit_v2",
        "rare_section_policy": "Two independent groups: train+validation; one group: train. No group splitting.",
        "section_targets_by_component": section_targets,
        "section_achieved_by_component": {s: {split: counts[split] for split in capacities}
                                          for s, counts in sorted(achieved.items())},
        "rare_sections": {s: len(keys) for s, keys in sorted(support.items()) if len(keys) < 3},
        "coverage_search_nodes": search_nodes,
    }
    return assignments, report


def assign_splits(pages: list[dict], seed: str = DEFAULT_SEED,
                  duplicate_span: int = 128) -> dict:
    """Leaf-grouped frozen 80/10/10 split; long duplicates join leaf components.

    Exact page duplicates and shared >=128-unit spans are grouped. Short repeated
    words/labels are expected structure and are audited, not deleted or grouped.
    """
    if duplicate_span < 16:
        raise ValueError("Duplicate grouping span must be at least 16 codepoints")
    parent = {p["leaf_id"]: p["leaf_id"] for p in pages}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    span_owner: dict[str, str] = {}
    full_owner: dict[str, str] = {}
    duplicate_links: set[tuple[str, str]] = set()
    for page in pages:
        key = page["leaf_id"]
        projection = duplicate_projection(page["text"])
        if projection in full_owner:
            other = full_owner[projection]
            join(key, other)
            if key != other:
                duplicate_links.add(tuple(sorted((key, other))))
        full_owner[projection] = key
        for i in range(max(0, len(projection) - duplicate_span + 1)):
            span = projection[i:i + duplicate_span]
            if span in span_owner and span_owner[span] != key:
                other = span_owner[span]
                join(key, other)
                duplicate_links.add(tuple(sorted((key, other))))
            else:
                span_owner[span] = key
    groups: dict[str, list[str]] = defaultdict(list)
    for key in parent:
        groups[root(key)].append(key)
    group_keys = sorted(groups, key=lambda key: sha256_bytes(f"{seed}:{','.join(sorted(groups[key]))}".encode()))
    if len(group_keys) < 3:
        raise ValueError("Need at least three independent leaf/duplicate groups for held-out splits")
    n_test = max(1, round(len(group_keys) * 0.1))
    n_validation = max(1, round(len(group_keys) * 0.1))
    capacities = {"train": len(group_keys) - n_test - n_validation,
                  "validation": n_validation, "test": n_test}
    group_assignments, stratification = _stratified_group_assignment(groups, pages, group_keys, capacities)
    assignments = {}
    for key, split in group_assignments.items():
        for leaf in groups[key]:
            assignments[leaf] = split
    for page in pages:
        page["split"] = assignments[page["leaf_id"]]
    return {"schema_version": 2, "split_version": "section_leaf_v2", "seed": seed,
            "unit": "leaf_with_f85_f86_rosettes_group", "stratification": stratification,
            "fractions_by_component": {"train": .8, "validation": .1, "test": .1},
            "duplicate_span": duplicate_span, "duplicate_projection": "remove_spaces_and_uncertainty_events_v1",
            "leaf_assignments": dict(sorted(assignments.items())),
            "duplicate_leaf_links": sorted(duplicate_links),
            "independent_groups": len(groups)}


def leakage_audit(pages: list[dict], span: int = 128) -> dict:
    """Fail on leaf/page/long-span leakage and report repeated shorter loci."""
    owners: dict[tuple[str, str], str] = {}
    short_lines: dict[str, set[str]] = defaultdict(set)
    conflicts: list[dict] = []
    for page in pages:
        split = page["split"]
        projection = duplicate_projection(page["text"])
        keys = [("leaf", page["leaf_id"]), ("page", projection)]
        keys.extend(("span", projection[i:i + span]) for i in range(max(0, len(projection) - span + 1)))
        for kind, key in keys:
            if (kind, key) in owners and owners[kind, key] != split:
                conflicts.append({"kind": kind, "page_id": page["page_id"]})
            owners[kind, key] = split
        for locus in page.get("loci", []):
            short_lines[duplicate_projection(locus["text"])].add(split)
    if conflicts:
        raise ValueError(f"Cross-split contamination: {conflicts[:5]}")
    overlaps = [line for line, splits in short_lines.items() if line and len(splits) > 1]
    return {"cross_split_leaf_page_long_span_conflicts": 0, "long_span_units": span,
            "cross_split_exact_locus_types": len(overlaps),
            "longest_cross_split_exact_locus_units": max(map(len, overlaps), default=0),
            "limitation": "Short recurring labels/words and approximate or topical overlap remain; no section-transfer claim."}


def load_pages(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(raw_path: str | Path, output_dir: str | Path, manifest_dir: str | Path,
            seed: str = DEFAULT_SEED) -> dict:
    raw_path, output_dir, manifest_dir = map(Path, (raw_path, output_dir, manifest_dir))
    source = json.loads((manifest_dir / "zl3b_source.json").read_text())
    raw = raw_path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != source["sha256"]:
        raise ValueError("Source checksum mismatch; explicitly review/version a new source before preparing")
    pages, parse_report = parse_ivtff(raw.decode("ascii"))
    invalid_codepoints = sorted({ord(char) for page in pages for char in page["text"]
                                if not ((char.isascii() and (char.isalpha() or char in "' \n"))
                                        or char in UNCERTAIN_CHARS or RARE_BASE + 128 <= ord(char) <= RARE_BASE + 255)})
    if invalid_codepoints:
        raise ValueError(f"Unexpected model-facing syntax: {invalid_codepoints}")
    split = assign_splits(pages, seed)
    audit = leakage_audit(pages, split["duplicate_span"])
    split.update({"source_sha256": digest, "pipeline_version": PIPELINE_VERSION})
    frozen_path = manifest_dir / "zl3b_split.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != split:
        raise ValueError("Frozen split differs. Use a new manifest directory/version; never silently overwrite holdouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = EVATokenizer.fit(p["text"] for p in pages if p["split"] == "train")
    tokenizer.save(output_dir / "tokenizer.json")
    stats = {}
    files = {}
    for split_name in ("train", "validation", "test"):
        selected = [p for p in pages if p["split"] == split_name]
        path = output_dir / f"{split_name}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=True, sort_keys=True) + "\n" for p in selected))
        ids = [i for p in selected for i in tokenizer.encode(p["text"])]
        stats[split_name] = {"pages": len(selected), "leaves": len({p["leaf_id"] for p in selected}),
                             "loci": sum(len(p["loci"]) for p in selected),
                             "text_codepoints": sum(len(p["text"]) for p in selected),
                             "tokens_with_bos_eos": len(ids), "unknown_vocab_tokens": ids.count(tokenizer.unk_id),
                             "uncertainty_tokens": sum(i in tokenizer.uncertainty_ids for i in ids),
                             "pages_by_currier_language": dict(sorted(Counter(
                                 p.get("metadata", {}).get("page_variables", {}).get("L", "unset")
                                 for p in selected).items())),
                             "pages_by_illustration_type": dict(sorted(Counter(
                                 p.get("metadata", {}).get("page_variables", {}).get("I", "unset")
                                 for p in selected).items()))}
        files[path.name] = sha256_bytes(path.read_bytes())
    files["tokenizer.json"] = sha256_bytes((output_dir / "tokenizer.json").read_bytes())
    counts = Counter(annotation["kind"] for page in pages for locus in page["loci"]
                     for annotation in locus["annotations"])
    report = {"schema_version": 1, "pipeline_version": PIPELINE_VERSION,
              "source_sha256": digest, "vocab_size": tokenizer.vocab_size,
              "split_version": split["split_version"],
              "text_syntax_audit": {"unexpected_codepoints": invalid_codepoints,
                                    "editorial_markup_characters_retained": 0},
              "parse": parse_report, "annotation_counts": dict(sorted(counts.items())),
              "splits": stats, "leakage_audit": audit, "derived_sha256": files}
    frozen_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (manifest_dir / "zl3b_preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/ZL3b-n.txt")
    parser.add_argument("--output-dir", default="data/processed/zl3b")
    parser.add_argument("--manifest-dir", default="data/manifests")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.output_dir, args.manifest_dir), indent=2))


if __name__ == "__main__":
    main()
