"""Inventory image-proposed plant-fragment pairs without reading their label strings."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/anchor0002"
SOURCES = {
    "knowles": ("knowles.html", "https://www.voynich.ninja/thread-4067-post-56211.html",
                "e745974055d9860241329efc3b7b0f5bb6f14114eb25bbc1f74b93bafbb4a578"),
    "q15": ("q15.html", "https://www.voynich.nu/q15/index.html",
            "25a8bb0083a2c6c09913910c52d03a699091c100294f8fe3c604e3846253f3a7"),
    "q19": ("q19.html", "https://www.voynich.nu/q19/index.html",
            "119fe32a005723833ec07a313fd87e1cd044a1f685ddd4fdd199e573c1dff1fb"),
}
REFERENCE = re.compile(r"(f\d+[rv]\d?)(?:\[(\d+)\s*,\s*(\d+)\])?")
ROW = re.compile(r'<I>(f\d+[rv]\d?), row (\d+)</I>\s*</P>\s*<TABLE\b.*?</TABLE>', re.S)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    text = {}
    provenance = {}
    for key, (filename, url, expected) in SOURCES.items():
        path = RAW / filename
        observed = sha(path)
        if observed != expected:
            raise ValueError(f"Source changed: {filename}")
        text[key] = path.read_text()
        provenance[key] = {"url": url, "sha256": observed, "ignored_raw_path": str(path.relative_to(ROOT))}
    return text, provenance


def numbered_fragments(htmls: dict[str, str]) -> dict[tuple[str, int], list[int]]:
    rows = {}
    fragments = set()
    for source in ("q15", "q19"):
        for match in ROW.finditer(htmls[source]):
            page, row = match.group(1), int(match.group(2))
            key = page, row
            if key in rows:
                raise ValueError(f"Duplicate source row: {key}")
            numbers = [int(value) for value in re.findall(r"<TD>\s*<B>(\d+)</B>", match.group())]
            if not numbers or any(number in fragments for number in numbers):
                raise ValueError(f"Missing/duplicate fragment number in {key}")
            fragments.update(numbers)
            rows[key] = numbers
    if len(rows) != 51 or len(fragments) != 239:
        raise ValueError("Unexpected source row/fragment counts")
    return rows


def numeric_locators() -> tuple[dict[int, str], dict[int, list[str]]]:
    labels = defaultdict(list)
    for split in ("train", "validation"):
        for line in (ROOT / f"data/processed/zl3b/{split}.jsonl").read_text().splitlines():
            page = json.loads(line)
            for locus in page["loci"]:
                if locus["locus_type"] != "Lf":
                    continue
                # Only editorial fragment numbers are read; no text field is used or emitted.
                for annotation in locus["annotations"]:
                    if annotation["kind"] != "editorial_comment":
                        continue
                    for value in re.findall(r"<!\s*(\d+)\s*>", annotation["raw"]):
                        number = int(value)
                        labels[number].append(locus["locus_id"])
    duplicates = {number: locus_ids for number, locus_ids in labels.items()
                  if len(locus_ids) > 1}
    unique = {number: locus_ids[0] for number, locus_ids in labels.items()
              if len(locus_ids) == 1}
    return unique, duplicates


def source_groups(html: str) -> list[str]:
    block = html.split('id="pid56205"', 1)[1].split("It needs tidying", 1)[0]
    lines = []
    for line in block.split("<br />"):
        references = list(REFERENCE.finditer(line))
        if len(references) >= 2:
            lines.append([match.group() for match in references])
    if len(lines) != 28 or sum(map(len, lines)) != 58:
        raise ValueError("Unexpected forum candidate inventory")
    return lines


def run() -> dict:
    htmls, provenance = load_sources()
    rows = numbered_fragments(htmls)
    locators, duplicate_locators = numeric_locators()
    split_manifest = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())
    groups = []
    for group_number, line in enumerate(source_groups(htmls["knowles"]), 1):
        members = []
        for ref in line:
            match = REFERENCE.fullmatch(ref)
            if match is None:
                raise ValueError(f"Bad source reference: {ref}")
            source_page = match.group(1)
            folio = re.match(r"f\d+", source_page).group()
            split = split_manifest["leaf_assignments"][folio]
            canonical = re.sub(r"(f101[rv])[12]$", r"\1", source_page)
            row = int(match.group(2)) if match.group(2) else None
            column = int(match.group(3)) if match.group(3) else None
            fragment = None
            if row is not None and column is not None:
                row_numbers = rows[(canonical, row)]
                if not 1 <= column <= len(row_numbers):
                    raise ValueError(f"Out-of-range column: {ref}")
                fragment = row_numbers[column - 1]
            locus_id = locators.get(fragment) if split != "test" else None
            members.append({"source_reference": ref, "page": canonical, "row": row,
                            "column": column, "fragment": fragment, "split": split,
                            "direct_lf_locus_id": locus_id})
        groups.append({"source_group": group_number, "members": members,
                       "all_members_direct_train_validation":
                           len(members) == 2 and all(m["direct_lf_locus_id"] for m in members)})
    qualified = [group["source_group"] for group in groups
                 if group["all_members_direct_train_validation"]]
    if qualified != [5, 16, 26]:
        raise ValueError("Numeric-locator feasibility drifted")
    result = {"id": "ANCHOR-0002", "status": "image-candidate-feasibility-only",
              "candidate_provenance": "transitive shared-large-plant matches, not verified duplicate drawings",
              "source_files": provenance,
              "split_manifest_sha256": sha(ROOT / "data/manifests/zl3b_split.json"),
              "corpus_sha256": {split: sha(ROOT / f"data/processed/zl3b/{split}.jsonl")
                                for split in ("train", "validation")},
              "numbered_fragment_count": 239, "source_row_count": 51,
              "source_group_count": len(groups), "fully_direct_group_ids": qualified,
              "duplicate_numeric_locators": duplicate_locators,
              "groups": groups,
              "limits": "No text-form comparison, no visual rerating, no test-text read."}
    output = ROOT / "data/manifests/anchor0002_candidates.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"source_groups": output["source_group_count"],
                      "numbered_fragments": output["numbered_fragment_count"],
                      "fully_direct_group_ids": output["fully_direct_group_ids"]}, indent=2))
