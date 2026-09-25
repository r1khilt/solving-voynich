"""Freeze a source-metadata-only class split for a large open-set herbal control.

The selection uses Commons chapter tags and page IDs only. It never reads
images, crops, feature distances, or Voynich manuscript text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data/manifests/herbal_open_inventory.json"
OUT = ROOT / "data/manifests/herbal_control_0003_panel.json"
PREVIOUS_1 = ROOT / "data/manifests/herbal_control_0001_images.json"
PREVIOUS_2 = ROOT / "data/manifests/herbal_control_0002_sources.json"
INVENTORY_SHA256 = "f6fcac85eaab9f882d1a0c6b0528bc2f03f1cb40f87cd8e662d3806768256d07"
SPLIT_SEED = "HERBAL-CONTROL-0003-2026-09-24-v1"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
ROLE_COUNTS = {
    "development_known": 24,
    "development_unknown": 12,
    "evaluation_known": 24,
    "evaluation_unknown": 12,
}
FULL_PAGE = re.compile(r"f\.\s*\d+[rv]\.jpe?g$", re.IGNORECASE)
EXPOSED_CLASSES = frozenset({
    "enula", "edera nigra", "herba vitis", "arthemisia",
    "arthemisia tagantes", "torbentilla", "ieribulbo", "aristologia",
    "terbentina",
})


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prior_page_titles() -> set[str]:
    first = json.loads(PREVIOUS_1.read_text())
    second = json.loads(PREVIOUS_2.read_text())
    titles = {"File:" + unquote(row["source_commons_file_page"].split("File:", 1)[1]).replace("_", " ")
              for row in first["rows"]}
    titles.update("File:" + row["commons_title"] for row in second["source_files"])
    return titles


def candidate_classes(inventory: dict, exposed_pages: set[str]) -> tuple[dict[str, dict], dict[str, int]]:
    candidates = {}
    rejected = {"exposed_class": 0, "nonunique_full_page": 0, "exposed_page": 0}
    page_by_id = {row["pageid"]: row for row in inventory["files"]}
    for chapter, members in inventory["three_manuscript_chapters"].items():
        if chapter in EXPOSED_CLASSES:
            rejected["exposed_class"] += 1
            continue
        pages = {}
        for manuscript in MANUSCRIPTS:
            matches = [row for row in members[manuscript] if FULL_PAGE.search(row["title"])]
            if len(matches) != 1:
                break
            source = matches[0]
            pages[manuscript] = {"pageid": source["pageid"], "title": source["title"],
                                 "page_chapters": page_by_id[source["pageid"]]["chapters"]}
        if len(pages) != len(MANUSCRIPTS):
            rejected["nonunique_full_page"] += 1
            continue
        if any(row["title"] in exposed_pages for row in pages.values()):
            rejected["exposed_page"] += 1
            continue
        candidates[chapter] = pages
    return candidates, rejected


def page_disjoint_set(candidates: dict[str, dict]) -> list[str]:
    page_frequency: dict[int, int] = {}
    for pages in candidates.values():
        for row in pages.values():
            pageid = row["pageid"]
            page_frequency[pageid] = page_frequency.get(pageid, 0) + 1
    # Prefer classes whose source pages collide with fewer other candidates.
    order = sorted(candidates, key=lambda chapter: (
        sum(page_frequency[row["pageid"]] for row in candidates[chapter].values()), chapter))
    chosen = []
    used_pages: set[int] = set()
    for chapter in order:
        pages = {row["pageid"] for row in candidates[chapter].values()}
        if pages.isdisjoint(used_pages):
            chosen.append(chapter)
            used_pages.update(pages)
    return chosen


def split_classes(chosen: list[str]) -> dict[str, list[str]]:
    order = sorted(chosen, key=lambda chapter: (sha256(f"{SPLIT_SEED}|{chapter}".encode()), chapter))
    if len(order) < sum(ROLE_COUNTS.values()):
        raise ValueError(f"only {len(order)} page-disjoint classes for {sum(ROLE_COUNTS.values())} roles")
    split = {}
    start = 0
    for role, count in ROLE_COUNTS.items():
        split[role] = order[start:start + count]
        start += count
    split["reserve"] = order[start:]
    return split


def run() -> dict:
    raw = INVENTORY.read_bytes()
    if sha256(raw) != INVENTORY_SHA256:
        raise ValueError("inventory metadata changed from the source snapshot for this split")
    inventory = json.loads(raw)
    exposed = prior_page_titles()
    candidates, rejected = candidate_classes(inventory, exposed)
    chosen = page_disjoint_set(candidates)
    split = split_classes(chosen)
    output = {
        "id": "HERBAL-CONTROL-0003",
        "status": "metadata-only-source-split-before-new-image-inspection-or-score",
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "source_inventory_sha256": INVENTORY_SHA256,
        "previous_panel_manifest_sha256": {
            "HERBAL-CONTROL-0001": sha256(PREVIOUS_1.read_bytes()),
            "HERBAL-CONTROL-0002": sha256(PREVIOUS_2.read_bytes()),
        },
        "split_seed": SPLIT_SEED,
        "selection_rule": "exactly one non-detail full folio JPEG per manuscript; exclude prior classes/pages; greedy lowest page-collision degree then lexicographic class; SHA256 class split",
        "rejected_counts": rejected,
        "eligible_classes": len(candidates),
        "page_disjoint_classes": len(chosen),
        "role_counts": {role: len(classes) for role, classes in split.items()},
        "roles": {role: [{"chapter": chapter, "pages": candidates[chapter]} for chapter in classes]
                  for role, classes in split.items()},
        "limitations": "Commons contributor chapter tags are historical labels, not verified plant species; each folio may contain multiple drawings and requires pre-score crop review. No feature or text score used for selection.",
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps({"eligible_classes": result["eligible_classes"],
                      "page_disjoint_classes": result["page_disjoint_classes"],
                      "role_counts": result["role_counts"]}, sort_keys=True))
