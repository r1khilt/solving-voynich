"""Independently replay the Commons metadata and HERBAL-CONTROL-0003 split.

No import from either inventory or selection implementation. No image/model
content is accessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data/manifests/herbal_open_inventory.json"
PANEL = ROOT / "data/manifests/herbal_control_0003_panel.json"
AUDIT = ROOT / "results/HERBAL-CONTROL-0003/source_audit.json"
MANUSCRIPT_CATEGORY = {
    "Category:BnF Latin 6823": "bnf",
    "Category:BL Egerton 747": "egerton",
    "Category:Casanatense 459": "casanatense",
}
MANUSCRIPTS = tuple(MANUSCRIPT_CATEGORY.values())
FULL_PAGE = re.compile(r"f\.\s*\d+[rv]\.jpe?g$", flags=re.IGNORECASE)
OLD_CLASSES = {
    "enula", "edera nigra", "herba vitis", "arthemisia", "arthemisia tagantes",
    "torbentilla", "ieribulbo", "aristologia", "terbentina",
}
ROLE_SIZES = (("development_known", 24), ("development_unknown", 12),
              ("evaluation_known", 24), ("evaluation_unknown", 12))


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def replay() -> dict:
    inventory_raw = INVENTORY.read_bytes()
    panel_raw = PANEL.read_bytes()
    inventory = json.loads(inventory_raw)
    panel = json.loads(panel_raw)
    check(panel["source_inventory_sha256"] == digest(inventory_raw), "inventory hash mismatch")
    members: dict[str, dict[int, str]] = {}
    categories: dict[int, list[str]] = {}
    check(len(inventory["source_responses"]) == 18, "wrong response count")
    for source in inventory["source_responses"]:
        raw = (ROOT / source["file"]).read_bytes()
        check(digest(raw) == source["sha256"] and len(raw) == source["bytes"],
              f"API cache mismatch: {source['file']}")
        obj = json.loads(raw)
        check("error" not in obj and "continue" not in obj, "incomplete API response")
        params = parse_qs(urlsplit(source["url"]).query)
        if params.get("list") == ["categorymembers"]:
            manuscript = MANUSCRIPT_CATEGORY[params["cmtitle"][0]]
            rows = obj["query"]["categorymembers"]
            check(all(row["ns"] == 6 for row in rows), "non-file category member")
            members[manuscript] = {row["pageid"]: row["title"] for row in rows}
            check(len(members[manuscript]) == len(rows), "duplicate category member")
        elif params.get("prop") == ["categories"]:
            expected_ids = {int(x) for x in params["pageids"][0].split("|")}
            rows = obj["query"]["pages"]
            check({row["pageid"] for row in rows} == expected_ids, "category page set mismatch")
            for row in rows:
                check("categories" in row and row["pageid"] not in categories,
                      "missing or repeated category page")
                categories[row["pageid"]] = [x["title"] for x in row["categories"]]
        else:
            raise AssertionError(f"unexpected API request: {source['url']}")
    check(set(members) == set(MANUSCRIPTS), "missing manuscript member list")
    membership_ids = set().union(*(set(rows) for rows in members.values()))
    check(len(membership_ids) == sum(len(rows) for rows in members.values()),
          "file appears in multiple manuscript lists")
    check(set(categories) == membership_ids, "not every member file has categories")

    chapters: dict[str, dict[str, list[tuple[int, str]]]] = {}
    files_with_tag = 0
    per_file_tags = {}
    for manuscript, rows in members.items():
        for pageid, title in rows.items():
            names = set()
            for category in categories[pageid]:
                normalized = category.replace("_", " ")
                prefix, suffix = "Category:De ", " in Tractatus de herbis"
                if normalized.casefold().startswith(prefix.casefold()) and normalized.casefold().endswith(suffix.casefold()):
                    names.add(normalized[len(prefix):-len(suffix)].casefold())
            per_file_tags[pageid] = sorted(names)
            files_with_tag += bool(names)
            for name in names:
                chapters.setdefault(name, {}).setdefault(manuscript, []).append((pageid, title))
    triples = {name: rows for name, rows in chapters.items() if set(rows) == set(MANUSCRIPTS)}
    actual_counts = {
        "files": len(membership_ids),
        "files_with_chapter_tag": files_with_tag,
        "distinct_chapter_tags": len(chapters),
        "three_manuscript_chapter_tags": len(triples),
    }
    check(actual_counts == inventory["counts"], "source inventory counts differ")
    observed_files = {(row["manuscript"], row["pageid"], row["title"],
                       tuple(row["chapters"])) for row in inventory["files"]}
    replay_files = {(m, pageid, title, tuple(per_file_tags[pageid]))
                    for m, rows in members.items() for pageid, title in rows.items()}
    check(observed_files == replay_files, "source inventory rows differ")
    for name, rows in triples.items():
        for manuscript, pairs in rows.items():
            expected = {(x["pageid"], x["title"]) for x in inventory["three_manuscript_chapters"][name][manuscript]}
            check(set(pairs) == expected, f"triple members differ for {name}/{manuscript}")
    check(set(inventory["three_manuscript_chapters"]) == set(triples), "triple set differs")

    previous_1 = (ROOT / "data/manifests/herbal_control_0001_images.json").read_bytes()
    previous_2 = (ROOT / "data/manifests/herbal_control_0002_sources.json").read_bytes()
    check(panel["previous_panel_manifest_sha256"]["HERBAL-CONTROL-0001"] == digest(previous_1),
          "prior control 1 hash differs")
    check(panel["previous_panel_manifest_sha256"]["HERBAL-CONTROL-0002"] == digest(previous_2),
          "prior control 2 hash differs")
    exposed = {"File:" + unquote(row["source_commons_file_page"].split("File:", 1)[1]).replace("_", " ")
               for row in json.loads(previous_1)["rows"]}
    exposed.update("File:" + row["commons_title"] for row in json.loads(previous_2)["source_files"])
    candidates = {}
    for name, by_manuscript in triples.items():
        if name in OLD_CLASSES:
            continue
        pages = {}
        for manuscript in MANUSCRIPTS:
            valid = [(pageid, title) for pageid, title in by_manuscript[manuscript]
                     if FULL_PAGE.search(title)]
            if len(valid) != 1:
                break
            pages[manuscript] = valid[0]
        if len(pages) == 3 and not any(title in exposed for _, title in pages.values()):
            candidates[name] = pages
    check(len(candidates) == panel["eligible_classes"], "eligible candidate count differs")
    page_degree = {pageid: sum(pageid in [p for p, _ in candidate.values()]
                               for candidate in candidates.values())
                   for candidate in candidates.values() for pageid, _ in candidate.values()}
    ordered = sorted(candidates, key=lambda name: (
        sum(page_degree[pageid] for pageid, _ in candidates[name].values()), name))
    chosen = []
    used_pages = set()
    for name in ordered:
        ids = {pageid for pageid, _ in candidates[name].values()}
        if not ids.intersection(used_pages):
            chosen.append(name)
            used_pages.update(ids)
    check(len(chosen) == panel["page_disjoint_classes"], "disjoint count differs")
    seed = panel["split_seed"]
    ranked = sorted(chosen, key=lambda name: (digest(f"{seed}|{name}".encode()), name))
    expected_roles = {}
    index = 0
    for role, count in ROLE_SIZES:
        expected_roles[role] = ranked[index:index + count]
        index += count
    expected_roles["reserve"] = ranked[index:]
    for role, names in expected_roles.items():
        rows = panel["roles"][role]
        check([row["chapter"] for row in rows] == names, f"role split differs: {role}")
        for row in rows:
            for manuscript in MANUSCRIPTS:
                recorded = row["pages"][manuscript]
                expected = candidates[row["chapter"]][manuscript]
                check((recorded["pageid"], recorded["title"]) == expected,
                      f"panel source differs: {row['chapter']}/{manuscript}")
                check(recorded["page_chapters"] == per_file_tags[recorded["pageid"]],
                      f"panel category list differs: {row['chapter']}/{manuscript}")
    return {
        "id": "HERBAL-CONTROL-0003-source-audit",
        "decision": "pass",
        "inventory_sha256": digest(inventory_raw),
        "panel_sha256": digest(panel_raw),
        "source_responses_checked": len(inventory["source_responses"]),
        "observed_counts": actual_counts,
        "eligible_classes": len(candidates),
        "page_disjoint_classes": len(chosen),
        "role_counts": {role: len(names) for role, names in expected_roles.items()},
        "distinct_selected_source_pages": len(used_pages),
        "image_or_model_scores_read": False,
    }


if __name__ == "__main__":
    outcome = replay()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": outcome["decision"],
                      "counts": outcome["observed_counts"],
                      "page_disjoint_classes": outcome["page_disjoint_classes"]}, sort_keys=True))
