"""Inventory Commons chapter tags across three historical herbal manuscripts.

Only metadata is fetched. API responses are cached under ignored data/raw/ so
an HTTP 429 does not discard successful batches. No images or model scores are
read, and this inventory is not an experiment's class split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/herbal_open_inventory"
OUT = ROOT / "data/manifests/herbal_open_inventory.json"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VoynichResearch/0.1 (historical herbal metadata study; github.com/r1khilt/solving-voynich)"
MANUSCRIPTS = {
    "bnf": "Category:BnF Latin 6823",
    "egerton": "Category:BL Egerton 747",
    "casanatense": "Category:Casanatense 459",
}
CHAPTER = re.compile(r"^Category:De (.+) in Tractatus de herbis$", re.IGNORECASE)
MIN_DELAY_SECONDS = 8.0
PAGE_BATCH = 50


class RateLimited(RuntimeError):
    """Commons requested that collection stop for now."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_url(params: dict[str, str]) -> str:
    return API + "?" + urlencode(params)


def fetch_or_cached(path: Path, url: str) -> tuple[dict, dict]:
    if path.exists():
        raw = path.read_bytes()
        source = "cached"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        command = [
            "curl", "--silent", "--show-error", "--location", "--max-time", "40",
            "--proto", "=https", "--proto-redir", "=https", "--user-agent", USER_AGENT,
            "--output", str(part), "--write-out", "%{http_code}", url,
        ]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        if run.returncode:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"curl failed ({run.returncode}): {run.stderr.strip()}")
        status = run.stdout.strip()
        if status == "429":
            part.unlink(missing_ok=True)
            raise RateLimited("Commons returned HTTP 429; stop and resume after cooldown")
        if status != "200":
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Commons returned HTTP {status}")
        raw = part.read_bytes()
        # Validate before promoting the API response to a reusable cache entry.
        json.loads(raw)
        part.replace(path)
        source = "fetched"
    data = json.loads(raw)
    if "error" in data or "continue" in data:
        raise RuntimeError(f"incomplete API response in {path.name}: {data.get('error') or data.get('continue')}")
    return data, {"file": str(path.relative_to(ROOT)), "url": url,
                  "sha256": sha256(raw), "bytes": len(raw), "source": source}


def parse_members(data: dict, manuscript: str) -> list[dict]:
    members = data["query"]["categorymembers"]
    if any(item.get("ns") != 6 for item in members):
        raise ValueError(f"non-file member in {manuscript}")
    if len(members) != len({item["pageid"] for item in members}):
        raise ValueError(f"duplicate member pageid in {manuscript}")
    return [{"manuscript": manuscript, "pageid": item["pageid"],
             "title": item["title"]} for item in members]


def parse_pages(data: dict, expected: list[dict]) -> dict[int, list[str]]:
    pages = data["query"]["pages"]
    expected_ids = {item["pageid"] for item in expected}
    seen = {page["pageid"] for page in pages}
    if len(pages) != len(expected) or seen != expected_ids:
        raise ValueError(f"category response pageids differ: {len(seen)} vs {len(expected_ids)}")
    if any("missing" in page or "categories" not in page for page in pages):
        raise ValueError("missing page or category list")
    return {page["pageid"]: [entry["title"] for entry in page["categories"]]
            for page in pages}


def chapter_tags(categories: list[str]) -> list[str]:
    return sorted({match.group(1).casefold() for title in categories
                   if (match := CHAPTER.fullmatch(title.replace("_", " ")))})


def build_inventory(records: list[dict], category_map: dict[int, list[str]],
                    requests: list[dict]) -> dict:
    files = []
    by_chapter: dict[str, dict[str, list[dict]]] = {}
    for item in records:
        tags = chapter_tags(category_map[item["pageid"]])
        row = {**item, "chapters": tags}
        files.append(row)
        for tag in tags:
            by_chapter.setdefault(tag, {}).setdefault(item["manuscript"], []).append(
                {"pageid": item["pageid"], "title": item["title"]})
    triples = {name: members for name, members in sorted(by_chapter.items())
               if len(members) == len(MANUSCRIPTS)}
    return {
        "id": "HERBAL-OPEN-INVENTORY-0001",
        "description": "Commons file-page category metadata only; chapter tags are contributor metadata, not botanical truth",
        "manuscript_categories": MANUSCRIPTS,
        "counts": {
            "files": len(files),
            "files_with_chapter_tag": sum(bool(row["chapters"]) for row in files),
            "distinct_chapter_tags": len(by_chapter),
            "three_manuscript_chapter_tags": len(triples),
        },
        "source_responses": requests,
        "files": files,
        "three_manuscript_chapters": triples,
    }


def run() -> dict:
    records = []
    requests = []
    last_fetch = 0.0
    for manuscript, category in MANUSCRIPTS.items():
        params = {"action": "query", "list": "categorymembers", "cmtitle": category,
                  "cmtype": "file", "cmlimit": "500", "format": "json"}
        path = RAW / f"members_{manuscript}.json"
        if not path.exists() and last_fetch:
            time.sleep(max(0.0, MIN_DELAY_SECONDS - (time.monotonic() - last_fetch)))
        data, provenance = fetch_or_cached(path, request_url(params))
        if provenance["source"] == "fetched":
            last_fetch = time.monotonic()
        requests.append({key: value for key, value in provenance.items() if key != "source"})
        records.extend(parse_members(data, manuscript))
    if len(records) != len({row["pageid"] for row in records}):
        raise ValueError("same file appears in multiple manuscript member lists")
    records.sort(key=lambda row: (row["manuscript"], row["pageid"]))
    category_map: dict[int, list[str]] = {}
    for start in range(0, len(records), PAGE_BATCH):
        batch = records[start:start + PAGE_BATCH]
        ids = [row["pageid"] for row in batch]
        digest = sha256("|".join(map(str, ids)).encode())[:12]
        path = RAW / f"categories_{start:04d}_{digest}.json"
        params = {"action": "query", "prop": "categories",
                  "pageids": "|".join(map(str, ids)), "cllimit": "max",
                  "clshow": "!hidden", "format": "json", "formatversion": "2"}
        if not path.exists() and last_fetch:
            time.sleep(max(0.0, MIN_DELAY_SECONDS - (time.monotonic() - last_fetch)))
        data, provenance = fetch_or_cached(path, request_url(params))
        if provenance["source"] == "fetched":
            last_fetch = time.monotonic()
        requests.append({key: value for key, value in provenance.items() if key != "source"})
        category_map.update(parse_pages(data, batch))
        print(f"cached {len(category_map)}/{len(records)} file-page category records", flush=True)
    output = build_inventory(records, category_map, requests)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    try:
        result = run()
    except RateLimited as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result["counts"], sort_keys=True))
