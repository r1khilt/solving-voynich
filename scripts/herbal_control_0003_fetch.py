"""Fetch the fixed 216 Commons page JPEGs for HERBAL-CONTROL-0003.

Bulk JPEGs live under ignored data/raw/. This script resumes from verified
local files, writes a compact manifest after every completed source, obeys a
hard byte cap, and stops on HTTP 429 without scoring or viewing the images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
OUT = ROOT / "data/manifests/herbal_control_0003_sources.json"
RAW = ROOT / "data/raw/herbal_control_0003/sources"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MAX_SOURCE_BYTES = 150 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_DELAY_SECONDS = 8.0
USER_AGENT = "VoynichResearch/0.1 (historical herbal image study; github.com/r1khilt/solving-voynich)"
ROLES = ("development_known", "development_unknown", "evaluation_known", "evaluation_unknown")
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


class RateLimited(RuntimeError):
    """Commons returned HTTP 429 and source acquisition must stop."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commons_upload_url(filename_on_commons: str) -> str:
    """Resolve the public upload path for an ASCII canonical Commons filename."""
    if not filename_on_commons.isascii() or "/" in filename_on_commons:
        raise ValueError("this bounded panel requires plain ASCII Commons filenames")
    route = hashlib.md5(filename_on_commons.encode()).hexdigest()
    return ("https://upload.wikimedia.org/wikipedia/commons/"
            f"{route[0]}/{route[:2]}/{quote(filename_on_commons, safe='')}")


def requested_sources(panel: dict) -> list[dict]:
    rows = []
    pageids = set()
    for role in ROLES:
        for item in panel["roles"][role]:
            for manuscript in MANUSCRIPTS:
                source = item["pages"][manuscript]
                pageid = source["pageid"]
                if pageid in pageids:
                    raise ValueError(f"repeated source page in frozen panel: {pageid}")
                pageids.add(pageid)
                title = source["title"]
                if not title.startswith("File:") or not title.lower().endswith(".jpg"):
                    raise ValueError(f"not a Commons JPEG title: {title}")
                filename = f"{manuscript}_{pageid}.jpg"
                filename_on_commons = title.removeprefix("File:").replace(" ", "_")
                # Direct public upload URL avoids a per-image Commons redirect.
                url = commons_upload_url(filename_on_commons)
                rows.append({
                    "role": role,
                    "chapter": item["chapter"],
                    "manuscript": manuscript,
                    "pageid": pageid,
                    "commons_title": title,
                    "commons_page": "https://commons.wikimedia.org/wiki/" + quote(
                        title.replace(" ", "_"), safe=":,()"),
                    "redirect_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/" + quote(
                        filename_on_commons, safe=""),
                    "download_url": url,
                    "file": str((RAW / filename).relative_to(ROOT)),
                })
    return rows


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    run = subprocess.run([
        "curl", "--silent", "--show-error", "--location", "--max-time", "60",
        "--max-filesize", str(MAX_IMAGE_BYTES),
        "--proto", "=https", "--proto-redir", "=https", "--user-agent", USER_AGENT,
        "--output", str(part), "--write-out", "%{http_code}", url,
    ], capture_output=True, text=True, check=False)
    if run.returncode:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"curl failed ({run.returncode}): {run.stderr.strip()}")
    status = run.stdout.strip()
    if status == "429":
        part.unlink(missing_ok=True)
        raise RateLimited("Commons returned HTTP 429; stop and resume after cooldown")
    if status != "200":
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Commons returned HTTP {status} for {url}")
    raw = part.read_bytes()
    if not raw.startswith(b"\xff\xd8\xff") or len(raw) > MAX_IMAGE_BYTES:
        part.unlink(missing_ok=True)
        raise ValueError(f"not a bounded JPEG: {path.name}, {len(raw)} bytes")
    part.replace(path)


def save_manifest(rows: list[dict], panel_sha256: str, complete: bool) -> dict:
    recorded = [row for row in rows if "sha256" in row]
    result = {
        "id": "HERBAL-CONTROL-0003",
        "status": "all-fixed-source-pages-downloaded" if complete else "partial-fixed-source-acquisition",
        "panel_manifest_sha256": panel_sha256,
        "requested_source_pages": len(rows),
        "recorded_source_pages": len(recorded),
        "recorded_total_bytes": sum(row["bytes"] for row in recorded),
        "max_total_bytes": MAX_SOURCE_BYTES,
        "sources": rows,
        "limitations": "Historical chapter metadata only; source JPEGs are Git-ignored, crop locations and feature scores are separate.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return result


def run(limit: int | None = None) -> dict:
    panel_raw = PANEL.read_bytes()
    if digest(panel_raw) != PANEL_SHA256:
        raise ValueError("frozen panel SHA-256 mismatch")
    panel = json.loads(panel_raw)
    rows = requested_sources(panel)
    if len(rows) != 216:
        raise ValueError(f"expected 216 source pages, got {len(rows)}")
    last_fetch = 0.0
    new_fetches = 0
    total = 0
    for index, row in enumerate(rows):
        path = ROOT / row["file"]
        if not path.exists():
            if limit is not None and new_fetches >= limit:
                break
            if total + MAX_IMAGE_BYTES > MAX_SOURCE_BYTES:
                save_manifest(rows, PANEL_SHA256, False)
                raise ValueError("150 MiB hard cap leaves insufficient room for another bounded JPEG")
            if last_fetch:
                time.sleep(max(0.0, MIN_DELAY_SECONDS - (time.monotonic() - last_fetch)))
            try:
                download(row["download_url"], path)
            except RateLimited:
                save_manifest(rows, PANEL_SHA256, False)
                raise
            last_fetch = time.monotonic()
            new_fetches += 1
        raw = path.read_bytes()
        if not raw.startswith(b"\xff\xd8\xff") or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"invalid cached JPEG: {path.name}")
        if total + len(raw) > MAX_SOURCE_BYTES:
            save_manifest(rows, PANEL_SHA256, False)
            raise ValueError(f"150 MiB source cap would be exceeded at {path.name}")
        total += len(raw)
        row["bytes"] = len(raw)
        row["sha256"] = digest(raw)
        save_manifest(rows, PANEL_SHA256, False)
        if index % 12 == 11 or index == len(rows) - 1:
            print(f"verified {index + 1}/{len(rows)} page JPEGs; {total} bytes", flush=True)
    return save_manifest(rows, PANEL_SHA256, all("sha256" in row for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="maximum new downloads in this invocation")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        manifest = run(arguments.limit)
    except RateLimited as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({key: manifest[key] for key in (
        "status", "requested_source_pages", "recorded_source_pages", "recorded_total_bytes")}, sort_keys=True))
