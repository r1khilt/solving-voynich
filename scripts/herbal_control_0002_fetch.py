"""Fetch the fixed 17 Commons copies for HERBAL-CONTROL-0002."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/herbal_control_0002"
OUT = ROOT / "data/manifests/herbal_control_0002_sources.json"
MAX_IMAGE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 20 * 1024 * 1024

# Fixed chapter/page inventory from the six Commons category pages in the
# registration. BnF f011v contains two chapters, so its JPEG is fetched once.
FILES = [
    ("bnf_f011v.jpg", "BnF Latin 6823, f.011v.jpg"),
    ("bnf_f011r.jpg", "BnF Latin 6823, f.011r.jpg"),
    ("bnf_f084r.jpg", "BnF Latin 6823, f.084r.jpg"),
    ("bnf_f157r.jpg", "BnF Latin 6823, f.157r.jpg"),
    ("bnf_f157v.jpg", "BnF Latin 6823, f.157v.jpg"),
    ("egerton_f007v.jpg", "BL Egerton 747, f.007v.jpg"),
    ("egerton_f008r.jpg", "BL Egerton 747, f.008r.jpg"),
    ("egerton_f102v.jpg", "BL Egerton 747, f.102v.jpg"),
    ("egerton_f050v.jpg", "BL Egerton 747, f.050v.jpg"),
    ("egerton_f007r.jpg", "BL Egerton 747, f.007r.jpg"),
    ("egerton_f102r.jpg", "BL Egerton 747, f.102r.jpg"),
    ("casanatense_f025r.jpg", "Casanatense 459, f.025r.jpg"),
    ("casanatense_f025v.jpg", "Casanatense 459, f.025v.jpg"),
    ("casanatense_f268v.jpg", "Casanatense 459, f.268v.jpg"),
    ("casanatense_f133v.jpg", "Casanatense 459, f.133v.jpg"),
    ("casanatense_f022v.jpg", "Casanatense 459, f.022v.jpg"),
    ("casanatense_f263r.jpg", "Casanatense 459, f.263r.jpg"),
]


def run(*, inventory_only: bool = False) -> dict:
    RAW.mkdir(exist_ok=True)
    records = []
    missing = []
    for filename, title in FILES:
        url = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(title.replace(' ', '_'))}"
        path = RAW / filename
        if not path.exists():
            if inventory_only:
                missing.append(filename)
                continue
            temporary = path.with_suffix(".part")
            subprocess.run(["curl", "-L", "--fail", "--max-time", "30", "-sS", url,
                            "-o", str(temporary)], check=True)
            temporary.replace(path)
            time.sleep(3)
        data = path.read_bytes()
        if not data.startswith(b"\xff\xd8\xff") or len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Not a bounded JPEG: {filename}, {len(data)} bytes")
        records.append({"file": str(path.relative_to(ROOT)), "commons_title": title,
                        "commons_file_page": f"https://commons.wikimedia.org/wiki/File:{quote(title.replace(' ', '_'))}",
                        "download_url": url, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    total_bytes = sum(r["bytes"] for r in records)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError(f"Source panel exceeds 20 MiB cap: {total_bytes} bytes")
    result = {"id": "HERBAL-CONTROL-0002", "status": "fixed-historical-source-images" if not missing else "partial-source-acquisition",
              "source_files": records, "total_source_bytes": total_bytes,
              "missing_files": missing,
              "limitations": "Commons copies with user-contributed category metadata; bulk bytes are Git-ignored. Crop and score records are separate artifacts."}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--inventory-only"):
        raise SystemExit("Usage: herbal_control_0002_fetch.py [--inventory-only]")
    result = run(inventory_only=len(sys.argv) == 2)
    print(json.dumps({"source_images": len(result["source_files"]),
                      "missing_images": len(result["missing_files"]),
                      "total_bytes": result["total_source_bytes"]}))
