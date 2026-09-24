"""Prepare source-pinned drawing-only crops for a historical herbal control."""

import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/herbal_control_0001"
OUT = RAW / "crops"
MANIFEST = ROOT / "data/manifests/herbal_control_0001_images.json"

# Wikimedia Commons copies credit the holding libraries. All bulk images stay
# ignored. The chapter categories are the known answers, not plant species IDs.
SOURCES = {
    "bnf_f060r.jpg": ("BnF Latin 6823, f.060r.jpg", "509a0dbb9f0d3403578057dfe43665803853764e8cea53a4062203d52efe36a2"),
    "bnf_f077v.jpg": ("BnF Latin 6823, f.077v.jpg", "487aa4fddbecb6f744eadecbb76270757da050e0dd58a8de075ed632eb48d756"),
    "egerton_f034v.jpg": ("BL Egerton 747, f.034v.jpg", "cfffd8cf880d7fbc997392333cc18cd0db8892634b0021509f44af3151737031"),
    "egerton_f037r.jpg": ("BL Egerton 747, f.037r.jpg", "3555ebdcb4d83657ef9e68d1c44853195f4e4888a1da4a3847165ef5b207f93f"),
    "egerton_f046r.jpg": ("BL Egerton 747, f.046r.jpg", "140ab4e3e77c586a1508da431391bea3fde23970a2d5ce9ed277f301c8e9e48f"),
    "casanatense_f094v.jpg": ("Casanatense 459, f.094v.jpg", "5ca3ca8e9c6d510821ed9da71031e331317417b5119160015738528768327d83"),
    "casanatense_f098r.jpg": ("Casanatense 459, f.098r.jpg", "2e4c8ee869d9c98379d633675ac55d8194b0dbc1f397711e6b813a8b22f720e2"),
    "casanatense_f131v.jpg": ("Casanatense 459, f.131v.jpg", "655fe5a8c1a903f7615ec22871f56c3b3ae0d4651504e71e64f34611c1a673c7"),
}

# Fixed after checking full page scans, before running any image feature score.
# Class names are manuscript chapter categories, not verified modern taxonomy.
CROPS = [
    ("enula", "bnf", "bnf_f060r.jpg", (35, 60, 450, 1260)),
    ("edera_nigra", "bnf", "bnf_f060r.jpg", (490, 385, 520, 925)),
    ("herba_vitis", "bnf", "bnf_f077v.jpg", (525, 600, 485, 705)),
    ("enula", "egerton", "egerton_f034v.jpg", (80, 565, 605, 820)),
    ("edera_nigra", "egerton", "egerton_f037r.jpg", (460, 480, 500, 620)),
    ("herba_vitis", "egerton", "egerton_f046r.jpg", (370, 695, 600, 720)),
    ("enula", "casanatense", "casanatense_f098r.jpg", (120, 15, 540, 520)),
    ("edera_nigra", "casanatense", "casanatense_f094v.jpg", (75, 65, 550, 610)),
    ("herba_vitis", "casanatense", "casanatense_f131v.jpg", (85, 55, 550, 585)),
]

CATEGORIES = {
    name: f"https://commons.wikimedia.org/wiki/Category:De_{name}_in_Tractatus_de_herbis"
    for name in ("enula", "edera_nigra", "herba_vitis")
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, manuscript, filename, (x, y, width, height) in CROPS:
        source_path = RAW / filename
        source_title, expected_sha = SOURCES[filename]
        if sha(source_path) != expected_sha:
            raise ValueError(f"Source bytes changed: {filename}")
        output_path = OUT / f"{manuscript}_{name}.png"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source_path),
            "-vf", f"crop={width}:{height}:{x}:{y},"
                   "scale=512:512:force_original_aspect_ratio=decrease,"
                   "pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
            "-frames:v", "1", str(output_path),
        ], check=True)
        rows.append({
            "chapter_class": name,
            "manuscript": manuscript,
            "source_file": str(source_path.relative_to(ROOT)),
            "source_file_sha256": expected_sha,
            "source_commons_file_page": f"https://commons.wikimedia.org/wiki/File:{quote(source_title.replace(' ', '_'))}",
            "crop_xywh": [x, y, width, height],
            "crop_file": str(output_path.relative_to(ROOT)),
            "crop_sha256": sha(output_path),
        })
    result = {
        "id": "HERBAL-CONTROL-0001",
        "status": "known-chapter-image-panel-prepared-no-scores",
        "category_pages": CATEGORIES,
        "rows": rows,
        "limitations": "Nine hand-selected chapter illustrations from three manuscripts; crop boxes selected with full images in view; chapter identity is not modern species identification; no visual feature score yet.",
    }
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps({"crops": len(run()["rows"])}))
