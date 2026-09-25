"""Freeze the 18 known-chapter drawing crops before Vision extraction."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/herbal_control_0002"
OUT = RAW / "crops"
SOURCES = ROOT / "data/manifests/herbal_control_0002_sources.json"
MANIFEST = ROOT / "data/manifests/herbal_control_0002_images.json"
SOURCE_MANIFEST_SHA256 = "8b47e4abe38d27c152a96546016b6e18d2492f962b6e3a481ecf253e0f961070"

# Hand-placed on complete pages before any feature score. Each box isolates the
# illustrated chapter subject where feasible, not a modern species identity.
# Some drawings overlap historical writing; the Casanatense terbentina chapter
# depicts a person and barrel rather than a tree.
CROPS = [
    ("arthemisia", "bnf", "bnf_f011v.jpg", (60, 70, 500, 1160), "left plant; overlaps text and right plant at the edge"),
    ("arthemisia_tagantes", "bnf", "bnf_f011v.jpg", (515, 495, 455, 735), "right plant on shared folio"),
    ("torbentilla", "bnf", "bnf_f157v.jpg", (105, 600, 455, 665), "left lower plant; some prior-page ghosting"),
    ("ieribulbo", "bnf", "bnf_f084r.jpg", (55, 350, 425, 610), "left bulb and purple stalks; animal behind"),
    ("aristologia", "bnf", "bnf_f011r.jpg", (440, 305, 515, 850), "right plant; historical text behind its upper leaves"),
    ("terbentina", "bnf", "bnf_f157r.jpg", (485, 70, 490, 1190), "right tree and lower harvest figure; text overlaps branches"),
    ("arthemisia", "egerton", "egerton_f007v.jpg", (150, 735, 570, 670), "lower-left plant; root included"),
    ("arthemisia_tagantes", "egerton", "egerton_f008r.jpg", (75, 245, 430, 550), "left of two adjacent plants"),
    ("torbentilla", "egerton", "egerton_f102v.jpg", (120, 550, 480, 605), "lower-left flowering plant"),
    ("ieribulbo", "egerton", "egerton_f050v.jpg", (670, 510, 290, 535), "right bulb; excludes left and lower plants"),
    ("aristologia", "egerton", "egerton_f007r.jpg", (100, 180, 390, 500), "upper-left broad-leaf plant and root; excludes right plant"),
    ("terbentina", "egerton", "egerton_f102r.jpg", (365, 165, 610, 800), "right feathery tree; excludes lower-left barrel illustration"),
    ("arthemisia", "casanatense", "casanatense_f025r.jpg", (100, 10, 520, 930), "large central plant; text overlaps trunk"),
    ("arthemisia_tagantes", "casanatense", "casanatense_f025v.jpg", (85, 5, 575, 660), "upper plant and root; decorated border partly included"),
    ("torbentilla", "casanatense", "casanatense_f268v.jpg", (105, 70, 530, 835), "main plant; text overlaps lower stems"),
    ("ieribulbo", "casanatense", "casanatense_f133v.jpg", (215, 60, 300, 550), "central bulb and blue stalk"),
    ("aristologia", "casanatense", "casanatense_f022v.jpg", (60, 55, 580, 510), "upper heart-leaf plant and pale root; monster mostly excluded"),
    ("terbentina", "casanatense", "casanatense_f263r.jpg", (205, 190, 365, 330), "chapter depicts resin preparation with a person and barrel, not a tree"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if sha(SOURCES) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Historical source manifest changed after crop placement")
    source_manifest = json.loads(SOURCES.read_text())
    if source_manifest["status"] != "fixed-historical-source-images" or source_manifest["missing_files"]:
        raise ValueError("Historical source inventory is incomplete")
    sources = {Path(item["file"]).name: item for item in source_manifest["source_files"]}
    if len(sources) != 17 or len(CROPS) != 18:
        raise ValueError("Registered panel size changed")
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, manuscript, filename, (x, y, width, height), note in CROPS:
        source = sources[filename]
        source_path = ROOT / source["file"]
        if sha(source_path) != source["sha256"]:
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
            "chapter_class": name, "manuscript": manuscript,
            "source_file": source["file"], "source_file_sha256": source["sha256"],
            "source_commons_file_page": source["commons_file_page"],
            "crop_xywh": [x, y, width, height], "crop_file": str(output_path.relative_to(ROOT)),
            "crop_sha256": sha(output_path), "depiction_note": note,
        })
    categories = {name: f"https://commons.wikimedia.org/wiki/Category:De_{name}_in_Tractatus_de_herbis"
                  for name in {r["chapter_class"] for r in rows}}
    result = {
        "id": "HERBAL-CONTROL-0002", "status": "known-chapter-image-panel-prepared-no-scores",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "category_pages": categories, "rows": rows,
        "limitations": "Hand-selected chapter illustrations, one reviewer placed crops; some text or adjoining decoration remains. Historical chapters are not modern species; Casanatense terbentina is a barrel scene.",
    }
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps({"crops": len(run()["rows"])}))
