"""Make image-only review sheets for the nine geometrically alignable plant pairs."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/anchor0002"
VIEWS = RAW / "yale_views"
OUT = RAW / "pair_sheets"
ALIGNMENT = ROOT / "data/manifests/anchor0002_label_alignment.json"

# Generous crop boxes on Yale IIIF 1600-pixel views. They isolate the drawing;
# reviewers should compare the source page when a crop truncates a plant part.
# These rectangles were selected without reading any transcription text.
BOXES = {
    26: ("f89r1.jpg", 740, 100, 800, 410),
    46: ("f89r.jpg", 405, 970, 205, 230),
    52: ("f89v_part1.jpg", 490, 80, 290, 330),
    53: ("f89v_part1.jpg", 790, 105, 210, 290),
    54: ("f89v_part1.jpg", 1080, 100, 450, 360),
    56: ("f89v_part1.jpg", 540, 480, 350, 360),
    77: ("yale_f99r_1600.jpg", 280, 280, 360, 190),
    98: ("f99v.jpg", 680, 45, 160, 300),
    102: ("f99v.jpg", 1360, 80, 200, 350),
    110: ("f99v.jpg", 505, 975, 250, 300),
    115: ("f100r.jpg", 160, 80, 300, 380),
    119: ("f100r.jpg", 960, 80, 320, 360),
    120: ("f100r.jpg", 130, 380, 310, 400),
    131: ("f100v.jpg", 400, 55, 230, 340),
    134: ("f100v.jpg", 955, 70, 300, 360),
    233: ("f102v_part1.jpg", 1190, 730, 350, 450),
    240: ("f102v_part2.jpg", 575, 1080, 360, 420),
    241: ("f102v_part2.jpg", 890, 1080, 485, 370),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    alignment = json.loads(ALIGNMENT.read_text())
    groups = alignment["potentially_alignable_groups"]
    if [row["source_group"] for row in groups] != [5, 7, 8, 10, 12, 16, 19, 23, 26]:
        raise ValueError("Unexpected pair inventory")
    OUT.mkdir(exist_ok=True)
    file_records = {}
    for group in groups:
        group_id = group["source_group"]
        for side, member in zip("ab", group["members"], strict=True):
            fragment = member["fragment"]
            filename, x, y, width, height = BOXES[fragment]
            input_path = (RAW / filename) if filename == "yale_f99r_1600.jpg" else (VIEWS / filename)
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            output_path = OUT / f"g{group_id:02d}_{side}_n{fragment}.png"
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(input_path),
                   "-vf", f"crop={width}:{height}:{x}:{y},"
                          "scale=400:400:force_original_aspect_ratio=decrease,"
                          "pad=400:400:(ow-iw)/2:(oh-ih)/2:white",
                   "-frames:v", "1", str(output_path)]
            subprocess.run(cmd, check=True)
            file_records[str(fragment)] = {"source_view": str(input_path.relative_to(ROOT)),
                                           "source_sha256": sha(input_path),
                                           "box_xywh_1600_view": [x, y, width, height],
                                           "crop": str(output_path.relative_to(ROOT)),
                                           "crop_sha256": sha(output_path)}
    sheets = []
    for sheet_number in range(3):
        part = groups[sheet_number * 3:(sheet_number + 1) * 3]
        inputs = []
        for group in part:
            group_id = group["source_group"]
            for side, member in zip("ab", group["members"], strict=True):
                inputs.append(OUT / f"g{group_id:02d}_{side}_n{member['fragment']}.png")
        list_path = OUT / f"sheet{sheet_number + 1}.txt"
        list_path.write_text("\n".join(f"file '{path.name}'" for path in inputs) + "\n")
        sheet_path = OUT / f"sheet{sheet_number + 1}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-r", "1", "-f", "concat",
                        "-safe", "0", "-i", str(list_path), "-vf", "tile=2x3",
                        "-frames:v", "1", str(sheet_path)], check=True)
        sheets.append({"file": str(sheet_path.relative_to(ROOT)),
                       "sha256": sha(sheet_path),
                       "group_ids_top_to_bottom": [row["source_group"] for row in part]})
    result = {"id": "ANCHOR-0002", "status": "image-only-review-assets",
              "alignment_manifest_sha256": sha(ALIGNMENT),
              "fragment_crops": file_records, "sheets": sheets,
              "limitations": "Single-researcher manual rectangles; pair similarity unrated; no label text used."}
    manifest_path = ROOT / "data/manifests/anchor0002_image_review.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"sheets": [sheet["file"] for sheet in result["sheets"]],
                      "crops": len(result["fragment_crops"])}))
