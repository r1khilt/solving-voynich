"""Package source-proposed plant pairs with balanced shuffled-image foils."""

import hashlib
import json
from pathlib import Path
import random
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/anchor0002"
CROPS = RAW / "pair_sheets"
OUT = RAW / "blind_review"
REVIEW_MANIFEST = ROOT / "data/manifests/anchor0002_image_review.json"
ALIGNMENT = ROOT / "data/manifests/anchor0002_label_alignment.json"
SEED = 20260924


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    review = json.loads(REVIEW_MANIFEST.read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    groups = alignment["potentially_alignable_groups"]
    if len(groups) != 9:
        raise ValueError("Expected nine source-proposed pairs")
    by_fragment = review["fragment_crops"]
    for fragment, record in by_fragment.items():
        path = ROOT / record["crop"]
        if sha(path) != record["crop_sha256"]:
            raise ValueError(f"Crop changed: {fragment}")
    OUT.mkdir(exist_ok=True)
    source_rows = []
    for i, group in enumerate(groups):
        a = group["members"][0]["fragment"]
        true_b = group["members"][1]["fragment"]
        foil_b1 = groups[(i + 1) % len(groups)]["members"][1]["fragment"]
        foil_b2 = groups[(i + 3) % len(groups)]["members"][1]["fragment"]
        if len({true_b, foil_b1, foil_b2}) != 3:
            raise ValueError("Foil collision")
        source_rows.append((group["source_group"], a, true_b,
                            [true_b, foil_b1, foil_b2]))
    rng = random.Random(SEED)
    rng.shuffle(source_rows)
    key = []
    packets = []
    for item_number, (group_id, a, true_b, choices) in enumerate(source_rows, 1):
        rng.shuffle(choices)
        paths = [ROOT / by_fragment[str(n)]["crop"] for n in [a, *choices]]
        packet = OUT / f"item_{item_number:02d}.png"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for path in paths:
            cmd += ["-i", str(path)]
        cmd += ["-filter_complex", "hstack=inputs=4", "-frames:v", "1", str(packet)]
        subprocess.run(cmd, check=True)
        key.append({"item": item_number, "source_group": group_id,
                    "query_fragment": a, "choice_fragments": choices,
                    "published_option_1_based": choices.index(true_b) + 1})
        packets.append({"item": item_number, "file": str(packet.relative_to(ROOT)),
                        "sha256": sha(packet)})
    key_path = OUT / "answer_key.json"
    key_path.write_text(json.dumps({"seed": SEED, "rows": key}, indent=2) + "\n")
    instructions = OUT / "README.txt"
    instructions.write_text(
        "Image-only fragment review. For each item_XX.png, compare the leftmost drawing "
        "to the three drawings to its right. Choose 1, 2 or 3 if one looks like the "
        "same plant or distinctive plant part; choose NONE if none does, or UNSURE "
        "if you cannot tell. Note the visible morphological feature supporting your "
        "choice. Do not use the nearby writing. Do not inspect answer_key.json or "
        "the source candidate list until all nine choices are fixed. The two other "
        "images are shuffled foils, not proven nonmatches.\n"
    )
    manifest = {"id": "ANCHOR-0002", "status": "image-only-review-packet-prepared",
                "seed": SEED, "alignment_manifest_sha256": sha(ALIGNMENT),
                "image_review_manifest_sha256": sha(REVIEW_MANIFEST),
                "packet_files": packets,
                "answer_key_sha256": sha(key_path),
                "instructions_sha256": sha(instructions),
                "foil_design": "Two balanced cyclic shifts of the nine candidate B images; each B is used once as published option and twice as a shuffled foil. Foils are not verified negatives.",
                "limits": "Packet generated, not independently rated; source-proposed pairs are not established duplicates; no label text or test corpus read."}
    output = ROOT / "data/manifests/anchor0002_blind_packet.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps({"items": len(result["packet_files"]),
                      "key_sha256": result["answer_key_sha256"]}))
