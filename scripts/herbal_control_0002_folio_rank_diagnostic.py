"""Exploratory metadata-only rank diagnostic on the exposed six-class panel.

This is not a registered holdout: all six old labels are already exposed. The
predictor sorts anonymous crop records by folio side, using crop x only to
break a same-page tie, and aligns equal ranks between manuscripts. Labels are
read only after the rank mapping is fixed, to tally diagnostic accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/manifests/herbal_control_0002_images.json"
OUT = ROOT / "results/HERBAL-CONTROL-0002/folio_rank_exploratory.json"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
FOLIO = re.compile(r"f\.(\d+)([rv])\b")


def rank_key(row: dict) -> tuple[int, int]:
    found = FOLIO.search(row["source_commons_file_page"])
    if found is None:
        raise ValueError(f"missing folio in {row['source_commons_file_page']}")
    side = 1 if found.group(2) == "v" else 0
    return 2 * int(found.group(1)) + side, row["crop_xywh"][0]


def run() -> dict:
    raw = IMAGES.read_bytes()
    rows = json.loads(raw)["rows"]
    if len(rows) != 18 or len({(r["manuscript"], r["chapter_class"]) for r in rows}) != 18:
        raise ValueError("expected 18 unique manuscript/class crops")
    by_manuscript = {}
    for manuscript in MANUSCRIPTS:
        selected = [row for row in rows if row["manuscript"] == manuscript]
        if len(selected) != 6 or len({rank_key(row) for row in selected}) != 6:
            raise ValueError(f"six distinct folio-side/x positions required: {manuscript}")
        by_manuscript[manuscript] = sorted(selected, key=rank_key)
    directions = []
    for query in MANUSCRIPTS:
        for reference in MANUSCRIPTS:
            if query == reference:
                continue
            matches = []
            for rank, (query_row, reference_row) in enumerate(
                    zip(by_manuscript[query], by_manuscript[reference], strict=True)):
                matches.append({
                    "rank": rank, "query_class": query_row["chapter_class"],
                    "predicted_reference_class": reference_row["chapter_class"],
                    "correct": query_row["chapter_class"] == reference_row["chapter_class"],
                })
            directions.append({"query": query, "reference": reference,
                               "correct": sum(item["correct"] for item in matches),
                               "total": len(matches), "matches": matches})
    result = {
        "id": "HERBAL-CONTROL-0002-folio-rank-exploratory",
        "status": "post-result-exploratory",
        "input_images_sha256": hashlib.sha256(raw).hexdigest(),
        "predictor": "anonymous folio-side rank, x-coordinate only for same-page tie",
        "uses_image_pixels": False,
        "uses_class_names_for_prediction": False,
        "correct": sum(direction["correct"] for direction in directions),
        "total": sum(direction["total"] for direction in directions),
        "directions": directions,
        "limitation": "Selected old panel with exposed labels; a diagnostic of confounding, not a fresh holdout or proof that a vision model reads folio numbers.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"correct": result["correct"], "total": result["total"],
                      "direction_correct": [d["correct"] for d in result["directions"]]}))
