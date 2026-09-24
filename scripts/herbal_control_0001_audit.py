"""Independently audit fixed cross-manuscript Vision distances and decision."""

import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/herbal_control_0001_images.json"
DISTANCES = ROOT / "results/HERBAL-CONTROL-0001/vision_distances.json"
SUMMARY = ROOT / "results/HERBAL-CONTROL-0001/summary.json"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
CLASSES = ("enula", "edera_nigra", "herba_vitis")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    raw = json.loads(DISTANCES.read_text())
    rows = manifest["rows"]
    if len(rows) != 9 or {(r["chapter_class"], r["manuscript"]) for r in rows} != {
        (name, ms) for name in CLASSES for ms in MANUSCRIPTS
    }:
        raise ValueError("Nine-cell known-answer panel drifted")
    if raw["feature_method"] != "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit":
        raise ValueError("Unexpected feature method")
    for row, reported in zip(rows, raw["row_order"], strict=True):
        for field in ("chapter_class", "manuscript", "crop_file", "crop_sha256"):
            if row[field] != reported[field]:
                raise ValueError(f"Feature row changed: {field}")
        for field, checksum in (("source_file", "source_file_sha256"),
                                ("crop_file", "crop_sha256")):
            if sha(ROOT / row[field]) != row[checksum]:
                raise ValueError(f"Input bytes changed: {row[field]}")
    matrix = raw["distance_matrix"]
    if len(matrix) != 9 or any(len(line) != 9 for line in matrix):
        raise ValueError("Distance shape changed")
    for i in range(9):
        if not math.isfinite(matrix[i][i]) or abs(matrix[i][i]) > 1e-4:
            raise ValueError("Duplicate image sanity check failed")
        for j in range(9):
            if not math.isfinite(matrix[i][j]) or matrix[i][j] < -1e-6:
                raise ValueError("Non-finite/negative distance")
            if abs(matrix[i][j] - matrix[j][i]) > 1e-4:
                raise ValueError("Asymmetric feature distance")

    retrieved = []
    for source in MANUSCRIPTS:
        for target in MANUSCRIPTS:
            if source == target:
                continue
            gallery = [j for j, r in enumerate(rows) if r["manuscript"] == target]
            for i, query in enumerate(rows):
                if query["manuscript"] != source:
                    continue
                ranked = sorted(gallery, key=lambda j: (matrix[i][j], j))
                nearest, second = ranked[:2]
                tied = math.isclose(matrix[i][nearest], matrix[i][second], rel_tol=0, abs_tol=1e-7)
                retrieved.append({"query_index": i, "gallery_manuscript": target,
                                  "nearest_index": nearest, "second_index": second,
                                  "nearest_distance": matrix[i][nearest],
                                  "second_distance": matrix[i][second],
                                  "margin_second_minus_first": matrix[i][second] - matrix[i][nearest],
                                  "tie": tied,
                                  "correct": not tied and query["chapter_class"] == rows[nearest]["chapter_class"]})
    if len(retrieved) != 18:
        raise ValueError("Expected 18 cross-manuscript queries")
    correct = sum(r["correct"] for r in retrieved)
    by_pair = {}
    for source in MANUSCRIPTS:
        for target in MANUSCRIPTS:
            if source != target:
                cells = [r for r in retrieved if rows[r["query_index"]]["manuscript"] == source
                         and r["gallery_manuscript"] == target]
                by_pair[f"{source}->{target}"] = sum(r["correct"] for r in cells)

    null_scores = []
    for chosen in product(permutations(CLASSES), repeat=len(MANUSCRIPTS)):
        renamed = {ms: dict(zip(CLASSES, perm, strict=True))
                   for ms, perm in zip(MANUSCRIPTS, chosen, strict=True)}
        score = 0
        for record in retrieved:
            if record["tie"]:
                continue
            a = rows[record["query_index"]]
            b = rows[record["nearest_index"]]
            score += renamed[a["manuscript"]][a["chapter_class"]] == renamed[b["manuscript"]][b["chapter_class"]]
        null_scores.append(score)
    if len(null_scores) != 216:
        raise ValueError("Permutation inventory changed")
    p = sum(score >= correct for score in null_scores) / len(null_scores)
    passed = correct >= 14 and all(n >= 2 for n in by_pair.values()) and p <= 0.05
    summary = {
        "id": "HERBAL-CONTROL-0001",
        "status": "known-answer-visual-control-scored",
        "source_manifest_sha256": sha(MANIFEST),
        "vision_distance_result_sha256": sha(DISTANCES),
        "feature_method": raw["feature_method"],
        "operating_system": raw["operating_system"],
        "primary_correct_out_of_18": correct,
        "ordered_manuscript_pair_correct_out_of_3": by_pair,
        "permutation_count": len(null_scores),
        "permutation_score_counts": {str(n): null_scores.count(n) for n in sorted(set(null_scores))},
        "one_sided_exact_p": p,
        "registered_pilot_feasibility_pass": passed,
        "rows": retrieved,
        "limitations": "Selective nine-image chapter panel; no random sampling, Voynich text, or botanical identification; Apple model proprietary; residual writing may enter crops.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"correct": result["primary_correct_out_of_18"],
                      "exact_p": result["one_sided_exact_p"],
                      "pass": result["registered_pilot_feasibility_pass"]}))
