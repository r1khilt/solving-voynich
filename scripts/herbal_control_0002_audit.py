"""Independently audit every input, match and fixed gate in HERBAL-CONTROL-0002.

This module deliberately does not import the feature extractor or match scorer.
"""

import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/manifests/herbal_control_0002_sources.json"
IMAGE_MANIFEST = ROOT / "data/manifests/herbal_control_0002_images.json"
DISTANCES = ROOT / "results/HERBAL-CONTROL-0002/vision_distances.json"
SUMMARY = ROOT / "results/HERBAL-CONTROL-0002/summary.json"
AUDIT = ROOT / "results/HERBAL-CONTROL-0002/audit.json"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
CLASSES = (
    "arthemisia", "arthemisia_tagantes", "torbentilla", "ieribulbo",
    "aristologia", "terbentina",
)
FEATURE_METHOD = "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not reproduce: {actual!r} != {expected!r}")


def med(values: list[float]) -> float:
    ordered = sorted(values)
    return (ordered[17] + ordered[18]) / 2


def recompute(rows: list[dict], distances: list[list[float]]) -> dict:
    """Rebuild the full score from distances and registered labels alone."""
    groups = {ms: [i for i, row in enumerate(rows) if row["manuscript"] == ms]
              for ms in MANUSCRIPTS}
    if any(len(indices) != 6 for indices in groups.values()):
        raise ValueError("Non-six-cell manuscript group")
    a, b, c = (groups[ms] for ms in MANUSCRIPTS)
    norms = {
        "bnf-egerton": med([distances[i][j] for i in a for j in b]),
        "bnf-casanatense": med([distances[i][j] for i in a for j in c]),
        "egerton-casanatense": med([distances[i][j] for i in b for j in c]),
    }
    if any(value <= 0 for value in norms.values()):
        raise ValueError("Bad normalizer")

    winner = None
    least = math.inf
    runner_up = math.inf
    assignment_count = 0
    for choice_b in permutations(b):
        for choice_c in permutations(c):
            assignment_count += 1
            value = sum(
                distances[a[k]][choice_b[k]] / norms["bnf-egerton"]
                + distances[a[k]][choice_c[k]] / norms["bnf-casanatense"]
                + distances[choice_b[k]][choice_c[k]] / norms["egerton-casanatense"]
                for k in range(6)
            )
            if value < least:
                runner_up, least, winner = least, value, (choice_b, choice_c)
            elif value < runner_up:
                runner_up = value
    require_equal(assignment_count, 518_400, "assignment enumeration count")
    if winner is None:
        raise ValueError("No winning assignment")
    chosen_b, chosen_c = winner
    triplets = [(a[k], chosen_b[k], chosen_c[k]) for k in range(6)]

    raw = []
    induced = []
    for source in MANUSCRIPTS:
        for target in MANUSCRIPTS:
            if target == source:
                continue
            source_col = MANUSCRIPTS.index(source)
            target_col = MANUSCRIPTS.index(target)
            for query in groups[source]:
                options = sorted(groups[target], key=lambda j: (distances[query][j], j))
                first, next_best = options[0], options[1]
                tied = distances[query][next_best] - distances[query][first] <= 1e-7
                raw.append({"query_index": query, "gallery_manuscript": target,
                            "match_index": first, "tie": tied,
                            "correct": not tied and rows[query]["chapter_class"] == rows[first]["chapter_class"]})
            for triple in triplets:
                query, match = triple[source_col], triple[target_col]
                induced.append({"query_index": query, "gallery_manuscript": target,
                                "match_index": match,
                                "correct": rows[query]["chapter_class"] == rows[match]["chapter_class"]})
    raw_correct = sum(record["correct"] for record in raw)
    induced_correct = sum(record["correct"] for record in induced)
    complete = sum(rows[x]["chapter_class"] == rows[y]["chapter_class"] == rows[z]["chapter_class"]
                   for x, y, z in triplets)

    # Relabel the two non-anchor manuscripts independently while the recovered
    # unlabeled triplets stay fixed. Build the exact tail without using the
    # scorer's positional-label representation.
    hist = {n: 0 for n in range(7)}
    for labels_b in permutations(CLASSES):
        b_map = dict(zip(b, labels_b, strict=True))
        for labels_c in permutations(CLASSES):
            c_map = dict(zip(c, labels_c, strict=True))
            count = sum(rows[x]["chapter_class"] == b_map[y] == c_map[z]
                        for x, y, z in triplets)
            hist[count] += 1
    null_count = sum(hist.values())
    require_equal(null_count, 518_400, "null enumeration count")
    p = sum(count for n, count in hist.items() if n >= complete) / null_count
    pair = {f"{source}->{target}": sum(record["correct"] for record in induced
            if rows[record["query_index"]]["manuscript"] == source
            and record["gallery_manuscript"] == target)
            for source in MANUSCRIPTS for target in MANUSCRIPTS if source != target}
    classes = {name: sum(record["correct"] for record in induced
               if rows[record["query_index"]]["chapter_class"] == name)
               for name in CLASSES}
    passed = complete >= 4 and induced_correct >= 28 and induced_correct - raw_correct >= 2 and p <= 0.05
    return {"triplets": [list(t) for t in triplets], "raw_matches": raw, "induced_matches": induced,
            "raw_pairwise_correct_out_of_36": raw_correct, "induced_correct_out_of_36": induced_correct,
            "improvement_out_of_36": induced_correct - raw_correct,
            "complete_triplets_out_of_6": complete, "best_assignment_cost": least,
            "second_assignment_cost": runner_up, "second_minus_best_cost": runner_up - least,
            "manuscript_pair_medians": norms, "exact_null_permutations": null_count,
            "exact_null_complete_triplet_histogram": {str(k): value for k, value in hist.items() if value},
            "one_sided_exact_p": p, "registered_fresh_panel_feasibility_pass": passed,
            "induced_correct_by_ordered_pair_out_of_6": pair,
            "induced_correct_by_query_class_out_of_6": classes}


def check_inputs(source_manifest: dict, image_manifest: dict, raw: dict) -> list[dict]:
    if image_manifest["source_manifest_sha256"] != sha(SOURCE_MANIFEST):
        raise ValueError("Source manifest checksum changed")
    if source_manifest["status"] != "fixed-historical-source-images" or source_manifest["missing_files"]:
        raise ValueError("Historical panel incomplete")
    if len(source_manifest["source_files"]) != 17 or source_manifest["total_source_bytes"] > 20 * 1024 * 1024:
        raise ValueError("Source panel size changed")
    by_path = {item["file"]: item for item in source_manifest["source_files"]}
    if len(by_path) != 17:
        raise ValueError("Duplicate source paths")
    for path, item in by_path.items():
        actual = ROOT / path
        if actual.stat().st_size != item["bytes"] or sha(actual) != item["sha256"]:
            raise ValueError(f"Source file changed: {path}")
    rows = image_manifest["rows"]
    expected = {(ms, name) for ms in MANUSCRIPTS for name in CLASSES}
    if len(rows) != 18 or {(r["manuscript"], r["chapter_class"]) for r in rows} != expected:
        raise ValueError("Chapter image cells changed")
    if raw["id"] != "HERBAL-CONTROL-0002" or raw["feature_method"] != FEATURE_METHOD:
        raise ValueError("Feature extraction method changed")
    require_equal(len(raw["row_order"]), 18, "feature row count")
    with tempfile.TemporaryDirectory(prefix="herbal-control-0002-audit-") as temporary:
        for index, (row, seen) in enumerate(zip(rows, raw["row_order"], strict=True)):
            source = by_path[row["source_file"]]
            if row["source_file_sha256"] != source["sha256"] or row["source_commons_file_page"] != source["commons_file_page"]:
                raise ValueError(f"Source attribution changed at row {index}")
            for field in ("chapter_class", "manuscript", "crop_file", "crop_sha256"):
                if row[field] != seen[field]:
                    raise ValueError(f"Vision row mismatch at {index}, {field}")
            crop = ROOT / row["crop_file"]
            data = crop.read_bytes()
            if sha(crop) != row["crop_sha256"] or data[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Crop bytes changed at row {index}")
            if int.from_bytes(data[16:20], "big") != 512 or int.from_bytes(data[20:24], "big") != 512:
                raise ValueError(f"Crop dimensions changed at row {index}")
            x, y, width, height = row["crop_xywh"]
            replay = Path(temporary) / f"{index}.png"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-i", str(ROOT / row["source_file"]),
                "-vf", f"crop={width}:{height}:{x}:{y},"
                       "scale=512:512:force_original_aspect_ratio=decrease,"
                       "pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
                "-frames:v", "1", str(replay),
            ], check=True)
            if sha(replay) != row["crop_sha256"]:
                raise ValueError(f"Crop replay failed at row {index}")
    matrix = raw["distance_matrix"]
    if len(matrix) != 18 or any(len(line) != 18 for line in matrix):
        raise ValueError("Feature matrix shape changed")
    for i in range(18):
        if not math.isfinite(matrix[i][i]) or abs(matrix[i][i]) > 1e-4:
            raise ValueError("Same-image distance is not near zero")
        for j in range(18):
            if not math.isfinite(matrix[i][j]) or matrix[i][j] < -1e-6:
                raise ValueError("Invalid Vision distance")
            if abs(matrix[i][j] - matrix[j][i]) > 1e-4:
                raise ValueError("Vision matrix is not symmetric")
    return rows


def run() -> dict:
    sources = json.loads(SOURCE_MANIFEST.read_text())
    images = json.loads(IMAGE_MANIFEST.read_text())
    raw = json.loads(DISTANCES.read_text())
    summary = json.loads(SUMMARY.read_text())
    rows = check_inputs(sources, images, raw)
    rebuilt = recompute(rows, raw["distance_matrix"])
    for key, value in rebuilt.items():
        reported = summary[key]
        if isinstance(value, float):
            if not math.isclose(value, reported, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Scored value changed: {key}")
        elif key == "manuscript_pair_medians":
            if any(not math.isclose(v, reported[k], rel_tol=0, abs_tol=1e-9)
                   for k, v in value.items()):
                raise ValueError("Normalizer changed")
        else:
            require_equal(reported, value, key)
    result = {"id": "HERBAL-CONTROL-0002", "audit_pass": True,
              "source_manifest_sha256": sha(SOURCE_MANIFEST),
              "image_manifest_sha256": sha(IMAGE_MANIFEST),
              "vision_distance_sha256": sha(DISTANCES), "summary_sha256": sha(SUMMARY),
              "source_files_checked": 17, "crops_checked_and_replayed": 18,
              "matrix_cells_checked": 324, "assignments_reenumerated": 518_400,
              "null_relabelings_reenumerated": 518_400,
              "raw_correct_out_of_36": rebuilt["raw_pairwise_correct_out_of_36"],
              "induced_correct_out_of_36": rebuilt["induced_correct_out_of_36"],
              "registered_gate_pass": rebuilt["registered_fresh_panel_feasibility_pass"]}
    AUDIT.parent.mkdir(exist_ok=True)
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
