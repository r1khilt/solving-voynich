"""Frozen three-manuscript matching rule for HERBAL-CONTROL-0002.

The assignment sees only Vision distances and manuscript membership. Chapter
names enter only after the assignment has been fixed, for evaluation.
"""

from collections import Counter
from itertools import permutations
import json
import math
from pathlib import Path
from statistics import median
import sys


MANUSCRIPTS = ("bnf", "egerton", "casanatense")
CLASSES = (
    "arthemisia", "arthemisia_tagantes", "torbentilla", "ieribulbo",
    "aristologia", "terbentina",
)
FEATURE_METHOD = "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"


def validate(rows: list[dict], matrix: list[list[float]]) -> dict[str, list[int]]:
    expected = {(manuscript, chapter) for manuscript in MANUSCRIPTS for chapter in CLASSES}
    if len(rows) != 18 or {(r["manuscript"], r["chapter_class"]) for r in rows} != expected:
        raise ValueError("Expected one image for each of 18 registered cells")
    if len(matrix) != 18 or any(len(line) != 18 for line in matrix):
        raise ValueError("Expected an 18 by 18 distance matrix")
    for i, line in enumerate(matrix):
        if not math.isfinite(line[i]) or abs(line[i]) > 1e-4:
            raise ValueError("Same-image distance is not approximately zero")
        for j, value in enumerate(line):
            if not math.isfinite(value) or value < -1e-6:
                raise ValueError("Non-finite or negative distance")
            if abs(value - matrix[j][i]) > 1e-4:
                raise ValueError("Asymmetric feature distance")
    return {name: [i for i, row in enumerate(rows) if row["manuscript"] == name]
            for name in MANUSCRIPTS}


def raw_retrieval(rows: list[dict], matrix: list[list[float]], groups: dict[str, list[int]]) -> list[dict]:
    records = []
    for source in MANUSCRIPTS:
        for target in MANUSCRIPTS:
            if source == target:
                continue
            for query in groups[source]:
                ranked = sorted(groups[target], key=lambda other: (matrix[query][other], other))
                first, second = ranked[:2]
                tie = math.isclose(matrix[query][first], matrix[query][second],
                                   rel_tol=0, abs_tol=1e-7)
                records.append({"query_index": query, "gallery_manuscript": target,
                                "match_index": first, "tie": tie,
                                "correct": not tie and rows[query]["chapter_class"] == rows[first]["chapter_class"]})
    return records


def best_triplets(matrix: list[list[float]], groups: dict[str, list[int]]) -> tuple[list[tuple[int, int, int]], float, float, dict[str, float]]:
    a, b, c = (groups[name] for name in MANUSCRIPTS)
    medians = {
        "bnf-egerton": median(matrix[i][j] for i in a for j in b),
        "bnf-casanatense": median(matrix[i][j] for i in a for j in c),
        "egerton-casanatense": median(matrix[i][j] for i in b for j in c),
    }
    if any(value <= 0 for value in medians.values()):
        raise ValueError("Cannot normalize by a nonpositive manuscript-pair median")
    ab = [[matrix[i][j] / medians["bnf-egerton"] for j in b] for i in a]
    ac = [[matrix[i][j] / medians["bnf-casanatense"] for j in c] for i in a]
    bc = [[matrix[i][j] / medians["egerton-casanatense"] for j in c] for i in b]
    index = tuple(range(6))
    best = math.inf
    second = math.inf
    best_pair = None
    # permutations() is lexicographic here, so keeping the first exact tie
    # implements the registered tie rule without consulting chapter labels.
    for pb in permutations(index):
        ab_total = sum(ab[i][pb[i]] for i in index)
        for pc in permutations(index):
            cost = ab_total + sum(ac[i][pc[i]] + bc[pb[i]][pc[i]] for i in index)
            if cost < best:
                second = best
                best = cost
                best_pair = (pb, pc)
            elif cost < second:
                second = cost
    if best_pair is None or not math.isfinite(second):
        raise AssertionError("Assignment enumeration was incomplete")
    pb, pc = best_pair
    return [(a[i], b[pb[i]], c[pc[i]]) for i in index], best, second, medians


def induced_retrieval(rows: list[dict], triplets: list[tuple[int, int, int]]) -> list[dict]:
    partner = {index: {MANUSCRIPTS[k]: triple[k] for k in range(3)}
               for triple in triplets for index in triple}
    records = []
    for source in MANUSCRIPTS:
        for target in MANUSCRIPTS:
            if source == target:
                continue
            for triple in triplets:
                query = triple[MANUSCRIPTS.index(source)]
                match = partner[query][target]
                records.append({"query_index": query, "gallery_manuscript": target,
                                "match_index": match,
                                "correct": rows[query]["chapter_class"] == rows[match]["chapter_class"]})
    return records


def exact_null(rows: list[dict], triplets: list[tuple[int, int, int]], observed: int) -> tuple[int, dict[str, int], float]:
    b = [rows[t[1]]["chapter_class"] for t in triplets]
    c = [rows[t[2]]["chapter_class"] for t in triplets]
    a = [rows[t[0]]["chapter_class"] for t in triplets]
    histogram: Counter[int] = Counter()
    total = 0
    for perm_b in permutations(b):
        for perm_c in permutations(c):
            complete = sum(a[i] == perm_b[i] == perm_c[i] for i in range(6))
            histogram[complete] += 1
            total += 1
    if total != 518_400:
        raise AssertionError("Exact null enumeration was incomplete")
    tail = sum(count for score, count in histogram.items() if score >= observed)
    return total, {str(k): histogram[k] for k in sorted(histogram)}, tail / total


def score(rows: list[dict], matrix: list[list[float]]) -> dict:
    groups = validate(rows, matrix)
    raw = raw_retrieval(rows, matrix, groups)
    triplets, best, second, medians = best_triplets(matrix, groups)
    induced = induced_retrieval(rows, triplets)
    raw_correct = sum(item["correct"] for item in raw)
    induced_correct = sum(item["correct"] for item in induced)
    complete = sum(len({rows[i]["chapter_class"] for i in triple}) == 1 for triple in triplets)
    null_count, histogram, p = exact_null(rows, triplets, complete)
    by_pair = {f"{source}->{target}": sum(item["correct"] for item in induced
               if rows[item["query_index"]]["manuscript"] == source
               and item["gallery_manuscript"] == target)
               for source in MANUSCRIPTS for target in MANUSCRIPTS if source != target}
    by_class = {name: sum(item["correct"] for item in induced
                if rows[item["query_index"]]["chapter_class"] == name) for name in CLASSES}
    passed = complete >= 4 and induced_correct >= 28 and induced_correct - raw_correct >= 2 and p <= 0.05
    return {"id": "HERBAL-CONTROL-0002", "status": "known-answer-three-manuscript-control-scored",
            "raw_pairwise_correct_out_of_36": raw_correct,
            "induced_correct_out_of_36": induced_correct,
            "improvement_out_of_36": induced_correct - raw_correct,
            "complete_triplets_out_of_6": complete,
            "best_assignment_cost": best, "second_assignment_cost": second,
            "second_minus_best_cost": second - best, "manuscript_pair_medians": medians,
            "exact_null_permutations": null_count, "exact_null_complete_triplet_histogram": histogram,
            "one_sided_exact_p": p, "registered_fresh_panel_feasibility_pass": passed,
            "triplets": [list(t) for t in triplets], "raw_matches": raw,
            "induced_matches": induced, "induced_correct_by_ordered_pair_out_of_6": by_pair,
            "induced_correct_by_query_class_out_of_6": by_class}


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: herbal_control_0002_match.py MANIFEST DISTANCES OUTPUT")
    manifest = json.loads(Path(sys.argv[1]).read_text())
    raw = json.loads(Path(sys.argv[2]).read_text())
    if manifest["id"] != "HERBAL-CONTROL-0002" or raw["id"] != "HERBAL-CONTROL-0002":
        raise ValueError("Experiment ID mismatch")
    if raw["feature_method"] != FEATURE_METHOD:
        raise ValueError("Feature method mismatch")
    rows = manifest["rows"]
    for row, reported in zip(rows, raw["row_order"], strict=True):
        for field in ("chapter_class", "manuscript", "crop_file", "crop_sha256"):
            if row[field] != reported[field]:
                raise ValueError(f"Vision input row mismatch: {field}")
    result = score(rows, raw["distance_matrix"])
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"raw": result["raw_pairwise_correct_out_of_36"],
                      "matched": result["induced_correct_out_of_36"],
                      "triplets": result["complete_triplets_out_of_6"],
                      "p": result["one_sided_exact_p"],
                      "pass": result["registered_fresh_panel_feasibility_pass"]}))


if __name__ == "__main__":
    main()
