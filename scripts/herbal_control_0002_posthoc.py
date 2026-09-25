"""Exploratory diagnostics after the registered HERBAL-CONTROL-0002 score.

These comparisons were not in the preregistered gate and cannot change it.
"""

import hashlib
import heapq
from itertools import combinations, permutations
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/herbal_control_0002_images.json"
DISTANCES = ROOT / "results/HERBAL-CONTROL-0002/vision_distances.json"
SUMMARY = ROOT / "results/HERBAL-CONTROL-0002/summary.json"
OUTPUT = ROOT / "results/HERBAL-CONTROL-0002/posthoc.json"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    rows = json.loads(MANIFEST.read_text())["rows"]
    matrix = json.loads(DISTANCES.read_text())["distance_matrix"]
    primary = json.loads(SUMMARY.read_text())
    groups = {ms: [i for i, row in enumerate(rows) if row["manuscript"] == ms]
              for ms in MANUSCRIPTS}
    pair_maps = {}
    pair_reports = []
    directed_correct = 0
    for left, right in combinations(MANUSCRIPTS, 2):
        a, b = groups[left], groups[right]
        cost, assigned = min((sum(matrix[a[k]][p[k]] for k in range(6)), p)
                             for p in permutations(b))
        mapping = dict(zip(a, assigned, strict=True))
        pair_maps[(left, right)] = mapping
        correct = sum(rows[i]["chapter_class"] == rows[j]["chapter_class"]
                      for i, j in mapping.items())
        directed_correct += 2 * correct
        pair_reports.append({"manuscripts": [left, right], "correct_pairs_out_of_6": correct,
                             "undirected_assignment_cost": cost,
                             "index_pairs": [[i, j] for i, j in mapping.items()]})
    cycles = sum(pair_maps[("egerton", "casanatense")][pair_maps[("bnf", "egerton")][i]]
                 == pair_maps[("bnf", "casanatense")][i] for i in groups["bnf"])

    a, b, c = (groups[ms] for ms in MANUSCRIPTS)
    norm_ab = median(matrix[i][j] for i in a for j in b)
    norm_ac = median(matrix[i][j] for i in a for j in c)
    norm_bc = median(matrix[i][j] for i in b for j in c)

    def cost(pb, pc):
        return sum(matrix[a[k]][pb[k]] / norm_ab + matrix[a[k]][pc[k]] / norm_ac
                   + matrix[pb[k]][pc[k]] / norm_bc for k in range(6))

    known_b = tuple(next(j for j in b if rows[j]["chapter_class"] == rows[i]["chapter_class"]) for i in a)
    known_c = tuple(next(j for j in c if rows[j]["chapter_class"] == rows[i]["chapter_class"]) for i in a)
    known_cost = cost(known_b, known_c)
    rank = 1
    top = []
    for pb in permutations(b):
        for pc in permutations(c):
            value = cost(pb, pc)
            if value < known_cost or (value == known_cost and (pb, pc) < (known_b, known_c)):
                rank += 1
            candidate = (-value, pb, pc)
            if len(top) < 3:
                heapq.heappush(top, candidate)
            elif candidate > top[0]:
                heapq.heapreplace(top, candidate)
    ranked = []
    for negative, pb, pc in sorted(top, reverse=True):
        triples = [[a[k], pb[k], pc[k]] for k in range(6)]
        complete = sum(len({rows[i]["chapter_class"] for i in triple}) == 1 for triple in triples)
        ranked.append({"cost": -negative, "complete_triplets_out_of_6": complete,
                       "triplets": triples})
    if ranked[0]["triplets"] != primary["triplets"]:
        raise ValueError("Posthoc reconstruction disagrees with frozen primary assignment")
    result = {
        "id": "HERBAL-CONTROL-0002", "status": "exploratory-post-result-diagnostic",
        "input_manifest_sha256": sha(MANIFEST), "vision_distance_sha256": sha(DISTANCES),
        "registered_summary_sha256": sha(SUMMARY),
        "registered_raw_nearest_neighbor_correct_out_of_36": primary["raw_pairwise_correct_out_of_36"],
        "registered_three_way_correct_out_of_36": primary["induced_correct_out_of_36"],
        "posthoc_pairwise_bijection_correct_out_of_36": directed_correct,
        "pairwise_bijection_reports": pair_reports,
        "pairwise_bijection_three_cycles_out_of_6": cycles,
        "three_way_top_three": ranked,
        "metadata_aligned_assignment_rank_out_of_518400": rank,
        "metadata_aligned_assignment_cost": known_cost,
        "interpretation_limit": "Pairwise bijection and top-rank checks were selected after seeing the registered result. They diagnose the source of gain and ambiguity; they do not replace the frozen gate or validate Voynich plant names.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"pairwise_bijection_correct": result["posthoc_pairwise_bijection_correct_out_of_36"],
                      "three_way_correct": result["registered_three_way_correct_out_of_36"],
                      "metadata_assignment_rank": result["metadata_aligned_assignment_rank_out_of_518400"]}))
