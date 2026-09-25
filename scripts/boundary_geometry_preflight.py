"""Train-only cross-fit: does physical gap add edge prediction beyond ZL label?

This is development evidence for deciding whether a geometry-aware unit model
deserves a frozen validation challenge. It is not a decipherment assay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.build_boundary_geometry_train import collapse, sha


ROOT = Path(__file__).resolve().parent.parent
ROWS_SHA = "9effc6e34f3f8ab4951d07e762070ca65fa88f912f96f75cfecf9c54b22c4a32"
KAPPA_BASE = 20.0
KAPPA_GAP = 80.0
PERMUTATIONS = 100
BOOTSTRAPS = 2000
SEED = 430043


def gap_bin(value: float) -> int:
    if value <= 0:
        return 0
    if value <= 0.075:
        return 1
    if value <= 0.15:
        return 2
    return 3


def fold(leaf: str) -> int:
    return int.from_bytes(hashlib.sha256(leaf.encode()).digest()[:4], "big") % 5


def prepare(path: Path) -> list[dict]:
    if sha(path) != ROWS_SHA:
        raise AssertionError("Geometry row input changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    lines: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        lines[(row["folio"], row["visual_line"])].append(index)
    positions = {}
    for members in lines.values():
        members.sort(key=lambda i: rows[i]["visual_left_index"])
        for j, index in enumerate(members):
            positions[index] = "first" if j == 0 else "last" if j == len(members) - 1 else "mid"
    for index, row in enumerate(rows):
        left, right = collapse(row["left_word"]), collapse(row["right_word"])
        row["last"] = left[-1]
        row["next_first"] = right[0]
        row["role"] = positions[index]
        row["fold"] = fold(row["leaf"])
        row["bin"] = gap_bin(row["gap_over_median_word_width"])
    return rows


def crossfit(rows: list[dict], bins: list[int]) -> tuple[float, dict[str, dict]]:
    if len(rows) != len(bins):
        raise ValueError("Rows and bins differ")
    by_leaf: dict[str, dict] = defaultdict(lambda: {"bits_sum": 0.0, "pairs": 0})
    alphabet = sorted({row["next_first"] for row in rows})
    for held in range(5):
        train = [i for i, row in enumerate(rows) if row["fold"] != held]
        evaluate = [i for i, row in enumerate(rows) if row["fold"] == held]
        global_counts = Counter(rows[i]["next_first"] for i in train)
        total = sum(global_counts.values())
        prior = {y: (global_counts[y] + 1) / (total + len(alphabet)) for y in alphabet}
        base: dict[tuple, Counter] = defaultdict(Counter)
        refined: dict[tuple, Counter] = defaultdict(Counter)
        for index in train:
            row = rows[index]
            key = (row["last"], row["label"], row["role"])
            base[key][row["next_first"]] += 1
            refined[key + (bins[index],)][row["next_first"]] += 1
        for index in evaluate:
            row = rows[index]
            y = row["next_first"]
            key = (row["last"], row["label"], row["role"])
            counts = base[key]
            p0 = (counts[y] + KAPPA_BASE * prior[y]) / (counts.total() + KAPPA_BASE)
            gap_counts = refined[key + (bins[index],)]
            p1 = (gap_counts[y] + KAPPA_GAP * p0) / (gap_counts.total() + KAPPA_GAP)
            entry = by_leaf[row["leaf"]]
            entry["bits_sum"] += math.log2(p1 / p0)
            entry["pairs"] += 1
    bits = sum(item["bits_sum"] for item in by_leaf.values())
    pairs = sum(item["pairs"] for item in by_leaf.values())
    return bits / pairs, dict(sorted(by_leaf.items()))


def shuffled_bins(rows: list[dict], bins: list[int], rng: np.random.Generator) -> list[int]:
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["folio"], row["label"], row["last"], row["role"])].append(index)
    result = list(bins)
    for members in strata.values():
        draws = rng.permutation([bins[index] for index in members])
        for index, value in zip(members, draws, strict=True):
            result[index] = int(value)
    return result


def bootstrap(by_leaf: dict[str, dict], rng: np.random.Generator) -> list[float]:
    values = list(by_leaf.values())
    n = len(values)
    scores = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)
        bits = sum(values[int(i)]["bits_sum"] for i in indices)
        pairs = sum(values[int(i)]["pairs"] for i in indices)
        scores.append(bits / pairs)
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def run() -> dict:
    rows = prepare(ROOT / "data/processed/boundary_geometry/train_pairs.jsonl")
    bins = [row["bin"] for row in rows]
    observed, by_leaf = crossfit(rows, bins)
    rng = np.random.default_rng(SEED)
    null = [crossfit(rows, shuffled_bins(rows, bins, rng))[0] for _ in range(PERMUTATIONS)]
    interval = bootstrap(by_leaf, rng)
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["folio"], row["label"], row["last"], row["role"])].append(index)
    shufflable = sum(len(indices) for indices in strata.values() if len(indices) > 1)
    report = {"status": "train_only_exploratory", "rows_sha256": ROWS_SHA,
              "rows": len(rows), "physical_leaves": len(by_leaf),
              "bin_counts": dict(Counter(bins)), "shufflable_rows": shufflable,
              "source": "physical box gaps plus ZL labels; no validation/test text",
              "model": {"base": "next initial | previous last, ZL label, visible-line role",
                        "addition": "four absolute gap-width bins, count shrinkage",
                        "kappa_base": KAPPA_BASE, "kappa_gap": KAPPA_GAP},
              "fold_rule": "sha256(physical leaf) first 4 bytes mod 5",
              "observed_gain_bits_per_pair": observed, "leaf_bootstrap95": interval,
              "null_mean": float(np.mean(null)), "null_max": max(null),
              "null_p_one_sided": (1 + sum(value >= observed for value in null)) / (PERMUTATIONS + 1),
              "null_gains": null, "per_leaf": by_leaf, "seed": SEED}
    output = ROOT / "results/BOUNDARY-CHANNEL-0001/train_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("rows", "physical_leaves", "shufflable_rows",
                       "observed_gain_bits_per_pair", "leaf_bootstrap95",
                       "null_mean", "null_max", "null_p_one_sided")}, indent=2))
    return report


if __name__ == "__main__":
    run()
