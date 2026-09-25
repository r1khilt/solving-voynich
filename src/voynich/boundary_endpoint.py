"""EXP-0037: position-aware boundary test with fixed line endpoints."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

from .boundary_order import ALPHABET, TRAIN_SHA, VALID_SHA, digest, load_groups, toy


def eligible(groups: list[dict]) -> list[dict]:
    return [group for group in groups if len(group["words"]) >= 4]


def edges(words: list[str]):
    n = len(words)
    for index in range(n - 1):
        role = "first" if index == 0 else "last" if index == n - 2 else "middle"
        yield role, words[index][-1], words[index + 1][0]


def fit(groups: list[dict], alphabet: tuple[str, ...] = ALPHABET) -> dict:
    tables = {role: {"joint": Counter(), "final": Counter(), "initial": Counter(), "n": 0}
              for role in ("first", "middle", "last")}
    for group in groups:
        for role, a, b in edges(group["words"]):
            if a not in alphabet or b not in alphabet:
                raise ValueError("Unknown codepoint")
            table = tables[role]
            table["joint"][(a, b)] += 1
            table["final"][a] += 1
            table["initial"][b] += 1
            table["n"] += 1
    if any(table["n"] <= 0 for table in tables.values()):
        raise ValueError("Missing boundary role")
    return {"tables": tables, "alphabet": alphabet}


def score(groups: list[dict], model: dict) -> dict[str, dict]:
    output = defaultdict(lambda: {"sum_bits": 0.0, "pairs": 0, "sections": set()})
    k = len(model["alphabet"])
    for group in groups:
        row = output[group["leaf"]]
        row["sections"].add(group["section"])
        for role, a, b in edges(group["words"]):
            table = model["tables"][role]
            p_cond = (table["joint"][(a, b)] + 1) / (table["final"][a] + k)
            p_base = (table["initial"][b] + 1) / (table["n"] + k)
            row["sum_bits"] += math.log2(p_cond / p_base)
            row["pairs"] += 1
    return {leaf: {"sum_bits": row["sum_bits"], "pairs": row["pairs"],
                   "sections": sorted(row["sections"])} for leaf, row in sorted(output.items())}


def permute(groups: list[dict], seed: int, mode: str) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for group in groups:
        words = group["words"][:]
        inside = words[1:-1]
        if mode == "endpoint":
            rng.shuffle(inside)
        elif mode == "half":
            split = len(inside) // 2
            a, b = inside[:split], inside[split:]
            rng.shuffle(a)
            rng.shuffle(b)
            inside = a + b
        else:
            raise ValueError(mode)
        result.append({**group, "words": words[:1] + inside + words[-1:]})
    return result


def aggregate(rows: dict[str, dict]) -> float:
    return sum(row["sum_bits"] for row in rows.values()) / sum(row["pairs"] for row in rows.values())


def bootstrap(real: dict[str, dict], null_leaf_sums: dict[str, list[float]]) -> list[float]:
    leaves = sorted(real)
    n_perm = len(next(iter(null_leaf_sums.values())))
    mean_null = {leaf: sum(null_leaf_sums[leaf]) / n_perm for leaf in leaves}
    rng = random.Random(370137)
    values = []
    for _ in range(2000):
        draw = rng.choices(leaves, k=len(leaves))
        count = sum(real[leaf]["pairs"] for leaf in draw)
        values.append(sum(real[leaf]["sum_bits"] - mean_null[leaf] for leaf in draw) / count)
    values.sort()
    return [values[49], values[1949]]


def run(root: Path) -> dict:
    start = time.monotonic()
    train_path = root / "data/processed/zl3b/train.jsonl"
    valid_path = root / "data/processed/zl3b/validation.jsonl"
    if digest(train_path) != TRAIN_SHA or digest(valid_path) != VALID_SHA:
        raise ValueError("Frozen corpus drift")
    train_all, _ = load_groups(train_path)
    valid_all, _ = load_groups(valid_path)
    train, valid = eligible(train_all), eligible(valid_all)
    model = fit(train)
    real = score(valid, model)
    observed = aggregate(real)
    controls = {}
    for mode, seed in (("endpoint", 370037), ("half", 370437)):
        means = []
        by_leaf = {leaf: [] for leaf in real}
        for i in range(200):
            row = score(permute(valid, seed + i, mode), model)
            means.append(aggregate(row))
            for leaf in real:
                if row[leaf]["pairs"] != real[leaf]["pairs"]:
                    raise ValueError("Pair count drift")
                by_leaf[leaf].append(row[leaf]["sum_bits"])
        null_mean = sum(means) / 200
        controls[mode] = {"seed": seed, "means": means, "by_leaf_sum_bits": by_leaf,
                          "mean": null_mean, "percentile95": sorted(means)[189],
                          "real_minus_null": observed - null_mean,
                          "leaf_bootstrap95": bootstrap(real, by_leaf)}
    toy_train, toy_valid = eligible(toy(360236, 100)), eligible(toy(360336, 30))
    toy_model = fit(toy_train, tuple("abcdefgh"))
    toy_real = aggregate(score(toy_valid, toy_model))
    toy_perm = sum(aggregate(score(permute(toy_valid, 370836 + i, "endpoint"), toy_model))
                   for i in range(100)) / 100
    toy_contrast = toy_real - toy_perm
    if toy_contrast < 0.1:
        decision = "invalid_positive_control"
    elif all(control["real_minus_null"] >= 0.02 and control["leaf_bootstrap95"][0] > 0
             and observed > control["percentile95"] for control in controls.values()):
        decision = "endpoint_robust_boundary_signal"
    else:
        decision = "not_supported"
    if time.monotonic() - start > 180:
        raise TimeoutError("EXP-0037 CPU cap exceeded")
    result = {"experiment": "EXP-0037", "status": "complete", "decision": decision,
              "date_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - start,
              "command": sys.argv, "python": platform.python_version(),
              "source_sha256": digest(Path(__file__)), "base_source_sha256": digest(root / "src/voynich/boundary_order.py"),
              "train_sha256": TRAIN_SHA, "validation_sha256": VALID_SHA,
              "train_groups": len(train), "train_pairs": sum(len(group["words"]) - 1 for group in train),
              "validation_groups": len(valid), "validation_pairs": sum(len(group["words"]) - 1 for group in valid),
              "real_by_leaf": real, "real_bits_per_pair": observed, "controls": controls,
              "toy_real_bits_per_pair": toy_real, "toy_permuted_bits_per_pair": toy_perm,
              "toy_contrast": toy_contrast,
              "limits": "Adaptive exposed-validation analysis; EVA codepoints; no mechanism or decipherment claim."}
    out = root / "results/EXP-0037/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    report = run(Path("."))
    print(json.dumps({"decision": report["decision"], "real": report["real_bits_per_pair"],
                      "endpoint_delta": report["controls"]["endpoint"]["real_minus_null"],
                      "half_delta": report["controls"]["half"]["real_minus_null"],
                      "endpoint_ci": report["controls"]["endpoint"]["leaf_bootstrap95"],
                      "half_ci": report["controls"]["half"]["leaf_bootstrap95"],
                      "toy_contrast": report["toy_contrast"]}, indent=2))
