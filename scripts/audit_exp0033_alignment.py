"""Independent recursive prefix/suffix replay of every EXP-0033 alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import cache
from pathlib import Path

import numpy as np

from voynich.runtime import write_json


FAMILIES = ("random_char", "periodic", "copy_mutate")
RANDOM_SEEDS = tuple(range(330400, 330420))
BOOT_SEED = 330033
BOOT_DRAWS = 2000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"{actual} != {expected}")


def recursive_alignment(observed: str, target: str) -> tuple[int, list[int]]:
    n, m = len(observed), len(target)

    @cache
    def prefix(i: int, j: int) -> int:
        if j == 0:
            return 1
        if i == 0:
            return 0
        answer = prefix(i - 1, j)
        if observed[i - 1] == target[j - 1]:
            answer += prefix(i - 1, j - 1)
        return answer

    @cache
    def suffix(i: int, j: int) -> int:
        if j == m:
            return 1
        if i == n:
            return 0
        answer = suffix(i + 1, j)
        if observed[i] == target[j]:
            answer += suffix(i + 1, j + 1)
        return answer

    total = prefix(n, m)
    if total != suffix(0, 0):
        raise AssertionError("independent prefix/suffix totals disagree")
    marginals = [sum(prefix(i, j) * suffix(i + 1, j + 1) for j in range(m) if observed[i] == target[j]) for i in range(n)]
    if sum(marginals) != m * total:
        raise AssertionError("marginal sum mismatch")
    return total, marginals


def f1(gold: list[int], pred: list[int]) -> float:
    tp = sum(g == 0 and p == 0 for g, p in zip(gold, pred, strict=True))
    fp = sum(g == 1 and p == 0 for g, p in zip(gold, pred, strict=True))
    fn = sum(g == 0 and p == 1 for g, p in zip(gold, pred, strict=True))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def audit(root: Path) -> dict:
    result_path = root / "results/EXP-0033/results.json"
    result = json.loads(result_path.read_text())
    source_path = root / "data/processed/exp0032/polish_holdout.jsonl"
    source_audit_path = root / "results/EXP-0032/data_audit.json"
    runner_path = root / "src/voynich/exp0033_alignment.py"
    for key, path in (("input_sha256", source_path), ("source_audit_sha256", source_audit_path), ("runner_sha256", runner_path)):
        if result[key] != digest(path):
            raise AssertionError(f"input/source hash mismatch: {key}")
    if result["experiment"] != "EXP-0033" or result["status"] != "exploratory_exposed_polish":
        raise AssertionError("experiment identity mismatch")
    if result["bootstrap_seed"] != BOOT_SEED or result["bootstrap_draws"] != BOOT_DRAWS or tuple(result["random_seeds"]) != RANDOM_SEEDS:
        raise AssertionError("frozen seed mismatch")
    source = [json.loads(line) for line in source_path.read_text().splitlines()]
    rows = result["rows"]
    if len(source) != 600 or len(rows) != 600:
        raise AssertionError("population mismatch")
    random_generators = [np.random.default_rng(seed) for seed in RANDOM_SEEDS]
    for i, (sample, saved) in enumerate(zip(source, rows, strict=True)):
        x, y, gold = sample["text"], sample["ciphered"], sample["mask"]
        if saved["row_index"] != i or saved["family"] != sample["filler_family"] or saved["source_block_index"] != sample["source_block_index"]:
            raise AssertionError("row/source identity mismatch")
        if "".join(ch for ch, bit in zip(x, gold, strict=True) if bit) != y:
            raise AssertionError("source-inverse mismatch")
        total, numerators = recursive_alignment(x, y)
        if str(total) != saved["total_alignments"]:
            raise AssertionError("exact alignment count mismatch")
        if total < 1:
            raise AssertionError("gold alignment missing")
        close(saved["log2_alignments"], math.log2(total))
        if saved["unique_alignment"] != (total == 1):
            raise AssertionError("uniqueness mismatch")
        if saved["observed_length"] != len(x) or saved["target_length"] != len(y):
            raise AssertionError("sequence length mismatch")
        forced = [count in (0, total) for count in numerators]
        nonspace = [j for j, ch in enumerate(x) if ch != " "]
        close(saved["forced_position_fraction"], sum(forced) / len(x))
        close(saved["forced_nonspace_fraction"], sum(forced[j] for j in nonspace) / len(nonspace))
        adjacent = sum(x[j] == x[j + 1] and gold[j] != gold[j + 1] for j in range(len(x) - 1))
        adjacent_nonspace = sum(x[j] != " " and x[j] == x[j + 1] and gold[j] != gold[j + 1] for j in range(len(x) - 1))
        if saved["adjacent_equal_swap_count"] != adjacent or saved["adjacent_equal_nonspace_swap_count"] != adjacent_nonspace:
            raise AssertionError("adjacent-swap certificate mismatch")
        sorted_positions = sorted(range(len(x)), key=lambda j: (-numerators[j], j))
        chosen = set(sorted_positions[: len(y)])
        top_mask = [int(j in chosen) for j in range(len(x))]
        if np.packbits(top_mask, bitorder="big").tobytes().hex() != saved["uniform_alignment_top_count_mask_hex"]:
            raise AssertionError("uniform alignment top-count mask mismatch")
        close(saved["uniform_alignment_top_count_null_f1"], f1(gold, top_mask))
        random_scores = []
        for rng in random_generators:
            chosen = set(map(int, rng.choice(len(x), size=len(y), replace=False)))
            random_scores.append(f1(gold, [int(j in chosen) for j in range(len(x))]))
        close(saved["matched_count_random_null_f1"], float(np.mean(random_scores)))
        trim = x.rstrip(" ")
        trim_gold = gold[: len(trim)]
        trim_y = "".join(ch for ch, bit in zip(trim, trim_gold, strict=True) if bit)
        trim_total, trim_numerators = recursive_alignment(trim, trim_y)
        if saved["trailing_spaces_stripped"] != len(x) - len(trim) or saved["trimmed_total_alignments"] != str(trim_total) or saved["trimmed_unique_alignment"] != (trim_total == 1):
            raise AssertionError("terminal-space sensitivity mismatch")
        trim_nonspace = [j for j, ch in enumerate(trim) if ch != " "]
        close(saved["trimmed_forced_nonspace_fraction"], sum(trim_numerators[j] in (0, trim_total) for j in trim_nonspace) / len(trim_nonspace))
    for family in FAMILIES:
        group = [r for r in rows if r["family"] == family]
        aggregate = result["families"][family]
        if aggregate["n"] != len(group):
            raise AssertionError("family count mismatch")
        expressions = {
            "unique_alignment_fraction": np.mean([r["unique_alignment"] for r in group]),
            "median_log2_alignments": np.median([r["log2_alignments"] for r in group]),
            "mean_forced_position_fraction": np.mean([r["forced_position_fraction"] for r in group]),
            "mean_forced_nonspace_fraction": np.mean([r["forced_nonspace_fraction"] for r in group]),
            "adjacent_equal_swap_row_fraction": np.mean([r["adjacent_equal_swap_count"] > 0 for r in group]),
            "adjacent_equal_nonspace_swap_row_fraction": np.mean([r["adjacent_equal_nonspace_swap_count"] > 0 for r in group]),
            "uniform_alignment_top_count_mean_null_f1": np.mean([r["uniform_alignment_top_count_null_f1"] for r in group]),
            "matched_count_random_mean_null_f1": np.mean([r["matched_count_random_null_f1"] for r in group]),
            "trimmed_unique_alignment_fraction": np.mean([r["trimmed_unique_alignment"] for r in group]),
            "trimmed_mean_forced_nonspace_fraction": np.mean([r["trimmed_forced_nonspace_fraction"] for r in group]),
        }
        for metric, value in expressions.items():
            close(aggregate[metric], float(value))
    copy = [r for r in rows if r["family"] == "copy_mutate"]
    indices = np.random.default_rng(BOOT_SEED).integers(0, len(copy), size=(BOOT_DRAWS, len(copy)))
    unique = np.asarray([r["unique_alignment"] for r in copy], dtype=float)
    nonspace = np.asarray([r["forced_nonspace_fraction"] for r in copy])
    for key, vector in (("copy_unique_alignment_ci95", unique), ("copy_mean_forced_nonspace_ci95", nonspace)):
        ci = np.quantile(vector[indices].mean(axis=1), [0.025, 0.975])
        for actual, expected in zip(result[key], ci, strict=True):
            close(actual, float(expected))
    substantial = result["families"]["copy_mutate"]["unique_alignment_fraction"] <= 0.5 and result["families"]["copy_mutate"]["mean_forced_nonspace_fraction"] < 0.9
    if result["substantial_combinatorial_ambiguity_by_registered_criterion"] != substantial:
        raise AssertionError("registered decision mismatch")
    return {
        "experiment": "EXP-0033",
        "passed": True,
        "result_sha256": digest(result_path),
        "auditor_sha256": digest(Path(__file__)),
        "rows_replayed": len(rows),
        "exact_count_and_marginal_rows": len(rows),
        "decision": substantial,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    output = args.root / "results/EXP-0033/audit.json"
    if output.exists():
        raise FileExistsError("EXP-0033 audit exists; refusing overwrite")
    report = audit(args.root)
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
