"""Exact source-conditioned subsequence ambiguity assay for EXP-0033."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
from pathlib import Path

import numpy as np

from voynich.runtime import write_json


EXPERIMENT = "EXP-0033"
INPUT_SHA256 = "ee78f08814a993c42e0558e5028cfdaa4d1c1e2812c9e56d82585a1e8b255664"
SOURCE_AUDIT_SHA256 = "b8f44e509a759d2661fdfde383ddf5a5fa4390b6a054d7c6c6c6b7cd0ab6b2a0"
FAMILIES = ("random_char", "periodic", "copy_mutate")
BOOTSTRAP_SEED = 330033
BOOTSTRAP_DRAWS = 2000
RANDOM_SEEDS = tuple(range(330400, 330420))
MAX_SECONDS = 180


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alignment_counts(observed: str, target: str) -> tuple[int, list[int]]:
    """Count all exact deletion alignments and alignments retaining each position."""
    n, m = len(observed), len(target)
    if m > n:
        return 0, [0] * n
    forward = [[0] * (m + 1) for _ in range(n + 1)]
    forward[0][0] = 1
    for i, glyph in enumerate(observed):
        for j in range(m + 1):
            count = forward[i][j]
            if count:
                forward[i + 1][j] += count
                if j < m and glyph == target[j]:
                    forward[i + 1][j + 1] += count
    backward = [[0] * (m + 1) for _ in range(n + 1)]
    backward[n][m] = 1
    for i in range(n - 1, -1, -1):
        for j in range(m, -1, -1):
            count = backward[i + 1][j]
            if j < m and observed[i] == target[j]:
                count += backward[i + 1][j + 1]
            backward[i][j] = count
    total = forward[n][m]
    if total != backward[0][0]:
        raise AssertionError("forward/backward alignment totals differ")
    keep = []
    for i, glyph in enumerate(observed):
        keep.append(sum(forward[i][j] * backward[i + 1][j + 1] for j in range(m) if glyph == target[j]))
    if sum(keep) != m * total:
        raise AssertionError("position marginals do not sum to target length")
    return total, keep


def null_f1(gold: list[int] | np.ndarray, pred: list[int] | np.ndarray) -> float:
    a, b = np.asarray(gold, dtype=int), np.asarray(pred, dtype=int)
    tp = int(((a == 0) & (b == 0)).sum())
    fp = int(((a == 1) & (b == 0)).sum())
    fn = int(((a == 0) & (b == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def row_summary(row: dict, index: int, random_generators: list[np.random.Generator]) -> dict:
    observed, target, gold = row["text"], row["ciphered"], row["mask"]
    if len(observed) != 128 or len(gold) != 128:
        raise ValueError("source or gold mask length drift")
    if "".join(glyph for glyph, keep in zip(observed, gold, strict=True) if keep) != target:
        raise AssertionError("gold mask does not recover independently audited source")
    total, keep_counts = alignment_counts(observed, target)
    if total < 1:
        raise AssertionError("gold alignment missing")
    forced = [count == 0 or count == total for count in keep_counts]
    nonspace = [i for i, ch in enumerate(observed) if ch != " "]
    ranked = sorted(range(len(observed)), key=lambda i: (-keep_counts[i], i))
    top = set(ranked[: len(target)])
    oracle_mask = [int(i in top) for i in range(len(observed))]
    random_f1 = []
    for rng in random_generators:
        selected = set(map(int, rng.choice(len(observed), size=len(target), replace=False)))
        random_f1.append(null_f1(gold, [int(i in selected) for i in range(len(observed))]))
    adjacent = sum(observed[i] == observed[i + 1] and gold[i] != gold[i + 1] for i in range(len(observed) - 1))
    adjacent_nonspace = sum(observed[i] != " " and observed[i] == observed[i + 1] and gold[i] != gold[i + 1] for i in range(len(observed) - 1))
    trimmed = observed.rstrip(" ")
    trimmed_gold = gold[: len(trimmed)]
    trimmed_target = "".join(ch for ch, keep in zip(trimmed, trimmed_gold, strict=True) if keep)
    trimmed_total, trimmed_keep = alignment_counts(trimmed, trimmed_target)
    trimmed_nonspace = [i for i, ch in enumerate(trimmed) if ch != " "]
    if trimmed_total < 1:
        raise AssertionError("trimmed gold alignment missing")
    return {
        "row_index": index,
        "family": row["filler_family"],
        "source_block_index": row["source_block_index"],
        "observed_length": len(observed),
        "target_length": len(target),
        "total_alignments": str(total),
        "log2_alignments": math.log2(total),
        "unique_alignment": total == 1,
        "forced_position_fraction": sum(forced) / len(forced),
        "forced_nonspace_fraction": sum(forced[i] for i in nonspace) / len(nonspace),
        "adjacent_equal_swap_count": adjacent,
        "adjacent_equal_nonspace_swap_count": adjacent_nonspace,
        "uniform_alignment_top_count_null_f1": null_f1(gold, oracle_mask),
        "matched_count_random_null_f1": float(np.mean(random_f1)),
        "uniform_alignment_top_count_mask_hex": np.packbits(oracle_mask, bitorder="big").tobytes().hex(),
        "trailing_spaces_stripped": len(observed) - len(trimmed),
        "trimmed_total_alignments": str(trimmed_total),
        "trimmed_unique_alignment": trimmed_total == 1,
        "trimmed_forced_nonspace_fraction": sum(trimmed_keep[i] in (0, trimmed_total) for i in trimmed_nonspace) / len(trimmed_nonspace),
    }


def summarize(rows: list[dict], family: str) -> dict:
    group = [row for row in rows if row["family"] == family]
    return {
        "n": len(group),
        "unique_alignment_fraction": float(np.mean([row["unique_alignment"] for row in group])),
        "median_log2_alignments": float(np.median([row["log2_alignments"] for row in group])),
        "mean_forced_position_fraction": float(np.mean([row["forced_position_fraction"] for row in group])),
        "mean_forced_nonspace_fraction": float(np.mean([row["forced_nonspace_fraction"] for row in group])),
        "adjacent_equal_swap_row_fraction": float(np.mean([row["adjacent_equal_swap_count"] > 0 for row in group])),
        "adjacent_equal_nonspace_swap_row_fraction": float(np.mean([row["adjacent_equal_nonspace_swap_count"] > 0 for row in group])),
        "uniform_alignment_top_count_mean_null_f1": float(np.mean([row["uniform_alignment_top_count_null_f1"] for row in group])),
        "matched_count_random_mean_null_f1": float(np.mean([row["matched_count_random_null_f1"] for row in group])),
        "trimmed_unique_alignment_fraction": float(np.mean([row["trimmed_unique_alignment"] for row in group])),
        "trimmed_mean_forced_nonspace_fraction": float(np.mean([row["trimmed_forced_nonspace_fraction"] for row in group])),
    }


def run(root: Path) -> dict:
    source = root / "data/processed/exp0032/polish_holdout.jsonl"
    source_audit = root / "results/EXP-0032/data_audit.json"
    if digest(source) != INPUT_SHA256 or digest(source_audit) != SOURCE_AUDIT_SHA256:
        raise ValueError("frozen input or source-audit digest mismatch")
    if not json.loads(source_audit.read_text())["passed"]:
        raise ValueError("source audit did not pass")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if len(rows) != 600:
        raise ValueError("Polish holdout population drift")
    random_generators = [np.random.default_rng(seed) for seed in RANDOM_SEEDS]
    summaries = [row_summary(row, i, random_generators) for i, row in enumerate(rows)]
    families = {family: summarize(summaries, family) for family in FAMILIES}
    if {family: data["n"] for family, data in families.items()} != {"random_char": 200, "periodic": 202, "copy_mutate": 198}:
        raise ValueError("family populations drift")
    copy = [row for row in summaries if row["family"] == "copy_mutate"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(copy), size=(BOOTSTRAP_DRAWS, len(copy)))
    unique = np.asarray([row["unique_alignment"] for row in copy], dtype=float)
    nonspace = np.asarray([row["forced_nonspace_fraction"] for row in copy], dtype=float)
    unique_ci = np.quantile(unique[indices].mean(axis=1), [0.025, 0.975]).astype(float).tolist()
    forced_ci = np.quantile(nonspace[indices].mean(axis=1), [0.025, 0.975]).astype(float).tolist()
    substantial = families["copy_mutate"]["unique_alignment_fraction"] <= 0.50 and families["copy_mutate"]["mean_forced_nonspace_fraction"] < 0.90
    return {
        "experiment": EXPERIMENT,
        "status": "exploratory_exposed_polish",
        "input_sha256": INPUT_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "runner_sha256": digest(Path(__file__)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "random_seeds": RANDOM_SEEDS,
        "copy_unique_alignment_ci95": unique_ci,
        "copy_mean_forced_nonspace_ci95": forced_ci,
        "substantial_combinatorial_ambiguity_by_registered_criterion": bool(substantial),
        "families": families,
        "rows": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"{EXPERIMENT} exceeded {MAX_SECONDS} seconds")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    out = args.root / "results/EXP-0033"
    if out.exists():
        raise FileExistsError("EXP-0033 output exists; refusing overwrite")
    result = run(args.root)
    out.mkdir(parents=True)
    write_json(out / "results.json", result)
    print(json.dumps({"experiment": EXPERIMENT, "copy": result["families"]["copy_mutate"], "copy_unique_ci95": result["copy_unique_alignment_ci95"], "copy_forced_nonspace_ci95": result["copy_mean_forced_nonspace_ci95"], "substantial": result["substantial_combinatorial_ambiguity_by_registered_criterion"]}, indent=2))


if __name__ == "__main__":
    main()
