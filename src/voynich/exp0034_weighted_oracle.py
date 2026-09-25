"""EXP-0034: exact source-conditioned posterior for the corrected copy channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
from functools import lru_cache
from pathlib import Path

import numpy as np

import voynich.latent_recovery as channel
from voynich.exp0032_data import clean_polish_raw
from voynich.runtime import write_json


EXPERIMENT = "EXP-0034"
POLISH_SHA = "ee78f08814a993c42e0558e5028cfdaa4d1c1e2812c9e56d82585a1e8b255664"
EXP0033_SHA = "4f6ac9db7343ac11a6d9e0d94b329b0fc4982cdbe00c5e5f4faa4e2c19d21626"
RAW_SHA = "d5521a54c38616e4b2255af8fe8deb6e95f4428e721d140973c672a482b28b5c"
BOOTSTRAP_SEED = 340034
BOOTSTRAP_DRAWS = 2000
MAX_SECONDS = 180
NEG_INF = float("-inf")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logadd(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def emission_prob(pool: str, alphabet: str, observed: str, remaining_null: int) -> tuple[float, float, float]:
    """Probability of a visible one-char token, two-char token, or crossing first char."""
    if not pool or remaining_null < 1:
        raise ValueError("empty copy pool or null budget")
    first = observed[0]
    frequency = pool.count(first) / len(pool)
    uniform = float(first in alphabet) / len(alphabet)
    if remaining_null == 1:
        return 0.6 * frequency + 0.4 * uniform, 0.0, 0.0
    one = 0.4 * frequency + 0.2 * uniform
    crossing = 0.2 * frequency + 0.2 * uniform
    two = 0.0
    if len(observed) >= 2:
        second = observed[1]
        two = 0.2 / len(alphabet) * (
            float(first in alphabet) * pool.count(second) / len(pool)
            + float(second in alphabet) * frequency
        )
    return one, two, crossing


def null_f1(gold: list[int], predicted: list[int]) -> float:
    tp = sum(a == 0 and b == 0 for a, b in zip(gold, predicted, strict=True))
    fp = sum(a == 1 and b == 0 for a, b in zip(gold, predicted, strict=True))
    fn = sum(a == 0 and b == 1 for a, b in zip(gold, predicted, strict=True))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def weighted_posterior(observed: str, full_source: str, alphabet: str, retained_count: int,
                       gold_mask: list[int]) -> dict:
    n = len(full_source)
    k_total = round(0.3 / 0.7 * n)
    length = n + k_total
    t = min(128, length)
    if len(observed) != 128 or len(gold_mask) != 128 or not n or len(set(alphabet)) != len(alphabet):
        raise ValueError("row length/source/alphabet invalid")
    if length < 128:
        pad = 128 - length
        if observed[length:] != " " * pad or gold_mask[length:] != [1] * pad:
            raise AssertionError("deterministic padding drift")
        target_j = n
        if retained_count != n + pad:
            raise AssertionError("retained count includes incorrect padding")
    else:
        target_j = retained_count
        if not 0 <= target_j <= n:
            raise AssertionError("retained source count out of range")
    if "".join(ch for ch, keep in zip(observed, gold_mask, strict=True) if keep) != (full_source[:target_j] + (" " * max(0, 128 - length))):
        raise AssertionError("gold visible source is not the full-source prefix")

    @lru_cache(maxsize=None)
    def edges(i: int, j: int) -> tuple[tuple[int, int, float, tuple[int, ...]], ...]:
        if i >= t:
            return ()
        used_null = i - j
        remaining_sig = n - j
        remaining_null = k_total - used_null
        if min(used_null, remaining_sig, remaining_null) < 0:
            return ()
        total = remaining_sig + remaining_null
        if total <= 0:
            return ()
        out: list[tuple[int, int, float, tuple[int, ...]]] = []
        if remaining_sig and observed[i] == full_source[j]:
            out.append((i + 1, j + 1, math.log(remaining_sig / total), (1,)))
        if remaining_null:
            pool = full_source[max(0, j - 12):j] if j else (full_source[:6] or alphabet[:8])
            one, two, crossing = emission_prob(pool, alphabet, observed[i:min(t, i + 2)], remaining_null)
            prior = math.log(remaining_null / total)
            if one > 0:
                out.append((i + 1, j, prior + math.log(one), (0,)))
            if remaining_null >= 2 and i + 2 <= t and two > 0:
                out.append((i + 2, j, prior + math.log(two), (0, 0)))
            if remaining_null >= 2 and i + 1 == t and length > t and crossing > 0:
                out.append((t, j, prior + math.log(crossing), (0,)))
        return tuple(out)

    forward: list[dict[int, float]] = [{} for _ in range(t + 1)]
    forward[0][0] = 0.0
    for i in range(t):
        for j, previous in list(forward[i].items()):
            for end_i, end_j, log_weight, _bits in edges(i, j):
                forward[end_i][end_j] = logadd(forward[end_i].get(end_j, NEG_INF), previous + log_weight)
    log_evidence = forward[t].get(target_j, NEG_INF)
    if not math.isfinite(log_evidence):
        raise AssertionError("gold source-conditioned alignment has zero likelihood")

    backward: list[dict[int, float]] = [{} for _ in range(t + 1)]
    backward[t][target_j] = 0.0
    for i in range(t - 1, -1, -1):
        for j in forward[i]:
            value = NEG_INF
            for end_i, end_j, log_weight, _bits in edges(i, j):
                value = logadd(value, log_weight + backward[end_i].get(end_j, NEG_INF))
            if math.isfinite(value):
                backward[i][j] = value
    if abs(backward[0].get(0, NEG_INF) - log_evidence) > 1e-9:
        raise AssertionError("weighted forward/backward mismatch")

    keep = [0.0] * 128
    coverage = [0.0] * t
    gold_forward: list[dict[int, float]] = [{} for _ in range(t + 1)]
    gold_forward[0][0] = 0.0
    for i in range(t):
        for j, previous in forward[i].items():
            suffix = backward[i].get(j, NEG_INF)
            for end_i, end_j, log_weight, bits in edges(i, j):
                future = backward[end_i].get(end_j, NEG_INF)
                if math.isfinite(suffix) and math.isfinite(future):
                    mass = math.exp(previous + log_weight + future - log_evidence)
                    for offset, bit in enumerate(bits):
                        coverage[i + offset] += mass
                        if bit:
                            keep[i + offset] += mass
                gold_previous = gold_forward[i].get(j, NEG_INF)
                if math.isfinite(gold_previous) and list(bits) == gold_mask[i:end_i]:
                    gold_forward[end_i][end_j] = logadd(
                        gold_forward[end_i].get(end_j, NEG_INF), gold_previous + log_weight
                    )
    if max(abs(value - 1) for value in coverage) > 1e-8:
        raise AssertionError("posterior position coverage not normalized")
    keep[t:] = [1.0] * (128 - t)
    if abs(sum(keep) - retained_count) > 1e-7:
        raise AssertionError("posterior retained count differs from conditioned count")
    log_gold = gold_forward[t].get(target_j, NEG_INF)
    if not math.isfinite(log_gold):
        raise AssertionError("saved gold mask is impossible under the generator")
    gold_probability = math.exp(log_gold - log_evidence)
    if not 0 < gold_probability <= 1 + 1e-8:
        raise AssertionError("gold mask posterior probability invalid")
    ranked = sorted(range(128), key=lambda i: (-keep[i], i))
    selected = set(ranked[:retained_count])
    predicted = [int(i in selected) for i in range(128)]
    return {
        "full_source_length": n,
        "null_budget": k_total,
        "generated_length": length,
        "visible_signal_count": target_j,
        "log_evidence": log_evidence,
        "gold_mask_posterior": min(gold_probability, 1.0),
        "mean_gold_label_probability": sum(p if bit else 1-p for p, bit in zip(keep, gold_mask, strict=True)) / 128,
        "posterior_top_count_null_f1": null_f1(gold_mask, predicted),
        "posterior_mask_hex": np.packbits(np.asarray(predicted, dtype=np.uint8), bitorder="big").tobytes().hex(),
        "posterior_keep_probabilities": [round(p, 12) for p in keep],
    }


def replay_polish(root: Path) -> list[tuple[int, dict, str]]:
    manifest = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    data_path = root / "data/processed/exp0032/polish_holdout.jsonl"
    raw_path = root / "data/raw/latent_corpora/polish_34635.txt"
    if digest(data_path) != POLISH_SHA or digest(raw_path) != RAW_SHA or digest(Path(channel.__file__)) != manifest["generator_sha256"]:
        raise ValueError("frozen generator, source or derived-data checksum drift")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    flat = re.sub(r"\s+", " ", clean_polish_raw(raw_path.read_text(encoding="utf-8-sig"))).strip()
    rng = np.random.default_rng(320034)
    blocks = rng.choice(len(flat) // 128, size=600, replace=False)
    result: list[tuple[int, dict, str]] = []
    for index, (saved, block) in enumerate(zip(rows, blocks, strict=True)):
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        source_seen: list[str] = []
        original = channel.insert_nulls

        def capture(source: str, *args: object, **kwargs: object) -> tuple[str, list[int], list[str]]:
            source_seen.append(source)
            return original(source, *args, **kwargs)

        channel.insert_nulls = capture
        try:
            fresh = channel.make_sample(flat[int(block) * 128:(int(block) + 1) * 128], rng, channel.WORLD_C, 0.3, alphabet,
                                        ("random_char", "periodic", "copy_mutate"))
        finally:
            channel.insert_nulls = original
        fresh["language"] = "polish_34635"
        fresh["source_block_index"] = int(block)
        if fresh != saved or len(source_seen) != 1:
            raise AssertionError("Polish source replay differs from frozen row")
        if saved["filler_family"] == "copy_mutate":
            result.append((index, saved, source_seen[0]))
    if len(result) != 198:
        raise AssertionError("copy family count drift")
    return result


def run(root: Path) -> dict:
    if digest(root / "results/EXP-0033/results.json") != EXP0033_SHA:
        raise ValueError("EXP-0033 uniform alignment source drift")
    uniform = json.loads((root / "results/EXP-0033/results.json").read_text())
    uniform_by_index = {row["row_index"]: row for row in uniform["rows"]}
    records = []
    for index, row, full_source in replay_polish(root):
        weighted = weighted_posterior(row["text"], full_source, row["alphabet"], len(row["ciphered"]), row["mask"])
        baseline = uniform_by_index[index]["uniform_alignment_top_count_null_f1"]
        records.append({"row_index": index, "source_block_index": row["source_block_index"],
                        "uniform_alignment_top_count_null_f1": baseline,
                        "weighted_minus_uniform_null_f1": weighted["posterior_top_count_null_f1"] - baseline,
                        **weighted})
    gains = np.asarray([row["weighted_minus_uniform_null_f1"] for row in records])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(records), size=(BOOTSTRAP_DRAWS, len(records)))
    ci = np.quantile(gains[indices].mean(axis=1), [0.025, 0.975]).tolist()
    subsets = {name: [row for row in records if (row["generated_length"] < 128) == short]
               for name, short in (("short_padded", True), ("cropped_prefix", False))}
    return {
        "experiment": EXPERIMENT,
        "status": "exploratory_exposed_polish_full_source_oracle",
        "input_sha256": POLISH_SHA,
        "uniform_result_sha256": EXP0033_SHA,
        "runner_sha256": digest(Path(__file__)),
        "generator_sha256": digest(Path(channel.__file__)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "n": len(records),
        "mean_weighted_null_f1": float(np.mean([row["posterior_top_count_null_f1"] for row in records])),
        "mean_uniform_null_f1": float(np.mean([row["uniform_alignment_top_count_null_f1"] for row in records])),
        "mean_weighted_minus_uniform_null_f1": float(np.mean(gains)),
        "gain_ci95": ci,
        "mean_gold_mask_posterior": float(np.mean([row["gold_mask_posterior"] for row in records])),
        "median_gold_mask_posterior": float(np.median([row["gold_mask_posterior"] for row in records])),
        "subsets": {name: {"n": len(group), "weighted_null_f1": float(np.mean([r["posterior_top_count_null_f1"] for r in group])),
                           "uniform_null_f1": float(np.mean([r["uniform_alignment_top_count_null_f1"] for r in group])),
                           "mean_gold_mask_posterior": float(np.mean([r["gold_mask_posterior"] for r in group]))}
                    for name, group in subsets.items()},
        "material_weighting_gain_by_registered_criterion": bool(float(np.mean(gains)) >= 0.02 and ci[0] > 0),
        "rows": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0034/results.json"
    if out.exists():
        raise FileExistsError("EXP-0034 result already exists; refusing overwrite")

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError("EXP-0034 exceeded 180-second CPU cap")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    try:
        result = run(args.root)
    finally:
        signal.alarm(0)
    out.parent.mkdir(parents=True, exist_ok=False)
    write_json(out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
