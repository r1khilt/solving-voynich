"""Independent full-row replay and explicit-branch posterior audit for EXP-0034."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

import voynich.latent_recovery as channel
from scripts.audit_exp0032_data import capture_sample, independent_clean_polish
from voynich.runtime import write_json


MAX_SECONDS = 180
NEG_INF = float("-inf")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_log(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def independent_replay(root: Path) -> list[tuple[int, dict, str]]:
    manifest = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    raw_path = root / "data/raw/latent_corpora/polish_34635.txt"
    data_path = root / "data/processed/exp0032/polish_holdout.jsonl"
    if digest(raw_path) != manifest["source_sha256"]["polish_34635"] or digest(data_path) != manifest["derived_sha256"]["polish_holdout"]:
        raise AssertionError("independent replay source digest drift")
    flat = re.sub(r"\s+", " ", independent_clean_polish(raw_path.read_text(encoding="utf-8-sig"))).strip()
    rng = np.random.default_rng(320034)
    blocks = rng.choice(len(flat) // 128, 600, replace=False)
    saved_rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    copy = []
    for index, (saved, block) in enumerate(zip(saved_rows, blocks, strict=True)):
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        fresh, full_source = capture_sample(flat[int(block) * 128:(int(block) + 1) * 128], rng, channel.WORLD_C, alphabet)
        fresh.update(language="polish_34635", source_block_index=int(block))
        if saved != fresh or full_source is None:
            raise AssertionError("independent replay row mismatch")
        if saved["filler_family"] == "copy_mutate":
            copy.append((index, saved, full_source))
    if len(copy) != 198:
        raise AssertionError("copy count drift")
    return copy


def explicit_posterior(observed: str, source: str, alphabet: str, visible_keep: int, gold: list[int]) -> dict:
    n, k = len(source), round(0.3 / 0.7 * len(source))
    generated = n + k
    t = min(128, generated)
    target_j = n if generated < 128 else visible_keep

    @lru_cache(maxsize=None)
    def null_distribution(j: int, final_slot: bool) -> dict[str, float]:
        pool = source[max(0, j - 12):j] if j else (source[:6] or alphabet[:8])
        token_probability: dict[str, float] = defaultdict(float)
        for tok in pool:
            token_probability[tok] += 0.4 / len(pool)
            for replacement in alphabet:
                token_probability[replacement] += 0.2 / (len(pool) * len(alphabet))
                token_probability[replacement + tok] += 0.2 / (len(pool) * len(alphabet))
                token_probability[tok + replacement] += 0.2 / (len(pool) * len(alphabet))
        if abs(sum(token_probability.values()) - 1) > 1e-12:
            raise AssertionError("explicit null emission mass does not sum to one")
        if not final_slot:
            return dict(token_probability)
        truncated: dict[str, float] = defaultdict(float)
        for token, probability in token_probability.items():
            truncated[token[0]] += probability
        return dict(truncated)

    @lru_cache(maxsize=None)
    def choices(i: int, j: int) -> tuple[tuple[int, int, float, int], ...]:
        if i >= t:
            return ()
        null_used = i - j
        remaining_s, remaining_n = n - j, k - null_used
        if min(null_used, remaining_s, remaining_n) < 0 or remaining_s + remaining_n == 0:
            return ()
        outcomes = []
        denominator = remaining_s + remaining_n
        if remaining_s and observed[i] == source[j]:
            outcomes.append((i + 1, j + 1, remaining_s / denominator, 1))
        if remaining_n:
            null_probability = remaining_n / denominator
            distribution = null_distribution(j, remaining_n == 1)
            one = distribution.get(observed[i], 0.0)
            if one:
                outcomes.append((i + 1, j, null_probability * one, 0))
            if remaining_n >= 2 and i + 2 <= t:
                two = distribution.get(observed[i:i + 2], 0.0)
                if two:
                    outcomes.append((i + 2, j, null_probability * two, 0))
            if remaining_n >= 2 and i + 1 == t and generated > t:
                crossing = sum(p for token, p in distribution.items() if len(token) == 2 and token[0] == observed[i])
                if crossing:
                    outcomes.append((t, j, null_probability * crossing, 0))
        return tuple(outcomes)

    @lru_cache(maxsize=None)
    def suffix(i: int, j: int) -> float:
        if i == t:
            return 0.0 if j == target_j else NEG_INF
        value = NEG_INF
        for end_i, end_j, weight, _keep in choices(i, j):
            future = suffix(end_i, end_j)
            if math.isfinite(future):
                value = add_log(value, math.log(weight) + future)
        return value

    evidence = suffix(0, 0)
    if not math.isfinite(evidence):
        raise AssertionError("independent oracle finds no gold-source path")
    forward: dict[tuple[int, int], float] = {(0, 0): 0.0}
    for i in range(t):
        for (at, j), previous in list(forward.items()):
            if at != i:
                continue
            for end_i, end_j, weight, _keep in choices(i, j):
                key = end_i, end_j
                forward[key] = add_log(forward.get(key, NEG_INF), previous + math.log(weight))
    if abs(forward.get((t, target_j), NEG_INF) - evidence) > 1e-8:
        raise AssertionError("independent forward/suffix disagree")
    keep_probability = [0.0] * 128
    coverage = [0.0] * t
    for (i, j), previous in forward.items():
        if i == t:
            continue
        for end_i, end_j, weight, keep in choices(i, j):
            future = suffix(end_i, end_j)
            if math.isfinite(future):
                mass = math.exp(previous + math.log(weight) + future - evidence)
                for pos in range(i, end_i):
                    coverage[pos] += mass
                    if keep:
                        keep_probability[pos] += mass
    if max(abs(value - 1) for value in coverage) > 1e-8:
        raise AssertionError("independent posterior position coverage failed")
    keep_probability[t:] = [1.0] * (128 - t)

    @lru_cache(maxsize=None)
    def gold_suffix(i: int, j: int) -> float:
        if i == t:
            return 0.0 if j == target_j else NEG_INF
        value = NEG_INF
        for end_i, end_j, weight, keep in choices(i, j):
            if all(gold[pos] == keep for pos in range(i, end_i)):
                future = gold_suffix(end_i, end_j)
                if math.isfinite(future):
                    value = add_log(value, math.log(weight) + future)
        return value

    gold_probability = math.exp(gold_suffix(0, 0) - evidence)
    ranked = sorted(range(128), key=lambda i: (-keep_probability[i], i))
    selected = set(ranked[:visible_keep])
    pred = [int(i in selected) for i in range(128)]
    tp = sum(a == b == 0 for a, b in zip(gold, pred, strict=True))
    fp = sum(a == 1 and b == 0 for a, b in zip(gold, pred, strict=True))
    fn = sum(a == 0 and b == 1 for a, b in zip(gold, pred, strict=True))
    return {
        "n": n, "k": k, "length": generated, "visible_j": target_j,
        "log_evidence": evidence, "gold_mask_posterior": gold_probability,
        "mean_gold_label_probability": sum(p if label else 1 - p for p, label in zip(keep_probability, gold, strict=True)) / 128,
        "null_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "mask_hex": np.packbits(np.asarray(pred, dtype=np.uint8), bitorder="big").tobytes().hex(),
        "keep": keep_probability,
    }


def audit(root: Path) -> dict:
    result_path = root / "results/EXP-0034/results.json"
    result = json.loads(result_path.read_text())
    if result["experiment"] != "EXP-0034" or result["n"] != 198:
        raise AssertionError("result identity/count drift")
    if result["runner_sha256"] != digest(root / "src/voynich/exp0034_weighted_oracle.py"):
        raise AssertionError("runner modified after output")
    if result["generator_sha256"] != digest(Path(channel.__file__)):
        raise AssertionError("generator modified after output")
    saved = {r["row_index"]: r for r in result["rows"]}
    if len(saved) != 198:
        raise AssertionError("duplicate/missing result rows")
    checked = []
    maximum_probability_error = 0.0
    for index, row, full_source in independent_replay(root):
        rec = saved[index]
        independent = explicit_posterior(row["text"], full_source, row["alphabet"], len(row["ciphered"]), row["mask"])
        for field, key in (("full_source_length", "n"), ("null_budget", "k"), ("generated_length", "length"),
                           ("visible_signal_count", "visible_j")):
            if rec[field] != independent[key]:
                raise AssertionError(f"row {index} exact field {field} mismatch")
        for field, key in (("log_evidence", "log_evidence"), ("gold_mask_posterior", "gold_mask_posterior"),
                           ("mean_gold_label_probability", "mean_gold_label_probability"),
                           ("posterior_top_count_null_f1", "null_f1")):
            if abs(rec[field] - independent[key]) > 1e-8:
                raise AssertionError(f"row {index} numeric field {field} mismatch")
        if rec["posterior_mask_hex"] != independent["mask_hex"]:
            raise AssertionError(f"row {index} selected mask mismatch")
        errors = [abs(a - b) for a, b in zip(rec["posterior_keep_probabilities"], independent["keep"], strict=True)]
        maximum_probability_error = max(maximum_probability_error, max(errors))
        if max(errors) > 1e-8:
            raise AssertionError(f"row {index} position posterior mismatch")
        checked.append(rec)
    gains = np.asarray([r["weighted_minus_uniform_null_f1"] for r in checked])
    rng = np.random.default_rng(340034)
    indices = rng.integers(0, len(checked), size=(2000, len(checked)))
    ci = np.quantile(gains[indices].mean(axis=1), [0.025, 0.975])
    if abs(float(np.mean(gains)) - result["mean_weighted_minus_uniform_null_f1"]) > 1e-10 or max(abs(ci - result["gain_ci95"])) > 1e-10:
        raise AssertionError("gain or bootstrap drift")
    if result["material_weighting_gain_by_registered_criterion"] != (float(np.mean(gains)) >= 0.02 and ci[0] > 0):
        raise AssertionError("registered decision drift")
    for key, field in (("mean_weighted_null_f1", "posterior_top_count_null_f1"),
                       ("mean_uniform_null_f1", "uniform_alignment_top_count_null_f1"),
                       ("mean_gold_mask_posterior", "gold_mask_posterior")):
        if abs(result[key] - float(np.mean([r[field] for r in checked]))) > 1e-10:
            raise AssertionError(f"aggregate {key} drift")
    if abs(result["median_gold_mask_posterior"] - float(np.median([r["gold_mask_posterior"] for r in checked]))) > 1e-10:
        raise AssertionError("median gold probability drift")
    for name, short in (("short_padded", True), ("cropped_prefix", False)):
        group = [r for r in checked if (r["generated_length"] < 128) == short]
        summary = result["subsets"][name]
        if summary["n"] != len(group):
            raise AssertionError("subset size drift")
        for key, field in (("weighted_null_f1", "posterior_top_count_null_f1"),
                           ("uniform_null_f1", "uniform_alignment_top_count_null_f1"),
                           ("mean_gold_mask_posterior", "gold_mask_posterior")):
            if abs(summary[key] - float(np.mean([r[field] for r in group]))) > 1e-10:
                raise AssertionError(f"subset aggregate {name}/{key} drift")
    return {"experiment": "EXP-0034", "passed": True, "result_sha256": digest(result_path),
            "auditor_sha256": digest(Path(__file__)), "rows_replayed": 600,
            "copy_posteriors_recomputed": 198, "maximum_position_probability_error": maximum_probability_error,
            "decision": result["material_weighting_gain_by_registered_criterion"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0034/audit.json"
    if out.exists():
        raise FileExistsError("EXP-0034 audit already exists")

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError("EXP-0034 auditor exceeded 180-second CPU cap")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    try:
        report = audit(args.root)
    finally:
        signal.alarm(0)
    write_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
