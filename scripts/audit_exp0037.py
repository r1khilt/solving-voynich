"""Independent replay of EXP-0037 endpoint and position controls."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from scripts.audit_exp0036 import read_segments, toy


ROOT = Path(".")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(segments):
    return [(leaf, section, words) for leaf, section, words in segments if len(words) >= 4]


def events(words):
    for index in range(len(words) - 1):
        role = 0 if index == 0 else 2 if index == len(words) - 2 else 1
        yield role, words[index][-1], words[index + 1][0]


def train(segments):
    joint = [Counter() for _ in range(3)]
    ends = [Counter() for _ in range(3)]
    starts = [Counter() for _ in range(3)]
    totals = [0, 0, 0]
    for _, _, words in segments:
        for role, a, b in events(words):
            joint[role][(a, b)] += 1
            ends[role][a] += 1
            starts[role][b] += 1
            totals[role] += 1
    return joint, ends, starts, totals


def evaluate(segments, model, size):
    joint, ends, starts, totals = model
    result = defaultdict(lambda: [0.0, 0])
    for leaf, _, words in segments:
        for role, a, b in events(words):
            conditional = (joint[role][(a, b)] + 1) / (ends[role][a] + size)
            marginal = (starts[role][b] + 1) / (totals[role] + size)
            result[leaf][0] += math.log2(conditional) - math.log2(marginal)
            result[leaf][1] += 1
    return dict(result)


def change(segments, seed, mode):
    rng = random.Random(seed)
    output = []
    for leaf, section, words in segments:
        interior = words[1:-1]
        if mode == "endpoint":
            rng.shuffle(interior)
        else:
            split = len(interior) // 2
            left, right = interior[:split], interior[split:]
            rng.shuffle(left)
            rng.shuffle(right)
            interior = left + right
        output.append((leaf, section, [words[0], *interior, words[-1]]))
    return output


def avg(rows):
    return sum(item[0] for item in rows.values()) / sum(item[1] for item in rows.values())


def close(a, b):
    if not math.isclose(a, b, abs_tol=1e-10, rel_tol=0):
        raise AssertionError((a, b))


def main():
    saved = json.loads((ROOT / "results/EXP-0037/results.json").read_text())
    assert saved["source_sha256"] == sha(ROOT / "src/voynich/boundary_endpoint.py")
    assert saved["base_source_sha256"] == sha(ROOT / "src/voynich/boundary_order.py")
    train_path = ROOT / "data/processed/zl3b/train.jsonl"
    valid_path = ROOT / "data/processed/zl3b/validation.jsonl"
    assert sha(train_path) == saved["train_sha256"]
    assert sha(valid_path) == saved["validation_sha256"]
    source, _ = read_segments(train_path)
    held, _ = read_segments(valid_path)
    source, held = select(source), select(held)
    assert len(source) == saved["train_groups"] and len(held) == saved["validation_groups"]
    assert sum(len(words) - 1 for _, _, words in source) == saved["train_pairs"]
    assert sum(len(words) - 1 for _, _, words in held) == saved["validation_pairs"]
    model = train(source)
    real = evaluate(held, model, 27)
    close(avg(real), saved["real_bits_per_pair"])
    for leaf, item in real.items():
        close(item[0], saved["real_by_leaf"][leaf]["sum_bits"])
        assert item[1] == saved["real_by_leaf"][leaf]["pairs"]
    all_control = {}
    for mode, seed in (("endpoint", 370037), ("half", 370437)):
        record = saved["controls"][mode]
        assert record["seed"] == seed
        repeats = []
        leaf_sums = {leaf: [] for leaf in real}
        for index in range(200):
            scored = evaluate(change(held, seed + index, mode), model, 27)
            repeats.append(avg(scored))
            close(repeats[-1], record["means"][index])
            for leaf in real:
                assert scored[leaf][1] == real[leaf][1]
                leaf_sums[leaf].append(scored[leaf][0])
                close(scored[leaf][0], record["by_leaf_sum_bits"][leaf][index])
        null_mean = sum(repeats) / 200
        contrast = avg(real) - null_mean
        close(null_mean, record["mean"])
        close(contrast, record["real_minus_null"])
        close(sorted(repeats)[189], record["percentile95"])
        rng = random.Random(370137)
        leaves = sorted(real)
        mean_null_leaf = {leaf: sum(leaf_sums[leaf]) / 200 for leaf in leaves}
        draws = []
        for _ in range(2000):
            sample = rng.choices(leaves, k=len(leaves))
            draws.append(sum(real[leaf][0] - mean_null_leaf[leaf] for leaf in sample)
                         / sum(real[leaf][1] for leaf in sample))
        draws.sort()
        for a, b in zip((draws[49], draws[1949]), record["leaf_bootstrap95"]):
            close(a, b)
        all_control[mode] = (contrast, draws[49], sorted(repeats)[189])
    toy_train, toy_valid = select(toy(360236, 100)), select(toy(360336, 30))
    toy_model = train(toy_train)
    toy_real = avg(evaluate(toy_valid, toy_model, 8))
    toy_null = sum(avg(evaluate(change(toy_valid, 370836 + i, "endpoint"), toy_model, 8))
                   for i in range(100)) / 100
    close(toy_real, saved["toy_real_bits_per_pair"])
    close(toy_null, saved["toy_permuted_bits_per_pair"])
    close(toy_real - toy_null, saved["toy_contrast"])
    if toy_real - toy_null < 0.1:
        decision = "invalid_positive_control"
    elif all(delta >= 0.02 and lower > 0 and avg(real) > percentile
             for delta, lower, percentile in all_control.values()):
        decision = "endpoint_robust_boundary_signal"
    else:
        decision = "not_supported"
    assert decision == saved["decision"]
    report = {"experiment": "EXP-0037", "audit": "pass", "decision": decision,
              "result_sha256": sha(ROOT / "results/EXP-0037/results.json"),
              "validation_pairs": saved["validation_pairs"], "permutations_per_control": 200,
              "bootstrap_draws_per_control": 2000, "independent_replay": True}
    path = ROOT / "results/EXP-0037/audit.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
