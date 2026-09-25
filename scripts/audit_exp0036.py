"""Independent source-to-result replay for EXP-0036."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import random
import re


ROOT = Path(".")
WORD = re.compile("[a-z']+\\Z")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_segments(path: Path) -> tuple[list[tuple[str, str, list[str]]], dict]:
    segments = []
    tally = Counter()
    leaves = set()
    for raw in path.read_text().splitlines():
        page = json.loads(raw)
        tally["pages"] += 1
        for locus in page["loci"]:
            tally["loci"] += 1
            words = locus["text"].split()
            tally["eligible_words"] += sum(bool(WORD.fullmatch(word)) for word in words)
            tally["excluded_words"] += sum(not WORD.fullmatch(word) for word in words)
            for eligible, run in groupby(words, key=lambda word: bool(WORD.fullmatch(word))):
                sequence = list(run)
                if eligible and len(sequence) > 1:
                    segments.append((page["leaf_id"], page["metadata"]["page_variables"].get("I", "?"), sequence))
                    leaves.add(page["leaf_id"])
    tally["groups"] = len(segments)
    tally["pairs"] = sum(len(words) - 1 for _, _, words in segments)
    tally["leaves"] = len(leaves)
    return segments, dict(tally)


def train_counts(segments: list[tuple[str, str, list[str]]]) -> tuple[Counter, Counter, Counter, int]:
    joint, finals, initials = Counter(), Counter(), Counter()
    for _, _, words in segments:
        for previous, following in zip(words, words[1:]):
            a, b = previous[-1], following[0]
            joint[(a, b)] += 1
            finals[a] += 1
            initials[b] += 1
    return joint, finals, initials, initials.total()


def score(segments: list[tuple[str, str, list[str]]], model: tuple, alphabet_size: int) -> dict[str, tuple[float, int]]:
    joint, finals, initials, n = model
    output = defaultdict(lambda: [0.0, 0])
    for leaf, _, words in segments:
        for previous, following in zip(words, words[1:]):
            a, b = previous[-1], following[0]
            p_cond = (joint[(a, b)] + 1) / (finals[a] + alphabet_size)
            p_base = (initials[b] + 1) / (n + alphabet_size)
            output[leaf][0] += math.log2(p_cond) - math.log2(p_base)
            output[leaf][1] += 1
    return dict(output)


def scramble(segments: list[tuple[str, str, list[str]]], seed: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(seed)
    output = []
    for leaf, section, words in segments:
        changed = words[:]
        rng.shuffle(changed)
        output.append((leaf, section, changed))
    return output


def mean(scored: dict[str, tuple[float, int]]) -> float:
    return sum(item[0] for item in scored.values()) / sum(item[1] for item in scored.values())


def toy(seed: int, n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(seed)
    result = []
    for index in range(n):
        words = []
        prior = rng.choice("abcdefgh")
        for _ in range(30):
            start = prior if rng.random() < 0.8 else rng.choice("abcdefgh")
            end = rng.choice("abcdefgh")
            words.append(start + end)
            prior = end
        result.append((str(index), "toy", words))
    return result


def close(a: float, b: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(a, b, abs_tol=tolerance, rel_tol=0):
        raise AssertionError((a, b))


def main() -> None:
    saved = json.loads((ROOT / "results/EXP-0036/results.json").read_text())
    assert saved["source_sha256"] == hash_file(ROOT / "src/voynich/boundary_order.py")
    train_path = ROOT / "data/processed/zl3b/train.jsonl"
    valid_path = ROOT / "data/processed/zl3b/validation.jsonl"
    assert hash_file(train_path) == saved["train_sha256"]
    assert hash_file(valid_path) == saved["validation_sha256"]
    train, train_diag = read_segments(train_path)
    valid, valid_diag = read_segments(valid_path)
    assert train_diag == saved["train"] and valid_diag == saved["validation"]
    model = train_counts(train)
    assert model[3] == saved["train_pair_count"]
    real = score(valid, model, 27)
    assert sorted(real) == sorted(saved["real_by_leaf"])
    for leaf, row in real.items():
        close(row[0], saved["real_by_leaf"][leaf]["sum_bits"])
        assert row[1] == saved["real_by_leaf"][leaf]["pairs"]
    close(mean(real), saved["real_bits_per_pair"])
    permutations = []
    for index in range(200):
        scored = score(scramble(valid, 360036 + index), model, 27)
        permutations.append(scored)
        close(mean(scored), saved["permutation_bits_per_pair"][index])
        for leaf, item in scored.items():
            close(item[0], saved["permuted_by_leaf"][index][leaf]["sum_bits"])
            assert item[1] == saved["permuted_by_leaf"][index][leaf]["pairs"]
    permutation_mean = sum(mean(item) for item in permutations) / 200
    close(permutation_mean, saved["permutation_mean"])
    contrast = mean(real) - permutation_mean
    close(contrast, saved["real_minus_permutation_mean"])
    threshold = sorted(mean(item) for item in permutations)[189]
    close(threshold, saved["permutation_95th"])
    rng = random.Random(360136)
    leaves = sorted(real)
    per_leaf_null = {leaf: sum(item[leaf][0] for item in permutations) / 200 for leaf in leaves}
    draws = []
    for _ in range(2000):
        selected = rng.choices(leaves, k=len(leaves))
        numerator = sum(real[leaf][0] - per_leaf_null[leaf] for leaf in selected)
        denominator = sum(real[leaf][1] for leaf in selected)
        draws.append(numerator / denominator)
    draws.sort()
    ci = [draws[49], draws[1949]]
    for actual, recorded in zip(ci, saved["leaf_bootstrap95"]):
        close(actual, recorded)
    neg_model = train_counts(scramble(train, 360536))
    close(mean(score(valid, neg_model, 27)), saved["shuffled_train_real_bits_per_pair"])
    positive_model = train_counts(toy(360236, 100))
    positive_valid = toy(360336, 30)
    positive_real = mean(score(positive_valid, positive_model, 8))
    positive_null = sum(mean(score(scramble(positive_valid, 360436 + i), positive_model, 8))
                        for i in range(100)) / 100
    close(positive_real, saved["toy_real_bits_per_pair"])
    close(positive_null, saved["toy_permutation_mean"])
    close(positive_real - positive_null, saved["toy_contrast"])
    if positive_real - positive_null < 0.1:
        decision = "invalid_positive_control"
    elif contrast >= 0.02 and ci[0] > 0 and mean(real) > threshold:
        decision = "boundary_order_signal"
    else:
        decision = "not_supported"
    assert decision == saved["decision"]
    audit = {"experiment": "EXP-0036", "audit": "pass", "result_sha256": hash_file(ROOT / "results/EXP-0036/results.json"),
             "leaves": len(leaves), "validation_pairs": valid_diag["pairs"], "permutations": 200,
             "bootstrap_draws": 2000, "decision": decision, "independent_replay": True}
    path = ROOT / "results/EXP-0036/audit.json"
    path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
