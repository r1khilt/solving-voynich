"""EXP-0036: held-out word-boundary dependence with within-locus controls."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import time


EXPERIMENT = "EXP-0036"
TRAIN_SHA = "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4"
VALID_SHA = "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3"
ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz'")
WORD = re.compile(r"[a-z']+\Z")
PERM_SEED = 360036
BOOT_SEED = 360136
N_PERM = 200
N_BOOT = 2000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_groups(path: Path) -> tuple[list[dict], dict]:
    groups = []
    diagnostics = Counter()
    for line in path.read_text().splitlines():
        page = json.loads(line)
        diagnostics["pages"] += 1
        for locus in page["loci"]:
            diagnostics["loci"] += 1
            run = []
            for word in locus["text"].split():
                if WORD.fullmatch(word):
                    run.append(word)
                    diagnostics["eligible_words"] += 1
                else:
                    diagnostics["excluded_words"] += 1
                    if len(run) >= 2:
                        groups.append({"leaf": page["leaf_id"], "page": page["page_id"],
                                       "section": page["metadata"]["page_variables"].get("I", "?"),
                                       "words": run})
                    run = []
            if len(run) >= 2:
                groups.append({"leaf": page["leaf_id"], "page": page["page_id"],
                               "section": page["metadata"]["page_variables"].get("I", "?"),
                               "words": run})
    diagnostics["groups"] = len(groups)
    diagnostics["pairs"] = sum(len(group["words"]) - 1 for group in groups)
    diagnostics["leaves"] = len({group["leaf"] for group in groups})
    return groups, dict(diagnostics)


def pairs(words: list[str]) -> list[tuple[str, str]]:
    return [(a[-1], b[0]) for a, b in zip(words, words[1:])]


def fit(groups: list[dict], alphabet: tuple[str, ...] = ALPHABET) -> dict:
    joint = Counter()
    initial = Counter()
    final = Counter()
    n = 0
    for group in groups:
        for a, b in pairs(group["words"]):
            if a not in alphabet or b not in alphabet:
                raise ValueError("Unexpected boundary symbol")
            joint[(a, b)] += 1
            initial[b] += 1
            final[a] += 1
            n += 1
    if n <= 0:
        raise ValueError("No training boundary pairs")
    return {"joint": joint, "initial": initial, "final": final, "n": n,
            "alphabet": alphabet}


def gain(pair: tuple[str, str], model: dict) -> float:
    a, b = pair
    k = len(model["alphabet"])
    conditional = (model["joint"][(a, b)] + 1) / (model["final"][a] + k)
    marginal = (model["initial"][b] + 1) / (model["n"] + k)
    return math.log2(conditional / marginal)


def score(groups: list[dict], model: dict) -> dict[str, dict]:
    by_leaf = defaultdict(lambda: {"sum_bits": 0.0, "pairs": 0, "sections": set()})
    for group in groups:
        record = by_leaf[group["leaf"]]
        record["sections"].add(group["section"])
        for pair in pairs(group["words"]):
            record["sum_bits"] += gain(pair, model)
            record["pairs"] += 1
    return {leaf: {"sum_bits": record["sum_bits"], "pairs": record["pairs"],
                   "sections": sorted(record["sections"])}
            for leaf, record in sorted(by_leaf.items())}


def shuffle(groups: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for group in groups:
        words = list(group["words"])
        rng.shuffle(words)
        result.append({**group, "words": words})
    return result


def aggregate(by_leaf: dict[str, dict]) -> float:
    count = sum(item["pairs"] for item in by_leaf.values())
    return sum(item["sum_bits"] for item in by_leaf.values()) / count


def bootstrap(real: dict[str, dict], permutations: list[dict[str, dict]]) -> list[float]:
    leaves = sorted(real)
    if any(sorted(perm) != leaves for perm in permutations):
        raise ValueError("Permutation leaf mismatch")
    mean_perm = {leaf: sum(perm[leaf]["sum_bits"] for perm in permutations) / len(permutations)
                 for leaf in leaves}
    rng = random.Random(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        sample = rng.choices(leaves, k=len(leaves))
        n = sum(real[leaf]["pairs"] for leaf in sample)
        draws.append(sum(real[leaf]["sum_bits"] - mean_perm[leaf] for leaf in sample) / n)
    draws.sort()
    return [draws[49], draws[1949]]


def toy(seed: int, n_groups: int) -> list[dict]:
    rng = random.Random(seed)
    alphabet = "abcdefgh"
    groups = []
    for index in range(n_groups):
        words = []
        prior_end = rng.choice(alphabet)
        for _ in range(30):
            start = prior_end if rng.random() < 0.8 else rng.choice(alphabet)
            end = rng.choice(alphabet)
            words.append(start + end)
            prior_end = end
        groups.append({"leaf": str(index), "page": str(index), "section": "toy", "words": words})
    return groups


def run(root: Path) -> dict:
    start = time.monotonic()
    train_path = root / "data/processed/zl3b/train.jsonl"
    valid_path = root / "data/processed/zl3b/validation.jsonl"
    if digest(train_path) != TRAIN_SHA or digest(valid_path) != VALID_SHA:
        raise ValueError("Frozen corpus drift")
    train, train_diag = load_groups(train_path)
    valid, valid_diag = load_groups(valid_path)
    model = fit(train)
    real = score(valid, model)
    perms = [score(shuffle(valid, PERM_SEED + i), model) for i in range(N_PERM)]
    real_mean = aggregate(real)
    perm_means = [aggregate(row) for row in perms]
    perm_mean = sum(perm_means) / N_PERM
    contrast = real_mean - perm_mean
    ci = bootstrap(real, perms)
    threshold = sorted(perm_means)[189]

    neg_model = fit(shuffle(train, 360536))
    negative = aggregate(score(valid, neg_model))
    toy_train = toy(360236, 100)
    toy_valid = toy(360336, 30)
    toy_model = fit(toy_train, tuple("abcdefgh"))
    toy_real = aggregate(score(toy_valid, toy_model))
    toy_perm = sum(aggregate(score(shuffle(toy_valid, 360436 + i), toy_model))
                   for i in range(100)) / 100
    toy_contrast = toy_real - toy_perm

    if toy_contrast < 0.1:
        decision = "invalid_positive_control"
    elif contrast >= 0.02 and ci[0] > 0 and real_mean > threshold:
        decision = "boundary_order_signal"
    else:
        decision = "not_supported"
    if time.monotonic() - start > 180:
        raise TimeoutError("EXP-0036 CPU cap exceeded")
    result = {
        "experiment": EXPERIMENT, "status": "complete", "decision": decision,
        "date_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - start,
        "command": sys.argv, "python": platform.python_version(),
        "source_sha256": digest(Path(__file__)),
        "train_sha256": TRAIN_SHA, "validation_sha256": VALID_SHA,
        "train": train_diag, "validation": valid_diag, "train_pair_count": model["n"],
        "real_by_leaf": real, "permuted_by_leaf": perms,
        "real_bits_per_pair": real_mean, "permutation_bits_per_pair": perm_means,
        "permutation_mean": perm_mean, "permutation_95th": threshold,
        "real_minus_permutation_mean": contrast, "leaf_bootstrap95": ci,
        "shuffled_train_real_bits_per_pair": negative,
        "toy_real_bits_per_pair": toy_real, "toy_permutation_mean": toy_perm,
        "toy_contrast": toy_contrast,
        "limits": "Exposed validation; EVA-codepoint and certain-word subset; no copying or meaning inference.",
    }
    out = root / "results/EXP-0036/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    report = run(Path("."))
    print(json.dumps({key: report[key] for key in ("decision", "real_bits_per_pair",
                                                    "permutation_mean", "real_minus_permutation_mean",
                                                    "leaf_bootstrap95", "toy_contrast")}, indent=2))
