"""Independent array-based replay of the train-only gap prediction preflight."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SEED = 430043


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay() -> dict:
    source = ROOT / "data/processed/boundary_geometry/train_pairs.jsonl"
    result_path = ROOT / "results/BOUNDARY-CHANNEL-0001/train_preflight.json"
    saved = json.loads(result_path.read_text())
    if digest(source) != saved["rows_sha256"]:
        raise AssertionError("Source rows changed")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    n = len(rows)
    by_line: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_line[(row["folio"], row["visual_line"])].append(i)
    role = np.zeros(n, dtype=int)
    for indices in by_line.values():
        order = sorted(indices, key=lambda i: rows[i]["visual_left_index"])
        role[order[0]] = 0
        role[order[-1]] = 2 if len(order) > 1 else 0
        for i in order[1:-1]:
            role[i] = 1
    last_value = [row["left_word"] for row in rows]
    next_value = [row["right_word"] for row in rows]

    def collapse(word: str) -> str:
        for old, new in (("cth", "T"), ("ckh", "K"), ("cph", "P"), ("cfh", "F"),
                         ("ch", "C"), ("sh", "S"), ("iin", "N"), ("in", "I"), ("ee", "E")):
            word = word.replace(old, new)
        return word

    last_symbols = [collapse(word)[-1] for word in last_value]
    target_symbols = [collapse(word)[0] for word in next_value]
    last_vocab = {word: i for i, word in enumerate(sorted(set(last_symbols)))}
    target_vocab = {word: i for i, word in enumerate(sorted(set(target_symbols)))}
    last = np.array([last_vocab[word] for word in last_symbols], dtype=int)
    target = np.array([target_vocab[word] for word in target_symbols], dtype=int)
    label = np.array([0 if row["label"] == "." else 1 for row in rows], dtype=int)
    values = np.array([row["gap_over_median_word_width"] for row in rows])
    bins = np.searchsorted(np.array([0, .075, .15]), values, side="left")
    folds = np.array([int.from_bytes(hashlib.sha256(row["leaf"].encode()).digest()[:4],
                                     "big") % 5 for row in rows], dtype=int)
    leaves = sorted({row["leaf"] for row in rows})
    leaf_map = {name: i for i, name in enumerate(leaves)}
    leaf_idx = np.array([leaf_map[row["leaf"]] for row in rows])
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        strata[(row["folio"], row["label"], last_symbols[i], int(role[i]))].append(i)

    def score(gap_bins: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        bits = np.zeros(n)
        alphabet = len(target_vocab)
        for held in range(5):
            fit = np.flatnonzero(folds != held)
            test = np.flatnonzero(folds == held)
            counts = np.bincount(target[fit], minlength=alphabet)
            q = (counts + 1) / (len(fit) + alphabet)
            base = np.zeros((len(last_vocab), 2, 3, alphabet), dtype=np.int64)
            refined = np.zeros((len(last_vocab), 2, 3, 4, alphabet), dtype=np.int64)
            np.add.at(base, (last[fit], label[fit], role[fit], target[fit]), 1)
            np.add.at(refined,
                      (last[fit], label[fit], role[fit], gap_bins[fit], target[fit]), 1)
            left = base[last[test], label[test], role[test]]
            right = refined[last[test], label[test], role[test], gap_bins[test]]
            p0 = (left[np.arange(len(test)), target[test]] + 20 * q[target[test]]) / (
                left.sum(axis=1) + 20)
            p1 = (right[np.arange(len(test)), target[test]] + 80 * p0) / (
                right.sum(axis=1) + 80)
            bits[test] = np.log2(p1 / p0)
        sums = np.bincount(leaf_idx, weights=bits, minlength=len(leaves))
        pairs = np.bincount(leaf_idx, minlength=len(leaves))
        return float(bits.mean()), sums, pairs

    observed, sums, pairs = score(bins)
    if abs(observed - saved["observed_gain_bits_per_pair"]) > 1e-10:
        raise AssertionError("Observed score differs")
    for name, i in leaf_map.items():
        item = saved["per_leaf"][name]
        if item["pairs"] != int(pairs[i]) or abs(item["bits_sum"] - sums[i]) > 1e-9:
            raise AssertionError("Physical-leaf sum differs")
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(100):
        shuffled = bins.copy()
        for indices in strata.values():
            shuffled[indices] = rng.permutation(bins[indices])
        null.append(score(shuffled)[0])
    if not np.allclose(null, saved["null_gains"], rtol=0, atol=1e-10):
        raise AssertionError("Conditional null replay differs")
    bootstrap = []
    for _ in range(2000):
        chosen = rng.integers(0, len(leaves), size=len(leaves))
        bootstrap.append(float(sums[chosen].sum() / pairs[chosen].sum()))
    interval = [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))]
    if not np.allclose(interval, saved["leaf_bootstrap95"], rtol=0, atol=1e-10):
        raise AssertionError("Bootstrap replay differs")
    audit = {"status": "pass", "rows": n, "leaves": len(leaves),
             "result_sha256": digest(result_path), "observed": observed,
             "max_null_replay_error": float(np.max(np.abs(np.array(null)-saved["null_gains"]))) }
    destination = ROOT / "results/BOUNDARY-CHANNEL-0001/preflight_audit.json"
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


if __name__ == "__main__":
    replay()
