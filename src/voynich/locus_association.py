"""EXP-0029: leaf-held-out matched locus association, beyond spelling."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from .word_copy_channel import eligible, fit_base, pages, sha256
from .word_copy_followup import GlobalEdit, static_prob


@lru_cache(maxsize=250000)
def edit_distance(a: str, b: str) -> int:
    old = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, old[j] + 1, old[j - 1] + (ca != cb)))
        old = new
    return old[-1]


def shared_edge(a: str, b: str) -> float:
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    return min(len(a), prefix + suffix) / max(len(a), len(b))


def p0_loci(page: dict) -> list[tuple[str, list[str]]]:
    return [(locus["locus_id"], [w for w in locus["text"].split() if eligible(w)])
            for locus in page["loci"] if locus["locus_type"] == "P0"]


def match_pairs(eval_pages: list[dict], counts: Counter) -> tuple[list[dict], dict]:
    pairs = []
    stats = Counter()
    for page in eval_pages:
        loci = p0_loci(page)
        for index, (locus_id, words) in enumerate(loci):
            if len(words) < 4:
                stats["short_locus"] += 1
                continue
            other = {word for j, (_, locus) in enumerate(loci) if j != index for word in locus}
            local = Counter(words)
            for slot, target in enumerate(words):
                if local[target] != 1:
                    stats["repeated_target"] += 1
                    continue
                if counts[target] < 3:
                    stats["rare_target"] += 1
                    continue
                context = tuple(w for j, w in enumerate(words) if j != slot)
                distance = min(edit_distance(target, w) for w in context)
                candidates = [word for word in other if word not in local and counts[word] >= 3
                              and len(word) == len(target)
                              and 0.5 <= counts[word] / counts[target] <= 2.0
                              and (word[0] == target[0] or word[-1] == target[-1])
                              and min(edit_distance(word, w) for w in context) == distance]
                if not candidates:
                    stats["no_matched_negative"] += 1
                    continue
                negative = min(candidates, key=lambda word: hashlib.sha256(
                    f"290029|{page['page_id']}|{locus_id}|{slot}|{word}".encode()).hexdigest())
                pairs.append({"page": page["page_id"], "leaf": page["leaf_id"],
                              "locus_id": locus_id, "slot": slot, "target": target,
                              "negative": negative, "context": context})
                stats["selected"] += 1
    return pairs, dict(sorted(stats.items()))


def page_seed(seed: int, page_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{page_id}".encode()).digest()[:8], "big")


def shuffled_p0(page: dict, seed: int) -> list[list[str]]:
    loci = [words for _, words in p0_loci(page)]
    flat = [word for locus in loci for word in locus]
    random.Random(page_seed(seed, page["page_id"])).shuffle(flat)
    result, offset = [], 0
    for locus in loci:
        result.append(flat[offset:offset + len(locus)])
        offset += len(locus)
    assert offset == len(flat)
    return result


class Association:
    def __init__(self, loci: list[list[str]]):
        self.loci = len(loci)
        self.single: Counter = Counter()
        self.pairs: Counter = Counter()
        for words in loci:
            unique = sorted(set(words))
            self.single.update(unique)
            for i, first in enumerate(unique):
                self.pairs.update((first, second) for second in unique[i + 1:])

    @classmethod
    def from_pages(cls, training_pages: list[dict], seed: int | None = None) -> Association:
        return cls([locus for page in training_pages
                    for locus in (shuffled_p0(page, seed) if seed is not None
                                  else [words for _, words in p0_loci(page)])])

    def score(self, candidate: str, context: tuple[str, ...]) -> float:
        if not context:
            return 0.0
        total = 0.0
        for word in set(context):
            marginal = (self.single[word] + 0.5) / (self.loci + 1)
            key = (candidate, word) if candidate < word else (word, candidate)
            joint = self.pairs[key] if candidate != word else self.single[candidate]
            conditional = (joint + 20 * marginal) / (self.single[candidate] + 20)
            total += math.log(conditional / marginal)
        return total / len(set(context))


def word_features(word: str, context: tuple[str, ...], base: tuple,
                  global_edit: GlobalEdit) -> tuple[float, float, float]:
    counts, n, char = base
    row = {"word": word, "count": counts[word], "char_p": char.prob(word)}
    p = static_prob(row, n, 1000000, 0.1, global_edit)
    mean_similarity = sum(1 - edit_distance(word, x) / max(len(word), len(x))
                          for x in context) / len(context)
    edge = max(shared_edge(word, x) for x in context)
    return math.log(p), mean_similarity, edge


def feature_matrix(pairs: list[dict], base: tuple, global_edit: GlobalEdit,
                   association: Association | None = None) -> np.ndarray:
    result = np.empty((len(pairs), 4 if association is not None else 3), dtype=np.float64)
    for i, pair in enumerate(pairs):
        context = pair["context"]
        good = word_features(pair["target"], context, base, global_edit)
        bad = word_features(pair["negative"], context, base, global_edit)
        result[i, :3] = np.subtract(good, bad)
        if association is not None:
            result[i, 3] = (association.score(pair["target"], context)
                            - association.score(pair["negative"], context))
    return result


def fit_readout(x: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(x * x, axis=0))
    rms = np.where(rms > 1e-12, rms, 1.0)
    z = x / rms
    weights = np.zeros(z.shape[1], dtype=np.float64)
    reg = 0.01

    def objective(w: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0, -(z @ w))) + 0.5 * reg * np.dot(w, w))

    for _ in range(30):
        margin = np.clip(z @ weights, -40, 40)
        negative = 1 / (1 + np.exp(margin))
        gradient = -(z.T @ negative) / len(z) + reg * weights
        variance = negative * (1 - negative)
        hessian = (z.T @ (z * variance[:, None])) / len(z) + reg * np.eye(z.shape[1])
        step = np.linalg.solve(hessian, gradient)
        old_loss, scale = objective(weights), 1.0
        while scale > 1 / 1024 and objective(weights - scale * step) > old_loss:
            scale /= 2
        weights -= scale * step
        if np.max(np.abs(scale * step)) < 1e-9:
            break
    return {"weights": weights.tolist(), "rms": rms.tolist(), "l2": reg}


def margins(x: np.ndarray, model: dict) -> np.ndarray:
    return (x / np.array(model["rms"])) @ np.array(model["weights"])


def losses(margin: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, -margin) / math.log(2)


def summary(form: np.ndarray, full: np.ndarray) -> dict:
    form_loss, full_loss = losses(form), losses(full)

    def accuracy(z: np.ndarray) -> float:
        return float(np.mean((z > 0) + 0.5 * (z == 0)))
    return {"n": len(form), "form_bits": float(np.mean(form_loss)),
            "full_bits": float(np.mean(full_loss)),
            "gain_bits": float(np.mean(form_loss - full_loss)),
            "form_accuracy": accuracy(form), "full_accuracy": accuracy(full)}


def leaf_bootstrap(pairs: list[dict], form: np.ndarray, full: np.ndarray) -> list[float]:
    by_leaf: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        by_leaf[pair["leaf"]].append(i)
    leaf_ids = sorted(by_leaf)
    delta = losses(form) - losses(full)
    rng = random.Random(290129)
    draws = []
    for _ in range(2000):
        indices = [i for _ in leaf_ids for i in by_leaf[rng.choice(leaf_ids)]]
        draws.append(float(np.mean(delta[indices])))
    draws.sort()
    return [draws[50], draws[1950]]


def run(root: Path) -> dict:
    started = time.monotonic()
    exp27 = json.loads((root / "results/EXP-0027/results.json").read_text())
    train_path = root / "data/processed/zl3b/train.jsonl"
    val_path = root / "data/processed/zl3b/validation.jsonl"
    if sha256(train_path) != exp27["train_sha256"] or sha256(val_path) != exp27["validation_sha256"]:
        raise ValueError("Pinned corpus drift")
    train, validation = pages(train_path), pages(val_path)
    fit_leaves = set(exp27["calibration"]["fit_leaves"])
    fit_pages = [p for p in train if p["leaf_id"] in fit_leaves]
    cal_pages = [p for p in train if p["leaf_id"] not in fit_leaves]
    if len(fit_leaves) != 65 or len({p["leaf_id"] for p in cal_pages}) != 16:
        raise ValueError("Training split drift")
    alphabet = exp27["alphabet"]
    fit_base_model = fit_base(fit_pages, alphabet)
    full_base_model = fit_base(train, alphabet)
    fit_global = GlobalEdit(fit_base_model[0], fit_base_model[1], alphabet)
    full_global = GlobalEdit(full_base_model[0], full_base_model[1], alphabet)
    cal_pairs, cal_selection = match_pairs(cal_pages, fit_base_model[0])
    val_pairs, val_selection = match_pairs(validation, full_base_model[0])
    if not cal_pairs or not val_pairs:
        raise ValueError("Matched pair pool empty")
    cal_form = feature_matrix(cal_pairs, fit_base_model, fit_global)
    val_form = feature_matrix(val_pairs, full_base_model, full_global)
    form_model = fit_readout(cal_form)
    val_form_margin = margins(val_form, form_model)
    fit_assoc = Association.from_pages(fit_pages)
    full_assoc = Association.from_pages(train)
    cal_full = np.column_stack((cal_form, feature_matrix(cal_pairs, fit_base_model, fit_global,
                                                         fit_assoc)[:, 3]))
    val_full = np.column_stack((val_form, feature_matrix(val_pairs, full_base_model, full_global,
                                                         full_assoc)[:, 3]))
    full_model = fit_readout(cal_full)
    val_full_margin = margins(val_full, full_model)
    observed = summary(val_form_margin, val_full_margin)
    ci = leaf_bootstrap(val_pairs, val_form_margin, val_full_margin)
    null_gains = []
    for replicate in range(100):
        seed = 290229 + replicate
        null_fit = Association.from_pages(fit_pages, seed)
        null_full = Association.from_pages(train, seed)
        cal_assoc = np.array([null_fit.score(p["target"], p["context"])
                              - null_fit.score(p["negative"], p["context"]) for p in cal_pairs])
        val_assoc = np.array([null_full.score(p["target"], p["context"])
                              - null_full.score(p["negative"], p["context"]) for p in val_pairs])
        null_model = fit_readout(np.column_stack((cal_form, cal_assoc)))
        null_margin = margins(np.column_stack((val_form, val_assoc)), null_model)
        null_gains.append(float(np.mean(losses(val_form_margin) - losses(null_margin))))
        if time.monotonic() - started > 1200:
            raise TimeoutError("EXP-0029 exceeded 20-minute wall cap")
    passed = (observed["gain_bits"] >= 0.02 and ci[0] > 0
              and observed["gain_bits"] > sorted(null_gains)[95]
              and observed["full_accuracy"] >= 0.55)
    pair_rows = [{"page": p["page"], "leaf": p["leaf"], "locus_id": p["locus_id"],
                  "slot": p["slot"], "target": p["target"], "negative": p["negative"],
                  "context": list(p["context"]), "form_margin": float(f), "full_margin": float(a)}
                 for p, f, a in zip(val_pairs, val_form_margin, val_full_margin)]
    result = {"experiment": "EXP-0029", "status": "complete",
              "decision": "locus_association_support" if passed else "no_support_under_registered_controls",
              "time_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
              "command": sys.argv, "python": platform.python_version(), "numpy": np.__version__,
              "source_sha256": sha256(Path(__file__)), "train_sha256": sha256(train_path),
              "validation_sha256": sha256(val_path),
              "calibration": {"selection": cal_selection, "pairs": len(cal_pairs),
                              "form_model": form_model, "full_model": full_model},
              "validation": {"selection": val_selection, "observed": observed,
                             "leaf_bootstrap_95_gain": ci, "pairs": pair_rows},
              "null": {"n": 100, "seed_rule": "290229+replicate", "gains": null_gains},
              "limits": "Already-exposed validation; a matched-word discrimination task, not a semantic label or decipherment."}
    out = root / "results/EXP-0029/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run(Path("."))
    print(json.dumps({"decision": result["decision"], "observed": result["validation"]["observed"],
                      "ci": result["validation"]["leaf_bootstrap_95_gain"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))
