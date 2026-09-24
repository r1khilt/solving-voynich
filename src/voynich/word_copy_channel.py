"""EXP-0027: proper word-string base and local one-edit cache likelihoods."""

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


WORD = re.compile(r"[a-z']+\Z")
WINDOWS = (4, 16, 64)
HALVES = (4, 16, 0)  # 0 denotes uniform recency weights.
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
BETAS = (1, 10, 100, 1000)
EXACT_WEIGHTS = (0.25, 0.5, 0.75)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pages(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def slots(page: dict) -> list[tuple[str, str]]:
    return [(word, locus["locus_type"])
            for locus in page["loci"] for word in locus["text"].split()]


def eligible(word: str) -> bool:
    return WORD.fullmatch(word) is not None


class CharModel:
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.events = alphabet + "$"
        self.counts = [defaultdict(Counter) for _ in range(4)]
        self.memo: dict[str, float] = {}

    def fit(self, words: list[str]) -> CharModel:
        for word in words:
            prior = "^^^"
            for target in word + "$":
                for order in range(4):
                    context = prior[-order:] if order else ""
                    self.counts[order][context][target] += 1
                prior += target
        return self

    def event_prob(self, prior: str, target: str) -> float:
        if target not in self.events:
            return 0.0
        p = (self.counts[0][""][target] + 0.1) / (
            self.counts[0][""].total() + 0.1 * len(self.events))
        for order in range(1, 4):
            counts = self.counts[order].get(prior[-order:])
            if counts:
                p = (counts[target] + 5.0 * p) / (counts.total() + 5.0)
        return p

    def prob(self, word: str) -> float:
        if word not in self.memo:
            prior, p = "^^^", 1.0
            for target in word + "$":
                p *= self.event_prob(prior, target)
                prior += target
            self.memo[word] = p
        return self.memo[word]


def fit_base(train_pages: list[dict], alphabet: str) -> tuple[Counter, int, CharModel]:
    words = [word for page in train_pages for word, _ in slots(page) if eligible(word)]
    if not words:
        raise ValueError("No eligible training words")
    return Counter(words), len(words), CharModel(alphabet).fit(words)


def edit_parts(source: str, target: str, alphabet_size: int) -> tuple[float, float, float, float]:
    """Normalized exact/substitute/insert/delete string-channel masses."""
    exact = float(source == target)
    sub = ins = delete = 0.0
    n = len(source)
    if len(target) == n and sum(a != b for a, b in zip(source, target)) == 1:
        sub = 1.0 / (n * (alphabet_size - 1))
    if len(target) == n + 1:
        paths = sum(target[:i] + target[i + 1:] == source for i in range(n + 1))
        ins = paths / ((n + 1) * alphabet_size)
    if len(target) == n - 1:
        paths = sum(source[:i] + source[i + 1:] == target for i in range(n))
        delete = paths / n
    return exact, sub, ins, delete


def cache_features(history: list[str], target: str, alphabet_size: int) -> dict[str, tuple[float, ...]]:
    prior = history[-64:][::-1]
    parts = [edit_parts(source, target, alphabet_size) for source in prior]
    result = {}
    for window in WINDOWS:
        subset = parts[:window]
        for half in HALVES:
            if not subset:
                result[f"{window}:{half}"] = (0.0, 0.0, 0.0, 0.0)
                continue
            weights = [1.0 if half == 0 else 2 ** (-(i + 1) / half)
                       for i in range(len(subset))]
            normalizer = sum(weights)
            result[f"{window}:{half}"] = tuple(
                sum(weight * part[k] for weight, part in zip(weights, subset)) / normalizer
                for k in range(4))
    return result


def make_rows(eval_pages: list[dict], base: tuple[Counter, int, CharModel], alphabet: str,
              slot_overrides: dict[str, list[tuple[str, str]]] | None = None) -> list[dict]:
    counts, _, char = base
    result = []
    for page in eval_pages:
        history: list[str] = []
        for word, locus_type in (slot_overrides or {}).get(page["page_id"], slots(page)):
            if not eligible(word):
                continue
            if any(c not in alphabet for c in word):
                raise ValueError(f"Scorable word has unseen alphabet character: {word!r}")
            result.append({"page": page["page_id"], "leaf": page["leaf_id"],
                           "locus_type": locus_type, "word": word, "count": counts[word],
                           "char_p": char.prob(word),
                           "has_history": bool(history),
                           "features": cache_features(history, word, len(alphabet))})
            history.append(word)
    return result


def probability(row: dict, n_train: int, params: dict) -> float:
    beta = params["beta"]
    base = (row["count"] + beta * row["char_p"]) / (n_train + beta)
    lam = params.get("lambda", 0.0)
    if lam == 0 or not row["has_history"]:
        return base
    features = row["features"][f'{params["window"]}:{params["half"]}']
    if params["arm"] == "exact":
        cache = features[0]
    else:
        q = params["q_exact"]
        cache = q * features[0] + (1 - q) * (
            0.5 * features[1] + 0.25 * features[2] + 0.25 * features[3])
    return (1 - lam) * base + lam * cache


def score(rows: list[dict], n_train: int, params: dict) -> float:
    return sum(-math.log2(probability(row, n_train, params)) for row in rows)


def select(rows: list[dict], base: tuple[Counter, int, CharModel]) -> dict[str, dict]:
    n = base[1]
    base_candidates = [{"arm": "base", "beta": beta, "lambda": 0.0}
                       for beta in BETAS]
    base_choice = min(base_candidates, key=lambda p: (score(rows, n, p), p["beta"]))
    result = {"base": base_choice}
    for arm in ("exact", "edit"):
        candidates = [{"arm": arm, "beta": base_choice["beta"], "lambda": lam,
                       "window": window, "half": half, **({"q_exact": q} if arm == "edit" else {})}
                      for window in WINDOWS for half in HALVES for lam in LAMBDAS
                      for q in (EXACT_WEIGHTS if arm == "edit" else (None,))]
        result[arm] = min(candidates, key=lambda p: score(rows, n, p))
    return result


def page_scores(rows: list[dict], n_train: int, selected: dict[str, dict]) -> list[dict]:
    by_page = {}
    for row in rows:
        page = by_page.setdefault(row["page"], {"page": row["page"], "leaf": row["leaf"],
                                              "count": 0, "p0_count": 0,
                                              "bits": {arm: 0.0 for arm in selected}})
        page["count"] += 1
        page["p0_count"] += row["locus_type"] == "P0"
        for arm, params in selected.items():
            page["bits"][arm] -= math.log2(probability(row, n_train, params))
    return sorted(by_page.values(), key=lambda item: item["page"])


def totals(page_rows: list[dict]) -> dict:
    count = sum(row["count"] for row in page_rows)
    bits = {arm: sum(row["bits"][arm] for row in page_rows) / count
            for arm in ("base", "exact", "edit")}
    return {"count": count, "bits_per_word": bits,
            "exact_minus_edit": bits["exact"] - bits["edit"],
            "base_minus_edit": bits["base"] - bits["edit"],
            "base_minus_exact": bits["base"] - bits["exact"]}


def bootstrap(page_rows: list[dict], seed: int = 270027, repetitions: int = 2000) -> list[float]:
    by_leaf: dict[str, list[dict]] = defaultdict(list)
    for row in page_rows:
        by_leaf[row["leaf"]].append(row)
    leaf_ids = sorted(by_leaf)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [row for _ in leaf_ids for row in by_leaf[rng.choice(leaf_ids)]]
        estimates.append(totals(sampled)["exact_minus_edit"])
    estimates.sort()
    return [estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions)]]


def permuted_slots(eval_pages: list[dict], seed: int) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    output = {}
    for page in eval_pages:
        original = slots(page)
        words = [word for word, _ in original if eligible(word)]
        rng.shuffle(words)
        iterator = iter(words)
        output[page["page_id"]] = [(next(iterator), kind) if eligible(word) else (word, kind)
                                    for word, kind in original]
    return output


def positive_slots(eval_pages: list[dict], alphabet: str, seed: int = 270227) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    output = {}
    for page in eval_pages:
        history: list[str] = []
        new = []
        for word, kind in slots(page):
            if eligible(word) and history and rng.random() < 0.30:
                source = rng.choice(history[-16:])
                ops = ["sub", "sub", "ins"] + (["del"] if len(source) > 1 else ["ins"])
                op = rng.choice(ops)
                if op == "sub":
                    i = rng.randrange(len(source))
                    chars = alphabet.replace(source[i], "")
                    word = source[:i] + rng.choice(chars) + source[i + 1:]
                elif op == "ins":
                    i = rng.randrange(len(source) + 1)
                    word = source[:i] + rng.choice(alphabet) + source[i:]
                else:
                    i = rng.randrange(len(source))
                    word = source[:i] + source[i + 1:]
            new.append((word, kind))
            if eligible(word):
                history.append(word)
        output[page["page_id"]] = new
    return output


def run(root: Path) -> dict:
    started = time.monotonic()
    train_path = root / "data/processed/zl3b/train.jsonl"
    val_path = root / "data/processed/zl3b/validation.jsonl"
    expected = {train_path: "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4",
                val_path: "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3"}
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Corpus digest drift: {path}")
    train, validation = pages(train_path), pages(val_path)
    leaves = sorted({p["leaf_id"] for p in train}, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    if len(leaves) != 81:
        raise ValueError("Training leaf count drift")
    fit_leaves = set(leaves[:-16])
    fit_pages = [p for p in train if p["leaf_id"] in fit_leaves]
    calibration_pages = [p for p in train if p["leaf_id"] not in fit_leaves]
    alphabet = "".join(sorted({c for page in train for word, _ in slots(page)
                               if eligible(word) for c in word}))
    calibration_base = fit_base(fit_pages, alphabet)
    calibration_rows = make_rows(calibration_pages, calibration_base, alphabet)
    selected = select(calibration_rows, calibration_base)
    full_base = fit_base(train, alphabet)
    real_rows = make_rows(validation, full_base, alphabet)
    per_page = page_scores(real_rows, full_base[1], selected)
    observed = totals(per_page)
    ci = bootstrap(per_page)
    nulls = []
    for replicate in range(100):
        altered = permuted_slots(validation, 270127 + replicate)
        rows = make_rows(validation, full_base, alphabet, altered)
        nulls.append(totals(page_scores(rows, full_base[1], selected)))
        if time.monotonic() - started > 1200:
            raise TimeoutError("EXP-0027 exceeded 20-minute wall cap")
    positive = positive_slots(validation, alphabet)
    positive_result = totals(page_scores(make_rows(validation, full_base, alphabet, positive),
                                         full_base[1], selected))
    sorted_null_gain = sorted(item["base_minus_edit"] for item in nulls)
    decision = (observed["exact_minus_edit"] >= 0.05 and ci[0] > 0
                and observed["base_minus_edit"] > sorted_null_gain[95]
                and positive_result["exact_minus_edit"] > 0)
    result = {"experiment": "EXP-0027", "status": "complete", "decision": "support_copy_channel_research" if decision else "no_support_under_registered_gate",
              "time_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
              "command": sys.argv, "python": platform.python_version(), "source_sha256": sha256(Path(__file__)),
              "train_sha256": expected[train_path],
              "validation_sha256": expected[val_path], "alphabet": alphabet,
              "calibration": {"fit_leaves": sorted(fit_leaves), "calibration_leaves": sorted(set(leaves) - fit_leaves),
                              "scored_words": len(calibration_rows), "selected": selected},
              "validation": {"observed": observed, "leaf_bootstrap_95_exact_minus_edit": ci,
                             "pages": per_page},
              "order_control": {"n": len(nulls), "seed_rule": "270127+replicate",
                                "base_minus_edit": [x["base_minus_edit"] for x in nulls],
                                "exact_minus_edit": [x["exact_minus_edit"] for x in nulls]},
              "positive_control": {"seed": 270227, "metrics": positive_result},
              "limits": "Already-exposed validation; EVA units; one-edit channel is not a historical mechanism or plaintext decoder."}
    path = root / "results/EXP-0027/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    report = run(Path("."))
    print(json.dumps({"decision": report["decision"], "observed": report["validation"]["observed"],
                      "ci": report["validation"]["leaf_bootstrap_95_exact_minus_edit"],
                      "wall_seconds": report["wall_seconds"]}, indent=2))
