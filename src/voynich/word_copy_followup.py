"""EXP-0028: global edit prior and layout-preserving cache controls."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

from .word_copy_channel import (
    HALVES, LAMBDAS, WINDOWS, bootstrap, edit_parts, eligible,
    fit_base, make_rows, pages, positive_slots, sha256, slots, totals,
)


BETAS = (1000, 10000, 100000, 1000000, 1000000000000)
GAMMAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
EDIT_WEIGHTS = (0.25, 0.5, 0.75)


class GlobalEdit:
    """Query only inverse one-edit neighbors; no giant output-vocabulary table."""

    def __init__(self, counts: Counter, n: int, alphabet: str):
        self.counts, self.n, self.alphabet = counts, n, alphabet
        self.memo: dict[str, float] = {}

    def prob(self, target: str) -> float:
        if target in self.memo:
            return self.memo[target]
        n, alphabet = len(target), self.alphabet
        sources = {target}
        for i in range(n):
            sources.update(target[:i] + c + target[i + 1:]
                           for c in alphabet if c != target[i])
            sources.add(target[:i] + target[i + 1:])
        for i in range(n + 1):
            sources.update(target[:i] + c + target[i:] for c in alphabet)
        mass = 0.0
        for source in sources:
            count = self.counts.get(source, 0)
            if count:
                exact, sub, ins, delete = edit_parts(source, target, len(alphabet))
                mass += count * (0.75 * exact + 0.125 * sub + 0.0625 * ins + 0.0625 * delete)
        self.memo[target] = mass / self.n
        return self.memo[target]


def static_prob(row: dict, n: int, beta: int, gamma: float, global_edit: GlobalEdit) -> float:
    base = (row["count"] + beta * row["char_p"]) / (n + beta)
    return (1 - gamma) * base + gamma * global_edit.prob(row["word"])


def predictive_prob(row: dict, n: int, params: dict, global_edit: GlobalEdit) -> float:
    base = static_prob(row, n, params["beta"], params["gamma"], global_edit)
    lam = params.get("lambda", 0.0)
    if not lam or not row["has_history"]:
        return base
    features = row["features"][f'{params["window"]}:{params["half"]}']
    if params["arm"] == "exact":
        local = features[0]
    else:
        q = params["q_exact"]
        local = q * features[0] + (1 - q) * (
            0.5 * features[1] + 0.25 * features[2] + 0.25 * features[3])
    return (1 - lam) * base + lam * local


def loss(rows: list[dict], n: int, params: dict, global_edit: GlobalEdit) -> float:
    return sum(-math.log2(predictive_prob(row, n, params, global_edit)) for row in rows)


def select(rows: list[dict], n: int, global_edit: GlobalEdit) -> dict[str, dict]:
    static_candidates = [{"arm": "base", "beta": beta, "gamma": gamma, "lambda": 0.0}
                         for beta in BETAS for gamma in GAMMAS]
    best_static = min(static_candidates, key=lambda p: loss(rows, n, p, global_edit))
    output = {"base": best_static}
    for arm in ("exact", "edit"):
        candidates = [{"arm": arm, "beta": best_static["beta"], "gamma": best_static["gamma"],
                       "lambda": lam, "window": window, "half": half,
                       **({"q_exact": q} if arm == "edit" else {})}
                      for window in WINDOWS for half in HALVES for lam in LAMBDAS
                      for q in (EDIT_WEIGHTS if arm == "edit" else (None,))]
        output[arm] = min(candidates, key=lambda p: loss(rows, n, p, global_edit))
    return output


def page_scores(rows: list[dict], n: int, models: dict[str, dict], global_edit: GlobalEdit) -> list[dict]:
    output: dict[str, dict] = {}
    for row in rows:
        item = output.setdefault(row["page"], {"page": row["page"], "leaf": row["leaf"],
                                                "count": 0, "bits": {arm: 0.0 for arm in models}})
        item["count"] += 1
        for arm, params in models.items():
            item["bits"][arm] -= math.log2(predictive_prob(row, n, params, global_edit))
    return sorted(output.values(), key=lambda x: x["page"])


def shuffled_slots(eval_pages: list[dict], seed: int, mode: str) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    output = {}
    for page in eval_pages:
        if mode == "type":
            original = slots(page)
            by_type: dict[str, list[str]] = {}
            for word, kind in original:
                if eligible(word):
                    by_type.setdefault(kind, []).append(word)
            for words in by_type.values():
                rng.shuffle(words)
            cursors = {kind: iter(words) for kind, words in by_type.items()}
            output[page["page_id"]] = [
                (next(cursors[kind]), kind) if eligible(word) else (word, kind)
                for word, kind in original]
        elif mode == "locus":
            altered = []
            for locus in page["loci"]:
                kind = locus["locus_type"]
                original = locus["text"].split()
                words = [w for w in original if eligible(w)]
                rng.shuffle(words)
                cursor = iter(words)
                altered.extend((next(cursor), kind) if eligible(w) else (w, kind)
                               for w in original)
            output[page["page_id"]] = altered
        else:
            raise ValueError(mode)
    return output


def run(root: Path) -> dict:
    started = time.monotonic()
    original = json.loads((root / "results/EXP-0027/results.json").read_text())
    train_path = root / "data/processed/zl3b/train.jsonl"
    val_path = root / "data/processed/zl3b/validation.jsonl"
    if sha256(train_path) != original["train_sha256"] or sha256(val_path) != original["validation_sha256"]:
        raise ValueError("Corpus drift from EXP-0027")
    train, validation = pages(train_path), pages(val_path)
    fit_leaves = set(original["calibration"]["fit_leaves"])
    fit_pages = [p for p in train if p["leaf_id"] in fit_leaves]
    calibration_pages = [p for p in train if p["leaf_id"] not in fit_leaves]
    if len(fit_leaves) != 65 or len({p["leaf_id"] for p in calibration_pages}) != 16:
        raise ValueError("Calibration split drift")
    alphabet = original["alphabet"]
    fit = fit_base(fit_pages, alphabet)
    fit_global = GlobalEdit(fit[0], fit[1], alphabet)
    calibration = make_rows(calibration_pages, fit, alphabet)
    selected = select(calibration, fit[1], fit_global)
    full = fit_base(train, alphabet)
    full_global = GlobalEdit(full[0], full[1], alphabet)
    real_rows = make_rows(validation, full, alphabet)
    per_page = page_scores(real_rows, full[1], selected, full_global)
    observed = totals(per_page)
    ci = bootstrap(per_page)
    controls = {}
    for mode, seed in (("type", 280127), ("locus", 280227)):
        values = []
        for replicate in range(100):
            altered = shuffled_slots(validation, seed + replicate, mode)
            rows = make_rows(validation, full, alphabet, altered)
            values.append(totals(page_scores(rows, full[1], selected, full_global)))
            if time.monotonic() - started > 1200:
                raise TimeoutError("EXP-0028 exceeded 20-minute wall cap")
        controls[mode] = {"seed_rule": f"{seed}+replicate", "n": 100,
                          "exact_minus_edit": [v["exact_minus_edit"] for v in values],
                          "base_minus_edit": [v["base_minus_edit"] for v in values]}
    positive = positive_slots(validation, alphabet, seed=280327)
    positive_score = totals(page_scores(make_rows(validation, full, alphabet, positive),
                                        full[1], selected, full_global))
    passed = (observed["exact_minus_edit"] >= 0.03 and ci[0] > 0
              and all(observed["exact_minus_edit"] > sorted(control["exact_minus_edit"])[95]
                      for control in controls.values())
              and positive_score["exact_minus_edit"] > 0)
    result = {"experiment": "EXP-0028", "status": "complete",
              "decision": "local_edit_order_support" if passed else "not_supported_under_stronger_controls",
              "time_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
              "command": sys.argv, "python": platform.python_version(),
              "source_sha256": sha256(Path(__file__)), "exp0027_sha256": sha256(root / "results/EXP-0027/results.json"),
              "train_sha256": sha256(train_path), "validation_sha256": sha256(val_path),
              "calibration": {"scored_words": len(calibration), "selected": selected},
              "validation": {"observed": observed, "leaf_bootstrap_95_exact_minus_edit": ci,
                             "pages": per_page},
              "order_controls": controls, "positive_control": positive_score,
              "limits": "Adaptive exposed validation; no historical production or decipherment claim."}
    out = root / "results/EXP-0028/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run(Path("."))
    print(json.dumps({"decision": result["decision"], "observed": result["validation"]["observed"],
                      "ci": result["validation"]["leaf_bootstrap_95_exact_minus_edit"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))
