"""EXP-0038: boundary-signal sensitivity to EVA segmentation and no-copy context."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import time

from .boundary_order import TRAIN_SHA, VALID_SHA, digest, toy


EXPERIMENT = "EXP-0038"
WORD = re.compile(r"[a-z']+\Z")
BASIC = ("cth", "ckh", "cph", "cfh", "ch", "sh")
MINIMS = ("iiin", "iiir", "iiil", "iiim", "iin", "iir", "iil", "iim",
          "in", "ir", "il", "im", "eeee", "eee", "ee")
VIEWS = ("codepoint", "basic_compound", "cuva_like", "clean_cuva_like")
FEATURES = ("last", "first", "length")
SYMBOLS = tuple("abcdefghijklmnopqrstuvwxyz'") + BASIC + MINIMS
UNCERTAIN = frozenset(("uncertain_space", "alternative", "rare_eva", "unreadable", "unknown_span"))
N_PERM, N_BOOT = 200, 2000


def load_groups(path: Path) -> tuple[list[dict], dict]:
    groups = []
    diagnostics = Counter()
    for line in path.read_text().splitlines():
        page = json.loads(line)
        diagnostics["pages"] += 1
        context = (page["metadata"]["page_variables"].get("I", "?"),
                   page["metadata"]["page_variables"].get("H", "?"))
        for locus in page["loci"]:
            diagnostics["loci"] += 1
            clean = not any(annotation["kind"] in UNCERTAIN for annotation in locus["annotations"])
            if clean:
                diagnostics["clean_loci"] += 1
            run = []
            for word in locus["text"].split():
                if WORD.fullmatch(word):
                    run.append(word)
                    diagnostics["eligible_words"] += 1
                else:
                    diagnostics["excluded_words"] += 1
                    if len(run) >= 4:
                        groups.append({"leaf": page["leaf_id"], "context": context,
                                       "locus": locus["locus_id"], "clean": clean, "words": run})
                    run = []
            if len(run) >= 4:
                groups.append({"leaf": page["leaf_id"], "context": context,
                               "locus": locus["locus_id"], "clean": clean, "words": run})
    diagnostics["groups"] = len(groups)
    diagnostics["clean_groups"] = sum(group["clean"] for group in groups)
    diagnostics["pairs"] = sum(len(group["words"]) - 1 for group in groups)
    diagnostics["clean_pairs"] = sum(len(group["words"]) - 1 for group in groups if group["clean"])
    diagnostics["leaves"] = len({group["leaf"] for group in groups})
    return groups, dict(diagnostics)


def tokenize(word: str, view: str) -> tuple[str, ...]:
    if view == "codepoint":
        return tuple(word)
    inventory = BASIC if view == "basic_compound" else BASIC + MINIMS
    order = sorted(inventory, key=lambda item: (-len(item), item))
    tokens = []
    index = 0
    while index < len(word):
        match = next((unit for unit in order if word.startswith(unit, index)), None)
        if match is None:
            match = word[index]
        tokens.append(match)
        index += len(match)
    if "".join(tokens) != word:
        raise ValueError("Non-reversible segmentation")
    return tuple(tokens)


def encode(groups: list[dict], view: str) -> list[dict]:
    cache = {}
    output = []
    for group in groups:
        words = []
        for word in group["words"]:
            if word not in cache:
                units = tokenize(word, view)
                cache[word] = (units[0], units[-1], min(len(units), 8))
            words.append(cache[word])
        output.append({**group, "items": words})
    return output


def select(groups: list[dict], view: str) -> list[dict]:
    return [group for group in groups if group["clean"]] if view == "clean_cuva_like" else groups


def events(items: list[tuple[str, str, int]]):
    for index, (previous, following) in enumerate(zip(items, items[1:])):
        role = "first" if index == 0 else "last" if index == len(items) - 2 else "middle"
        values = {"last": previous[1], "first": previous[0], "length": previous[2]}
        yield role, values, following[0]


def fit(groups: list[dict]) -> dict:
    global_counts, global_n = defaultdict(Counter), Counter()
    context_counts, context_n = defaultdict(Counter), Counter()
    feature_counts, feature_n = defaultdict(Counter), Counter()
    for group in groups:
        context = tuple(group["context"])
        for role, values, target in events(group["items"]):
            if target not in SYMBOLS:
                raise ValueError("Target outside frozen alphabet")
            global_counts[role][target] += 1
            global_n[role] += 1
            key = (context, role)
            context_counts[key][target] += 1
            context_n[key] += 1
            for feature, value in values.items():
                fkey = (context, role, feature, value)
                feature_counts[fkey][target] += 1
                feature_n[fkey] += 1
    if any(global_n[role] == 0 for role in ("first", "middle", "last")):
        raise ValueError("Missing training role")
    return {"global_counts": global_counts, "global_n": global_n,
            "context_counts": context_counts, "context_n": context_n,
            "feature_counts": feature_counts, "feature_n": feature_n, "cache": {}}


def probability(model: dict, context: tuple[str, str], role: str,
                feature: str, value: str | int, target: str) -> tuple[float, float]:
    cache_key = (context, role, feature, value, target)
    if cache_key not in model["cache"]:
        k = len(SYMBOLS)
        global_prior = (model["global_counts"][role][target] + 1) / (model["global_n"][role] + k)
        ckey = (context, role)
        base = (model["context_counts"][ckey][target] + 50 * global_prior) / (model["context_n"][ckey] + 50)
        fkey = (context, role, feature, value)
        conditional = (model["feature_counts"][fkey][target] + 20 * base) / (model["feature_n"][fkey] + 20)
        model["cache"][cache_key] = (conditional, base)
    return model["cache"][cache_key]


def score(groups: list[dict], model: dict) -> dict[str, dict]:
    output = defaultdict(lambda: {"pairs": 0, "bits": {feature: 0.0 for feature in FEATURES}})
    for group in groups:
        record = output[group["leaf"]]
        context = tuple(group["context"])
        for role, values, target in events(group["items"]):
            record["pairs"] += 1
            for feature, value in values.items():
                conditional, base = probability(model, context, role, feature, value, target)
                record["bits"][feature] += math.log2(conditional / base)
    return dict(sorted(output.items()))


def shuffle(groups: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    changed = []
    for group in groups:
        inside = list(group["items"][1:-1])
        split = len(inside) // 2
        first, second = inside[:split], inside[split:]
        rng.shuffle(first)
        rng.shuffle(second)
        changed.append({**group, "items": [group["items"][0], *first, *second, group["items"][-1]]})
    return changed


def mean(rows: dict[str, dict], feature: str) -> float:
    return sum(row["bits"][feature] for row in rows.values()) / sum(row["pairs"] for row in rows.values())


def bootstrap(real: dict[str, dict], null_last: dict[str, list[float]]) -> dict[str, list[float]]:
    leaves = sorted(real)
    null_leaf_mean = {leaf: sum(null_last[leaf]) / N_PERM for leaf in leaves}
    rng = random.Random(380138)
    outputs = {name: [] for name in ("last_minus_null", "last_minus_first", "last_minus_length")}
    for _ in range(N_BOOT):
        sample = rng.choices(leaves, k=len(leaves))
        n = sum(real[leaf]["pairs"] for leaf in sample)
        outputs["last_minus_null"].append(sum(real[leaf]["bits"]["last"] - null_leaf_mean[leaf]
                                              for leaf in sample) / n)
        for other in ("first", "length"):
            outputs[f"last_minus_{other}"].append(sum(real[leaf]["bits"]["last"] - real[leaf]["bits"][other]
                                                      for leaf in sample) / n)
    for values in outputs.values():
        values.sort()
    return {name: [values[49], values[1949]] for name, values in outputs.items()}


def run(root: Path) -> dict:
    start = time.monotonic()
    train_path = root / "data/processed/zl3b/train.jsonl"
    valid_path = root / "data/processed/zl3b/validation.jsonl"
    if digest(train_path) != TRAIN_SHA or digest(valid_path) != VALID_SHA:
        raise ValueError("Frozen data drift")
    raw_train, train_diag = load_groups(train_path)
    raw_valid, valid_diag = load_groups(valid_path)
    results = {}
    for view in VIEWS:
        train = select(encode(raw_train, view), view)
        valid_all = encode(raw_valid, view)
        valid = select(valid_all, view)
        model = fit(train)
        real = score(valid, model)
        real_mean = {feature: mean(real, feature) for feature in FEATURES}
        null_means = {feature: [] for feature in FEATURES}
        null_last = {leaf: [] for leaf in real}
        for rep in range(N_PERM):
            shuffled_all = shuffle(valid_all, 380038 + rep)
            shuffled = select(shuffled_all, view)
            scored = score(shuffled, model)
            for feature in FEATURES:
                null_means[feature].append(mean(scored, feature))
            for leaf in real:
                if scored[leaf]["pairs"] != real[leaf]["pairs"]:
                    raise ValueError("Permutation pair drift")
                null_last[leaf].append(scored[leaf]["bits"]["last"])
        last_null_mean = sum(null_means["last"]) / N_PERM
        intervals = bootstrap(real, null_last)
        results[view] = {"train_groups": len(train), "validation_groups": len(valid),
                         "train_pairs": sum(len(group["items"]) - 1 for group in train),
                         "validation_pairs": sum(len(group["items"]) - 1 for group in valid),
                         "real_by_leaf": real, "real_mean": real_mean,
                         "null_means": null_means, "null_last_by_leaf": null_last,
                         "last_null_mean": last_null_mean,
                         "last_null_95th": sorted(null_means["last"])[189],
                         "last_minus_null": real_mean["last"] - last_null_mean,
                         "last_minus_first": real_mean["last"] - real_mean["first"],
                         "last_minus_length": real_mean["last"] - real_mean["length"],
                         "bootstrap95": intervals}
        if time.monotonic() - start > 180:
            raise TimeoutError("EXP-0038 CPU cap exceeded")
    toy_train = [{"leaf": group["leaf"], "context": ("toy", "toy"), "clean": True,
                  "words": group["words"]} for group in toy(360236, 100)]
    toy_valid = [{"leaf": group["leaf"], "context": ("toy", "toy"), "clean": True,
                  "words": group["words"]} for group in toy(360336, 30)]
    toy_model = fit(encode(toy_train, "codepoint"))
    toy_encoded = encode(toy_valid, "codepoint")
    toy_real = score(toy_encoded, toy_model)
    toy_last = mean(toy_real, "last")
    toy_first = mean(toy_real, "first")
    toy_null = sum(mean(score(shuffle(toy_encoded, 380838 + rep), toy_model), "last")
                   for rep in range(100)) / 100
    toy_controls = {"last_minus_null": toy_last - toy_null, "last_minus_first": toy_last - toy_first}
    valid_toy = all(value >= 0.1 for value in toy_controls.values())
    robust = valid_toy and all(row["last_minus_null"] >= 0.02 and row["bootstrap95"]["last_minus_null"][0] > 0
                               and row["real_mean"]["last"] > row["last_null_95th"] for row in results.values())
    specific = valid_toy and all(row[f"last_minus_{other}"] >= 0.02 and
                                 row["bootstrap95"][f"last_minus_{other}"][0] > 0
                                 for row in results.values() for other in ("first", "length"))
    report = {"experiment": EXPERIMENT, "status": "complete",
              "decision": "invalid_positive_control" if not valid_toy else "assessed",
              "representation_robust": robust, "terminal_specific": specific,
              "date_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - start,
              "command": sys.argv, "python": platform.python_version(),
              "source_sha256": digest(Path(__file__)), "train_sha256": TRAIN_SHA, "validation_sha256": VALID_SHA,
              "train": train_diag, "validation": valid_diag, "toy": toy_controls,
              "views": results, "limits": "Adaptive exposed-validation sensitivity study on one ZL3b transcription; no historical mechanism inferred."}
    out = root / "results/EXP-0038/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run(Path("."))
    print(json.dumps({"decision": result["decision"], "representation_robust": result["representation_robust"],
                      "terminal_specific": result["terminal_specific"], "toy": result["toy"],
                      "views": {name: {key: row[key] for key in ("validation_pairs", "last_minus_null",
                                                                     "last_minus_first", "last_minus_length", "bootstrap95")}
                                for name, row in result["views"].items()}}, indent=2))
