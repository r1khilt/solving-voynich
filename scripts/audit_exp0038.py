"""Independent input-to-decision replay of EXP-0038."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import random
import re

from scripts.audit_exp0036 import toy


ROOT = Path(".")
VIEWS = ("codepoint", "basic_compound", "cuva_like", "clean_cuva_like")
FEATURES = ("last", "first", "length")
BASIC = ("cth", "ckh", "cph", "cfh", "ch", "sh")
EXTRA = ("iiin", "iiir", "iiil", "iiim", "iin", "iir", "iil", "iim",
         "in", "ir", "il", "im", "eeee", "eee", "ee")
SYMBOLS = tuple("abcdefghijklmnopqrstuvwxyz'") + BASIC + EXTRA
FORBIDDEN = {"uncertain_space", "alternative", "rare_eva", "unreadable", "unknown_span"}
WORD = re.compile("[a-z']+\\Z")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(path):
    output = []
    counts = Counter()
    leaves = set()
    for line in path.read_text().splitlines():
        page = json.loads(line)
        counts["pages"] += 1
        metadata = page["metadata"]["page_variables"]
        context = (metadata.get("I", "?"), metadata.get("H", "?"))
        for locus in page["loci"]:
            counts["loci"] += 1
            clean = not ({a["kind"] for a in locus["annotations"]} & FORBIDDEN)
            counts["clean_loci"] += clean
            tokens = locus["text"].split()
            counts["eligible_words"] += sum(bool(WORD.fullmatch(word)) for word in tokens)
            counts["excluded_words"] += sum(not WORD.fullmatch(word) for word in tokens)
            for is_word, block in groupby(tokens, key=lambda word: bool(WORD.fullmatch(word))):
                run = list(block)
                if is_word and len(run) >= 4:
                    output.append((page["leaf_id"], context, locus["locus_id"], clean, run))
                    leaves.add(page["leaf_id"])
    counts["groups"] = len(output)
    counts["clean_groups"] = sum(group[3] for group in output)
    counts["pairs"] = sum(len(group[4]) - 1 for group in output)
    counts["clean_pairs"] = sum(len(group[4]) - 1 for group in output if group[3])
    counts["leaves"] = len(leaves)
    return output, dict(counts)


def convert(word, view):
    if view == "codepoint":
        return tuple(word)
    inventory = BASIC if view == "basic_compound" else BASIC + EXTRA
    expression = re.compile("|".join(re.escape(part) for part in sorted(inventory, key=lambda x: (-len(x), x))) + "|[a-z']")
    symbols = tuple(expression.findall(word))
    if "".join(symbols) != word:
        raise AssertionError("Tokenization not reversible")
    return symbols


def view_groups(source, view):
    rendered = []
    for leaf, context, locus, clean, words in source:
        symbols = [convert(word, view) for word in words]
        cells = [(units[0], units[-1], min(len(units), 8)) for units in symbols]
        rendered.append((leaf, context, locus, clean, cells))
    return rendered


def choose(source, view):
    return [group for group in source if group[3]] if view == "clean_cuva_like" else source


def edges(cells):
    for index in range(len(cells) - 1):
        role = 0 if index == 0 else 2 if index == len(cells) - 2 else 1
        old, new = cells[index], cells[index + 1]
        yield role, (old[1], old[0], old[2]), new[0]


def fit(source):
    glob = [Counter() for _ in range(3)]
    glob_n = [0, 0, 0]
    context = defaultdict(Counter)
    context_n = Counter()
    feature = defaultdict(Counter)
    feature_n = Counter()
    for _, ctx, _, _, cells in source:
        for role, features, target in edges(cells):
            glob[role][target] += 1
            glob_n[role] += 1
            ckey = (ctx, role)
            context[ckey][target] += 1
            context_n[ckey] += 1
            for index, value in enumerate(features):
                fkey = (ctx, role, index, value)
                feature[fkey][target] += 1
                feature_n[fkey] += 1
    return glob, glob_n, context, context_n, feature, feature_n


def score(source, model):
    glob, glob_n, context, context_n, feature, feature_n = model
    result = defaultdict(lambda: {"pairs": 0, "bits": [0.0, 0.0, 0.0]})
    n_symbols = len(SYMBOLS)
    for leaf, ctx, _, _, cells in source:
        row = result[leaf]
        for role, features, target in edges(cells):
            p_global = (glob[role][target] + 1) / (glob_n[role] + n_symbols)
            ckey = (ctx, role)
            p_base = (context[ckey][target] + 50 * p_global) / (context_n[ckey] + 50)
            row["pairs"] += 1
            for index, value in enumerate(features):
                fkey = (ctx, role, index, value)
                p_cond = (feature[fkey][target] + 20 * p_base) / (feature_n[fkey] + 20)
                row["bits"][index] += math.log2(p_cond) - math.log2(p_base)
    return dict(result)


def scramble(source, seed):
    rng = random.Random(seed)
    output = []
    for leaf, ctx, locus, clean, cells in source:
        interior = cells[1:-1]
        mid = len(interior) // 2
        a, b = interior[:mid], interior[mid:]
        rng.shuffle(a)
        rng.shuffle(b)
        output.append((leaf, ctx, locus, clean, [cells[0], *a, *b, cells[-1]]))
    return output


def average(scored, feature):
    return sum(row["bits"][feature] for row in scored.values()) / sum(row["pairs"] for row in scored.values())


def close(a, b):
    if not math.isclose(a, b, abs_tol=1e-10, rel_tol=0):
        raise AssertionError((a, b))


def verify_view(view, source, held, recorded):
    train_all = view_groups(source, view)
    valid_all = view_groups(held, view)
    train = choose(train_all, view)
    valid = choose(valid_all, view)
    assert len(train) == recorded["train_groups"] and len(valid) == recorded["validation_groups"]
    assert sum(len(g[4]) - 1 for g in train) == recorded["train_pairs"]
    assert sum(len(g[4]) - 1 for g in valid) == recorded["validation_pairs"]
    model = fit(train)
    real = score(valid, model)
    assert sorted(real) == sorted(recorded["real_by_leaf"])
    for leaf, row in real.items():
        assert row["pairs"] == recorded["real_by_leaf"][leaf]["pairs"]
        for index, name in enumerate(FEATURES):
            close(row["bits"][index], recorded["real_by_leaf"][leaf]["bits"][name])
            close(average(real, index), recorded["real_mean"][name])
    null_means = [[] for _ in FEATURES]
    null_leaf = {leaf: [] for leaf in real}
    for rep in range(200):
        perm = choose(scramble(valid_all, 380038 + rep), view)
        scored = score(perm, model)
        for index, name in enumerate(FEATURES):
            value = average(scored, index)
            null_means[index].append(value)
            close(value, recorded["null_means"][name][rep])
        for leaf in real:
            assert scored[leaf]["pairs"] == real[leaf]["pairs"]
            null_leaf[leaf].append(scored[leaf]["bits"][0])
            close(scored[leaf]["bits"][0], recorded["null_last_by_leaf"][leaf][rep])
    last_null = sum(null_means[0]) / 200
    close(last_null, recorded["last_null_mean"])
    close(sorted(null_means[0])[189], recorded["last_null_95th"])
    close(average(real, 0) - last_null, recorded["last_minus_null"])
    close(average(real, 0) - average(real, 1), recorded["last_minus_first"])
    close(average(real, 0) - average(real, 2), recorded["last_minus_length"])
    rng = random.Random(380138)
    leaves = sorted(real)
    null_by_leaf = {leaf: sum(null_leaf[leaf]) / 200 for leaf in leaves}
    draws = [[], [], []]
    for _ in range(2000):
        sample = rng.choices(leaves, k=len(leaves))
        count = sum(real[leaf]["pairs"] for leaf in sample)
        draws[0].append(sum(real[leaf]["bits"][0] - null_by_leaf[leaf] for leaf in sample) / count)
        draws[1].append(sum(real[leaf]["bits"][0] - real[leaf]["bits"][1] for leaf in sample) / count)
        draws[2].append(sum(real[leaf]["bits"][0] - real[leaf]["bits"][2] for leaf in sample) / count)
    for index, name in enumerate(("last_minus_null", "last_minus_first", "last_minus_length")):
        draws[index].sort()
        close(draws[index][49], recorded["bootstrap95"][name][0])
        close(draws[index][1949], recorded["bootstrap95"][name][1])


def main():
    saved = json.loads((ROOT / "results/EXP-0038/results.json").read_text())
    assert saved["source_sha256"] == sha(ROOT / "src/voynich/boundary_robustness.py")
    train_path = ROOT / "data/processed/zl3b/train.jsonl"
    valid_path = ROOT / "data/processed/zl3b/validation.jsonl"
    assert sha(train_path) == saved["train_sha256"]
    assert sha(valid_path) == saved["validation_sha256"]
    source, train_diag = groups(train_path)
    held, valid_diag = groups(valid_path)
    assert train_diag == saved["train"] and valid_diag == saved["validation"]
    for view in VIEWS:
        verify_view(view, source, held, saved["views"][view])
    def toy_groups(seed, n):
        return [(str(i), ("toy", "toy"), str(i), True, words)
                for i, (_, _, words) in enumerate(toy(seed, n))]
    toy_train = view_groups(toy_groups(360236, 100), "codepoint")
    toy_valid = view_groups(toy_groups(360336, 30), "codepoint")
    toy_model = fit(toy_train)
    real = score(toy_valid, toy_model)
    null = sum(average(score(scramble(toy_valid, 380838 + rep), toy_model), 0)
               for rep in range(100)) / 100
    toy_out = {"last_minus_null": average(real, 0) - null,
               "last_minus_first": average(real, 0) - average(real, 1)}
    for key, value in toy_out.items():
        close(value, saved["toy"][key])
    toy_validity = all(value >= 0.1 for value in toy_out.values())
    robust = toy_validity and all(row["last_minus_null"] >= 0.02 and row["bootstrap95"]["last_minus_null"][0] > 0
                                  and row["real_mean"]["last"] > row["last_null_95th"]
                                  for row in saved["views"].values())
    specific = toy_validity and all(row[f"last_minus_{other}"] >= 0.02 and
                                    row["bootstrap95"][f"last_minus_{other}"][0] > 0
                                    for row in saved["views"].values() for other in ("first", "length"))
    assert robust == saved["representation_robust"]
    assert specific == saved["terminal_specific"]
    assert saved["decision"] == ("assessed" if toy_validity else "invalid_positive_control")
    report = {"experiment": "EXP-0038", "audit": "pass", "result_sha256": sha(ROOT / "results/EXP-0038/results.json"),
              "views": len(VIEWS), "permutations_per_view": 200, "bootstrap_draws_per_view": 2000,
              "representation_robust": robust, "terminal_specific": specific, "independent_replay": True}
    path = ROOT / "results/EXP-0038/audit.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
