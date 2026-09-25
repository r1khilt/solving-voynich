"""Independent raw-input replay of EXP-0039, without importing its runner."""

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
NAMES = ("last", "first", "length", "penult", "identity", "two_back")
COMPOUNDS = ("cth", "ckh", "cph", "cfh", "ch", "sh")
GC_HASH = "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f"
SPLIT_HASH = "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e"
TRAIN_HASH = "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4"
VALID_HASH = "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gc_chunks(raw):
    # Explicit separator stream, independent of the runner's character walker.
    marked = re.sub(r"<[^>]+>", ",", raw)
    parts = re.split(r"([.,])", marked)
    result = []
    run = []
    for index in range(0, len(parts), 2):
        token = parts[index].strip()
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if token and "?" not in token:
            atoms = re.findall(r"@\d{3};|.", token)
            if "".join(atoms) != token or any(atom in "[]{}>=O" for atom in atoms):
                raise AssertionError("Unsupported GC symbol")
            if any(atom.startswith("@") and not 128 <= int(atom[1:4]) <= 255 for atom in atoms):
                raise AssertionError("Invalid GC code")
            run.append(tuple(atoms))
        else:
            if len(run) >= 4:
                result.append(list(run))
            run = []
        if separator != ".":
            if len(run) >= 4:
                result.append(list(run))
            run = []
    return result


def gc_source(path, split):
    raw_lines = path.read_text(encoding="ascii").splitlines()
    assert raw_lines[0] == "#=IVTFF v101 2.0 M 6"
    selected = None
    meta = {}
    hand = "?"
    output = []
    ids = set()
    counts = Counter()
    for raw in raw_lines[1:]:
        if not raw or raw.startswith("#"):
            continue
        page = re.fullmatch(r"<(f(?:\d+[rv]\d*|Ros))>\s*(?:<!([^>]*)>)?\s*", raw)
        if page:
            leaf = re.match(r"f\d+", page[1])
            selected = page[1] if leaf and split.get(leaf.group()) in {"train", "validation"} else None
            meta = dict(re.findall(r"\$([A-Z])=([A-Za-z0-9@])", page[2] or ""))
            hand = meta.get("H", "?")
            if selected:
                counts[f"{split[leaf.group()]}_pages"] += 1
            continue
        entry = re.match(r"<(f(?:\d+[rv]\d*|Ros))\.(\d+),([^>]*)>\s*(.*)$", raw)
        assert entry is not None, raw[:50]
        if selected is None:
            continue
        assert entry[1] == selected
        field = entry[3]
        assert re.fullmatch(r"[@+*=&~/!][PLCR][a-z0-9](?:;[A-Za-z0-9])?", field)
        if field[1:3] != "P0" or field.startswith("!"):
            continue
        locus_id = f"{selected}.{entry[2]}"
        assert locus_id not in ids
        ids.add(locus_id)
        leaf = re.match(r"f\d+", selected).group()
        counts[f"{split[leaf]}_p0_loci"] += 1
        changed = re.findall(r"<@H=([A-Za-z0-9@])>", entry[4])
        assert len(changed) <= 1
        if changed:
            assert re.match(r"^(?:<%>)?<@H=[A-Za-z0-9@]>", entry[4])
            hand = changed[0]
        for words in gc_chunks(entry[4]):
            output.append((leaf, locus_id, (meta.get("I", "?"), hand), words))
    counts["groups"] = len(output)
    counts["pairs"] = sum(len(row[3]) - 1 for row in output)
    counts["leaves"] = len({row[0] for row in output})
    return output, dict(counts), ids


def zl_source(path, common_ids):
    blocks = []
    counts = Counter()
    pattern = re.compile("|".join(re.escape(x) for x in sorted(COMPOUNDS, key=lambda v: (-len(v), v))) + "|[a-z']")
    for raw in path.read_text().splitlines():
        page = json.loads(raw)
        counts["pages"] += 1
        base_meta = page["metadata"]["page_variables"]
        for locus in page["loci"]:
            if locus["locus_type"] != "P0" or locus["locus_id"] not in common_ids:
                continue
            counts["matched_p0_loci"] += 1
            hand = locus["text_tags"].get("H", base_meta.get("H", "?"))
            ctx = (base_meta.get("I", "?"), hand)
            tokens = locus["text"].split()
            for valid, candidates in groupby(tokens, lambda x: bool(re.fullmatch(r"[a-z']+", x))):
                if not valid:
                    continue
                rendered = []
                for word in candidates:
                    atoms = tuple(pattern.findall(word))
                    assert "".join(atoms) == word
                    rendered.append(atoms)
                if len(rendered) >= 4:
                    blocks.append((page["leaf_id"], locus["locus_id"], ctx, rendered))
    counts["groups"] = len(blocks)
    counts["pairs"] = sum(len(row[3]) - 1 for row in blocks)
    counts["leaves"] = len({row[0] for row in blocks})
    return blocks, dict(counts)


def pairs(words):
    for index in range(len(words) - 1):
        prior = words[index]
        role = 0 if index == 0 else 2 if index == len(words) - 2 else 1
        values = (prior[-1], prior[0], min(len(prior), 8),
                  prior[-2] if len(prior) > 1 else "<single>", prior,
                  (prior[-1], words[index - 1][-1] if index else "<start>"))
        yield role, values, words[index + 1][0]


def model_fit(source):
    alphabet = {atom for _, _, _, words in source for word in words for atom in word}
    global_counts = [Counter() for _ in range(3)]
    global_total = [0, 0, 0]
    base_counts = defaultdict(Counter)
    base_total = Counter()
    conditional_counts = defaultdict(Counter)
    conditional_total = Counter()
    for _, _, ctx, words in source:
        for role, values, outcome in pairs(words):
            global_counts[role][outcome] += 1
            global_total[role] += 1
            base_counts[(ctx, role)][outcome] += 1
            base_total[(ctx, role)] += 1
            for index, feature in enumerate(values):
                key = (ctx, role, index, feature)
                conditional_counts[key][outcome] += 1
                conditional_total[key] += 1
    return (alphabet, global_counts, global_total, base_counts, base_total,
            conditional_counts, conditional_total)


def score(source, model):
    alphabet, globals_, globals_n, base, base_n, features, features_n = model
    scored = defaultdict(lambda: {"pairs": 0, "bits": [0.0] * 6})
    for leaf, _, ctx, words in source:
        row = scored[leaf]
        for role, values, target in pairs(words):
            target = target if target in alphabet else "<unk>"
            prior = (globals_[role][target] + 1) / (globals_n[role] + len(alphabet) + 1)
            contextual = (base[(ctx, role)][target] + 50 * prior) / (base_n[(ctx, role)] + 50)
            last_key = (ctx, role, 0, values[0])
            last = (features[last_key][target] + 20 * contextual) / (features_n[last_key] + 20)
            row["pairs"] += 1
            row["bits"][0] += math.log2(last) - math.log2(contextual)
            for index in (1, 2, 3, 4, 5):
                key = (ctx, role, index, values[index])
                inherited = contextual if index <= 3 else last
                estimate = (features[key][target] + 20 * inherited) / (features_n[key] + 20)
                row["bits"][index] += math.log2(estimate) - math.log2(inherited)
    return dict(scored)


def scramble(rows, seed):
    generator = random.Random(seed)
    result = []
    for leaf, locus, context, words in rows:
        interior = list(words[1:-1])
        split = len(interior) // 2
        a = interior[:split]
        b = interior[split:]
        generator.shuffle(a)
        generator.shuffle(b)
        result.append((leaf, locus, context, [words[0], *a, *b, words[-1]]))
    return result


def aggregate(rows, index):
    return sum(row["bits"][index] for row in rows.values()) / sum(row["pairs"] for row in rows.values())


def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0), (actual, expected)


def replay(name, source, held, seed, saved):
    assert len(source) == saved["train_groups"]
    assert sum(len(row[3]) - 1 for row in source) == saved["train_pairs"]
    assert len(held) == saved["validation_groups"]
    assert sum(len(row[3]) - 1 for row in held) == saved["validation_pairs"]
    model = model_fit(source)
    assert len(model[0]) + 1 == saved["alphabet_size_train_plus_unk"]
    unseen = sum(target not in model[0] for _, _, _, words in held for _, _, target in pairs(words))
    assert unseen == saved["validation_unseen_target_pairs"]
    real = score(held, model)
    assert sorted(real) == sorted(saved["real_by_leaf"])
    for leaf, row in real.items():
        assert row["pairs"] == saved["real_by_leaf"][leaf]["pairs"]
        for index, feature in enumerate(NAMES):
            close(row["bits"][index], saved["real_by_leaf"][leaf]["bits"][feature])
            close(aggregate(real, index), saved["real_mean"][feature])
    null = []
    by_leaf = {leaf: [] for leaf in real}
    for rep in range(200):
        scrambled = score(scramble(held, seed + rep), model)
        null.append(aggregate(scrambled, 0))
        for leaf in real:
            by_leaf[leaf].append(scrambled[leaf]["bits"][0])
    null_mean = sum(null) / 200
    close(null_mean, saved["null_last_mean"])
    close(sorted(null)[189], saved["null_last_95th"])
    avg_leaf = {leaf: sum(values) / 200 for leaf, values in by_leaf.items()}
    for leaf, value in avg_leaf.items():
        close(value, saved["null_last_by_leaf_mean"][leaf])
    close(aggregate(real, 0) - null_mean, saved["contrasts"]["last_minus_null"])
    for index, feature in enumerate(NAMES[1:4], start=1):
        close(aggregate(real, 0) - aggregate(real, index), saved["contrasts"][f"last_minus_{feature}"])
    leaves = sorted(real)
    chooser = random.Random(seed + 1000)
    samples = [chooser.choices(leaves, k=len(leaves)) for _ in range(2000)]
    expected = ["last_minus_null", "last_minus_first", "last_minus_length",
                "last_minus_penult", "identity_over_last", "two_back_over_last"]
    for index, key in enumerate(expected):
        differences = []
        for sample in samples:
            pairs_n = sum(real[leaf]["pairs"] for leaf in sample)
            if index == 0:
                total = sum(real[leaf]["bits"][0] - avg_leaf[leaf] for leaf in sample)
            elif index < 4:
                total = sum(real[leaf]["bits"][0] - real[leaf]["bits"][index] for leaf in sample)
            else:
                total = sum(real[leaf]["bits"][index] for leaf in sample)
            differences.append(total / pairs_n)
        differences.sort()
        close(differences[49], saved["bootstrap95"][key][0])
        close(differences[1949], saved["bootstrap95"][key][1])
    print(f"{name}: independent source-to-score replay pass")


def main():
    paths = [ROOT / "data/raw/v101/GC2a-n.txt", ROOT / "data/manifests/zl3b_split.json",
             ROOT / "data/processed/zl3b/train.jsonl", ROOT / "data/processed/zl3b/validation.jsonl"]
    assert [digest(path) for path in paths] == [GC_HASH, SPLIT_HASH, TRAIN_HASH, VALID_HASH]
    report = json.loads((ROOT / "results/EXP-0039/results.json").read_text())
    assert report["source_sha256"] == digest(ROOT / "src/voynich/boundary_cross_transcription.py")
    split = json.loads(paths[1].read_text())["leaf_assignments"]
    all_gc, gc_diag, ids = gc_source(paths[0], split)
    assert gc_diag == report["gc_diagnostics"]
    gc_fit = [row for row in all_gc if split[row[0]] == "train"]
    gc_hold = [row for row in all_gc if split[row[0]] == "validation"]
    z_fit, z_fit_diag = zl_source(paths[2], ids)
    z_hold, z_hold_diag = zl_source(paths[3], ids)
    assert z_fit_diag == report["zl_train_diagnostics"]
    assert z_hold_diag == report["zl_validation_diagnostics"]
    assert z_fit_diag["matched_p0_loci"] + z_hold_diag["matched_p0_loci"] == len(ids)
    replay("gc_v101", gc_fit, gc_hold, 390039, report["views"]["gc_v101"])
    replay("zl_basic_matched", z_fit, z_hold, 390139, report["views"]["zl_basic_matched"])

    def toy_rows(seed, count):
        return [(group["leaf"], group["leaf"], ("toy", "toy"),
                 [tuple(word) for word in group["words"]]) for group in toy(seed, count)]
    t_fit = toy_rows(390239, 100)
    t_hold = toy_rows(390339, 30)
    t_model = model_fit(t_fit)
    t_real = score(t_hold, t_model)
    t_null = sum(aggregate(score(scramble(t_hold, 390439 + rep), t_model), 0)
                 for rep in range(200)) / 200
    t_values = {"last_minus_null": aggregate(t_real, 0) - t_null,
                "last_minus_first": aggregate(t_real, 0) - aggregate(t_real, 1)}
    for key, value in t_values.items():
        close(value, report["toy"][key])
    valid = all(value >= 0.1 for value in t_values.values())
    gc = report["views"]["gc_v101"]
    coverage = gc["validation_pairs"] >= 1000 and len(gc["real_by_leaf"]) >= 8
    assert coverage == report["coverage"]
    cross = valid and coverage and all(
        view["contrasts"]["last_minus_null"] >= .02 and
        view["bootstrap95"]["last_minus_null"][0] > 0 and
        view["real_mean"]["last"] > view["null_last_95th"]
        for view in report["views"].values())
    specific = valid and coverage and all(
        view["contrasts"][f"last_minus_{name}"] >= .02 and
        view["bootstrap95"][f"last_minus_{name}"][0] > 0
        for view in report["views"].values() for name in ("first", "length", "penult"))
    assert cross == report["cross_transcription_signal"]
    assert specific == report["terminal_specific"]
    assert report["decision"] == ("assessed" if valid else "invalid_toy_control")
    audit = {"experiment": "EXP-0039", "audit": "pass", "independent_replay": True,
             "result_sha256": digest(ROOT / "results/EXP-0039/results.json"),
             "views": 2, "permutations_per_view": 200,
             "bootstrap_draws_per_view": 2000,
             "cross_transcription_signal": cross, "terminal_specific": specific}
    output = ROOT / "results/EXP-0039/audit.json"
    output.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
