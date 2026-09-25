"""EXP-0039: independent GC-v101 transcription boundary-order challenge."""

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
import time

from .boundary_order import toy


EXPERIMENT = "EXP-0039"
GC_SHA = "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f"
SPLIT_SHA = "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e"
ZL_TRAIN_SHA = "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4"
ZL_VALID_SHA = "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3"
PAGE_ID = r"(?:f\d+[rv]\d*|fRos)"
PAGE = re.compile(rf"^<({PAGE_ID})>\s*(?:<!([^>]*)>)?\s*$")
LOCUS = re.compile(rf"^<({PAGE_ID})\.(\d+),([^>]*)>\s*(.*)$")
HIGH = re.compile(r"@(?:1[2-9]\d|2(?:[0-4]\d|5[0-5]));")
ZL_WORD = re.compile(r"[a-z']+\Z")
BASIC = ("cth", "ckh", "cph", "cfh", "ch", "sh")
FEATURES = ("last", "first", "length", "penult", "identity", "two_back")
N_PERM = 200
N_BOOT = 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_gc_body(body: str) -> list[list[tuple[str, ...]]]:
    """Return certain dot-linked runs; commas/unknowns/markup sever adjacency.

    v101 printable characters other than IVTFF delimiters are atomic glyphs.
    The @nnn; encoding is one glyph. Never silently resolve unreadable signs.
    """
    runs: list[list[tuple[str, ...]]] = []
    run: list[tuple[str, ...]] = []
    word: list[str] = []
    invalid = False

    def end_word() -> None:
        nonlocal invalid
        if word and not invalid:
            run.append(tuple(word))
        word.clear()
        invalid = False

    def end_run() -> None:
        end_word()
        if len(run) >= 4:
            runs.append(list(run))
        run.clear()

    pos = 0
    while pos < len(body):
        char = body[pos]
        if char.isspace():
            pos += 1
        elif char == "<":
            end_run()
            close = body.find(">", pos)
            if close < 0:
                raise ValueError("Unclosed GC markup")
            marker = body[pos:close + 1]
            if marker not in {"<%>", "<$>", "<->", "<~>"} and not (
                marker.startswith("<!") or re.fullmatch(r"<@[A-Z]=[A-Za-z0-9@]>", marker)
            ):
                raise ValueError(f"Unsupported GC markup {marker!r}")
            pos = close + 1
        elif char == ".":
            if invalid or not word:
                end_run()
            else:
                end_word()
            pos += 1
        elif char == ",":
            end_run()
            pos += 1
        elif char == "?":
            # The entire uncertain token is excluded, including any suffix.
            word.clear()
            invalid = True
            if run:
                if len(run) >= 4:
                    runs.append(list(run))
                run.clear()
            pos += 1
        elif char == "@":
            match = HIGH.match(body, pos)
            if match is None:
                raise ValueError("Invalid GC high-ASCII glyph encoding")
            word.append(match.group())
            pos = match.end()
        elif char in "[]{}>=O":
            raise ValueError(f"Unsupported GC syntax {char!r}")
        elif char.isascii() and char.isprintable():
            word.append(char)
            pos += 1
        else:
            raise ValueError(f"Non-ASCII GC glyph {char!r}")
    end_run()
    return runs


def load_gc(path: Path, split: dict[str, str]) -> tuple[list[dict], dict, set[str]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "#=IVTFF v101 2.0 M 6":
        raise ValueError("Expected official GC IVTFF 2.0 header")
    groups = []
    counts = Counter()
    ids: set[str] = set()
    current = None
    metadata: dict[str, str] = {}
    active_hand = "?"
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        page = PAGE.fullmatch(line)
        if page:
            leaf = re.fullmatch(r"(f\d+)[rv]\d*", page[1])
            current = page[1] if leaf is not None and split.get(leaf[1]) in {"train", "validation"} else None
            metadata = dict(re.findall(r"\$([A-Z])=([A-Za-z0-9@])", page[2] or ""))
            active_hand = metadata.get("H", "?")
            if current:
                counts[f"{split[leaf[1]]}_pages"] += 1
            continue
        locus = LOCUS.fullmatch(line)
        if locus is None:
            raise ValueError(f"Malformed GC line {line[:50]!r}")
        if current is None:
            continue  # Reserved physical leaves are never parsed or scored.
        if locus[1] != current:
            raise ValueError("GC locus/page mismatch")
        if not re.fullmatch(r"[@+*=&~/!][PLCR][a-z0-9](?:;[A-Za-z0-9])?", locus[3]):
            raise ValueError("Unsupported GC locus header")
        kind = re.search(r"[PLCR][a-z0-9]", locus[3])
        if kind is None or kind.group() != "P0" or locus[3].startswith("!"):
            continue
        locus_id = f"{current}.{locus[2]}"
        if locus_id in ids:
            raise ValueError("Duplicate GC locus")
        ids.add(locus_id)
        counts[f"{split[re.match(r'f\d+', current).group()]}_p0_loci"] += 1
        hand_tags = re.findall(r"<@H=([A-Za-z0-9@])>", locus[4])
        if hand_tags:
            if len(hand_tags) != 1 or not re.match(r"^(?:<%>)?<@H=[A-Za-z0-9@]>", locus[4]):
                raise ValueError("Unsupported mid-locus GC hand change")
            active_hand = hand_tags[0]
        context = (metadata.get("I", "?"), active_hand)
        for words in parse_gc_body(locus[4]):
            groups.append({"leaf": re.match(r"f\d+", current).group(), "locus": locus_id,
                           "context": context, "words": words})
    counts["groups"] = len(groups)
    counts["pairs"] = sum(len(group["words"]) - 1 for group in groups)
    counts["leaves"] = len({group["leaf"] for group in groups})
    return groups, dict(counts), ids


def zl_symbols(word: str) -> tuple[str, ...]:
    order = sorted(BASIC, key=lambda item: (-len(item), item))
    result = []
    pos = 0
    while pos < len(word):
        found = next((unit for unit in order if word.startswith(unit, pos)), word[pos])
        result.append(found)
        pos += len(found)
    if "".join(result) != word:
        raise ValueError("ZL tokenization not reversible")
    return tuple(result)


def load_zl(path: Path, matched_ids: set[str]) -> tuple[list[dict], dict]:
    groups = []
    counts = Counter()
    for raw in path.read_text().splitlines():
        page = json.loads(raw)
        counts["pages"] += 1
        meta = page["metadata"]["page_variables"]
        for locus in page["loci"]:
            if locus["locus_type"] != "P0" or locus["locus_id"] not in matched_ids:
                continue
            counts["matched_p0_loci"] += 1
            context = (meta.get("I", "?"), locus["text_tags"].get("H", meta.get("H", "?")))
            run = []
            for candidate in locus["text"].split():
                if ZL_WORD.fullmatch(candidate):
                    run.append(zl_symbols(candidate))
                else:
                    if len(run) >= 4:
                        groups.append({"leaf": page["leaf_id"], "locus": locus["locus_id"],
                                       "context": context, "words": run})
                    run = []
            if len(run) >= 4:
                groups.append({"leaf": page["leaf_id"], "locus": locus["locus_id"],
                               "context": context, "words": run})
    counts["groups"] = len(groups)
    counts["pairs"] = sum(len(group["words"]) - 1 for group in groups)
    counts["leaves"] = len({group["leaf"] for group in groups})
    return groups, dict(counts)


def events(words: list[tuple[str, ...]]):
    for index, (previous, following) in enumerate(zip(words, words[1:])):
        role = "first" if index == 0 else "last" if index == len(words) - 2 else "middle"
        last = previous[-1]
        yield role, {
            "last": last,
            "first": previous[0],
            "length": min(len(previous), 8),
            "penult": previous[-2] if len(previous) > 1 else "<single>",
            "identity": previous,
            "two_back": (last, words[index - 1][-1] if index else "<start>"),
        }, following[0]


def fit(groups: list[dict]) -> dict:
    alphabet = sorted({glyph for group in groups for word in group["words"] for glyph in word})
    if not alphabet:
        raise ValueError("Empty train alphabet")
    global_counts = defaultdict(Counter)
    global_n = Counter()
    context_counts = defaultdict(Counter)
    context_n = Counter()
    feature_counts = defaultdict(Counter)
    feature_n = Counter()
    for group in groups:
        context = tuple(group["context"])
        for role, features, target in events(group["words"]):
            global_counts[role][target] += 1
            global_n[role] += 1
            ckey = (context, role)
            context_counts[ckey][target] += 1
            context_n[ckey] += 1
            for name, value in features.items():
                fkey = (context, role, name, value)
                feature_counts[fkey][target] += 1
                feature_n[fkey] += 1
    return {"alphabet": set(alphabet), "n_symbols": len(alphabet) + 1,
            "global_counts": global_counts, "global_n": global_n,
            "context_counts": context_counts, "context_n": context_n,
            "feature_counts": feature_counts, "feature_n": feature_n}


def score(groups: list[dict], model: dict) -> dict[str, dict]:
    output = defaultdict(lambda: {"pairs": 0, "bits": {name: 0.0 for name in FEATURES}})
    for group in groups:
        leaf = group["leaf"]
        context = tuple(group["context"])
        for role, features, raw_target in events(group["words"]):
            target = raw_target if raw_target in model["alphabet"] else "<unk>"
            glob = (model["global_counts"][role][target] + 1) / (
                model["global_n"][role] + model["n_symbols"])
            ckey = (context, role)
            base = (model["context_counts"][ckey][target] + 50 * glob) / (
                model["context_n"][ckey] + 50)
            last_key = (context, role, "last", features["last"])
            last = (model["feature_counts"][last_key][target] + 20 * base) / (
                model["feature_n"][last_key] + 20)
            output[leaf]["pairs"] += 1
            output[leaf]["bits"]["last"] += math.log2(last / base)
            for name in ("first", "length", "penult"):
                key = (context, role, name, features[name])
                conditional = (model["feature_counts"][key][target] + 20 * base) / (
                    model["feature_n"][key] + 20)
                output[leaf]["bits"][name] += math.log2(conditional / base)
            for name in ("identity", "two_back"):
                key = (context, role, name, features[name])
                conditional = (model["feature_counts"][key][target] + 20 * last) / (
                    model["feature_n"][key] + 20)
                output[leaf]["bits"][name] += math.log2(conditional / last)
    return dict(sorted(output.items()))


def shuffle(groups: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for group in groups:
        interior = list(group["words"][1:-1])
        mid = len(interior) // 2
        a, b = interior[:mid], interior[mid:]
        rng.shuffle(a)
        rng.shuffle(b)
        output.append({**group, "words": [group["words"][0], *a, *b, group["words"][-1]]})
    return output


def average(rows: dict[str, dict], name: str) -> float:
    return sum(row["bits"][name] for row in rows.values()) / sum(row["pairs"] for row in rows.values())


def interval(real: dict[str, dict], null_by_leaf: dict[str, float] | None,
             feature: str, other: str | None, seed: int) -> list[float]:
    leaves = sorted(real)
    rng = random.Random(seed)
    draws = []
    for _ in range(N_BOOT):
        sample = rng.choices(leaves, k=len(leaves))
        pairs = sum(real[leaf]["pairs"] for leaf in sample)
        if null_by_leaf is not None:
            numerator = sum(real[leaf]["bits"][feature] - null_by_leaf[leaf] for leaf in sample)
        else:
            numerator = sum(real[leaf]["bits"][feature] -
                            (real[leaf]["bits"][other] if other else 0.0) for leaf in sample)
        draws.append(numerator / pairs)
    draws.sort()
    return [draws[49], draws[1949]]


def evaluate(groups_train: list[dict], groups_valid: list[dict], seed: int) -> dict:
    model = fit(groups_train)
    real = score(groups_valid, model)
    null_scores = []
    null_leaf = {leaf: [] for leaf in real}
    for rep in range(N_PERM):
        shuffled = score(shuffle(groups_valid, seed + rep), model)
        null_scores.append(average(shuffled, "last"))
        for leaf, row in shuffled.items():
            null_leaf[leaf].append(row["bits"]["last"])
    null_mean = sum(null_scores) / N_PERM
    leaf_null_mean = {leaf: sum(values) / N_PERM for leaf, values in null_leaf.items()}
    real_mean = {name: average(real, name) for name in FEATURES}
    contrasts = {"last_minus_null": real_mean["last"] - null_mean,
                 "last_minus_first": real_mean["last"] - real_mean["first"],
                 "last_minus_length": real_mean["last"] - real_mean["length"],
                 "last_minus_penult": real_mean["last"] - real_mean["penult"]}
    ci = {"last_minus_null": interval(real, leaf_null_mean, "last", None, seed + 1000)}
    for name in ("first", "length", "penult"):
        ci[f"last_minus_{name}"] = interval(real, None, "last", name, seed + 1000)
    for name in ("identity", "two_back"):
        ci[f"{name}_over_last"] = interval(real, None, name, None, seed + 1000)
    return {"alphabet_size_train_plus_unk": model["n_symbols"],
            "validation_unseen_target_pairs": sum(
                1 for group in groups_valid for _, _, target in events(group["words"])
                if target not in model["alphabet"]),
            "train_groups": len(groups_train), "train_pairs": sum(len(g["words"]) - 1 for g in groups_train),
            "validation_groups": len(groups_valid), "validation_pairs": sum(len(g["words"]) - 1 for g in groups_valid),
            "real_by_leaf": real, "real_mean": real_mean,
            "null_last_mean": null_mean, "null_last_95th": sorted(null_scores)[189],
            "null_last_by_leaf_mean": leaf_null_mean, "contrasts": contrasts,
            "bootstrap95": ci}


def run(root: Path) -> dict:
    started = time.monotonic()
    gc = root / "data/raw/v101/GC2a-n.txt"
    split_path = root / "data/manifests/zl3b_split.json"
    zl_train = root / "data/processed/zl3b/train.jsonl"
    zl_valid = root / "data/processed/zl3b/validation.jsonl"
    observed = [sha(path) for path in (gc, split_path, zl_train, zl_valid)]
    if observed != [GC_SHA, SPLIT_SHA, ZL_TRAIN_SHA, ZL_VALID_SHA]:
        raise ValueError("EXP-0039 pinned input drift")
    split = json.loads(split_path.read_text())["leaf_assignments"]
    all_gc, gc_diag, matched_ids = load_gc(gc, split)
    gc_train = [group for group in all_gc if split[group["leaf"]] == "train"]
    gc_valid = [group for group in all_gc if split[group["leaf"]] == "validation"]
    zl_train_groups, zl_train_diag = load_zl(zl_train, matched_ids)
    zl_valid_groups, zl_valid_diag = load_zl(zl_valid, matched_ids)
    if zl_train_diag["matched_p0_loci"] + zl_valid_diag["matched_p0_loci"] != len(matched_ids):
        raise ValueError("GC/ZL locus-set mismatch")
    views = {"gc_v101": evaluate(gc_train, gc_valid, 390039),
             "zl_basic_matched": evaluate(zl_train_groups, zl_valid_groups, 390139)}
    toy_train = [{"leaf": g["leaf"], "context": ("toy", "toy"),
                  "words": [tuple(w) for w in g["words"]]} for g in toy(390239, 100)]
    toy_valid = [{"leaf": g["leaf"], "context": ("toy", "toy"),
                  "words": [tuple(w) for w in g["words"]]} for g in toy(390339, 30)]
    toy_result = evaluate(toy_train, toy_valid, 390439)
    toy_gate = (toy_result["contrasts"]["last_minus_null"] >= 0.1 and
                toy_result["contrasts"]["last_minus_first"] >= 0.1)
    gc_view = views["gc_v101"]
    zl_view = views["zl_basic_matched"]
    coverage = gc_view["validation_pairs"] >= 1000 and len(gc_view["real_by_leaf"]) >= 8
    cross = toy_gate and coverage and all(
        row["contrasts"]["last_minus_null"] >= 0.02 and
        row["bootstrap95"]["last_minus_null"][0] > 0 and
        row["real_mean"]["last"] > row["null_last_95th"] for row in (gc_view, zl_view))
    specificity = toy_gate and coverage and all(
        row["contrasts"][f"last_minus_{name}"] >= 0.02 and
        row["bootstrap95"][f"last_minus_{name}"][0] > 0
        for row in (gc_view, zl_view) for name in ("first", "length", "penult"))
    if time.monotonic() - started > 180:
        raise TimeoutError("EXP-0039 CPU cap exceeded")
    return {"experiment": EXPERIMENT, "status": "complete",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "assessed" if toy_gate else "invalid_toy_control",
            "cross_transcription_signal": cross, "terminal_specific": specificity,
            "coverage": coverage, "toy": {key: toy_result["contrasts"][key]
                                      for key in ("last_minus_null", "last_minus_first")},
            "inputs_sha256": {"gc": GC_SHA, "split": SPLIT_SHA,
                              "zl_train": ZL_TRAIN_SHA, "zl_validation": ZL_VALID_SHA},
            "source_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "wall_seconds": time.monotonic() - started,
            "gc_diagnostics": gc_diag, "zl_train_diagnostics": zl_train_diag,
            "zl_validation_diagnostics": zl_valid_diag, "views": views,
            "limits": "Adaptive exposed-ZL-leaf study; independent transcriber, shared manuscript and metadata; no historical mechanism."}


def main() -> None:
    root = Path.cwd()
    report = run(root)
    path = root / "results/EXP-0039/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      {"views", "gc_diagnostics", "zl_train_diagnostics", "zl_validation_diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
