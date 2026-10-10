"""Train-leaf cross-fitting of an observable terminal-state transport hypothesis.

This is a conditional emission assay, not a complete ciphertext likelihood or
a plaintext decoder. No hyperparameter is selected from cross-line scores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .boundary_cross_transcription import LOCUS, PAGE, zl_symbols
from .data import leaf_id

EXPERIMENT = "PARAGRAPH-CARRY-001"
FOLD_SALT = "paragraph-carry-training-leaf-v1-20261010"
N_FOLDS, N_BOOT, N_PERM = 5, 1000, 100
UNKNOWN = "<unknown>"
UNCERTAIN = {"uncertain_space", "alternative", "rare_eva", "unreadable", "unknown_span",
             "drawing_interruption", "text_tag"}
MODES = ("paragraph_carry", "line_reset", "always_carry", "copy_mutate", "iid", "paragraph_common_cause")
CONTROL_ALLOCATION = (("iid_terminal", 97601), ("iid_terminal", 97609),
                      ("markov_terminal", 97611), ("markov_terminal", 97619))
EXCLUSIONS = ("excluded_layout", "excluded_locator", "excluded_paragraph_markers",
              "excluded_annotation", "excluded_hand", "excluded_word_shape")


@dataclass(frozen=True)
class Line:
    leaf: str
    page: str
    number: int
    context: tuple[str, str, str]
    words: tuple[tuple[str, ...], ...]
    start: bool
    end: bool
    depth: int = 0
    locator: str = "@"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extraction_counts(counts: Counter, lines: list[Line]) -> dict:
    counts["retained_lines"] = len(lines)
    for name in EXCLUSIONS:
        counts.setdefault(name, 0)
    if counts["loci_seen"] != len(lines) + sum(counts[name] for name in EXCLUSIONS):
        raise ValueError("Incomplete locus accounting")
    return dict(counts)


def load_zl_train(path: Path) -> tuple[list[Line], dict]:
    """Do not accept a validation/test file even if supplied through the CLI."""
    lines, counts = [], Counter()
    for raw in path.read_text().splitlines():
        page = json.loads(raw)
        if page["split"] != "train":
            raise ValueError("Only train-assigned manuscript records are allowed")
        meta = page["metadata"]["page_variables"]
        depth = -1
        for locus in page["loci"]:
            counts["loci_seen"] += 1
            counts["locus_type_seen:" + locus["locus_type"]] += 1
            counts["locator_seen:" + locus["locator"]] += 1
            annotations = {a["kind"] for a in locus["annotations"]}
            if locus["locus_type"].startswith("P"):
                depth = 0 if "paragraph_start" in annotations else depth + 1 if depth >= 0 else -1
            if locus["locus_type"] != "P0":
                counts["excluded_layout"] += 1
                continue
            if locus["locator"] not in ("@", "+", "*"):
                counts["excluded_locator"] += 1
                continue
            starts = [a for a in locus["annotations"] if a["kind"] == "paragraph_start"]
            ends = [a for a in locus["annotations"] if a["kind"] == "paragraph_end"]
            if (len(starts) > 1 or len(ends) > 1 or any(a["start"] != 0 for a in starts)
                    or any(a["end"] != len(locus["text"]) for a in ends)):
                counts["excluded_paragraph_markers"] += 1
                continue
            hand = locus["text_tags"].get("H", meta.get("H", "?"))
            if hand not in "12345" or len(hand) != 1:
                counts["excluded_hand"] += 1
                continue
            if annotations & UNCERTAIN:
                counts["excluded_annotation"] += 1
                continue
            words = locus["text"].split()
            if len(words) < 2 or any(re.fullmatch(r"[a-z']+", word) is None for word in words):
                counts["excluded_word_shape"] += 1
                continue
            lines.append(Line(page["leaf_id"], page["page_id"], int(locus["locus_id"].split(".")[1]),
                              (meta.get("I", "?"), meta.get("L", "?"), hand),
                              tuple(zl_symbols(word) for word in words),
                              "paragraph_start" in annotations, "paragraph_end" in annotations,
                              depth, locus["locator"]))
    return lines, extraction_counts(counts, lines)


def load_gc_train(path: Path, leaf_assignment: dict[str, str]) -> tuple[list[Line], dict]:
    """Skip every nontraining page's body before interpreting glyphs or tags."""
    rows, counts = [], Counter()
    current, metadata, hand, depth = None, {}, "?", -1
    raw_lines = path.read_text(encoding="ascii").splitlines()
    if raw_lines[0] != "#=IVTFF v101 2.0 M 6":
        raise ValueError("Wrong GC source header")
    for raw in raw_lines[1:]:
        if not raw or raw.startswith("#"):
            continue
        page = PAGE.fullmatch(raw)
        if page:
            leaf = leaf_id(page[1])
            current = page[1] if leaf_assignment.get(leaf) == "train" else None
            if current:
                metadata = dict(re.findall(r"\$([A-Z])=([A-Za-z0-9@])", page[2] or ""))
                hand = metadata.get("H", "?")
                depth = -1
            continue
        if current is None:
            continue
        locus = LOCUS.fullmatch(raw)
        if locus is None or locus[1] != current:
            raise ValueError("Malformed selected GC locus")
        changed = re.findall(r"<@H=([A-Za-z0-9@])>", locus[4])
        if changed:
            if len(changed) != 1 or re.match(r"^(?:<%>)?<@H=[A-Za-z0-9@]>", locus[4]) is None:
                raise ValueError("Unsupported hand change")
            hand = changed[0]
        counts["loci_seen"] += 1
        identifier = re.fullmatch(r"([@+*=&~/!])([A-Z][a-z0-9])(?:;[A-Za-z0-9])?", locus[3])
        if identifier is None:
            raise ValueError("Unsupported selected GC locus identifier")
        locator, layout = identifier.groups()
        counts["locus_type_seen:" + layout] += 1
        counts["locator_seen:" + locator] += 1
        if layout.startswith("P"):
            depth = 0 if "<%>" in locus[4] else depth + 1 if depth >= 0 else -1
        if layout != "P0":
            counts["excluded_layout"] += 1
            continue
        if locator not in ("@", "+", "*"):
            counts["excluded_locator"] += 1
            continue
        body = locus[4]
        if len(hand) != 1 or hand not in "12345":
            counts["excluded_hand"] += 1
            continue
        # Editorial comments are not glyphs, uncertainty or hand metadata.
        marker_view = re.sub(r"<![^>]*>", "", body).strip()
        if changed or any(mark in marker_view for mark in (",", "?", "[", "]", "{", "}", "<->", "<~>", "@")):
            counts["excluded_annotation"] += 1
            continue
        start, end = "<%>" in body, "<$>" in body
        if ((start and (body.count("<%>") != 1 or not marker_view.startswith("<%>")))
                or (end and (body.count("<$>") != 1 or not marker_view.endswith("<$>")))):
            counts["excluded_paragraph_markers"] += 1
            continue
        # Editorial comments never become glyphs. Other markup must be one of
        # the already recognized paragraph markers, otherwise fail closed.
        for mark in re.findall(r"<[^>]+>", body):
            if mark not in ("<%>", "<$>") and not mark.startswith("<!"):
                raise ValueError("Unsupported selected GC markup")
        plain = re.sub(r"<[^>]+>", "", body).strip()
        words = plain.split(".")
        if len(words) < 2 or any(not word or any(c.isspace() or c in ">=O" for c in word) for word in words):
            counts["excluded_word_shape"] += 1
            continue
        if any(not c.isascii() or not c.isprintable() for word in words for c in word):
            raise ValueError("Invalid selected GC symbol")
        rows.append(Line(leaf_id(current), current, int(locus[2]),
                         (metadata.get("I", "?"), metadata.get("L", "?"), hand),
                         tuple(tuple(word) for word in words), start, end, depth, locator))
    return rows, extraction_counts(counts, rows)


def matched_views(zl: list[Line], gc: list[Line]) -> tuple[dict[str, list[Line]], dict]:
    def indexed(rows):
        output = {(row.page, row.number): row for row in rows}
        if len(output) != len(rows):
            raise ValueError("Duplicate locus")
        return output
    a, b = indexed(zl), indexed(gc)
    keys = sorted(key for key in a.keys() & b.keys()
                  if (a[key].context, a[key].start, a[key].end, a[key].depth, a[key].locator) ==
                  (b[key].context, b[key].start, b[key].end, b[key].depth, b[key].locator))
    return {"zl": [a[k] for k in keys], "gc": [b[k] for k in keys]}, {
        "zl_clean_lines": len(a), "gc_clean_lines": len(b), "matched_lines": len(keys),
        "zl_only_clean_lines": len(a.keys() - b.keys()), "gc_only_clean_lines": len(b.keys() - a.keys()),
        "excluded_metadata_conflicts": len(a.keys() & b.keys()) - len(keys)}


def folds(lines: list[Line]) -> dict[str, int]:
    leaves = sorted({line.leaf for line in lines},
                    key=lambda leaf: hashlib.sha256(f"{FOLD_SALT}:{leaf}".encode()).hexdigest())
    if len(leaves) < N_FOLDS:
        raise ValueError("Too few physical leaves")
    return {leaf: rank % N_FOLDS for rank, leaf in enumerate(leaves)}


def junctions(lines: list[Line]) -> list[dict]:
    lookup = {(line.page, line.number): line for line in lines}
    if len(lookup) != len(lines):
        raise ValueError("Duplicate locus")
    output = []
    for previous in sorted(lines, key=lambda row: (row.page, row.number)):
        following = lookup.get((previous.page, previous.number + 1))
        if (following is None or following.locator not in ("+", "*")
                or previous.context != following.context or previous.leaf != following.leaf):
            continue
        if previous.end != following.start:
            continue  # Disagreeing paragraph markers are not resolved by us.
        output.append({"leaf": previous.leaf, "page": previous.page, "left": previous.number,
                       "right": following.number, "context": following.context,
                       "kind": "paragraph" if following.start else "continuation",
                       "last": previous.words[-1][-1], "target": following.words[0][0],
                       "stratum": (min(following.depth, 3), min(len(previous.words) // 4, 3),
                                   min(len(following.words) // 4, 3))})
    return output


def fit(lines: list[Line]) -> dict:
    alphabet = tuple(sorted({word[0] for row in lines for word in row.words})) + (UNKNOWN,)
    if not lines or len(alphabet) > 513:
        raise ValueError("Empty or excessively large training alphabet")
    counts = defaultdict(Counter)
    for row in lines:
        role = "paragraph" if row.start else "continuation"
        first = row.words[0][0]
        counts[("start", role)][first] += 1
        counts[("start", role, row.context)][first] += 1
        for left, right in zip(row.words, row.words[1:]):
            initial, last = right[0], left[-1]
            counts[("mid",)][initial] += 1
            counts[("mid", row.context)][initial] += 1
            counts[("edge", row.context, last)][initial] += 1
    return {"alphabet": alphabet, "counts": counts, "cache": {}}


def distributions(model: dict, context: tuple[str, ...], kind: str, last: str) -> tuple[dict, dict]:
    if kind not in ("continuation", "paragraph"):
        raise ValueError("Unknown junction type")
    key = context, kind, last
    if key not in model["cache"]:
        alpha, counts = model["alphabet"], model["counts"]
        def global_p(key):
            c = counts[key]
            return {a: (c[a] + 1) / (sum(c.values()) + len(alpha)) for a in alpha}
        def shrunk(key, parent, concentration):
            c = counts[key]
            n = sum(c.values()) + concentration
            return {a: (c[a] + concentration * parent[a]) / n for a in alpha}
        mid = shrunk(("mid", context), global_p(("mid",)), 50)
        edge = shrunk(("edge", context, last), mid, 20)
        baseline = shrunk(("start", kind, context), global_p(("start", kind)), 50)
        raw = {a: baseline[a] * edge[a] / mid[a] for a in alpha}
        normalizer = math.fsum(raw.values())
        carry = {a: raw[a] / normalizer for a in alpha}
        model["cache"][key] = baseline, carry
    return model["cache"][key]


def gain(model: dict, event: dict) -> float:
    base, carry = distributions(model, tuple(event["context"]), event["kind"], event["last"])
    target = event["target"] if event["target"] in base else UNKNOWN
    return math.log2(carry[target] / base[target])


def cross_fit(lines: list[Line]) -> tuple[list[dict], dict]:
    assignment = folds(lines)
    events = junctions(lines)
    ledger, manifests = [], []
    for fold in range(N_FOLDS):
        training = [row for row in lines if assignment[row.leaf] != fold]
        testing = [row for row in events if assignment[row["leaf"]] == fold]
        model = fit(training)
        counts_manifest = sorted((repr(key), sorted(value.items())) for key, value in model["counts"].items())
        manifests.append({"fold": fold, "training_leaves": sorted({x.leaf for x in training}),
                          "test_leaves": sorted(leaf for leaf, f in assignment.items() if f == fold),
                          "training_lines": len(training), "alphabet": list(model["alphabet"]),
                          "count_digest": json_sha(counts_manifest)})
        for event in testing:
            ledger.append({**event, "fold": fold, "gain": gain(model, event)})
    ledger.sort(key=lambda x: (x["page"], x["left"]))
    return ledger, {"leaf_fold": assignment, "models": manifests, "lines": len(lines)}


def summarize(ledger: list[dict], seed: int, n_boot: int = N_BOOT) -> dict:
    leaves = sorted({row["leaf"] for row in ledger})
    stats = {leaf: {kind: [0, 0.] for kind in ("continuation", "paragraph")} for leaf in leaves}
    for row in ledger:
        stats[row["leaf"]][row["kind"]][0] += 1
        stats[row["leaf"]][row["kind"]][1] += row["gain"]
    def mean(selected, kind):
        n = sum(stats[leaf][kind][0] for leaf in selected)
        return sum(stats[leaf][kind][1] for leaf in selected) / n if n else None
    observed = {kind: mean(leaves, kind) for kind in ("continuation", "paragraph")}
    rng = random.Random(seed)
    samples = {"continuation": [], "paragraph": [], "difference": []}
    for _ in range(n_boot):
        draw = [rng.choice(leaves) for _ in leaves]
        a, b = mean(draw, "continuation"), mean(draw, "paragraph")
        if a is not None and b is not None:
            samples["continuation"].append(a)
            samples["paragraph"].append(b)
            samples["difference"].append(a - b)
    def interval(values):
        if not values:
            return None
        values.sort()
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
    intervals = {key: interval(values) for key, values in samples.items()}
    counts = {kind: sum(stats[leaf][kind][0] for leaf in leaves) for kind in observed}
    enough = counts["continuation"] >= 100 and counts["paragraph"] >= 25 and len(leaves) >= 5
    a, b = observed["continuation"], observed["paragraph"]
    calibrated = enough and all(intervals.values())
    paragraph = bool(calibrated and a >= .02 and intervals["continuation"][0] > 0
                     and a - b >= .05 and intervals["difference"][0] > 0
                     and b <= 0 and intervals["paragraph"][1] < .02)
    reset = bool(calibrated and a <= 0 and intervals["continuation"][1] < .02)
    unbroken = bool(calibrated and a >= .02 and b >= .02
                    and intervals["continuation"][0] > 0 and intervals["paragraph"][0] > 0)
    verdict = "paragraph_carry" if paragraph else "line_reset_compatible" if reset else (
        "unbroken_carry" if unbroken else "undetermined")
    return {"counts": counts, "leaf_count": len(leaves), "by_leaf": stats, "gain_bits": observed,
            "difference_bits": a - b if a is not None and b is not None else None,
            "bootstrap_intervals": intervals, "valid_bootstraps": len(samples["difference"]),
            "support_sufficient": enough, "classification": verdict}


def permutation_null(lines: list[Line], ledger: list[dict], seed: int, n_perm: int = N_PERM) -> dict:
    """Permute only source terminal units within held-out leaf/context/kind."""
    assignment, events = folds(lines), junctions(lines)
    models = {f: fit([row for row in lines if assignment[row.leaf] != f]) for f in range(N_FOLDS)}
    groups = defaultdict(list)
    for index, event in enumerate(events):
        groups[(event["leaf"], tuple(event["context"]), event["kind"], tuple(event["stratum"]))].append(index)
    rng = random.Random(seed)
    contrasts = []
    permutable = sum(len(indices) for indices in groups.values() if len(indices) > 1)
    movable = sum(len(indices) for indices in groups.values()
                  if len({events[i]["last"] for i in indices}) > 1)
    for _ in range(n_perm):
        changed = [dict(event) for event in events]
        for indices in groups.values():
            previous = [events[i]["last"] for i in indices]
            rng.shuffle(previous)
            for i, last in zip(indices, previous):
                changed[i]["last"] = last
        sums, ns = Counter(), Counter()
        for event in changed:
            sums[event["kind"]] += gain(models[assignment[event["leaf"]]], event)
            ns[event["kind"]] += 1
        contrasts.append(sums["continuation"] / ns["continuation"] - sums["paragraph"] / ns["paragraph"]
                         if ns["continuation"] and ns["paragraph"] else None)
    real = summarize(ledger, 0, n_boot=0)["difference_bits"]
    finite = [value for value in contrasts if value is not None]
    p = (1 + sum(value >= real for value in finite)) / (1 + len(finite)) if real is not None and finite else None
    return {"n_permutations": n_perm, "events": len(events), "permutable_events": permutable,
            "movable_events": movable, "contrasts": contrasts,
            "one_sided_p": p}


def controls(family: str, seed: int, mode: str, n_leaves: int = 40, n_lines: int = 24) -> list[Line]:
    """New keyed, disjoint observable source families; never Voynich plaintext.

    Family differences are terminal-generation rules, not natural languages.
    The seed specifies one document-wide emission key, never a leaf-specific key.
    """
    if family not in ("iid_terminal", "markov_terminal") or mode not in MODES:
        raise ValueError("Unknown control")
    rng = random.Random(seed)
    alphabet = tuple("abcdefgh")
    permutation = list(alphabet)
    rng.shuffle(permutation)
    mapping = dict(zip(alphabet, permutation))
    output = []
    for leaf_index in range(n_leaves):
        previous, source_last = None, rng.choice(alphabet)
        page = f"control{leaf_index}r"
        for line_index in range(n_lines):
            start, end = line_index % 4 == 0, line_index % 4 == 3
            if start:
                style = rng.choice(alphabet)
            words = []
            for position in range(6):
                reset = start and position == 0 or mode == "line_reset" and position == 0
                if mode == "always_carry":
                    reset = previous is None
                if mode == "copy_mutate" and previous is not None and not reset and rng.random() < .8:
                    word = list(previous)
                    word[1] = rng.choice(alphabet)
                else:
                    initial = rng.choice(alphabet)
                    if mode in ("paragraph_carry", "line_reset", "always_carry") and previous is not None and not reset:
                        if rng.random() < .9:
                            initial = mapping[previous[-1]]
                    terminal = rng.choice(alphabet)
                    if family == "markov_terminal" and rng.random() < .75:
                        terminal = source_last
                    source_last = terminal
                    if mode == "paragraph_common_cause":
                        # No previous-terminal-driven initial update: a shared
                        # paragraph cause independently influences both edges.
                        if rng.random() < .8:
                            initial = mapping[style]
                        if rng.random() < .8:
                            terminal = style
                    word = [initial, rng.choice(alphabet), rng.choice(alphabet), terminal]
                previous = tuple(word)
                words.append(previous)
            output.append(Line(f"control{leaf_index}", page, line_index + 1, ("H", "A", "1"),
                               tuple(words), start, end, line_index % 4,
                               "@" if line_index == 0 else "*" if start else "+"))
    return output
