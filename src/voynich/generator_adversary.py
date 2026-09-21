"""EXP-0019: generator-adversary search for HYP-005 (R3).

Evolutionary search may use only the optimizable surface metrics.
Held-out diagnostics are scored once on the frozen winner.
Not a decipherment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PUA_RE = re.compile(r"[\ue000-\uf8ff]")
WORD_RE = re.compile(r"[a-z]+")

SEED = 4019
POP_SIZE = 24
N_GENERATIONS = 20
N_ELITE = 4

OPT_METRICS = (
    "zipf_slope_top500",
    "adjacent_repeat_rate",
    "char_h2_bits",
    "top1_word_share",
)
HELDOUT_METRICS = (
    "line_initial_type_share",
    "ab_top50_jaccard",
    "word_h2_bits",
)
FLOORS = {
    "zipf_slope_top500": 0.15,
    "adjacent_repeat_rate": 0.005,
    "char_h2_bits": 0.25,
    "top1_word_share": 0.010,
    "line_initial_type_share": 0.05,
    "ab_top50_jaccard": 0.15,
    "word_h2_bits": 0.25,
}
DEFAULT_GENOME = {
    "window": 20,
    "p_cite": 0.72,
    "p_mutate_given_cite": 0.55,
    "p_splice_given_cite": 0.25,
}


@dataclass(frozen=True)
class Genome:
    window: int
    p_cite: float
    p_mutate_given_cite: float
    p_splice_given_cite: float

    def repaired(self) -> Genome:
        window = int(max(5, min(40, self.window)))
        p_cite = float(max(0.30, min(0.95, self.p_cite)))
        p_m = float(max(0.05, min(0.85, self.p_mutate_given_cite)))
        p_s = float(max(0.05, min(0.85, self.p_splice_given_cite)))
        if p_m + p_s > 0.95:
            scale = 0.95 / (p_m + p_s)
            p_m *= scale
            p_s *= scale
        return Genome(window, p_cite, p_m, p_s)

    def key(self) -> tuple:
        g = self.repaired()
        return (
            g.window,
            round(g.p_cite, 6),
            round(g.p_mutate_given_cite, 6),
            round(g.p_splice_given_cite, 6),
        )

    def as_dict(self) -> dict[str, float | int]:
        g = self.repaired()
        return {
            "window": g.window,
            "p_cite": g.p_cite,
            "p_mutate_given_cite": g.p_mutate_given_cite,
            "p_splice_given_cite": g.p_splice_given_cite,
        }


def clean_text(text: str) -> str:
    text = PUA_RE.sub("", text).lower()
    return "".join(ch if ch in "abcdefghijklmnopqrstuvwxyz \n" else " " for ch in text)


def load_pages(path: Path) -> list[dict]:
    pages = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        pages.append(
            {
                "page_id": page["page_id"],
                "L": page.get("metadata", {}).get("page_variables", {}).get("L", "?"),
                "text": clean_text(page["text"]),
            }
        )
    return pages


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def lines_of(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def zipf_slope(words: list[str], top: int = 500) -> float:
    counts = Counter(words)
    if len(counts) < 2:
        return float("nan")
    freqs = np.array([c for _, c in counts.most_common(min(top, len(counts)))], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(freqs)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def adjacent_repeat_rate(words: list[str]) -> float:
    if len(words) < 2:
        return float("nan")
    return sum(a == b for a, b in zip(words, words[1:])) / (len(words) - 1)


def char_h2_bits(text: str) -> float:
    stream = text.replace("\n", " ")
    stream = re.sub(r" +", " ", stream)
    if len(stream) < 2:
        return float("nan")
    pairs = Counter(stream[i : i + 2] for i in range(len(stream) - 1))
    singles = Counter(stream[:-1])
    total = sum(pairs.values())
    h = 0.0
    for pair, c in pairs.items():
        p_xy = c / total
        p_x = singles[pair[0]] / total
        h -= p_xy * math.log2(p_xy / p_x)
    return h


def word_h2_bits(words: list[str]) -> float:
    if len(words) < 2:
        return float("nan")
    pairs = Counter(zip(words[:-1], words[1:]))
    singles = Counter(words[:-1])
    total = sum(pairs.values())
    h = 0.0
    for (a, _b), c in pairs.items():
        p_xy = c / total
        p_x = singles[a] / total
        h -= p_xy * math.log2(p_xy / p_x)
    return h


def ab_top50_jaccard(pages: list[dict]) -> float:
    a_words, b_words = [], []
    for page in pages:
        if page["L"] == "A":
            a_words.extend(words_of(page["text"]))
        elif page["L"] == "B":
            b_words.extend(words_of(page["text"]))
    if not a_words or not b_words:
        return float("nan")
    a_top = {w for w, _ in Counter(a_words).most_common(50)}
    b_top = {w for w, _ in Counter(b_words).most_common(50)}
    union = a_top | b_top
    if not union:
        return float("nan")
    return len(a_top & b_top) / len(union)


def top1_word_share(words: list[str]) -> float:
    if not words:
        return float("nan")
    return Counter(words).most_common(1)[0][1] / len(words)


def line_initial_type_share(pages: list[dict], top_n: int = 20) -> float:
    initials = []
    for page in pages:
        for line in lines_of(page["text"]):
            ws = words_of(line)
            if ws:
                initials.append(ws[0])
    if not initials:
        return float("nan")
    top = sum(c for _, c in Counter(initials).most_common(top_n))
    return top / len(initials)


def metrics_from_pages(pages: list[dict]) -> dict[str, Any]:
    words: list[str] = []
    for page in pages:
        words.extend(words_of(page["text"]))
    text = "\n".join(page["text"] for page in pages)
    counts = Counter(words)
    return {
        "n_pages": len(pages),
        "n_tokens": len(words),
        "n_types": len(counts),
        "zipf_slope_top500": zipf_slope(words),
        "adjacent_repeat_rate": adjacent_repeat_rate(words),
        "char_h2_bits": char_h2_bits(text),
        "ab_top50_jaccard": ab_top50_jaccard(pages),
        "top1_word_share": top1_word_share(words),
        "line_initial_type_share": line_initial_type_share(pages),
        "word_h2_bits": word_h2_bits(words),
        "top1_word": counts.most_common(1)[0][0] if counts else None,
        "hapax_frac_types_exploratory": (
            sum(1 for c in counts.values() if c == 1) / len(counts) if counts else float("nan")
        ),
        "mean_word_length_exploratory": float(np.mean([len(w) for w in words])) if words else float("nan"),
    }


def page_shapes(pages: list[dict]) -> list[dict]:
    shapes = []
    for page in pages:
        line_word_counts = [len(words_of(line)) for line in lines_of(page["text"])]
        shapes.append({"page_id": page["page_id"], "L": page["L"], "line_word_counts": line_word_counts})
    return shapes


def sample_from_counts(rng: np.random.Generator, counter: Counter):
    items = list(counter.keys())
    weights = np.array([counter[i] for i in items], dtype=float)
    weights /= weights.sum()
    return items[int(rng.choice(len(items), p=weights))]


def fit_char_unigram(pages: list[dict]) -> Counter:
    counts: Counter = Counter()
    for page in pages:
        for ch in page["text"].replace("\n", " "):
            if ch in "abcdefghijklmnopqrstuvwxyz ":
                counts[ch] += 1
    return counts


def fit_bigram_and_lengths(pages: list[dict]) -> tuple[dict[str, Counter], Counter, Counter]:
    starts: Counter = Counter()
    bigrams: dict[str, Counter] = {}
    lengths: Counter = Counter()
    for page in pages:
        for word in words_of(page["text"]):
            lengths[len(word)] += 1
            starts[word[0]] += 1
            for a, b in zip(word, word[1:]):
                bigrams.setdefault(a, Counter())[b] += 1
    return bigrams, starts, lengths


def fresh_word(rng: np.random.Generator, bigrams: dict[str, Counter], starts: Counter, lengths: Counter) -> str:
    length = sample_from_counts(rng, lengths)
    ch = sample_from_counts(rng, starts)
    out = [ch]
    for _ in range(length - 1):
        nxt = bigrams.get(out[-1])
        if not nxt:
            ch = sample_from_counts(rng, starts)
        else:
            ch = sample_from_counts(rng, nxt)
        out.append(ch)
    return "".join(out)


def mutate_word(rng: np.random.Generator, word: str, alphabet: list[str]) -> str:
    if not word:
        return word
    op = int(rng.integers(4))
    chars = list(word)
    if op == 0:
        i = int(rng.integers(len(chars)))
        chars[i] = alphabet[int(rng.integers(len(alphabet)))]
    elif op == 1 and len(chars) > 1:
        del chars[int(rng.integers(len(chars)))]
    elif op == 2:
        i = int(rng.integers(len(chars) + 1))
        chars.insert(i, alphabet[int(rng.integers(len(alphabet)))])
    else:
        i = int(rng.integers(len(chars)))
        chars.insert(i, chars[i])
    return "".join(chars) or word


def splice_words(rng: np.random.Generator, a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    cut_a = int(rng.integers(1, len(a) + 1))
    cut_b = int(rng.integers(0, len(b)))
    out = a[:cut_a] + b[cut_b:]
    return out or a


def generate_iid_char(rng: np.random.Generator, shapes: list[dict], unigram: Counter) -> list[dict]:
    pages = []
    alphabet = list(unigram.keys())
    probs = np.array([unigram[c] for c in alphabet], dtype=float)
    probs /= probs.sum()
    for shape in shapes:
        lines = []
        for n_words in shape["line_word_counts"]:
            words = []
            while len(words) < max(n_words, 1):
                chars = []
                while True:
                    ch = alphabet[int(rng.choice(len(alphabet), p=probs))]
                    if ch == " ":
                        if chars:
                            break
                        continue
                    chars.append(ch)
                    if len(chars) > 24:
                        break
                words.append("".join(chars))
            lines.append(" ".join(words[:n_words] if n_words else words[:1]))
        pages.append({"page_id": f"iid-{shape['page_id']}", "L": shape["L"], "text": "\n".join(lines)})
    return pages


def generate_selfcite(
    rng: np.random.Generator,
    shapes: list[dict],
    style_models: dict[str, dict],
    default_style_probs: dict[str, float],
    genome: Genome,
) -> list[dict]:
    g = genome.repaired()
    pages = []
    buffers = {"A": [], "B": []}
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    for shape in shapes:
        style = shape["L"] if shape["L"] in style_models else (
            "A" if rng.random() < default_style_probs["A"] else "B"
        )
        model = style_models[style]
        buf = buffers[style]
        lines = []
        for n_words in shape["line_word_counts"]:
            words = []
            for _ in range(max(n_words, 0)):
                if buf and rng.random() < g.p_cite:
                    recent = buf[-g.window :]
                    src = recent[int(rng.integers(len(recent)))]
                    roll = rng.random()
                    if roll < g.p_mutate_given_cite:
                        word = mutate_word(rng, src, alphabet)
                    elif roll < g.p_mutate_given_cite + g.p_splice_given_cite and len(recent) >= 2:
                        other = recent[int(rng.integers(len(recent)))]
                        word = splice_words(rng, src, other)
                    else:
                        word = src
                else:
                    word = fresh_word(rng, model["bigrams"], model["starts"], model["lengths"])
                words.append(word)
                buf.append(word)
            lines.append(" ".join(words))
        out_L = shape["L"] if shape["L"] in {"A", "B"} else style
        pages.append(
            {
                "page_id": f"selfcite-{shape['page_id']}",
                "L": out_L if shape["L"] in {"A", "B"} else style,
                "text": "\n".join(lines),
            }
        )
    return pages


def metric_verdicts(train_m: dict, val_m: dict, gen_m: dict, names: tuple[str, ...]) -> dict:
    verdicts = {}
    matches = 0
    excess = 0.0
    for name in names:
        r = val_m[name]
        t = train_m[name]
        g = gen_m[name]
        floor = FLOORS[name]
        tau = max(abs(t - r), floor)
        delta = abs(g - r)
        matched = delta <= tau
        matches += int(matched)
        excess += max(0.0, delta / tau - 1.0)
        verdicts[name] = {
            "validation": r,
            "train": t,
            "generator": g,
            "abs_delta_vs_validation": delta,
            "tolerance": tau,
            "verdict": "match" if matched else "separate",
        }
    return {
        "per_metric": verdicts,
        "n_match": matches,
        "n_metrics": len(names),
        "excess": excess,
    }


def genome_seed(base: int, genome: Genome) -> int:
    payload = json.dumps(genome.as_dict(), sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return (base ^ int(digest[:8], 16)) & 0x7FFFFFFF


def random_genome(rng: np.random.Generator) -> Genome:
    window = int(rng.integers(5, 41))
    p_cite = float(rng.uniform(0.30, 0.95))
    p_m = float(rng.uniform(0.05, 0.85))
    p_s = float(rng.uniform(0.05, 0.85))
    return Genome(window, p_cite, p_m, p_s).repaired()


def mutate_genome(rng: np.random.Generator, genome: Genome) -> Genome:
    g = genome.repaired()
    window = g.window + int(rng.choice([-2, -1, 0, 1, 2]))
    p_cite = g.p_cite + float(rng.normal(0, 0.08 * 0.65))
    p_m = g.p_mutate_given_cite + float(rng.normal(0, 0.08 * 0.80))
    p_s = g.p_splice_given_cite + float(rng.normal(0, 0.08 * 0.80))
    return Genome(window, p_cite, p_m, p_s).repaired()


def crossover(rng: np.random.Generator, a: Genome, b: Genome) -> Genome:
    a, b = a.repaired(), b.repaired()
    return Genome(
        window=a.window if rng.random() < 0.5 else b.window,
        p_cite=a.p_cite if rng.random() < 0.5 else b.p_cite,
        p_mutate_given_cite=a.p_mutate_given_cite if rng.random() < 0.5 else b.p_mutate_given_cite,
        p_splice_given_cite=a.p_splice_given_cite if rng.random() < 0.5 else b.p_splice_given_cite,
    ).repaired()


def evaluate_selfcite(
    genome: Genome,
    shapes: list[dict],
    style_models: dict[str, dict],
    default_style_probs: dict[str, float],
    train_m: dict,
    val_m: dict,
    base_seed: int,
) -> dict:
    rng = np.random.default_rng(genome_seed(base_seed, genome))
    pages = generate_selfcite(rng, shapes, style_models, default_style_probs, genome)
    gen_m = metrics_from_pages(pages)
    opt = metric_verdicts(train_m, val_m, gen_m, OPT_METRICS)
    return {
        "genome": genome.as_dict(),
        "metrics": {k: gen_m[k] for k in list(OPT_METRICS) + list(HELDOUT_METRICS)},
        "exploratory": {
            "hapax_frac_types": gen_m["hapax_frac_types_exploratory"],
            "mean_word_length": gen_m["mean_word_length_exploratory"],
            "top1_word": gen_m["top1_word"],
            "n_tokens": gen_m["n_tokens"],
            "n_types": gen_m["n_types"],
        },
        "optimizable": opt,
        # held-out payload computed but MUST NOT guide search; caller strips before ranking
        "_heldout_raw_metrics": {k: gen_m[k] for k in HELDOUT_METRICS},
    }


def fitness_key(result: dict) -> tuple:
    opt = result["optimizable"]
    g = result["genome"]
    return (
        opt["n_match"],
        -opt["excess"],
        -g["window"],
        -g["p_cite"],
        -g["p_mutate_given_cite"],
        -g["p_splice_given_cite"],
    )


def decide_pass(
    winner_opt_n_match: int,
    winner_heldout_n_separate: int,
    iid_opt_n_separate: int,
) -> dict:
    reasons = []
    mode = None
    if iid_opt_n_separate < 3:
        mode = "iid_control_failed"
        reasons.append(
            f"iid_char separated on only {iid_opt_n_separate}/4 optimizable metrics (need ≥3)"
        )
        passed = False
    elif winner_opt_n_match < 3:
        mode = "surface_unmatched"
        reasons.append(
            f"winner matched only {winner_opt_n_match}/4 optimizable metrics (need ≥3)"
        )
        passed = False
    elif winner_heldout_n_separate < 2:
        mode = "heldout_also_matched"
        reasons.append(
            f"winner separated on only {winner_heldout_n_separate}/3 held-out diagnostics (need ≥2)"
        )
        passed = False
    else:
        mode = "surface_match_heldout_fail"
        reasons.append("all EXP-0019 gates cleared")
        passed = True
    return {
        "passed": passed,
        "mode": mode,
        "reasons": reasons,
        "rule": "EXP-0019",
        "gates": {
            "optimizable_match_min": 3,
            "heldout_separate_min": 2,
            "iid_optimizable_separate_min": 3,
            "winner_optimizable_n_match": winner_opt_n_match,
            "winner_heldout_n_separate": winner_heldout_n_separate,
            "iid_optimizable_n_separate": iid_opt_n_separate,
        },
    }


def run_search(
    train_m: dict,
    val_m: dict,
    shapes: list[dict],
    style_models: dict[str, dict],
    default_style_probs: dict[str, float],
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    population: list[Genome] = [Genome(**DEFAULT_GENOME).repaired()]
    while len(population) < POP_SIZE:
        population.append(random_genome(rng))

    history = []
    scored_cache: dict[tuple, dict] = {}

    def score(genome: Genome) -> dict:
        key = genome.key()
        if key not in scored_cache:
            scored_cache[key] = evaluate_selfcite(
                genome, shapes, style_models, default_style_probs, train_m, val_m, seed
            )
        return scored_cache[key]

    best_ever = None
    for gen in range(N_GENERATIONS):
        scored = [score(g) for g in population]
        ranked = sorted(scored, key=fitness_key, reverse=True)
        gen_best = ranked[0]
        if best_ever is None or fitness_key(gen_best) > fitness_key(best_ever):
            best_ever = gen_best
        hit_perfect = gen_best["optimizable"]["n_match"] >= 4
        history.append(
            {
                "generation": gen,
                "best_optimizable_n_match": gen_best["optimizable"]["n_match"],
                "best_excess": gen_best["optimizable"]["excess"],
                "best_genome": gen_best["genome"],
                "mean_optimizable_n_match": float(
                    np.mean([r["optimizable"]["n_match"] for r in ranked])
                ),
                "early_stop_after_4of4": hit_perfect,
            }
        )
        if hit_perfect:
            break

        elites = [Genome(**r["genome"]).repaired() for r in ranked[:N_ELITE]]
        next_pop = list(elites)
        while len(next_pop) < POP_SIZE:
            if rng.random() < 0.5 and len(elites) >= 2:
                a = elites[int(rng.integers(len(elites)))]
                b = elites[int(rng.integers(len(elites)))]
                child = crossover(rng, a, b)
            else:
                parent = elites[int(rng.integers(len(elites)))]
                child = mutate_genome(rng, parent)
            next_pop.append(child)
        population = next_pop

    assert best_ever is not None
    winner_genome = Genome(**best_ever["genome"]).repaired()
    winner_full = score(winner_genome)
    heldout = metric_verdicts(
        train_m,
        val_m,
        {k: winner_full["metrics"][k] for k in HELDOUT_METRICS},
        HELDOUT_METRICS,
    )

    defaults_ref = score(Genome(**DEFAULT_GENOME).repaired())
    defaults_heldout = metric_verdicts(
        train_m,
        val_m,
        {
            **{k: defaults_ref["metrics"][k] for k in OPT_METRICS},
            **{k: defaults_ref["metrics"][k] for k in HELDOUT_METRICS},
        },
        HELDOUT_METRICS,
    )

    return {
        "winner": {
            "genome": winner_full["genome"],
            "optimizable": winner_full["optimizable"],
            "heldout": heldout,
            "metrics": winner_full["metrics"],
            "exploratory": winner_full["exploratory"],
        },
        "defaults_reference": {
            "genome": defaults_ref["genome"],
            "optimizable": defaults_ref["optimizable"],
            "heldout": defaults_heldout,
            "metrics": defaults_ref["metrics"],
        },
        "history": history,
        "n_unique_genomes_scored": len(scored_cache),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    data_root = args.data_root or (root / "data" / "processed" / "zl3b")
    out_dir = root / "results" / "EXP-0019"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train = load_pages(data_root / "train.jsonl")
    validation = load_pages(data_root / "validation.jsonl")
    test_path = data_root / "test.jsonl"
    test_pages = sum(1 for _ in test_path.open()) if test_path.exists() else 0

    train_m = metrics_from_pages(train)
    val_m = metrics_from_pages(validation)
    shapes = page_shapes(validation)

    train_a = [p for p in train if p["L"] == "A"]
    train_b = [p for p in train if p["L"] == "B"]
    style_models = {}
    for label, subset in (("A", train_a), ("B", train_b)):
        bigrams, starts, lengths = fit_bigram_and_lengths(subset)
        style_models[label] = {"bigrams": bigrams, "starts": starts, "lengths": lengths}
    n_ab = len(train_a) + len(train_b)
    default_style_probs = {"A": len(train_a) / n_ab, "B": len(train_b) / n_ab}
    unigram = fit_char_unigram(train)

    # iid control (optimizable only for gate; held-out reported for audit)
    iid_rng = np.random.default_rng(args.seed + 17)
    iid_pages = generate_iid_char(iid_rng, shapes, unigram)
    iid_m = metrics_from_pages(iid_pages)
    iid_opt = metric_verdicts(train_m, val_m, iid_m, OPT_METRICS)
    iid_held = metric_verdicts(train_m, val_m, iid_m, HELDOUT_METRICS)

    search = run_search(
        train_m, val_m, shapes, style_models, default_style_probs, seed=args.seed
    )
    winner = search["winner"]
    heldout_n_separate = sum(
        1 for v in winner["heldout"]["per_metric"].values() if v["verdict"] == "separate"
    )
    iid_opt_n_separate = iid_opt["n_metrics"] - iid_opt["n_match"]

    decision = decide_pass(
        winner_opt_n_match=winner["optimizable"]["n_match"],
        winner_heldout_n_separate=heldout_n_separate,
        iid_opt_n_separate=iid_opt_n_separate,
    )
    decision["winner_genome"] = winner["genome"]
    decision["optimizable"] = winner["optimizable"]
    decision["heldout"] = winner["heldout"]
    decision["iid_optimizable"] = iid_opt
    decision["thresholds"] = {
        "floors": FLOORS,
        "optimizable_metrics": list(OPT_METRICS),
        "heldout_metrics": list(HELDOUT_METRICS),
        "match_rule": "abs(G-R) <= max(|T-R|, floor)",
        "population": POP_SIZE,
        "generations": N_GENERATIONS,
        "seed": args.seed,
    }

    elapsed = time.perf_counter() - t0
    results = {
        "experiment": "EXP-0019",
        "question": (
            "Can evolutionary copy-mutate search force optimizable ZL3b surface-stat match "
            "and then fail held-out joint-futures / line-initial / A/B Jaccard diagnostics?"
        ),
        "linked": ["HYP-005", "R3", "NB-0022"],
        "seed": args.seed,
        "data_root": str(data_root),
        "test_pages_present_unscored": test_pages,
        "elapsed_seconds": elapsed,
        "reference_metrics": {"train": train_m, "validation": val_m},
        "iid_char": {
            "role": "negative_control",
            "optimizable": iid_opt,
            "heldout_audit_only": iid_held,
            "metrics": {k: iid_m[k] for k in list(OPT_METRICS) + list(HELDOUT_METRICS)},
        },
        "search": {
            "population": POP_SIZE,
            "generations_max": N_GENERATIONS,
            "elite": N_ELITE,
            "n_unique_genomes_scored": search["n_unique_genomes_scored"],
            "history": search["history"],
        },
        "defaults_reference": search["defaults_reference"],
        "winner": winner,
        "decision": decision,
        "non_claims": [
            "Not a decipherment",
            "Not a language ID",
            "Surface-stat match alone is not a HYP-005 proof",
        ],
    }

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    results_s = _sanitize(results)
    decision_s = _sanitize(decision)
    (out_dir / "results.json").write_text(json.dumps(results_s, indent=2) + "\n")
    (out_dir / "decision.json").write_text(json.dumps(decision_s, indent=2) + "\n")

    manifest = {
        "experiment": "EXP-0019",
        "seed": args.seed,
        "data_root": str(data_root),
        "train_pages": len(train),
        "validation_pages": len(validation),
        "test_pages_unscored": test_pages,
        "optimizable_metrics": list(OPT_METRICS),
        "heldout_metrics": list(HELDOUT_METRICS),
        "floors": FLOORS,
        "passed": decision["passed"],
        "mode": decision["mode"],
    }
    man_path = root / "data" / "manifests" / "exp0019_data.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "wrote": str(out_dir),
                "passed": decision["passed"],
                "mode": decision["mode"],
                "winner_opt_match": winner["optimizable"]["n_match"],
                "winner_heldout_separate": heldout_n_separate,
                "iid_opt_separate": iid_opt_n_separate,
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
