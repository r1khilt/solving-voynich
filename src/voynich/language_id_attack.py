"""EXP-0022: monoalphabetic / soft-homophonic language-ID attack with null controls (R5)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

EXPERIMENT_ID = "EXP-0022"
DATA_SEED = 4022
SECTION_SEED = 4023
NULL_SEED = 4024
HILLCLIMB_PROPOSALS = 2500
HOMO_PASSES = 8
MARGIN_BITS = 0.020
LETTER_BUDGET = 12_000
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
ALPHA_IDX = {c: i for i, c in enumerate(ALPHABET)}
N_SYM = len(ALPHABET)
PUA_RE = __import__("re").compile(r"[\ue000-\uf8ff]")

PANEL: list[tuple[str, str]] = [
    ("heb", "heb__udhr_heb.txt"),
    ("lat", "lat__udhr_lat_1.txt"),
    ("eng", "eng__udhr_eng.txt"),
    ("spa", "spa__udhr_spa.txt"),
    ("deu", "deu__udhr_deu_1996.txt"),
    ("fra", "fra__udhr_fra.txt"),
    ("ita", "ita__udhr_ita.txt"),
    ("rus", "rus__udhr_rus.txt"),
    ("ell", "ell__udhr_ell_monotonic.txt"),
    ("arb", "arb__udhr_arb.txt"),
    ("fin", "fin__udhr_fin.txt"),
    ("hun", "hun__udhr_hun.txt"),
    ("tur", "tur__udhr_tur.txt"),
    ("ind", "ind__udhr_ind.txt"),
    ("eus", "eus__udhr_eus.txt"),
    ("hin", "hin__udhr_hin.txt"),
    ("swh", "swh__udhr_swh.txt"),
    ("tgl", "tgl__udhr_tgl.txt"),
    ("amh", "amh__udhr_amh.txt"),
    ("quz", "quz__udhr_quz.txt"),
    ("vie", "vie__udhr_vie.txt"),
    ("pol", "pol__udhr_pol.txt"),
    ("gle", "gle__udhr_gle.txt"),
    ("hau", "hau__udhr_hau_NG.txt"),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = PUA_RE.sub("", text).lower()
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch in ALPHA_IDX and ch != " ":
            out.append(ch)
            prev_space = False
        elif ch == "\n":
            out.append("\n")
            prev_space = False
        elif ch.isspace() or ch not in ALPHA_IDX:
            if not prev_space and out and out[-1] != "\n":
                out.append(" ")
                prev_space = True
    return "".join(out)


def load_pages(path: Path) -> list[dict]:
    pages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        pages.append(
            {
                "page_id": page["page_id"],
                "I": page.get("metadata", {}).get("page_variables", {}).get("I", "?"),
                "text": clean_text(page["text"]),
            }
        )
    return pages


def concat_pages(pages: list[dict]) -> str:
    return "\n".join(p["text"] for p in sorted(pages, key=lambda p: p["page_id"]))


def letter_prefix(text: str, budget: int = LETTER_BUDGET) -> str:
    n = 0
    for i, ch in enumerate(text):
        if ch in "abcdefghijklmnopqrstuvwxyz":
            n += 1
            if n >= budget:
                return text[: i + 1]
    raise RuntimeError(f"ciphertext_too_short:{n}")


def letter_count(text: str) -> int:
    return sum(ch in "abcdefghijklmnopqrstuvwxyz" for ch in text)


def encode_stream(text: str) -> np.ndarray:
    """Encode to symbol ids; newlines become -1 boundary markers."""
    ids = []
    for ch in text:
        if ch == "\n":
            ids.append(-1)
        elif ch in ALPHA_IDX:
            ids.append(ALPHA_IDX[ch])
        # other chars already cleaned away
    return np.asarray(ids, dtype=np.int16)


class BigramLM:
    def __init__(self, text: str):
        counts = np.ones((N_SYM, N_SYM), dtype=np.float64)
        prev = None
        for ch in text:
            if ch == "\n":
                prev = None
                continue
            if ch not in ALPHA_IDX:
                continue
            j = ALPHA_IDX[ch]
            if prev is not None:
                counts[prev, j] += 1.0
            prev = j
        self.logp = np.log2(counts / counts.sum(axis=1, keepdims=True))
        uni = counts.sum(axis=0)
        self.unigram = uni / uni.sum()


def cipher_bigram_pairs(cipher_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (prev_cipher_id, curr_cipher_id) for each scored bigram step."""
    prevs: list[int] = []
    currs: list[int] = []
    prev = None
    for c_id in cipher_ids.tolist():
        if c_id < 0:
            prev = None
            continue
        if prev is not None:
            prevs.append(prev)
            currs.append(c_id)
        prev = c_id
    if not prevs:
        return np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int16)
    return np.asarray(prevs, dtype=np.int16), np.asarray(currs, dtype=np.int16)


def score_table(lm: BigramLM, prev_c: np.ndarray, curr_c: np.ndarray, table: np.ndarray) -> float:
    if prev_c.size == 0:
        return float("inf")
    prev_p = table[prev_c]
    curr_p = table[curr_c]
    return float((-lm.logp[prev_p, curr_p]).mean())


def frequency_rank_table(cipher_ids: np.ndarray, lang_unigram: np.ndarray) -> np.ndarray:
    table = np.full(N_SYM, ALPHA_IDX[" "], dtype=np.int16)
    table[ALPHA_IDX[" "]] = ALPHA_IDX[" "]
    counts = np.bincount(cipher_ids[cipher_ids >= 0], minlength=N_SYM)
    cipher_letters = [i for i in np.argsort(-counts[:26]) if counts[i] > 0]
    lang_order = [i for i in np.argsort(-lang_unigram[:26])]
    for i, c in enumerate(cipher_letters):
        table[c] = lang_order[i % len(lang_order)]
    return table


def mono_attack(cipher_ids: np.ndarray, lm: BigramLM, rng: np.random.Generator) -> float:
    prev_c, curr_c = cipher_bigram_pairs(cipher_ids)
    table = frequency_rank_table(cipher_ids, lm.unigram)
    best = score_table(lm, prev_c, curr_c, table)
    letters = np.array(sorted({int(x) for x in cipher_ids.tolist() if 0 <= x < 26}), dtype=np.int16)
    if letters.size < 2:
        return best
    for _ in range(HILLCLIMB_PROPOSALS):
        a, b = rng.choice(letters, size=2, replace=False)
        table[a], table[b] = int(table[b]), int(table[a])
        score = score_table(lm, prev_c, curr_c, table)
        if score < best:
            best = score
        else:
            table[a], table[b] = int(table[b]), int(table[a])
    return float(best)


def homo_attack(cipher_ids: np.ndarray, lm: BigramLM) -> float:
    prev_c, curr_c = cipher_bigram_pairs(cipher_ids)
    counts = np.bincount(cipher_ids[cipher_ids >= 0], minlength=N_SYM)
    cipher_order = [i for i in np.argsort(-counts[:26]) if counts[i] > 0]
    if not cipher_order:
        return float("inf")
    lang_order = [i for i in np.argsort(-lm.unigram[:26])]
    table = np.full(N_SYM, ALPHA_IDX[" "], dtype=np.int16)
    table[ALPHA_IDX[" "]] = ALPHA_IDX[" "]
    for i, c in enumerate(cipher_order):
        table[c] = lang_order[min(i, len(lang_order) - 1)]
    for _ in range(HOMO_PASSES):
        for c in cipher_order:
            best_letter = int(table[c])
            best_score = score_table(lm, prev_c, curr_c, table)
            for letter in lang_order:
                table[c] = letter
                score = score_table(lm, prev_c, curr_c, table)
                if score < best_score:
                    best_score = score
                    best_letter = letter
            table[c] = best_letter
    return float(score_table(lm, prev_c, curr_c, table))


def scrambled_glyph(text: str, seed: int) -> str:
    letters = sorted({ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz"})
    rng = np.random.default_rng(seed)
    perm = list(letters)
    rng.shuffle(perm)
    table = dict(zip(letters, perm))
    return "".join(table.get(ch, ch) for ch in text)


def section_shuffled(pages: list[dict], seed: int) -> str:
    rng = np.random.default_rng(seed)
    by_section: dict[str, list[dict]] = {}
    for p in pages:
        by_section.setdefault(p["I"], []).append(p)
    remapped = {p["page_id"]: p["text"] for p in pages}
    for _sec, group in by_section.items():
        if len(group) < 2:
            continue
        ids = [g["page_id"] for g in group]
        texts = [g["text"] for g in group]
        order = np.arange(len(texts))
        rng.shuffle(order)
        for page_id, idx in zip(ids, order):
            remapped[page_id] = texts[int(idx)]
    return "\n".join(remapped[pid] for pid in sorted(remapped))


def zipf_length_null(text: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    chars = [ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz"]
    rng.shuffle(chars)
    it = iter(chars)
    return "".join(next(it) if ch in "abcdefghijklmnopqrstuvwxyz" else ch for ch in text)


def preferred_languages(scores: dict[str, float], margin: float = MARGIN_BITS) -> list[str]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    if len(ordered) < 2:
        return [ordered[0][0]] if ordered else []
    best_lang, best = ordered[0]
    second = ordered[1][1]
    if second - best >= margin:
        return [best_lang]
    return []


def decide(preferred: dict[str, list[str]]) -> dict:
    w_null = preferred["zipf_length_null"]
    if "heb" in w_null or "lat" in w_null:
        return {
            "pass": False,
            "mode": "hebrew_latin_null_preferred",
            "interpretation": (
                "Attack prefers Hebrew/Latin on length+unigram-matched null; "
                "method invalid for language ID."
            ),
        }
    w_true = set(preferred["true_eva"])
    w_scram = set(preferred["scrambled_glyph"])
    w_sect = set(preferred["section_shuffled"])
    leads = sorted(w_true - w_scram - w_sect)
    if leads:
        return {
            "pass": False,
            "mode": "lead_open",
            "leads": leads,
            "interpretation": (
                "Language(s) preferred on true EVA but not on scrambled-glyph or section-shuffled; "
                "artifact-resistant lead under this attack only — not a decipherment."
            ),
        }
    return {
        "pass": True,
        "mode": "simple_language_id_rejected",
        "interpretation": (
            "No panel language uniquely preferred on true EVA over both null controls "
            "(or none preferred on true EVA). Simple monoalphabetic language-ID class "
            "rejected under frozen gates."
        ),
    }


def run(root: Path) -> dict:
    t0 = time.time()
    romanized = root / "data" / "raw" / "exp0016_corpora" / "romanized"
    zl3b = root / "data" / "processed" / "zl3b"
    train = load_pages(zl3b / "train.jsonl")
    val = load_pages(zl3b / "validation.jsonl")
    test_path = zl3b / "test.jsonl"
    pages = train + val

    full_true = concat_pages(pages)
    true_eva = letter_prefix(full_true, LETTER_BUDGET)
    full_section = section_shuffled(pages, SECTION_SEED)
    section_text = letter_prefix(full_section, LETTER_BUDGET)
    corpora = {
        "true_eva": true_eva,
        "scrambled_glyph": scrambled_glyph(true_eva, DATA_SEED),
        "section_shuffled": section_text,
        "zipf_length_null": zipf_length_null(true_eva, NULL_SEED),
    }
    corpus_ids = {name: encode_stream(text) for name, text in corpora.items()}

    languages: dict[str, dict] = {}
    for iso, rel in PANEL:
        path = romanized / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing panel file: {path}")
        text = clean_text(path.read_text(encoding="utf-8"))
        n_letters = letter_count(text)
        if n_letters < 2000:
            raise RuntimeError(f"panel_too_short:{iso}:{n_letters}")
        languages[iso] = {
            "path": str(path.relative_to(root)),
            "n_letters": n_letters,
            "sha256": sha256_text(text),
            "lm": BigramLM(text),
        }

    mono_tables: dict[str, dict[str, float]] = {}
    homo_tables: dict[str, dict[str, float]] = {}
    preferred: dict[str, list[str]] = {}
    top3: dict[str, list[dict]] = {}

    for corp_name, ids in corpus_ids.items():
        mono_scores: dict[str, float] = {}
        homo_scores: dict[str, float] = {}
        for iso, meta in languages.items():
            rng = np.random.default_rng(
                DATA_SEED + sum(ord(c) for c in iso) * 17 + sum(ord(c) for c in corp_name) * 31
            )
            mono_scores[iso] = mono_attack(ids, meta["lm"], rng)
            homo_scores[iso] = homo_attack(ids, meta["lm"])
        mono_tables[corp_name] = mono_scores
        homo_tables[corp_name] = homo_scores
        preferred[corp_name] = preferred_languages(mono_scores)
        ordered = sorted(mono_scores.items(), key=lambda kv: kv[1])
        top3[corp_name] = [
            {"iso": iso, "mono_bits": score, "homo_bits": homo_scores[iso]}
            for iso, score in ordered[:3]
        ]

    decision = decide(preferred)
    elapsed = time.time() - t0
    results = {
        "experiment": EXPERIMENT_ID,
        "seeds": {
            "data": DATA_SEED,
            "section": SECTION_SEED,
            "null": NULL_SEED,
            "hillclimb_proposals": HILLCLIMB_PROPOSALS,
            "homo_passes": HOMO_PASSES,
            "margin_bits": MARGIN_BITS,
            "letter_budget": LETTER_BUDGET,
        },
        "panel": {
            iso: {
                "path": languages[iso]["path"],
                "n_letters": languages[iso]["n_letters"],
                "sha256": languages[iso]["sha256"],
            }
            for iso in languages
        },
        "corpora": {
            name: {
                "sha256": sha256_text(text),
                "n_chars": len(text),
                "n_letters": letter_count(text),
            }
            for name, text in corpora.items()
        },
        "zl3b": {
            "train_pages": len(train),
            "validation_pages": len(val),
            "test_opened_for_scoring": False,
            "test_file_present": test_path.exists(),
        },
        "mono_bits": mono_tables,
        "homo_bits": homo_tables,
        "preferred_mono": preferred,
        "top3_mono": top3,
        "decision": decision,
        "elapsed_seconds": elapsed,
        "non_claim": "Not a decipherment. No frozen reading. No held-out leaf semantic test.",
    }
    return results


def write_outputs(root: Path, results: dict) -> None:
    out = root / "results" / EXPERIMENT_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision = {
        "experiment": EXPERIMENT_ID,
        "pass": results["decision"]["pass"],
        "mode": results["decision"]["mode"],
        "preferred_mono": results["preferred_mono"],
        "top3_mono": results["top3_mono"],
        "interpretation": results["decision"].get("interpretation"),
        "leads": results["decision"].get("leads"),
        "non_claim": results["non_claim"],
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "experiment": EXPERIMENT_ID,
        "panel": results["panel"],
        "corpora": results["corpora"],
        "seeds": results["seeds"],
        "zl3b": results["zl3b"],
    }
    (root / "data" / "manifests" / "exp0022_data.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    results = run(root)
    write_outputs(root, results)
    print(
        json.dumps(
            {
                "mode": results["decision"]["mode"],
                "pass": results["decision"]["pass"],
                "preferred": results["preferred_mono"],
                "elapsed": results["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
