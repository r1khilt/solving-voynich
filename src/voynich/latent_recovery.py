"""EXP-0011: recover latent ciphered signal under structured null insertion.

Pipeline: L -> C(L) -> N(C(L)). Fresh cipher per sample. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from voynich.recovery_metrics import (
    FixedRankBigram, binary_mask, edit_distance, fixed_rank_deletion_diagnostic,
    reconstruction_diagnostics, summarize_reconstruction,
)
from voynich.runtime import digest, environment, resolve_device, write_json

# ---------------------------------------------------------------------------
# Frozen registration constants (EXP-0011)
# ---------------------------------------------------------------------------

DATA_SEED = 4011
MODEL_SEED = 42
FINNISH_SEED = 4012
PRIMARY_FILLER_RATE = 0.30
SECONDARY_FILLER_RATE = 0.15
MAX_UPDATES = 3000
SEQ_LEN = 128
BATCH_SIZE = 32
PASS_MASK_F1_MIN = 0.65
PASS_MARGIN_VS_WEAK = 0.08
PASS_MARGIN_VS_VOCAB = 0.05
PASS_VOCAB_F1_MAX = 0.55
PASS_RECON_MARGIN = 0.08
PASS_PRED_BITS_MARGIN = 0.05
MATCHED_RANDOM_SEEDS = 20

# EXP-0012 frozen thresholds (do not alter EXP-0011 constants above)
PASS12_NULL_RECALL_MIN = 0.55
PASS12_NULL_PRECISION_MIN = 0.50
PASS12_PRED_NULL_RATE_MIN = 0.15
PASS12_PRED_NULL_RATE_MAX = 0.45
PASS12_RECON_MARGIN = 0.08
PASS12_MASK_ACC_MARGIN = 0.05
PASS12_PRED_BITS_MARGIN = 0.03
PASS12_VOCAB_F1_MAX = 0.55
NULL_LOSS_WEIGHT = 3.5  # recall-seeking vs signal class

# EXP-0013 frozen thresholds (do not alter EXP-0011 / 0012 constants above)
PASS13_NULL_RECALL_MIN = 0.50
PASS13_NULL_PRECISION_MIN = 0.50
PASS13_PRED_NULL_RATE_MIN = 0.15
PASS13_PRED_NULL_RATE_MAX = 0.45
PASS13_VOCAB_F1_MAX = 0.55
ALIGN_CTC_WEIGHT = 1.0
ALIGN_MASK_BCE_WEIGHT = 0.25
ALIGN_RATE_WEIGHT = 0.15
ALIGN_WORLD_WEIGHT = 0.10

LATIN_PLAIN = "abcdefghijklmnopqrstuvwxyz "
# Cipher pool deliberately excludes latin a-z so world B/C never looks like English spelling.
CIPHER_POOL = (
    "αβγδεζηθικλμνξοπρστυφχψω"
    "абвгдежзиклмнопрстуфхцчшщъыьэюя"
    "אבּגדהוזחטיכלמנסעפצקרשת"
    "åäöüßĉĝĥĵŝŭþð"
)
assert not set(CIPHER_POOL) & set("abcdefghijklmnopqrstuvwxyz")

FILLER_FAMILIES = (
    "random_char",
    "random_pseudoword",
    "copy_mutate",
    "periodic",
    "shift_phase",
    "stateful",
)
EASY_FILLER_FAMILIES = ("random_char", "periodic")
COPY_MUTATE_ONLY = ("copy_mutate",)


def parse_filler_families(spec: str | None) -> tuple[str, ...]:
    """Parse comma-separated filler family names; None/empty → full EXP-0011 set."""
    if not spec or not str(spec).strip():
        return FILLER_FAMILIES
    names = tuple(x.strip() for x in str(spec).split(",") if x.strip())
    unknown = [n for n in names if n not in FILLER_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown filler families: {unknown}; allowed={FILLER_FAMILIES}")
    if not names:
        raise ValueError("filler families list is empty")
    return names

WORLD_A, WORLD_B, WORLD_C, WORLD_D = 0, 1, 2, 3
WORLD_NAMES = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Text cleaning / corpora
# ---------------------------------------------------------------------------

GUTENBERG_MARKERS = (
    re.compile(r"\*\*\*\s*START OF.+?\*\*\*", re.I | re.S),
    re.compile(r"\*\*\*\s*END OF.+?\*\*\*", re.I | re.S),
)


def clean_plaintext(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for pat in GUTENBERG_MARKERS[:1]:
        m = pat.search(text)
        if m:
            text = text[m.end() :]
    for pat in GUTENBERG_MARKERS[1:]:
        m = pat.search(text)
        if m:
            text = text[: m.start()]
    text = text.lower()
    text = "".join(ch if ch in LATIN_PLAIN or ch == "\n" else " " for ch in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_language_text(path: Path) -> str:
    return clean_plaintext(path.read_text(encoding="utf-8", errors="replace"))


def chunks_from_text(text: str, rng: np.random.Generator, n: int, length: int = SEQ_LEN) -> list[str]:
    # Prefer space-delimited windows so samples are mostly contiguous prose.
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) < length + 10:
        raise ValueError(f"Corpus too short: {len(flat)} chars")
    out = []
    for _ in range(n):
        start = int(rng.integers(0, len(flat) - length))
        piece = flat[start : start + length]
        # Snap to next space when cheap to reduce mid-word starts.
        sp = piece.find(" ")
        if 0 < sp < 20 and start + sp + 1 + length <= len(flat):
            piece = flat[start + sp + 1 : start + sp + 1 + length]
        out.append(piece[:length].ljust(length)[:length])
    return out


# ---------------------------------------------------------------------------
# Cipher C and null N
# ---------------------------------------------------------------------------


@dataclass
class CipherSpec:
    mapping: dict[str, str]
    homophones: dict[str, tuple[str, str]]
    merges: dict[str, str]  # digram -> symbol
    drop_vowels: bool
    break_spaces: str  # 'none' | 'delete' | 'insert'


def sample_cipher(rng: np.random.Generator, alphabet: str) -> CipherSpec:
    plain_letters = [c for c in LATIN_PLAIN if c != " "]
    symbols = list(alphabet)
    rng.shuffle(symbols)
    mapping = {p: symbols[i] for i, p in enumerate(plain_letters)}
    mapping[" "] = " "
    # Homophones for ~20% of letters when alphabet has spare symbols.
    homophones: dict[str, tuple[str, str]] = {}
    spare = symbols[len(plain_letters) :]
    if spare and rng.random() < 0.7:
        targets = list(rng.choice(plain_letters, size=min(5, len(spare)), replace=False))
        for i, p in enumerate(targets):
            if i >= len(spare):
                break
            homophones[p] = (mapping[p], spare[i])
    merges: dict[str, str] = {}
    if spare and rng.random() < 0.4:
        # merge two frequent-ish letters into one unused symbol if available
        unused = [s for s in spare if s not in {h[1] for h in homophones.values()}]
        if unused:
            a, b = rng.choice(plain_letters, size=2, replace=False)
            merges[a + b] = unused[0]
    drop_vowels = bool(rng.random() < 0.35)
    break_spaces = str(rng.choice(["none", "none", "delete", "insert"]))
    return CipherSpec(mapping, homophones, merges, drop_vowels, break_spaces)


def apply_cipher(text: str, spec: CipherSpec, rng: np.random.Generator) -> str:
    t = text
    if spec.drop_vowels:
        t = "".join(ch for ch in t if ch not in "aeiou")
        t = re.sub(r" {2,}", " ", t)
    if spec.break_spaces == "delete":
        chars = list(t)
        for i in range(len(chars)):
            if chars[i] == " " and rng.random() < 0.25:
                chars[i] = ""
        t = "".join(chars)
    elif spec.break_spaces == "insert":
        chars = []
        for ch in t:
            chars.append(ch)
            if ch != " " and rng.random() < 0.08:
                chars.append(" ")
        t = "".join(chars)
    out = []
    i = 0
    while i < len(t):
        if i + 1 < len(t) and t[i : i + 2] in spec.merges:
            out.append(spec.merges[t[i : i + 2]])
            i += 2
            continue
        ch = t[i]
        if ch in spec.homophones:
            out.append(spec.homophones[ch][int(rng.integers(2))])
        elif ch in spec.mapping:
            out.append(spec.mapping[ch])
        elif ch == "\n":
            out.append("\n")
        else:
            out.append(" ")
        i += 1
    return "".join(out)


def _mutate_token(tok: str, rng: np.random.Generator, alphabet: str) -> str:
    if not tok:
        return tok
    chars = list(tok)
    op = int(rng.integers(3))
    if op == 0 and chars:
        chars[int(rng.integers(len(chars)))] = alphabet[int(rng.integers(len(alphabet)))]
    elif op == 1 and len(chars) > 1:
        chars.pop(int(rng.integers(len(chars))))
    else:
        chars.insert(int(rng.integers(len(chars) + 1)), alphabet[int(rng.integers(len(alphabet)))])
    return "".join(chars) or alphabet[0]


def insert_nulls(
    ciphered: str,
    rng: np.random.Generator,
    filler_rate: float,
    alphabet: str,
    filler_families: tuple[str, ...] | None = None,
) -> tuple[str, list[int], list[str]]:
    """Return noisy text, binary mask (1=signal), and per-token filler family or ''."""
    # Operate on characters; spaces retained as tokens.
    signal_chars = list(ciphered.replace("\n", " "))
    if not signal_chars:
        signal_chars = [" "]
    # Target length so that null fraction ≈ filler_rate of final tokens.
    # n_null / (n_sig + n_null) = filler_rate => n_null = filler_rate/(1-f) * n_sig
    n_sig = len(signal_chars)
    n_null = int(round(filler_rate / max(1e-6, 1 - filler_rate) * n_sig))
    families = list(filler_families) if filler_families is not None else list(FILLER_FAMILIES)
    if not families:
        raise ValueError("filler_families must be non-empty")
    family = str(rng.choice(families))
    # Build insertion plan: positions in final stream.
    out_chars: list[str] = []
    mask: list[int] = []
    fam_tags: list[str] = []
    sig_i = 0
    phase = int(rng.integers(3, 8))
    state = 0
    state_sym = alphabet[int(rng.integers(len(alphabet)))]
    inserts_left = n_null
    total_target = n_sig + n_null

    def emit_null() -> str:
        nonlocal state, state_sym
        if family == "random_char":
            return alphabet[int(rng.integers(len(alphabet)))]
        if family == "random_pseudoword":
            # Emit one char of a short pseudoword burst tracked via state
            nuclei = "αειουаеиоу"
            if state == 0:
                state = int(rng.integers(2, 5))
            state -= 1
            if state % 2 == 0:
                return alphabet[int(rng.integers(len(alphabet)))]
            return nuclei[int(rng.integers(len(nuclei)))] if nuclei else alphabet[0]
        if family == "copy_mutate":
            # Copy from nearby already-emitted signal-ish chars
            pool = [c for c, m in zip(out_chars, mask) if m == 1][-12:]
            if not pool:
                pool = signal_chars[max(0, sig_i - 6) : sig_i + 6] or list(alphabet[:8])
            tok = pool[int(rng.integers(len(pool)))]
            if rng.random() < 0.6:
                return _mutate_token(tok, rng, alphabet)
            return tok
        if family == "periodic":
            return alphabet[phase % len(alphabet)]
        if family == "shift_phase":
            # Phase shifts every ~line (every 16 chars)
            local = (len(out_chars) // 16 + phase) % len(alphabet)
            return alphabet[local]
        # stateful Markov filler
        if rng.random() < 0.3:
            state_sym = alphabet[int(rng.integers(len(alphabet)))]
        return state_sym

    # Interleave by walking a random merge of signal and nulls
    sig_left = n_sig
    while sig_left + inserts_left > 0:
        take_null = inserts_left > 0 and (
            sig_left == 0 or rng.random() < inserts_left / (sig_left + inserts_left)
        )
        if take_null:
            out_chars.append(emit_null())
            mask.append(0)
            fam_tags.append(family)
            inserts_left -= 1
        else:
            out_chars.append(signal_chars[sig_i])
            mask.append(1)
            fam_tags.append("")
            sig_i += 1
            sig_left -= 1
        if len(out_chars) > total_target + 5:
            break
    return "".join(out_chars), mask, fam_tags


def structured_pseudotext(rng: np.random.Generator, length: int, alphabet: str) -> tuple[str, list[int]]:
    """World D: copy/mutate + periodic filler, no latent language."""
    chars: list[str] = []
    buf: list[str] = [alphabet[int(rng.integers(len(alphabet)))] for _ in range(4)]
    phase = int(rng.integers(3, 9))
    for i in range(length):
        if i % phase == 0:
            ch = alphabet[(i // phase) % len(alphabet)]
        elif rng.random() < 0.55 and buf:
            src = buf[int(rng.integers(len(buf)))]
            ch = _mutate_token(src, rng, alphabet) if rng.random() < 0.5 else src
        else:
            ch = alphabet[int(rng.integers(len(alphabet)))]
        chars.append(ch)
        buf.append(ch)
        if len(buf) > 20:
            buf.pop(0)
        if rng.random() < 0.12:
            chars.append(" ")
    text = "".join(chars)[:length].ljust(length)[:length]
    return text, [0] * len(text)


def _fit_len(text: str, mask: list[int], length: int = SEQ_LEN) -> tuple[str, list[int]]:
    if len(text) >= length:
        text = text[:length]
        mask = list(mask[:length])
    else:
        pad = length - len(text)
        text = text + (" " * pad)
        mask = list(mask) + [1] * pad
    if len(mask) != length:
        mask = (list(mask) + [1] * length)[:length]
    return text, mask


def make_sample(
    plaintext: str,
    rng: np.random.Generator,
    world: int,
    filler_rate: float,
    alphabet: str | None = None,
    filler_families: tuple[str, ...] | None = None,
) -> dict:
    alphabet = alphabet or "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
    if world == WORLD_A:
        text = plaintext.replace("\n", " ")
        text = re.sub(r" {2,}", " ", text)
        text, mask = _fit_len(text, [1] * len(text))
        return {
            "world": world,
            "text": text,
            "mask": mask,
            "ciphered": text,
            "plaintext": plaintext[:SEQ_LEN],
            "alphabet": LATIN_PLAIN.strip(),
            "filler_family": "",
            "filler_rate": 0.0,
        }
    if world == WORLD_D:
        text, mask = structured_pseudotext(rng, SEQ_LEN, alphabet)
        text, mask = _fit_len(text, mask)
        return {
            "world": world,
            "text": text,
            "mask": mask,
            "ciphered": "",
            "plaintext": "",
            "alphabet": alphabet,
            "filler_family": "pure_pseudo",
            "filler_rate": 1.0,
        }
    spec = sample_cipher(rng, alphabet)
    ciphered = apply_cipher(plaintext, spec, rng)
    ciphered = re.sub(r"\s+", " ", ciphered).strip()
    if len(ciphered) < SEQ_LEN // 2:
        ciphered = (ciphered + " ") * (SEQ_LEN // max(1, len(ciphered)))
    ciphered = ciphered[: SEQ_LEN * 2]
    if world == WORLD_B:
        text, mask = _fit_len(ciphered, [1] * len(ciphered))
        family = ""
        rate = 0.0
        kept = text
    else:
        noisy, mask_full, fams = insert_nulls(
            ciphered, rng, filler_rate, alphabet, filler_families=filler_families
        )
        text, mask = _fit_len(noisy, mask_full)
        family = next((f for f in fams[: len(text)] if f), "mixed")
        rate = filler_rate
        kept = "".join(ch for ch, m in zip(text, mask) if m == 1)
    return {
        "world": world,
        "text": text,
        "mask": mask,
        "ciphered": kept if world == WORLD_C else text,
        "plaintext": plaintext[:SEQ_LEN],
        "alphabet": alphabet,
        "filler_family": family,
        "filler_rate": rate,
    }


# ---------------------------------------------------------------------------
# Vocab / encoding for the neural model
# ---------------------------------------------------------------------------


def build_vocab() -> dict[str, int]:
    # PAD=0, UNK=1, then all possible symbols we emit
    chars = ["<pad>", "<unk>", " ", "\n"] + list(dict.fromkeys(LATIN_PLAIN.strip() + CIPHER_POOL))
    return {c: i for i, c in enumerate(chars)}


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def recon_accuracy(true_ciphered: str, text: str, pred_mask: np.ndarray) -> float:
    """Primary reconstruction metric (unchanged since EXP-0011).

    Kept subsequence vs true C(L): prefix match rate × length ratio penalty.
    Not exact full-string equality.
    """
    pred = "".join(ch for ch, m in zip(text, pred_mask) if m == 1)
    n = min(len(pred), len(true_ciphered))
    if n == 0:
        return 1.0 if true_ciphered == pred else 0.0
    matches = sum(a == b for a, b in zip(pred[:n], true_ciphered[:n]))
    # Penalize length mismatch
    length_pen = min(len(pred), len(true_ciphered)) / max(len(pred), len(true_ciphered), 1)
    return float(matches / n * length_pen)


def levenshtein(a: str, b: str) -> int:
    """Backward-compatible public name for the shared unit-cost distance."""
    return edit_distance(a, b)


def recon_edit_similarity(true_ciphered: str, text: str, pred_mask: np.ndarray) -> float:
    """Secondary: 1 - Levenshtein(pred, C(L)) / max(len). Labeled secondary only."""
    pred = "".join(ch for ch, m in zip(text, pred_mask) if m == 1)
    denom = max(len(pred), len(true_ciphered), 1)
    return float(1.0 - levenshtein(pred, true_ciphered) / denom)


def rank_bucket_sequence(text: str, n_buckets: int = 8) -> list[int]:
    from collections import Counter

    counts = Counter(ch for ch in text if ch != " ")
    # Rank chars by frequency; map to buckets
    ranked = [c for c, _ in counts.most_common()]
    bucket = {}
    for i, ch in enumerate(ranked):
        bucket[ch] = min(n_buckets - 1, i * n_buckets // max(1, len(ranked)))
    return [bucket.get(ch, 0) if ch != " " else n_buckets for ch in text]


def bigram_bits(seq: list[int], model: dict) -> float:
    """Cross-entropy bits/symbol under a bigram model dict with keys (a,b)->prob."""
    if len(seq) < 2:
        return 0.0
    total = 0.0
    n = 0
    uni = model["uni"]
    bi = model["bi"]
    v = model["v"]
    for a, b in zip(seq, seq[1:]):
        p = bi.get((a, b), 0.0)
        if p <= 0:
            p = 0.1 * uni.get(b, 1.0 / v) / v
        total += -math.log2(max(p, 1e-12))
        n += 1
    return total / max(n, 1)


def fit_rank_bigram(texts: list[str]) -> dict:
    from collections import Counter

    uni: Counter = Counter()
    bi: Counter = Counter()
    for text in texts:
        seq = rank_bucket_sequence(text)
        uni.update(seq)
        bi.update(zip(seq, seq[1:]))
    v = max(8, len(uni))
    uni_p = {k: (c + 0.1) / (sum(uni.values()) + 0.1 * v) for k, c in uni.items()}
    bi_p = {}
    row = Counter()
    for (a, b), c in bi.items():
        row[a] += c
    for (a, b), c in bi.items():
        bi_p[(a, b)] = (c + 0.1) / (row[a] + 0.1 * v)
    return {"uni": uni_p, "bi": bi_p, "v": v}


def pred_bits_of_selected(text: str, mask: np.ndarray, model: dict) -> float:
    selected = "".join(ch for ch, m in zip(text, mask) if m == 1)
    if len(selected) < 4:
        return 10.0
    return bigram_bits(rank_bucket_sequence(selected), model)


# ---------------------------------------------------------------------------
# Classical 2-state Viterbi (copy/rank aware)
# ---------------------------------------------------------------------------


def _token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def _edit_dist1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    # a shorter by 1
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diffs += 1
            j += 1
            if diffs > 1:
                return False
        else:
            i += 1
            j += 1
    return True


def copy_aware_features(text: str) -> np.ndarray:
    """Per-position features: [is_space, copy_char, copy_token, periodic]."""
    n = len(text)
    feats = np.zeros((n, 4), dtype=np.float32)
    recent_chars: list[str] = []
    spans = _token_spans(text)
    # Mark token-level copy/mutate against previous tokens
    token_copy = np.zeros(n, dtype=np.float32)
    seen: list[str] = []
    for start, end, tok in spans:
        score = 0.0
        for prev in seen[-12:]:
            if _edit_dist1(tok, prev):
                score = 1.0
                break
        token_copy[start:end] = score
        seen.append(tok)
    for i, ch in enumerate(text):
        feats[i, 0] = 1.0 if ch == " " else 0.0
        window = recent_chars[-20:]
        feats[i, 1] = 1.0 if ch != " " and ch in window else 0.0
        feats[i, 2] = token_copy[i]
        feats[i, 3] = 1.0 if (i % 5 == 0 or i % 7 == 0) else 0.0
        recent_chars.append(ch)
    return feats


def classical_null_mask(text: str, filler_rate: float = PRIMARY_FILLER_RATE) -> np.ndarray:
    """Interpretable 2-state Viterbi with token-level copy/mutate features."""
    n = len(text)
    if n == 0:
        return np.zeros(0, dtype=int)
    feats = copy_aware_features(text)
    from collections import Counter

    counts = Counter(ch for ch in text if ch != " ")
    ranked = {ch: i for i, (ch, _) in enumerate(counts.most_common())}
    emit_s = np.zeros(n)
    emit_n = np.zeros(n)
    for i, ch in enumerate(text):
        rank = ranked.get(ch, len(ranked))
        zipf_s = -abs(min(rank, 8) - 2) * 0.1
        space_s = 0.5 * feats[i, 0]
        copy = 1.5 * feats[i, 1] + 2.2 * feats[i, 2]
        periodic = 0.35 * feats[i, 3]
        emit_s[i] = zipf_s + space_s - 0.8 * copy
        emit_n[i] = copy + periodic - space_s
    log_stay_s, log_to_n = math.log(0.88), math.log(0.12)
    log_stay_n, log_to_s = math.log(0.72), math.log(0.28)
    neg = -1e9
    dp_s = np.full(n, neg)
    dp_n = np.full(n, neg)
    ptr_s = np.zeros(n, dtype=int)
    ptr_n = np.zeros(n, dtype=int)
    dp_s[0] = math.log(1 - filler_rate) + emit_s[0]
    dp_n[0] = math.log(filler_rate) + emit_n[0]
    for i in range(1, n):
        s_from_s, s_from_n = dp_s[i - 1] + log_stay_s, dp_n[i - 1] + log_to_s
        if s_from_s >= s_from_n:
            dp_s[i], ptr_s[i] = s_from_s + emit_s[i], 1
        else:
            dp_s[i], ptr_s[i] = s_from_n + emit_s[i], 0
        n_from_n, n_from_s = dp_n[i - 1] + log_stay_n, dp_s[i - 1] + log_to_n
        if n_from_n >= n_from_s:
            dp_n[i], ptr_n[i] = n_from_n + emit_n[i], 0
        else:
            dp_n[i], ptr_n[i] = n_from_s + emit_n[i], 1
    mask = np.zeros(n, dtype=int)
    state = 1 if dp_s[-1] >= dp_n[-1] else 0
    for i in range(n - 1, -1, -1):
        mask[i] = state
        state = ptr_s[i] if state == 1 else ptr_n[i]
        if i == 0:
            break
    # Calibrate null rate to target by score thresholding (keeps relative ranking)
    scores = emit_n - emit_s
    k = int(round(filler_rate * n))
    idx = np.argsort(-scores)[:k]
    mask = np.ones(n, dtype=int)
    mask[idx] = 0
    return mask


def classical_hsmm_null_mask(text: str, filler_rate: float = PRIMARY_FILLER_RATE) -> np.ndarray:
    """Run-aware 2-state Viterbi: keep decoded path (no top-k overwrite).

    Sticky null self-transitions encode geometric run lengths. Periodic and rarity
    emissions target easy fillers; token-copy score is anti-null.
    """
    n = len(text)
    if n == 0:
        return np.zeros(0, dtype=int)
    feats = copy_aware_features(text)
    from collections import Counter

    counts = Counter(ch for ch in text if ch != " ")
    ranked = {ch: i for i, (ch, _) in enumerate(counts.most_common())}
    period_hit = np.zeros(n, dtype=np.float32)
    for p in (3, 4, 5, 6, 7, 8):
        for i in range(p, n):
            if text[i] != " " and text[i] == text[i - p]:
                period_hit[i] += 1.0
    emit_s = np.zeros(n)
    emit_n = np.zeros(n)
    for i, ch in enumerate(text):
        rank = ranked.get(ch, len(ranked))
        rarity = min(rank, 12) / 12.0
        space_s = 0.6 * feats[i, 0]
        copy = 1.2 * feats[i, 1] + 1.8 * feats[i, 2]
        periodic = 0.55 * feats[i, 3] + 0.45 * min(period_hit[i], 3.0)
        emit_s[i] = -0.15 * rarity + space_s + 0.4 * copy - 0.35 * periodic
        emit_n[i] = 0.55 * rarity + periodic - space_s - 0.7 * copy
    # Geometric null runs: high stay-null probability
    p_to_n = min(0.35, max(0.08, filler_rate * 0.55))
    p_to_s = min(0.40, max(0.10, 0.22))
    log_stay_s, log_to_n = math.log(1 - p_to_n), math.log(p_to_n)
    log_stay_n, log_to_s = math.log(1 - p_to_s), math.log(p_to_s)
    neg = -1e9
    dp_s = np.full(n, neg)
    dp_n = np.full(n, neg)
    ptr_s = np.zeros(n, dtype=int)
    ptr_n = np.zeros(n, dtype=int)
    dp_s[0] = math.log(max(1e-6, 1 - filler_rate)) + emit_s[0]
    dp_n[0] = math.log(max(1e-6, filler_rate)) + emit_n[0]
    for i in range(1, n):
        s_from_s, s_from_n = dp_s[i - 1] + log_stay_s, dp_n[i - 1] + log_to_s
        if s_from_s >= s_from_n:
            dp_s[i], ptr_s[i] = s_from_s + emit_s[i], 1
        else:
            dp_s[i], ptr_s[i] = s_from_n + emit_s[i], 0
        n_from_n, n_from_s = dp_n[i - 1] + log_stay_n, dp_s[i - 1] + log_to_n
        if n_from_n >= n_from_s:
            dp_n[i], ptr_n[i] = n_from_n + emit_n[i], 0
        else:
            dp_n[i], ptr_n[i] = n_from_s + emit_n[i], 1
    mask = np.zeros(n, dtype=int)
    state = 1 if dp_s[-1] >= dp_n[-1] else 0
    for i in range(n - 1, -1, -1):
        mask[i] = state
        state = ptr_s[i] if state == 1 else ptr_n[i]
        if i == 0:
            break
    return mask


def null_f1_from_metrics(m: dict) -> float:
    p, r = m.get("null_precision", 0.0), m.get("null_recall", 0.0)
    if p + r <= 0:
        return 0.0
    return float(2 * p * r / (p + r))


# ---------------------------------------------------------------------------
# Tiny BiLSTM with copy-aware side features
# ---------------------------------------------------------------------------


class TinySignalModel(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 64, n_feats: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.feat_proj = nn.Linear(n_feats, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.mask_head = nn.Linear(hidden * 2, 1)
        self.recon_head = nn.Linear(hidden * 2, vocab_size)
        self.world_head = nn.Linear(hidden * 2, 4)

    def forward(self, x: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.emb(x) + self.feat_proj(feats))
        mask_logit = self.mask_head(h).squeeze(-1)
        recon_logit = self.recon_head(h)
        world_logit = self.world_head(h.mean(1))
        return mask_logit, recon_logit, world_logit

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def copy_constrained_ctc_log_probs(
    mask_logit: torch.Tensor,
    x: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """(T, N, C) log-probs: blank = delete, else copy observed char only.

    Higher mask_logit ⇒ keep (emit observed token). Blank index = vocab_size.
    """
    n_batch, t_len = mask_logit.shape
    blank = vocab_size
    c_size = vocab_size + 1
    log_keep = F.logsigmoid(mask_logit)
    log_blank = F.logsigmoid(-mask_logit)
    # Start near -inf; fill blank and observed-char channels.
    log_probs = mask_logit.new_full((n_batch, t_len, c_size), -1.0e4)
    log_probs[:, :, blank] = log_blank
    # If observed id collides with blank index it cannot; blank is vocab_size.
    log_probs.scatter_(2, x.unsqueeze(-1).clamp(0, vocab_size - 1), log_keep.unsqueeze(-1))
    return log_probs.transpose(0, 1).contiguous()


def ctc_deletion_loss(
    mask_logit: torch.Tensor,
    x: torch.Tensor,
    target_ids: list[list[int]],
    vocab_size: int,
) -> torch.Tensor:
    """CTC loss forcing kept subsequence to match target token ids (C(L))."""
    if not target_ids:
        return mask_logit.new_zeros(())
    log_probs = copy_constrained_ctc_log_probs(mask_logit, x, vocab_size)
    t_len, n_batch, _ = log_probs.shape
    # Drop empty targets (CTC undefined); keep a zero contribution via mask.
    usable = [i for i, tgt in enumerate(target_ids) if len(tgt) > 0]
    if not usable:
        return mask_logit.new_zeros(())
    log_probs_u = log_probs[:, usable, :]
    targets_flat: list[int] = []
    target_lengths = []
    for i in usable:
        tgt = target_ids[i]
        targets_flat.extend(tgt)
        target_lengths.append(len(tgt))
    targets = torch.tensor(targets_flat, dtype=torch.long, device=mask_logit.device)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=mask_logit.device)
    input_lengths = torch.full((len(usable),), t_len, dtype=torch.long, device=mask_logit.device)
    ctc = nn.CTCLoss(blank=vocab_size, zero_infinity=True, reduction="mean")
    return ctc(log_probs_u, targets, input_lengths, target_lengths_t)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    corpora: dict[str, str],
    n_train: int,
    n_val: int,
    seed: int,
    filler_rate: float = PRIMARY_FILLER_RATE,
    filler_families: tuple[str, ...] | None = None,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    langs = [k for k in ("english", "latin") if k in corpora]
    if not langs:
        raise ValueError("Need english and/or latin corpora")
    samples_train, samples_val = [], []
    # Fix 1 (post-fail): oversample world C so null-detection is not drowned by easy A/B/D.
    # Mix still includes A/B/D for the Naibbe trap (cipher ≠ filler).
    world_p = np.array([0.10, 0.15, 0.60, 0.15], dtype=float)

    def build(n, store):
        for _ in range(n):
            world = int(rng.choice(4, p=world_p))
            lang = str(rng.choice(langs))
            piece = chunks_from_text(corpora[lang], rng, 1)[0]
            alphabet = "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
            sample = make_sample(
                piece, rng, world, filler_rate, alphabet, filler_families=filler_families
            )
            sample["language"] = lang
            store.append(sample)

    build(n_train, samples_train)
    build(n_val, samples_val)
    return samples_train, samples_val


def generate_finnish_holdout(
    finnish_text: str,
    n: int,
    seed: int,
    filler_rate: float = PRIMARY_FILLER_RATE,
    filler_families: tuple[str, ...] | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Unseen alphabet subset: take from the end of the pool / reshuffle with dedicated seed
    alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
    pieces = chunks_from_text(finnish_text, rng, n)
    out = []
    for piece in pieces:
        sample = make_sample(
            piece, rng, WORLD_C, filler_rate, alphabet, filler_families=filler_families
        )
        sample["language"] = "finnish"
        out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def batchify(samples: list[dict], vocab: dict[str, int], idxs: np.ndarray) -> dict:
    texts = [samples[i]["text"] for i in idxs]
    x = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long)
    feats = torch.tensor(np.stack([copy_aware_features(t) for t in texts]), dtype=torch.float32)
    mask = torch.tensor([samples[i]["mask"] for i in idxs], dtype=torch.float32)
    world = torch.tensor([samples[i]["world"] for i in idxs], dtype=torch.long)
    recon = x.clone()
    recon[mask < 0.5] = 0
    # Alignment targets: C(L) for world C; full surface for A/B; empty for D.
    targets: list[list[int]] = []
    for i in idxs:
        s = samples[int(i)]
        if s["world"] == WORLD_D:
            targets.append([])
        elif s["world"] == WORLD_C:
            targets.append(encode(s.get("ciphered") or "", vocab))
        else:
            targets.append(encode(s["text"], vocab))
    return {"x": x, "feats": feats, "mask": mask, "world": world, "recon": recon, "ctc_targets": targets}


@torch.no_grad()
def predict_masks(model: TinySignalModel, samples: list[dict], vocab: dict[str, int], device: str) -> list[np.ndarray]:
    model.eval()
    out = []
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start : start + BATCH_SIZE]
        texts = [s["text"] for s in batch]
        x = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long, device=device)
        feats = torch.tensor(np.stack([copy_aware_features(t) for t in texts]), dtype=torch.float32, device=device)
        logits, _, _ = model(x, feats)
        pred = (logits.sigmoid() >= 0.5).long().cpu().numpy()
        out.extend(pred)
    return out


@torch.no_grad()
def predict_mask_probs(model: TinySignalModel, samples: list[dict], vocab: dict[str, int], device: str) -> list[np.ndarray]:
    """Return P(signal) per position."""
    model.eval()
    out = []
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start : start + BATCH_SIZE]
        texts = [s["text"] for s in batch]
        x = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long, device=device)
        feats = torch.tensor(np.stack([copy_aware_features(t) for t in texts]), dtype=torch.float32, device=device)
        logits, _, _ = model(x, feats)
        out.extend(logits.sigmoid().cpu().numpy())
    return out


def calibrate_signal_threshold(
    probs: list[np.ndarray],
    target_null_rate: float = PRIMARY_FILLER_RATE,
) -> float:
    """Frozen calibration: choose signal threshold so mean pred null rate ≈ target."""
    flat = np.concatenate([np.asarray(p).reshape(-1) for p in probs])
    if flat.size == 0:
        return 0.5
    # pred null when p_signal < thr ⇒ null_rate = mean(p < thr)
    # Want thr such that mean(p < thr) ≈ target_null_rate
    thr = float(np.quantile(flat, target_null_rate))
    return float(np.clip(thr, 0.05, 0.95))


def probs_to_masks(probs: list[np.ndarray], signal_threshold: float) -> list[np.ndarray]:
    return [(np.asarray(p) >= signal_threshold).astype(int) for p in probs]


def evaluate_masks(
    samples: list[dict],
    pred_masks: list[np.ndarray],
    rank_model: dict,
    *,
    fixed_rank_model: FixedRankBigram | None = None,
    diagnostic_random_replicates: int = MATCHED_RANDOM_SEEDS,
) -> dict:
    """Legacy gate metrics plus additive diagnostics; no implicit new LM fit."""
    if not samples or len(samples) != len(pred_masks):
        raise ValueError("one prediction mask is required per nonempty sample list")
    f1s, accs, recons, gains, edit_sims = [], [], [], [], []
    reconstruction_rows, rank_diagnostics = [], []
    null_tps = null_fps = null_fns = null_tns = 0
    pred_nulls = true_nulls = 0
    total_tok = 0
    for s, pred in zip(samples, pred_masks):
        true = np.array(s["mask"], dtype=int)
        pred = np.asarray(pred, dtype=int)[: len(true)]
        if len(pred) < len(true):
            pred = np.pad(pred, (0, len(true) - len(pred)))
        true_c = s.get("ciphered") or "".join(ch for ch, m in zip(s["text"], true) if m == 1)
        f1s.append(f1_binary(true, pred))
        accs.append(float((true == pred).mean()))
        recons.append(recon_accuracy(true_c, s["text"], pred))
        diagnostic = reconstruction_diagnostics(true_c, s["text"], pred)
        reconstruction_rows.append(diagnostic)
        edit_sims.append(diagnostic["aligned_similarity"])
        if fixed_rank_model is not None:
            rank_diagnostics.append(fixed_rank_deletion_diagnostic(
                s["text"], pred, fixed_rank_model, n_random=diagnostic_random_replicates,
                seed=91021 + len(reconstruction_rows) - 1))
        # pred_bits_gain: lower bits than full text is good; report full_bits - selected_bits
        full_bits = bigram_bits(rank_bucket_sequence(s["text"]), rank_model)
        sel_bits = pred_bits_of_selected(s["text"], pred, rank_model)
        gains.append(full_bits - sel_bits)
        # Null class = 0 in mask. Exploratory for EXP-0011*; required in later ids.
        null_tps += int(((true == 0) & (pred == 0)).sum())
        null_fps += int(((true == 1) & (pred == 0)).sum())
        null_fns += int(((true == 0) & (pred == 1)).sum())
        null_tns += int(((true == 1) & (pred == 1)).sum())
        pred_nulls += int((pred == 0).sum())
        true_nulls += int((true == 0).sum())
        total_tok += len(true)
    null_prec = null_tps / max(null_tps + null_fps, 1)
    null_rec = null_tps / max(null_tps + null_fns, 1)
    reconstruction = summarize_reconstruction(reconstruction_rows)
    result = {
        "mask_f1": float(np.mean(f1s)),
        "mask_acc": float(np.mean(accs)),
        "recon_acc": float(np.mean(recons)),
        "recon_edit_sim": float(np.mean(edit_sims)),
        "recon_exact_match_rate": reconstruction["exact_sequence_match_rate"],
        "reconstruction_diagnostics": reconstruction,
        "pred_bits_gain": float(np.mean(gains)),
        "n": len(samples),
        "null_precision": float(null_prec),
        "null_recall": float(null_rec),
        "pred_null_rate": float(pred_nulls / max(total_tok, 1)),
        "true_null_rate": float(true_nulls / max(total_tok, 1)),
    }
    if fixed_rank_model is not None:
        result["fixed_rank_deletion_diagnostics"] = {
            "model": fixed_rank_model.metadata(), "per_sample": rank_diagnostics,
            "role": "Additive offline diagnostic; never read by historical pass rules.",
        }
    return result


def matched_random_baseline(samples: list[dict], rank_model: dict, n_seeds: int = MATCHED_RANDOM_SEEDS,
                            *, reference_masks: list[np.ndarray] | None = None) -> dict:
    """Legacy gold-count baseline, or opt-in prediction-count matching.

    Omitting reference_masks preserves historical random draws and gate meaning.
    Supplying masks matches each prediction's retained count instead; this is a
    separately named diagnostic, not a replacement for old archived baselines.
    """
    if not samples or not isinstance(n_seeds, int) or isinstance(n_seeds, bool) or n_seeds < 1:
        raise ValueError("nonempty samples and a positive integer seed count are required")
    if reference_masks is not None and len(reference_masks) != len(samples):
        raise ValueError("one reference mask is required per sample")
    reference_null_counts = None if reference_masks is None else [
        int((~binary_mask(sample["text"], mask)).sum())
        for sample, mask in zip(samples, reference_masks, strict=True)]
    f1s, recons, gains = [], [], []
    reconstruction_rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        pf1, pr, pg = [], [], []
        for sample_index, s in enumerate(samples):
            true = np.array(s["mask"], dtype=int)
            n_null = (int((true == 0).sum()) if reference_null_counts is None
                      else reference_null_counts[sample_index])
            pred = np.ones(len(true), dtype=int)
            if n_null > 0:
                idx = rng.choice(len(true), size=n_null, replace=False)
                pred[idx] = 0
            pf1.append(f1_binary(true, pred))
            target = s.get("ciphered") or "".join(
                ch for ch, m in zip(s["text"], true) if m == 1
            )
            pr.append(recon_accuracy(target, s["text"], pred))
            reconstruction_rows.append(reconstruction_diagnostics(target, s["text"], pred))
            full_bits = bigram_bits(rank_bucket_sequence(s["text"]), rank_model)
            sel_bits = pred_bits_of_selected(s["text"], pred, rank_model)
            pg.append(full_bits - sel_bits)
        f1s.append(float(np.mean(pf1)))
        recons.append(float(np.mean(pr)))
        gains.append(float(np.mean(pg)))
    reconstruction = summarize_reconstruction(reconstruction_rows)
    return {
        "mask_f1": float(np.mean(f1s)),
        "mask_f1_std": float(np.std(f1s)),
        "recon_acc": float(np.mean(recons)),
        "recon_edit_sim": reconstruction["aligned_similarity_mean"],
        "recon_exact_match_rate": reconstruction["exact_sequence_match_rate"],
        "reconstruction_diagnostics": reconstruction,
        "pred_bits_gain": float(np.mean(gains)),
        "matched_count_source": "gold_signal_mask" if reference_masks is None else "supplied_prediction_mask",
        "n_samples": len(samples),
        "n_random_seeds": n_seeds,
    }


def majority_baseline(samples: list[dict], rank_model: dict) -> dict:
    preds = [np.ones(len(s["mask"]), dtype=int) for s in samples]
    return evaluate_masks(samples, preds, rank_model)


def vocab_filter_baseline(samples: list[dict], train_token_vocab: set[str], rank_model: dict) -> dict:
    """Delete space-delimited tokens absent from training world-B/C surface token vocabularies."""
    preds = []
    for s in samples:
        text = s["text"]
        pred = np.ones(len(text), dtype=int)
        for part in re.finditer(r"\S+", text):
            tok = part.group()
            if tok not in train_token_vocab:
                pred[part.start() : part.end()] = 0
        preds.append(pred)
    return evaluate_masks(samples, preds, rank_model)


def train_model(
    train: list[dict],
    val: list[dict],
    vocab: dict[str, int],
    device: str,
    updates: int = MAX_UPDATES,
    balanced_null_loss: bool = False,
    alignment_ctc: bool = False,
) -> tuple[TinySignalModel, dict]:
    torch.manual_seed(MODEL_SEED)
    model = TinySignalModel(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    rng = np.random.default_rng(MODEL_SEED)
    # Index pools for world-C-focused batching
    by_world = {w: [i for i, s in enumerate(train) if s["world"] == w] for w in range(4)}
    for w, idxs in by_world.items():
        if not idxs:
            by_world[w] = list(range(len(train)))
    val_c = [s for s in val if s["world"] == WORLD_C] or val
    vocab_size = len(vocab)
    model.train()
    t0 = time.time()
    for step in range(1, updates + 1):
        # 70% world C, remainder mixed A/B/D (Naibbe trap retained)
        n_c = int(0.70 * BATCH_SIZE)
        idxs_c = rng.choice(by_world[WORLD_C], size=n_c, replace=True)
        other_pool = by_world[WORLD_A] + by_world[WORLD_B] + by_world[WORLD_D]
        idxs_o = rng.choice(other_pool, size=BATCH_SIZE - n_c, replace=True)
        idxs = np.concatenate([idxs_c, idxs_o])
        rng.shuffle(idxs)
        batch = batchify(train, vocab, idxs)
        x = batch["x"].to(device)
        feats = batch["feats"].to(device)
        mask = batch["mask"].to(device)
        world = batch["world"].to(device)
        recon = batch["recon"].to(device)
        mask_logit, recon_logit, world_logit = model(x, feats)
        if alignment_ctc:
            # Primary: kept subsequence must CTC-align to C(L) (copy-constrained).
            loss_ctc = ctc_deletion_loss(mask_logit, x, batch["ctc_targets"], vocab_size)
            w = torch.where(mask > 0.5, torch.ones_like(mask), torch.full_like(mask, NULL_LOSS_WEIGHT))
            loss_mask = F.binary_cross_entropy_with_logits(mask_logit, mask, weight=w)
            # Rate regularizer on world-C rows only
            c_rows = world == WORLD_C
            if c_rows.any():
                pred_null = torch.sigmoid(-mask_logit[c_rows]).mean()
                loss_rate = (pred_null - PRIMARY_FILLER_RATE) ** 2
            else:
                loss_rate = mask_logit.new_zeros(())
            loss_world = F.cross_entropy(world_logit, world)
            loss = (
                ALIGN_CTC_WEIGHT * loss_ctc
                + ALIGN_MASK_BCE_WEIGHT * loss_mask
                + ALIGN_RATE_WEIGHT * loss_rate
                + ALIGN_WORLD_WEIGHT * loss_world
            )
        elif balanced_null_loss:
            # Upweight null positions (mask==0) so delete-nothing cannot win the loss.
            w = torch.where(mask > 0.5, torch.ones_like(mask), torch.full_like(mask, NULL_LOSS_WEIGHT))
            loss_mask = F.binary_cross_entropy_with_logits(mask_logit, mask, weight=w)
            flat_logits = recon_logit.reshape(-1, recon_logit.size(-1))
            flat_tgt = recon.reshape(-1)
            flat_w = mask.reshape(-1)
            loss_recon = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
            loss_recon = (loss_recon * flat_w).sum() / flat_w.sum().clamp_min(1.0)
            loss_world = F.cross_entropy(world_logit, world)
            loss = loss_mask + 0.5 * loss_recon + 0.15 * loss_world
        else:
            loss_mask = F.binary_cross_entropy_with_logits(mask_logit, mask)
            flat_logits = recon_logit.reshape(-1, recon_logit.size(-1))
            flat_tgt = recon.reshape(-1)
            flat_w = mask.reshape(-1)
            loss_recon = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
            loss_recon = (loss_recon * flat_w).sum() / flat_w.sum().clamp_min(1.0)
            loss_world = F.cross_entropy(world_logit, world)
            loss = loss_mask + 0.5 * loss_recon + 0.15 * loss_world
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == updates:
            rank_model = fit_rank_bigram(
                [s["text"] for s in train if s["world"] == WORLD_B][:200] or [s["text"] for s in train[:200]]
            )
            if alignment_ctc or balanced_null_loss:
                probs = predict_mask_probs(model, val_c, vocab, device)
                thr = calibrate_signal_threshold(probs, PRIMARY_FILLER_RATE)
                preds = probs_to_masks(probs, thr)
                metrics = evaluate_masks(val_c, preds, rank_model)
                # EXP-0013 checkpoints on recon_acc (localization); 0012 on null F1.
                if alignment_ctc:
                    score = metrics["recon_acc"]
                    metrics["checkpoint_metric"] = "recon_acc"
                else:
                    score = null_f1_from_metrics(metrics)
                    metrics["checkpoint_metric"] = "null_f1"
                metrics["signal_threshold"] = thr
                metrics["null_f1"] = null_f1_from_metrics(metrics)
            else:
                preds = predict_masks(model, val_c, vocab, device)
                metrics = evaluate_masks(val_c, preds, rank_model)
                score = metrics["mask_f1"]
                thr = 0.5
                metrics["checkpoint_metric"] = "signal_mask_f1"
            metrics["step"] = step
            metrics["loss"] = float(loss.item())
            metrics["val_slice"] = "world_C_only"
            history.append(metrics)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_threshold = thr
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    if alignment_ctc:
        fix = "exp0013_ctc_deletion_alignment_checkpoint_recon_acc_calibrated_threshold"
    elif balanced_null_loss:
        fix = "exp0012_balanced_null_loss_checkpoint_null_f1_calibrated_threshold"
    else:
        fix = "worldC_oversample_copy_features_checkpoint_on_worldC_val"
    summary = {
        "param_count": model.param_count(),
        "best_val_score": best_score,
        "best_val_mask_f1": history[-1]["mask_f1"] if history else -1.0,
        "signal_threshold": best_threshold,
        "updates": updates,
        "seconds": time.time() - t0,
        "history": history,
        "device": device,
        "fix": fix,
        "balanced_null_loss": balanced_null_loss,
        "alignment_ctc": alignment_ctc,
        "null_loss_weight": NULL_LOSS_WEIGHT if (balanced_null_loss or alignment_ctc) else 1.0,
    }
    return model, summary


def apply_pass_rule(neural: dict, classical: dict, majority: dict, random_b: dict, vocab_b: dict) -> dict:
    def check(name: str, m: dict) -> dict:
        reasons = []
        ok = True
        if m["mask_f1"] < PASS_MASK_F1_MIN:
            ok = False
            reasons.append(f"mask_f1 {m['mask_f1']:.4f} < {PASS_MASK_F1_MIN}")
        if m["mask_acc"] < majority["mask_acc"] + PASS_MARGIN_VS_WEAK:
            ok = False
            reasons.append(
                f"mask_acc {m['mask_acc']:.4f} not >= majority_acc+{PASS_MARGIN_VS_WEAK} "
                f"(majority_acc={majority['mask_acc']:.4f})"
            )
        if m["mask_f1"] < random_b["mask_f1"] + PASS_MARGIN_VS_WEAK:
            ok = False
            reasons.append(
                f"mask_f1 {m['mask_f1']:.4f} not >= matched_random_f1+{PASS_MARGIN_VS_WEAK} "
                f"(random_f1={random_b['mask_f1']:.4f})"
            )
        if m["mask_f1"] < vocab_b["mask_f1"] + PASS_MARGIN_VS_VOCAB:
            ok = False
            reasons.append("mask_f1 not above vocab filter margin")
        if vocab_b["mask_f1"] > PASS_VOCAB_F1_MAX:
            ok = False
            reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS_VOCAB_F1_MAX} (cheat surface)")
        if m["recon_acc"] < random_b["recon_acc"] + PASS_RECON_MARGIN:
            ok = False
            reasons.append("recon_acc margin vs matched random failed")
        if m["pred_bits_gain"] < random_b["pred_bits_gain"] + PASS_PRED_BITS_MARGIN:
            ok = False
            reasons.append("pred_bits_gain margin vs matched random failed")
        return {"model": name, "pass": ok, "reasons": reasons, "metrics": m}

    n_res = check("neural", neural)
    c_res = check("classical", classical)
    passed = n_res["pass"] or c_res["pass"]
    winner = None
    if n_res["pass"] and c_res["pass"]:
        winner = "classical" if classical["mask_f1"] >= neural["mask_f1"] else "neural"
    elif n_res["pass"]:
        winner = "neural"
    elif c_res["pass"]:
        winner = "classical"
    return {
        "passed": passed,
        "winner": winner,
        "neural_check": n_res,
        "classical_check": c_res,
        "thresholds": {
            "mask_f1_min": PASS_MASK_F1_MIN,
            "margin_vs_weak": PASS_MARGIN_VS_WEAK,
            "margin_vs_vocab": PASS_MARGIN_VS_VOCAB,
            "vocab_f1_max": PASS_VOCAB_F1_MAX,
            "recon_margin": PASS_RECON_MARGIN,
            "pred_bits_margin": PASS_PRED_BITS_MARGIN,
        },
    }


def apply_pass_rule_v12(neural: dict, classical: dict, majority: dict, random_b: dict, vocab_b: dict) -> dict:
    """EXP-0012 rule: null-class recall/precision + recon; blocks delete-nothing."""

    def check(name: str, m: dict) -> dict:
        reasons = []
        ok = True
        nr = float(m.get("null_recall", 0.0))
        np_ = float(m.get("null_precision", 0.0))
        pr = float(m.get("pred_null_rate", 0.0))
        if nr < PASS12_NULL_RECALL_MIN:
            ok = False
            reasons.append(f"null_recall {nr:.4f} < {PASS12_NULL_RECALL_MIN}")
        if np_ < PASS12_NULL_PRECISION_MIN:
            ok = False
            reasons.append(f"null_precision {np_:.4f} < {PASS12_NULL_PRECISION_MIN}")
        if not (PASS12_PRED_NULL_RATE_MIN <= pr <= PASS12_PRED_NULL_RATE_MAX):
            ok = False
            reasons.append(
                f"pred_null_rate {pr:.4f} not in "
                f"[{PASS12_PRED_NULL_RATE_MIN}, {PASS12_PRED_NULL_RATE_MAX}]"
            )
        if m["recon_acc"] < random_b["recon_acc"] + PASS12_RECON_MARGIN:
            ok = False
            reasons.append("recon_acc margin vs matched random failed")
        if m["mask_acc"] < majority["mask_acc"] + PASS12_MASK_ACC_MARGIN:
            ok = False
            reasons.append(
                f"mask_acc {m['mask_acc']:.4f} not >= majority_acc+{PASS12_MASK_ACC_MARGIN} "
                f"(majority_acc={majority['mask_acc']:.4f})"
            )
        if vocab_b["mask_f1"] > PASS12_VOCAB_F1_MAX:
            ok = False
            reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS12_VOCAB_F1_MAX} (cheat surface)")
        if m["pred_bits_gain"] < random_b["pred_bits_gain"] + PASS12_PRED_BITS_MARGIN:
            ok = False
            reasons.append("pred_bits_gain margin vs matched random failed")
        return {
            "model": name,
            "pass": ok,
            "reasons": reasons,
            "metrics": m,
            "null_f1": null_f1_from_metrics(m),
        }

    n_res = check("neural", neural)
    c_res = check("classical", classical)
    passed = n_res["pass"] or c_res["pass"]
    winner = None
    if n_res["pass"] and c_res["pass"]:
        winner = "classical" if c_res["null_f1"] >= n_res["null_f1"] else "neural"
    elif n_res["pass"]:
        winner = "neural"
    elif c_res["pass"]:
        winner = "classical"
    return {
        "passed": passed,
        "winner": winner,
        "rule": "EXP-0012",
        "neural_check": n_res,
        "classical_check": c_res,
        "thresholds": {
            "null_recall_min": PASS12_NULL_RECALL_MIN,
            "null_precision_min": PASS12_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS12_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS12_PRED_NULL_RATE_MAX,
            "recon_margin": PASS12_RECON_MARGIN,
            "mask_acc_margin": PASS12_MASK_ACC_MARGIN,
            "pred_bits_margin": PASS12_PRED_BITS_MARGIN,
            "vocab_f1_max": PASS12_VOCAB_F1_MAX,
        },
    }


def apply_pass_rule_v13(neural: dict, classical: dict, majority: dict, random_b: dict, vocab_b: dict) -> dict:
    """EXP-0013 rule: null gates + recon must strictly beat matched-random deletion."""

    def check(name: str, m: dict) -> dict:
        reasons = []
        ok = True
        nr = float(m.get("null_recall", 0.0))
        np_ = float(m.get("null_precision", 0.0))
        pr = float(m.get("pred_null_rate", 0.0))
        if nr < PASS13_NULL_RECALL_MIN:
            ok = False
            reasons.append(f"null_recall {nr:.4f} < {PASS13_NULL_RECALL_MIN}")
        if np_ < PASS13_NULL_PRECISION_MIN:
            ok = False
            reasons.append(f"null_precision {np_:.4f} < {PASS13_NULL_PRECISION_MIN}")
        if not (PASS13_PRED_NULL_RATE_MIN <= pr <= PASS13_PRED_NULL_RATE_MAX):
            ok = False
            reasons.append(
                f"pred_null_rate {pr:.4f} not in "
                f"[{PASS13_PRED_NULL_RATE_MIN}, {PASS13_PRED_NULL_RATE_MAX}]"
            )
        if m["recon_acc"] <= random_b["recon_acc"]:
            ok = False
            reasons.append(
                f"recon_acc {m['recon_acc']:.4f} not > matched_random "
                f"{random_b['recon_acc']:.4f}"
            )
        if vocab_b["mask_f1"] > PASS13_VOCAB_F1_MAX:
            ok = False
            reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS13_VOCAB_F1_MAX} (cheat surface)")
        return {
            "model": name,
            "pass": ok,
            "reasons": reasons,
            "metrics": m,
            "null_f1": null_f1_from_metrics(m),
        }

    n_res = check("neural", neural)
    c_res = check("classical", classical)
    passed = n_res["pass"] or c_res["pass"]
    winner = None
    if n_res["pass"] and c_res["pass"]:
        winner = "classical" if classical["recon_acc"] >= neural["recon_acc"] else "neural"
    elif n_res["pass"]:
        winner = "neural"
    elif c_res["pass"]:
        winner = "classical"
    return {
        "passed": passed,
        "winner": winner,
        "rule": "EXP-0013",
        "neural_check": n_res,
        "classical_check": c_res,
        "thresholds": {
            "null_recall_min": PASS13_NULL_RECALL_MIN,
            "null_precision_min": PASS13_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
            "recon_vs_matched_random": "strictly_greater",
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        },
        "note": "majority/mask_acc and bits_gain reported but not pass criteria in EXP-0013",
        "majority_acc_ref": majority.get("mask_acc"),
    }


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

CORPUS_SOURCES = {
    "english": {
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "license": "Project Gutenberg license (public domain in the USA)",
        "language": "english",
    },
    "latin": {
        "url": "https://www.gutenberg.org/files/213/213-0.txt",
        "title": "The History of Rome, Book I (Latin?)",
        "author": "Livy (check)",
        "license": "Project Gutenberg license (public domain in the USA)",
        "language": "latin",
    },
    "finnish": {
        "url": "https://www.gutenberg.org/files/7000/7000-0.txt",
        "title": "Kalevala (check after download)",
        "author": "check",
        "license": "Project Gutenberg license (public domain in the USA)",
        "language": "finnish",
    },
}


def download_corpora(raw_root: Path) -> dict:
    import urllib.request

    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}, "files": {}}
    # Prefer known-good PG text IDs; fall back to alternate URLs if needed.
    sources = {
        "english": {
            "urls": [
                "https://www.gutenberg.org/cache/epub/11/pg11.txt",
                "https://www.gutenberg.org/ebooks/11.txt.utf-8",
                "https://www.gutenberg.org/files/11/11-0.txt",
            ],
            "title": "Alice's Adventures in Wonderland",
            "author": "Lewis Carroll",
            "license": "Project Gutenberg License; public domain in the USA",
        },
        "latin": {
            "urls": [
                "https://www.gutenberg.org/cache/epub/218/pg218.txt",
                "https://www.gutenberg.org/ebooks/218.txt.utf-8",
            ],
            "title": "C. Iuli Caesaris De Bello Gallico, I-IV (Latin)",
            "author": "Julius Caesar",
            "license": "Project Gutenberg License; public domain in the USA",
        },
        "finnish": {
            "urls": [
                "https://www.gutenberg.org/cache/epub/7000/pg7000.txt",
                "https://www.gutenberg.org/ebooks/7000.txt.utf-8",
            ],
            "title": "Kalevala (Finnish, 1849)",
            "author": "Elias Lönnrot (compiler)",
            "license": "Project Gutenberg License; public domain in the USA",
        },
    }
    texts = {}
    for lang, meta in sources.items():
        dest = raw_root / f"{lang}.txt"
        if dest.exists() and dest.stat().st_size > 1000:
            raw = dest.read_text(encoding="utf-8", errors="replace")
        else:
            raw = None
            last_err = None
            for url in meta["urls"]:
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                    dest.write_text(raw, encoding="utf-8")
                    meta = {**meta, "url_used": url}
                    break
                except Exception as exc:  # noqa: BLE001 — try next mirror
                    last_err = exc
            if raw is None:
                raise RuntimeError(f"Failed to download {lang}: {last_err}")
        cleaned = clean_plaintext(raw)
        clean_path = raw_root / f"{lang}.clean.txt"
        clean_path.write_text(cleaned, encoding="utf-8")
        texts[lang] = cleaned
        manifest["sources"][lang] = {
            "title": meta["title"],
            "author": meta["author"],
            "license": meta["license"],
            "url_used": meta.get("url_used", "local-cache"),
            "raw_sha256": digest(dest),
            "clean_sha256": digest(clean_path),
            "raw_bytes": dest.stat().st_size,
            "clean_chars": len(cleaned),
        }
        # Language sanity: Finnish should have ä/ö more than English after cleaning we stripped them...
        # Our cleaner maps to latin a-z only, so Finnish is still Finnish vocabulary/syntax without diacritics.
    return {"texts": texts, "manifest": manifest}


# ---------------------------------------------------------------------------
# Voynich label-free (gated)
# ---------------------------------------------------------------------------


def label_free_voynich(pages: list[str], rank_model: dict, classical_rate: float = 0.30) -> dict:
    results = []
    rng = np.random.default_rng(9011)
    for text in pages:
        text = text[:512]
        if len(text) < 32:
            continue
        pred = classical_null_mask(text, classical_rate)
        full_bits = bigram_bits(rank_bucket_sequence(text), rank_model)
        sel_bits = pred_bits_of_selected(text, pred, rank_model)
        # matched random
        n_null = int((pred == 0).sum())
        rand_gains = []
        for s in range(20):
            r = np.random.default_rng(9011 + s)
            m = np.ones(len(text), dtype=int)
            if n_null:
                m[r.choice(len(text), size=min(n_null, len(text)), replace=False)] = 0
            rb = pred_bits_of_selected(text, m, rank_model)
            rand_gains.append(full_bits - rb)
        results.append({
            "full_bits": full_bits,
            "selected_bits": sel_bits,
            "gain": full_bits - sel_bits,
            "random_gain_mean": float(np.mean(rand_gains)),
            "null_fraction": float(1 - pred.mean()),
            "beats_random": (full_bits - sel_bits) > float(np.mean(rand_gains)) + 0.05,
        })
    if not results:
        return {"n": 0, "fraction_beats_random": 0.0}
    return {
        "n": len(results),
        "mean_gain": float(np.mean([r["gain"] for r in results])),
        "mean_random_gain": float(np.mean([r["random_gain_mean"] for r in results])),
        "fraction_beats_random": float(np.mean([r["beats_random"] for r in results])),
        "pages": results[:20],  # compact
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    experiment_id = str(getattr(args, "experiment_id", None) or "EXP-0011")
    filler_families = parse_filler_families(getattr(args, "filler_families", None))
    eid = experiment_id.upper().replace("_", "")
    use_v13 = eid in {"EXP-0013", "EXP0013", "EXP-0013B", "EXP0013B"}
    use_v12 = eid in {"EXP-0012", "EXP0012"} and not use_v13
    slug = experiment_id.lower().replace("_", "").replace("-", "")
    # Keep readable dirs: exp0012 / exp0011a / exp0013
    if experiment_id.upper().startswith("EXP-"):
        slug = experiment_id.lower().replace("-", "")
    raw_root = root / "data" / "raw" / "latent_corpora"
    proc_root = root / "data" / "processed" / slug
    out_root = root / "outputs" / experiment_id
    res_root = root / "results" / experiment_id
    for p in (proc_root, out_root, res_root, root / "data" / "manifests"):
        p.mkdir(parents=True, exist_ok=True)

    downloaded = download_corpora(raw_root)
    # Corpora manifest shared; per-run data manifest is experiment-specific.
    write_json(root / "data" / "manifests" / "exp0011_corpora.json", downloaded["manifest"])
    texts = downloaded["texts"]
    for required in ("english", "latin", "finnish"):
        if required not in texts or len(texts[required]) < 5000:
            raise RuntimeError(f"Corpus {required} missing or too short ({len(texts.get(required, ''))})")

    train, val = generate_dataset(
        {"english": texts["english"], "latin": texts["latin"]},
        n_train=args.n_train,
        n_val=args.n_val,
        seed=DATA_SEED,
        filler_rate=PRIMARY_FILLER_RATE,
        filler_families=filler_families,
    )
    holdout = generate_finnish_holdout(
        texts["finnish"],
        n=args.n_holdout,
        seed=FINNISH_SEED,
        filler_families=filler_families,
    )
    # Persist compact derived samples (not huge)
    def dump(name, rows):
        path = proc_root / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "world": r["world"],
                    "language": r.get("language"),
                    "text": r["text"],
                    "mask": r["mask"],
                    "ciphered": r.get("ciphered", ""),
                    "filler_family": r.get("filler_family", ""),
                    "filler_rate": r.get("filler_rate", 0),
                }, ensure_ascii=False) + "\n")
        return digest(path)

    digests = {
        "train": dump("train.jsonl", train),
        "validation": dump("validation.jsonl", val),
        "finnish_holdout": dump("finnish_holdout.jsonl", holdout),
    }
    if use_v13:
        pass_rule_meta = asdict_pass_thresholds_v13()
        pass_rule_source = "EXP-0013"
    elif use_v12:
        pass_rule_meta = asdict_pass_thresholds_v12()
        pass_rule_source = "EXP-0012"
    else:
        pass_rule_meta = asdict_pass_thresholds()
        pass_rule_source = "EXP-0011 frozen thresholds (unchanged)"
    write_json(root / "data" / "manifests" / f"{slug}_data.json", {
        "experiment": experiment_id,
        "data_seed": DATA_SEED,
        "finnish_seed": FINNISH_SEED,
        "filler_rate_primary": PRIMARY_FILLER_RATE,
        "filler_families": list(filler_families),
        "seq_len": SEQ_LEN,
        "n_train": len(train),
        "n_val": len(val),
        "n_holdout": len(holdout),
        "worlds": WORLD_NAMES,
        "derived_sha256": digests,
        "pass_rule": pass_rule_meta,
        "pass_rule_source": pass_rule_source,
        "escalation": use_v12 or use_v13,
        "alignment_ctc": use_v13,
    })

    vocab = build_vocab()
    device = resolve_device(args.device)
    # Warm rank model from training world-B surfaces
    world_b_texts = [s["text"] for s in train if s["world"] == WORLD_B]
    rank_model = fit_rank_bigram(world_b_texts[:400] or [s["text"] for s in train[:400]])

    model, train_summary = train_model(
        train,
        val,
        vocab,
        device,
        updates=args.updates,
        balanced_null_loss=use_v12,
        alignment_ctc=use_v13,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "summary": train_summary,
            "signal_threshold": train_summary.get("signal_threshold", 0.5),
        },
        out_root / "model.pt",
    )

    # Holdout evaluation — first time looking at Finnish metrics for this run id
    if use_v12 or use_v13:
        holdout_probs = predict_mask_probs(model, holdout, vocab, device)
        thr = float(train_summary.get("signal_threshold", 0.5))
        neural_preds = probs_to_masks(holdout_probs, thr)
        classical_preds = [classical_hsmm_null_mask(s["text"], PRIMARY_FILLER_RATE) for s in holdout]
    else:
        neural_preds = predict_masks(model, holdout, vocab, device)
        classical_preds = [classical_null_mask(s["text"], PRIMARY_FILLER_RATE) for s in holdout]
    neural_metrics = evaluate_masks(holdout, neural_preds, rank_model)
    classical_metrics = evaluate_masks(holdout, classical_preds, rank_model)
    neural_metrics["null_f1"] = null_f1_from_metrics(neural_metrics)
    classical_metrics["null_f1"] = null_f1_from_metrics(classical_metrics)
    if use_v12 or use_v13:
        neural_metrics["signal_threshold"] = thr
    # Teacher-forced secondary (gold mask) — sanity only; not a pass criterion.
    if use_v13:
        gold_preds = [np.array(s["mask"], dtype=int) for s in holdout]
        tf_metrics = evaluate_masks(holdout, gold_preds, rank_model)
        neural_metrics["teacher_forced_recon_acc"] = tf_metrics["recon_acc"]
        neural_metrics["teacher_forced_recon_edit_sim"] = tf_metrics["recon_edit_sim"]
    majority = majority_baseline(holdout, rank_model)
    random_b = matched_random_baseline(holdout, rank_model)
    # Training surface token vocab from world B/C only (ciphered surfaces)
    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)

    if use_v13:
        decision = apply_pass_rule_v13(neural_metrics, classical_metrics, majority, random_b, vocab_b)
    elif use_v12:
        decision = apply_pass_rule_v12(neural_metrics, classical_metrics, majority, random_b, vocab_b)
    else:
        decision = apply_pass_rule(neural_metrics, classical_metrics, majority, random_b, vocab_b)

    voynich_result = None
    if decision["passed"] and args.voynich_if_pass:
        # Load ZL3b validation only
        val_path = root / "data" / "processed" / "zl3b" / "validation.jsonl"
        pages = []
        if val_path.exists():
            for line in val_path.read_text().splitlines():
                if not line.strip():
                    continue
                page = json.loads(line)
                # glyph stream: strip PUA
                text = re.sub(r"[\ue000-\uf8ff]", "", page["text"]).lower()
                pages.append(text)
        # Fit rank model on train ZL3b for bits measure (structure relative to manuscript train)
        train_path = root / "data" / "processed" / "zl3b" / "train.jsonl"
        train_pages = []
        if train_path.exists():
            for line in train_path.read_text().splitlines():
                if line.strip():
                    train_pages.append(re.sub(r"[\ue000-\uf8ff]", "", json.loads(line)["text"]).lower())
        v_rank = fit_rank_bigram(train_pages[:177] or pages)
        voynich_result = {
            "glyph": label_free_voynich(pages, v_rank),
            "token": label_free_voynich(
                [" ".join(re.findall(r"[a-z]+", p)) for p in pages],
                v_rank,
            ),
            "note": "Label-free structure test on ZL3b validation only; not a decipherment. Test split untouched.",
        }

    if use_v13:
        parent_rule = "EXP-0013"
        classical_note = (
            "Classical is sticky-null Viterbi with periodic/rarity emissions "
            "(path kept; no top-k overwrite)"
        )
        extra_notes = [
            "EXP-0013 copy-constrained CTC deletion; checkpoint on val recon_acc; "
            "val-calibrated signal threshold; free-running mask is primary",
            "Teacher-forced gold-mask recon reported as secondary sanity only",
            "Secondary recon_edit_sim = 1 - Levenshtein/max_len (non-deciding)",
        ]
    elif use_v12:
        parent_rule = "EXP-0012"
        classical_note = (
            "Classical is sticky-null Viterbi with periodic/rarity emissions "
            "(path kept; no top-k overwrite)"
        )
        extra_notes = [
            f"EXP-0012 null-loss weight {NULL_LOSS_WEIGHT}; checkpoint on null F1; "
            "val-calibrated signal threshold",
        ]
    else:
        parent_rule = "EXP-0011"
        classical_note = (
            "Classical model is a 2-state feature Viterbi with token-level "
            "copy/mutate scores, not a full 20-state HSMM"
        )
        extra_notes = []

    report = {
        "experiment": experiment_id,
        "parent_pass_rule": parent_rule,
        "filler_families": list(filler_families),
        "environment": environment(),
        "train_summary": {k: v for k, v in train_summary.items() if k != "history"},
        "train_history_tail": train_summary["history"][-5:],
        "param_count": train_summary["param_count"],
        "holdout_language": "finnish",
        "filler_rate": PRIMARY_FILLER_RATE,
        "neural": neural_metrics,
        "classical": classical_metrics,
        "baselines": {
            "majority": majority,
            "matched_random": random_b,
            "vocab_filter": vocab_b,
        },
        "decision": decision,
        "voynich_label_free": voynich_result,
        "data_digests": digests,
        "simplifications": [
            "Character-level BiLSTM with copy-aware side features (~tens of k params), not a large transformer",
            "Primary reconstruction target is C(L) via mask deletion, not full plaintext cryptanalysis",
            classical_note,
            "Finnish diacritics folded to ASCII a-z by shared cleaner (syntax/vocab still Finnish)",
            "Single primary filler rate 0.30 in the scored holdout",
            f"Ablation filler families: {list(filler_families)}",
            *extra_notes,
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
    return report


def asdict_pass_thresholds() -> dict:
    return {
        "mask_f1_min": PASS_MASK_F1_MIN,
        "margin_vs_weak": PASS_MARGIN_VS_WEAK,
        "margin_vs_vocab": PASS_MARGIN_VS_VOCAB,
        "vocab_f1_max": PASS_VOCAB_F1_MAX,
        "recon_margin": PASS_RECON_MARGIN,
        "pred_bits_margin": PASS_PRED_BITS_MARGIN,
    }


def asdict_pass_thresholds_v12() -> dict:
    return {
        "null_recall_min": PASS12_NULL_RECALL_MIN,
        "null_precision_min": PASS12_NULL_PRECISION_MIN,
        "pred_null_rate_min": PASS12_PRED_NULL_RATE_MIN,
        "pred_null_rate_max": PASS12_PRED_NULL_RATE_MAX,
        "recon_margin": PASS12_RECON_MARGIN,
        "mask_acc_margin": PASS12_MASK_ACC_MARGIN,
        "pred_bits_margin": PASS12_PRED_BITS_MARGIN,
        "vocab_f1_max": PASS12_VOCAB_F1_MAX,
    }


def asdict_pass_thresholds_v13() -> dict:
    return {
        "null_recall_min": PASS13_NULL_RECALL_MIN,
        "null_precision_min": PASS13_NULL_PRECISION_MIN,
        "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
        "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
        "recon_vs_matched_random": "strictly_greater",
        "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        "primary_metric": "recon_acc (prefix×length_pen; unchanged)",
        "secondary_metrics": ["recon_edit_sim", "mask_acc", "pred_bits_gain", "teacher_forced_recon_acc"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-holdout", type=int, default=300)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--experiment-id", default="EXP-0011")
    parser.add_argument(
        "--filler-families",
        default="",
        help="Comma-separated subset of FILLER_FAMILIES; empty = full EXP-0011 set",
    )
    parser.add_argument("--voynich-if-pass", action="store_true", default=True)
    parser.add_argument("--no-voynich", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.no_voynich:
        args.voynich_if_pass = False
    if args.download_only:
        raw_root = Path(args.root) / "data" / "raw" / "latent_corpora"
        downloaded = download_corpora(raw_root)
        write_json(Path(args.root) / "data" / "manifests" / "exp0011_corpora.json", downloaded["manifest"])
        print(json.dumps(downloaded["manifest"], indent=2)[:2000])
        return
    report = run_experiment(args)
    print(json.dumps({
        "experiment": report["experiment"],
        "filler_families": report["filler_families"],
        "passed": report["decision"]["passed"],
        "winner": report["decision"]["winner"],
        "param_count": report["param_count"],
        "neural": {
            "mask_f1": report["neural"]["mask_f1"],
            "mask_acc": report["neural"]["mask_acc"],
            "recon_acc": report["neural"]["recon_acc"],
            "recon_edit_sim": report["neural"].get("recon_edit_sim"),
            "pred_bits_gain": report["neural"]["pred_bits_gain"],
            "null_recall": report["neural"].get("null_recall"),
            "null_precision": report["neural"].get("null_precision"),
            "null_f1": report["neural"].get("null_f1"),
            "pred_null_rate": report["neural"].get("pred_null_rate"),
            "teacher_forced_recon_acc": report["neural"].get("teacher_forced_recon_acc"),
        },
        "classical": {
            "mask_f1": report["classical"]["mask_f1"],
            "mask_acc": report["classical"]["mask_acc"],
            "recon_acc": report["classical"]["recon_acc"],
            "recon_edit_sim": report["classical"].get("recon_edit_sim"),
            "pred_bits_gain": report["classical"]["pred_bits_gain"],
            "null_recall": report["classical"].get("null_recall"),
            "null_precision": report["classical"].get("null_precision"),
            "null_f1": report["classical"].get("null_f1"),
            "pred_null_rate": report["classical"].get("pred_null_rate"),
        },
        "baselines": {
            "majority_f1": report["baselines"]["majority"]["mask_f1"],
            "matched_random_f1": report["baselines"]["matched_random"]["mask_f1"],
            "matched_random_recon": report["baselines"]["matched_random"]["recon_acc"],
            "vocab_filter_f1": report["baselines"]["vocab_filter"]["mask_f1"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
