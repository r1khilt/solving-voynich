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
    families = list(FILLER_FAMILIES)
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


def make_sample(
    plaintext: str,
    rng: np.random.Generator,
    world: int,
    filler_rate: float,
    alphabet: str | None = None,
) -> dict:
    alphabet = alphabet or "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
    if world == WORLD_A:
        text = plaintext.replace("\n", " ")
        text = re.sub(r" {2,}", " ", text)[:SEQ_LEN].ljust(SEQ_LEN)[:SEQ_LEN]
        mask = [1] * len(text)
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
        text = ciphered[:SEQ_LEN].ljust(SEQ_LEN)[:SEQ_LEN]
        mask = [1] * len(text)
        family = ""
        rate = 0.0
        kept = text
    else:
        noisy, mask_full, fams = insert_nulls(ciphered, rng, filler_rate, alphabet)
        # Truncate/pad to SEQ_LEN
        if len(noisy) >= SEQ_LEN:
            text = noisy[:SEQ_LEN]
            mask = mask_full[:SEQ_LEN]
            family = next((f for f in fams[:SEQ_LEN] if f), "mixed")
        else:
            text = noisy.ljust(SEQ_LEN)[:SEQ_LEN]
            mask = mask_full + [1] * (SEQ_LEN - len(mask_full))
            family = next((f for f in fams if f), "mixed")
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
    pred = "".join(ch for ch, m in zip(text, pred_mask) if m == 1)
    n = min(len(pred), len(true_ciphered))
    if n == 0:
        return 1.0 if true_ciphered == pred else 0.0
    matches = sum(a == b for a, b in zip(pred[:n], true_ciphered[:n]))
    # Penalize length mismatch
    length_pen = min(len(pred), len(true_ciphered)) / max(len(pred), len(true_ciphered), 1)
    return float(matches / n * length_pen)


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


def classical_null_mask(text: str, filler_rate: float = PRIMARY_FILLER_RATE) -> np.ndarray:
    """Interpretable signal/null decode. ~features, not neural."""
    n = len(text)
    if n == 0:
        return np.zeros(0, dtype=int)
    # Emission scores for SIGNAL vs NULL at each position
    recent: list[str] = []
    emit_s = np.zeros(n)
    emit_n = np.zeros(n)
    from collections import Counter

    counts = Counter(text)
    ranked = {ch: i for i, (ch, _) in enumerate(counts.most_common())}
    for i, ch in enumerate(text):
        rank = ranked.get(ch, n)
        # Language-like: mid-frequency ranks preferred over rare continuous noise
        zipf_s = -abs(rank - 3) * 0.15
        copy = 0.0
        window = recent[-16:]
        if ch in window:
            copy += 1.2
        # single-edit proximity to a recent char
        for prev in window[-8:]:
            if prev != ch and len(prev) == 1 and len(ch) == 1:
                copy += 0.15
        periodic = 0.4 if (i % 5 == 0 or i % 7 == 0) else 0.0
        # Spaces more often signal in ciphered language
        space_s = 0.3 if ch == " " else 0.0
        emit_s[i] = zipf_s + space_s - 0.5 * copy
        emit_n[i] = copy + periodic - space_s
        recent.append(ch)
    # Viterbi with prior favoring signal fraction (1 - filler_rate)
    log_stay_s = math.log(0.85)
    log_to_n = math.log(0.15)
    log_stay_n = math.log(0.70)
    log_to_s = math.log(0.30)
    neg = -1e9
    dp_s = np.full(n, neg)
    dp_n = np.full(n, neg)
    ptr_s = np.zeros(n, dtype=int)
    ptr_n = np.zeros(n, dtype=int)
    dp_s[0] = math.log(1 - filler_rate) + emit_s[0]
    dp_n[0] = math.log(filler_rate) + emit_n[0]
    for i in range(1, n):
        s_from_s = dp_s[i - 1] + log_stay_s
        s_from_n = dp_n[i - 1] + log_to_s
        if s_from_s >= s_from_n:
            dp_s[i] = s_from_s + emit_s[i]
            ptr_s[i] = 1
        else:
            dp_s[i] = s_from_n + emit_s[i]
            ptr_s[i] = 0
        n_from_n = dp_n[i - 1] + log_stay_n
        n_from_s = dp_s[i - 1] + log_to_n
        if n_from_n >= n_from_s:
            dp_n[i] = n_from_n + emit_n[i]
            ptr_n[i] = 0
        else:
            dp_n[i] = n_from_s + emit_n[i]
            ptr_n[i] = 1
    mask = np.zeros(n, dtype=int)
    state = 1 if dp_s[-1] >= dp_n[-1] else 0
    for i in range(n - 1, -1, -1):
        mask[i] = state
        state = ptr_s[i] if state == 1 else ptr_n[i]
    # Soft rate match: if predicted null rate far from target, threshold-adjust
    pred_null = 1 - mask.mean()
    if abs(pred_null - filler_rate) > 0.2:
        scores = emit_n - emit_s
        k = int(round(filler_rate * n))
        idx = np.argsort(-scores)[:k]
        mask = np.ones(n, dtype=int)
        mask[idx] = 0
    return mask


# ---------------------------------------------------------------------------
# Tiny BiLSTM
# ---------------------------------------------------------------------------


class TinySignalModel(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 48):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.mask_head = nn.Linear(hidden * 2, 1)
        self.recon_head = nn.Linear(hidden * 2, vocab_size)
        self.world_head = nn.Linear(hidden * 2, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.emb(x))
        mask_logit = self.mask_head(h).squeeze(-1)
        recon_logit = self.recon_head(h)
        world_logit = self.world_head(h.mean(1))
        return mask_logit, recon_logit, world_logit

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    corpora: dict[str, str],
    n_train: int,
    n_val: int,
    seed: int,
    filler_rate: float = PRIMARY_FILLER_RATE,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    langs = [k for k in ("english", "latin") if k in corpora]
    if not langs:
        raise ValueError("Need english and/or latin corpora")
    samples_train, samples_val = [], []

    def build(n, store):
        for _ in range(n):
            world = int(rng.integers(0, 4))
            lang = str(rng.choice(langs))
            piece = chunks_from_text(corpora[lang], rng, 1)[0]
            alphabet = "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
            sample = make_sample(piece, rng, world, filler_rate, alphabet)
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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Unseen alphabet subset: take from the end of the pool / reshuffle with dedicated seed
    alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
    pieces = chunks_from_text(finnish_text, rng, n)
    out = []
    for piece in pieces:
        sample = make_sample(piece, rng, WORLD_C, filler_rate, alphabet)
        sample["language"] = "finnish"
        out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def batchify(samples: list[dict], vocab: dict[str, int], idxs: np.ndarray) -> dict:
    texts = [samples[i]["text"] for i in idxs]
    x = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long)
    mask = torch.tensor([samples[i]["mask"] for i in idxs], dtype=torch.float32)
    world = torch.tensor([samples[i]["world"] for i in idxs], dtype=torch.long)
    # Reconstruction target: original char id if signal else pad
    recon = x.clone()
    recon[mask < 0.5] = 0
    return {"x": x, "mask": mask, "world": world, "recon": recon}


@torch.no_grad()
def predict_masks(model: TinySignalModel, samples: list[dict], vocab: dict[str, int], device: str) -> list[np.ndarray]:
    model.eval()
    out = []
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start : start + BATCH_SIZE]
        x = torch.tensor([encode(s["text"], vocab) for s in batch], dtype=torch.long, device=device)
        logits, _, _ = model(x)
        pred = (logits.sigmoid() >= 0.5).long().cpu().numpy()
        out.extend(pred)
    return out


def evaluate_masks(
    samples: list[dict],
    pred_masks: list[np.ndarray],
    rank_model: dict,
) -> dict:
    f1s, accs, recons, gains = [], [], [], []
    for s, pred in zip(samples, pred_masks):
        true = np.array(s["mask"], dtype=int)
        pred = np.asarray(pred, dtype=int)[: len(true)]
        if len(pred) < len(true):
            pred = np.pad(pred, (0, len(true) - len(pred)))
        f1s.append(f1_binary(true, pred))
        accs.append(float((true == pred).mean()))
        recons.append(recon_accuracy(s.get("ciphered") or "".join(
            ch for ch, m in zip(s["text"], true) if m == 1
        ), s["text"], pred))
        # pred_bits_gain: lower bits than full text is good; report full_bits - selected_bits
        full_bits = bigram_bits(rank_bucket_sequence(s["text"]), rank_model)
        sel_bits = pred_bits_of_selected(s["text"], pred, rank_model)
        gains.append(full_bits - sel_bits)
    return {
        "mask_f1": float(np.mean(f1s)),
        "mask_acc": float(np.mean(accs)),
        "recon_acc": float(np.mean(recons)),
        "pred_bits_gain": float(np.mean(gains)),
        "n": len(samples),
    }


def matched_random_baseline(samples: list[dict], rank_model: dict, n_seeds: int = MATCHED_RANDOM_SEEDS) -> dict:
    f1s, recons, gains = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        pf1, pr, pg = [], [], []
        for s in samples:
            true = np.array(s["mask"], dtype=int)
            n_null = int((true == 0).sum())
            pred = np.ones(len(true), dtype=int)
            if n_null > 0:
                idx = rng.choice(len(true), size=n_null, replace=False)
                pred[idx] = 0
            pf1.append(f1_binary(true, pred))
            pr.append(recon_accuracy(s.get("ciphered") or "".join(
                ch for ch, m in zip(s["text"], true) if m == 1
            ), s["text"], pred))
            full_bits = bigram_bits(rank_bucket_sequence(s["text"]), rank_model)
            sel_bits = pred_bits_of_selected(s["text"], pred, rank_model)
            pg.append(full_bits - sel_bits)
        f1s.append(float(np.mean(pf1)))
        recons.append(float(np.mean(pr)))
        gains.append(float(np.mean(pg)))
    return {
        "mask_f1": float(np.mean(f1s)),
        "mask_f1_std": float(np.std(f1s)),
        "recon_acc": float(np.mean(recons)),
        "pred_bits_gain": float(np.mean(gains)),
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
) -> tuple[TinySignalModel, dict]:
    torch.manual_seed(MODEL_SEED)
    model = TinySignalModel(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_f1 = -1.0
    best_state = None
    rng = np.random.default_rng(MODEL_SEED)
    model.train()
    t0 = time.time()
    for step in range(1, updates + 1):
        idxs = rng.integers(0, len(train), size=BATCH_SIZE)
        batch = batchify(train, vocab, idxs)
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        world = batch["world"].to(device)
        recon = batch["recon"].to(device)
        mask_logit, recon_logit, world_logit = model(x)
        loss_mask = F.binary_cross_entropy_with_logits(mask_logit, mask)
        # Reconstruction only on signal positions
        flat_logits = recon_logit.reshape(-1, recon_logit.size(-1))
        flat_tgt = recon.reshape(-1)
        flat_w = mask.reshape(-1)
        loss_recon = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
        loss_recon = (loss_recon * flat_w).sum() / flat_w.sum().clamp_min(1.0)
        loss_world = F.cross_entropy(world_logit, world)
        loss = loss_mask + 0.5 * loss_recon + 0.2 * loss_world
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == updates:
            rank_model = fit_rank_bigram([s["text"] for s in train if s["world"] == WORLD_B][:200] or [s["text"] for s in train[:200]])
            preds = predict_masks(model, val, vocab, device)
            metrics = evaluate_masks(val, preds, rank_model)
            metrics["step"] = step
            metrics["loss"] = float(loss.item())
            history.append(metrics)
            if metrics["mask_f1"] > best_f1:
                best_f1 = metrics["mask_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "param_count": model.param_count(),
        "best_val_mask_f1": best_f1,
        "updates": updates,
        "seconds": time.time() - t0,
        "history": history,
        "device": device,
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
    raw_root = root / "data" / "raw" / "latent_corpora"
    proc_root = root / "data" / "processed" / "exp0011"
    out_root = root / "outputs" / "EXP-0011"
    res_root = root / "results" / "EXP-0011"
    for p in (proc_root, out_root, res_root, root / "data" / "manifests"):
        p.mkdir(parents=True, exist_ok=True)

    downloaded = download_corpora(raw_root)
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
    )
    holdout = generate_finnish_holdout(texts["finnish"], n=args.n_holdout, seed=FINNISH_SEED)
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
    write_json(root / "data" / "manifests" / "exp0011_data.json", {
        "experiment": "EXP-0011",
        "data_seed": DATA_SEED,
        "finnish_seed": FINNISH_SEED,
        "filler_rate_primary": PRIMARY_FILLER_RATE,
        "seq_len": SEQ_LEN,
        "n_train": len(train),
        "n_val": len(val),
        "n_holdout": len(holdout),
        "worlds": WORLD_NAMES,
        "filler_families": FILLER_FAMILIES,
        "derived_sha256": digests,
        "pass_rule": asdict_pass_thresholds(),
    })

    vocab = build_vocab()
    device = resolve_device(args.device)
    # Warm rank model from training world-B surfaces
    world_b_texts = [s["text"] for s in train if s["world"] == WORLD_B]
    rank_model = fit_rank_bigram(world_b_texts[:400] or [s["text"] for s in train[:400]])

    model, train_summary = train_model(train, val, vocab, device, updates=args.updates)
    torch.save({"model": model.state_dict(), "vocab": vocab, "summary": train_summary}, out_root / "model.pt")

    # Holdout evaluation — first time looking at Finnish metrics
    neural_preds = predict_masks(model, holdout, vocab, device)
    neural_metrics = evaluate_masks(holdout, neural_preds, rank_model)
    classical_preds = [classical_null_mask(s["text"], PRIMARY_FILLER_RATE) for s in holdout]
    classical_metrics = evaluate_masks(holdout, classical_preds, rank_model)
    majority = majority_baseline(holdout, rank_model)
    random_b = matched_random_baseline(holdout, rank_model)
    # Training surface token vocab from world B/C only (ciphered surfaces)
    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)

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

    report = {
        "experiment": "EXP-0011",
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
            "Character-level BiLSTM (~tens of k params), not a large transformer",
            "Primary reconstruction target is C(L) via mask deletion, not full plaintext cryptanalysis",
            "Classical model is a 2-state feature Viterbi (copy/rank), not a full 20-state HSMM",
            "Finnish diacritics folded to ASCII a-z by shared cleaner (syntax/vocab still Finnish)",
            "Single primary filler rate 0.30 in the scored holdout",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-holdout", type=int, default=300)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
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
        "passed": report["decision"]["passed"],
        "winner": report["decision"]["winner"],
        "param_count": report["param_count"],
        "neural_f1": report["neural"]["mask_f1"],
        "classical_f1": report["classical"]["mask_f1"],
        "baselines": {k: v["mask_f1"] for k, v in report["baselines"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
