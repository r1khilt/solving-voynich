"""EXP-0016 thousand-language family-holdout latent recovery.

Preregistration: docs/experiments/EXP-0016.md. Not a Voynich decipherment.
Supersedes EXP-0015 as the session bar.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from voynich.latent_recovery import (
    ALIGN_CTC_WEIGHT,
    ALIGN_MASK_BCE_WEIGHT,
    ALIGN_RATE_WEIGHT,
    ALIGN_WORLD_WEIGHT,
    BATCH_SIZE,
    CIPHER_POOL,
    EASY_FILLER_FAMILIES,
    MATCHED_RANDOM_SEEDS,
    NULL_LOSS_WEIGHT,
    PRIMARY_FILLER_RATE,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    WORLD_D,
    batchify,
    build_vocab,
    chunks_from_text,
    ctc_deletion_loss,
    evaluate_masks,
    fit_rank_bigram,
    majority_baseline,
    make_sample,
    matched_random_baseline,
    null_f1_from_metrics,
    parse_filler_families,
    predict_mask_probs,
    vocab_filter_baseline,
)
from voynich.multilang_recovery import exact_count_masks
from voynich.romanize import ROMANIZATION_DOC, fold_latin, romanize
from voynich.runtime import digest, environment, resolve_device, write_json

# ---------------------------------------------------------------------------
# Frozen EXP-0016 constants (do not alter after seeing scores)
# ---------------------------------------------------------------------------

DATA_SEED = 4016
MODEL_SEED = 42
HOLDOUT_SEED = 4017

PASS16_MIN_LANGS_BEAT_RANDOM = 100
PASS16_MIN_FAMILIES_IN_PASSERS = 5
PASS16_MIN_NON_IE_PASSER_FRAC = 0.50
PASS16_NULL_RECALL_MIN = 0.50
PASS16_NULL_PRECISION_MIN = 0.50
PASS16_PRED_NULL_RATE_MIN = 0.15
PASS16_PRED_NULL_RATE_MAX = 0.45

MIN_POOL_LANGUAGES = 1000
MIN_HOLDOUT_LANGUAGES = 150
MIN_HOLDOUT_FAMILIES = 8
MIN_TRAIN_SUBSET = 400
MIN_CHARS = 800

# Scaled model (~8–12M params depending on vocab).
MODEL_EMB = 192
MODEL_HIDDEN = 384
MODEL_LAYERS = 2
MAX_UPDATES_DEFAULT = 4000
N_TRAIN_DEFAULT = 12000
N_VAL_DEFAULT = 800
N_HOLDOUT_PER_LANG = 40  # short samples × many languages

# Entire families held out of training (Glottolog family Name strings).
# Exact ISO membership resolved from Glottolog at acquire time; frozen into manifest.
HOLDOUT_FAMILY_NAMES = [
    "Uralic",
    "Basque",  # isolate family label; also catch Is_Isolate Basque
    "Afro-Asiatic",  # includes Semitic → Hebrew unseen with Arabic also out of train
    "Austronesian",
    "Quechuan",
    "Tupian",
    "Dravidian",
    "Kartvelian",
    # Expanded so held-out set can reach ≥150 once the pool is filled:
    "Mayan",
    "Otomanguean",
    "Aymaran",
]

# Extra isolate codes always held out even if family map misses them.
FORCE_HOLDOUT_ISO = {"eus", "eu"}  # Basque

IE_FAMILY_NAMES = {"Indo-European"}


def is_indo_european(family: str) -> bool:
    return family in IE_FAMILY_NAMES or family.startswith("Indo-European")


# ---------------------------------------------------------------------------
# Glottolog family map
# ---------------------------------------------------------------------------


def load_glottolog_family_map(path: Path) -> dict[str, dict]:
    """ISO 639-3 → {family, glottocode, name, isolate}."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_id = {r["ID"]: r for r in rows}
    out: dict[str, dict] = {}
    for r in rows:
        iso = (r.get("ISO639P3code") or "").strip().lower()
        if not iso:
            continue
        isolate = (r.get("Is_Isolate") or "").lower() in {"true", "1", "yes"}
        fid = (r.get("Family_ID") or "").strip()
        if isolate:
            family = "isolate"
        elif fid and fid in by_id:
            family = by_id[fid]["Name"]
        else:
            family = "unknown"
        # Prefer language-level rows over dialects when colliding: keep first
        if iso not in out or r.get("Level") == "language":
            out[iso] = {
                "family": family,
                "glottocode": r.get("Glottocode") or r["ID"],
                "name": r.get("Name") or iso,
                "isolate": isolate,
                "macroarea": r.get("Macroarea") or "",
            }
    # Basque under isolate + force family label "Basque" for holdout matching
    if "eus" in out:
        out["eus"]["family"] = "Basque"
        out["eus"]["isolate"] = True
    return out


# ---------------------------------------------------------------------------
# Script detection + romanization
# ---------------------------------------------------------------------------

_SCRIPT_RANGES = [
    ("cyrillic", (0x0400, 0x04FF)),
    ("greek", (0x0370, 0x03FF)),
    ("hebrew", (0x0590, 0x05FF)),
    ("arabic", (0x0600, 0x06FF)),
    ("devanagari", (0x0900, 0x097F)),
]


def detect_script(text: str) -> str:
    counts: Counter[str] = Counter()
    for ch in text:
        o = ord(ch)
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            counts["latin"] += 1
            continue
        for name, (lo, hi) in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] += 1
                break
        else:
            # CJK / Hangul / other
            if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
                counts["cjk"] += 1
            elif not ch.isspace() and not ch.isdigit():
                counts["other"] += 1
    if not counts:
        return "latin"
    top = counts.most_common(1)[0][0]
    return top


def romanize_auto(text: str, script_hint: str | None = None) -> tuple[str, str, bool]:
    """Return (romanized, scheme, ok). ok=False ⇒ skip language."""
    hint = (script_hint or "").lower()
    if hint in {"latn", "latin", "latin-1"}:
        return fold_latin(text), "latin_fold", True
    detected = detect_script(text)
    if hint in {"cyrl", "cyrillic"} or detected == "cyrillic":
        return romanize(text, "cyrillic"), "cyrillic_iso9like", True
    if hint in {"grek", "greek"} or detected == "greek":
        return romanize(text, "greek"), "greek_classicizing", True
    if hint in {"hebr", "hebrew"} or detected == "hebrew":
        return romanize(text, "hebrew"), "hebrew_iso259ish", True
    if hint in {"arab", "arabic"} or detected == "arabic":
        return romanize(text, "arabic"), "arabic_buckwalterish", True
    if hint in {"deva", "devanagari"} or detected == "devanagari":
        return romanize(text, "devanagari"), "devanagari_iastish", True
    if detected == "latin" or hint.endswith("latn"):
        return fold_latin(text), "latin_fold", True
    if detected == "cjk":
        return "", "skipped_cjk_no_romanizer", False
    # Other scripts: try Unidecode (documented ASCII transliteration) if installed.
    try:
        from unidecode import unidecode as _unidecode

        folded = fold_latin(_unidecode(text))
        letters = sum(1 for ch in folded if ch.isalpha())
        if letters >= max(200, int(0.10 * max(len(text), 1))):
            return folded, "unidecode_then_latin_fold", True
    except Exception:  # noqa: BLE001
        pass
    # Unknown script: try Latin fold; if too few letters remain, skip
    folded = fold_latin(text)
    letters = sum(1 for ch in folded if ch.isalpha())
    if letters >= max(200, int(0.15 * max(len(text), 1))):
        return folded, "latin_fold_fallback", True
    return "", f"skipped_unromanizable:{detected}", False


# ---------------------------------------------------------------------------
# Corpus parsers
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "solving-voynich/EXP-0016 (research; github.com/r1khilt/solving-voynich)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_udhr(txt_dir: Path) -> dict[str, dict]:
    index = ET.parse(txt_dir / "index.xml").getroot()
    out: dict[str, dict] = {}
    for el in index.findall("udhr"):
        iso = (el.attrib.get("iso639-3") or "").lower()
        if not iso or iso == "und":
            continue
        f = el.attrib.get("f")
        path = txt_dir / f"udhr_{f}.txt"
        if not path.exists():
            # also try bcp47-named files
            bcp = el.attrib.get("bcp47") or iso
            alt = txt_dir / f"udhr_{bcp}.txt"
            path = alt if alt.exists() else path
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        # drop project header before ----- if present
        if "-----" in raw:
            raw = raw.split("-----", 1)[-1]
        script = el.attrib.get("iso15924") or "Latn"
        key = f"udhr_{iso}_{f}"
        out[key] = {
            "iso6393": iso,
            "name": el.attrib.get("n") or iso,
            "script": script,
            "source": "udhr",
            "license": "Unicode UDHR assembly; OHCHR source text; attribution required",
            "text": raw,
            "bcp47": el.attrib.get("bcp47") or iso,
            "udhr_f": f,
        }
    return out


def parse_flores(dev_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(dev_dir.glob("*.dev")):
        code = path.name[: -len(".dev")]  # e.g. eng_Latn
        parts = code.split("_", 1)
        iso = parts[0].lower()
        script = parts[1] if len(parts) > 1 else "Latn"
        text = path.read_text(encoding="utf-8", errors="replace")
        out[f"flores_{code}"] = {
            "iso6393": iso,
            "name": code,
            "script": script,
            "source": "flores200_dev",
            "license": "CC BY-SA 4.0 (FLORES-200)",
            "text": text,
            "bcp47": code.replace("_", "-"),
        }
    return out


def parse_tatoeba(sentences_csv: Path, min_chars: int = MIN_CHARS) -> dict[str, dict]:
    """sentences.csv: id \\t lang \\t text. lang is ISO 639-3 (sometimes 2-letter)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = Counter()
    with sentences_csv.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            lang = parts[1].strip().lower()
            if not lang or lang == "\\N":
                continue
            text = parts[2]
            if sizes[lang] >= 12000:
                continue
            buckets[lang].append(text)
            sizes[lang] += len(text) + 1
    out: dict[str, dict] = {}
    for lang, sents in buckets.items():
        text = " ".join(sents)
        if len(text) < min_chars:
            continue
        out[f"tatoeba_{lang}"] = {
            "iso6393": lang,
            "name": lang,
            "script": "unknown",
            "source": "tatoeba",
            "license": "CC BY 2.0 FR (Tatoeba; attribution)",
            "text": text[:50000],
            "bcp47": lang,
        }
    return out


def fetch_wikipedia_extract(lang: str, titles: list[str], dest: Path) -> str | None:
    chunks = []
    # Prefer random article extracts for coverage when titles are thin.
    try:
        qs = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": 5,
                "prop": "extracts",
                "explaintext": 1,
                "exlimit": 5,
                "format": "json",
            }
        )
        url = f"https://{lang}.wikipedia.org/w/api.php?{qs}"
        time.sleep(0.25)
        payload = json.loads(_http_get(url, timeout=45))
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract") or ""
            if len(extract) > 150:
                chunks.append(extract)
    except Exception:  # noqa: BLE001
        pass
    for title in titles:
        if not title:
            continue
        qs = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exlimit": 1,
                "titles": title,
                "format": "json",
                "redirects": 1,
            }
        )
        url = f"https://{lang}.wikipedia.org/w/api.php?{qs}"
        try:
            time.sleep(0.25)
            payload = json.loads(_http_get(url, timeout=45))
            pages = payload.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("missing") is not None:
                    continue
                extract = page.get("extract") or ""
                if len(extract) > 200:
                    chunks.append(extract)
        except Exception:  # noqa: BLE001
            continue
    if not chunks:
        return None
    text = "\n\n".join(chunks)
    dest.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Merge / romanize / split
# ---------------------------------------------------------------------------


def merge_and_romanize(
    records: dict[str, dict],
    family_map: dict[str, dict],
    min_chars: int = MIN_CHARS,
) -> tuple[dict[str, dict], dict]:
    """Collapse to one entry per ISO 639-3 (prefer longer text). Return texts + stats."""
    by_iso: dict[str, dict] = {}
    skipped = []
    for key, rec in records.items():
        iso = rec["iso6393"].lower()
        # Map ISO 639-1 → 639-3 via family_map keys / common map
        if len(iso) == 2:
            iso = _iso1_to3.get(iso, iso)
        rom, scheme, ok = romanize_auto(rec["text"], rec.get("script"))
        if not ok or len(rom) < min_chars:
            skipped.append(
                {
                    "key": key,
                    "iso6393": iso,
                    "reason": scheme if not ok else f"too_short:{len(rom)}",
                    "source": rec["source"],
                }
            )
            continue
        meta = family_map.get(iso, {"family": "unknown", "name": iso, "isolate": False, "glottocode": ""})
        entry = {
            "iso6393": iso,
            "name": rec.get("name") or meta.get("name") or iso,
            "family": meta.get("family") or "unknown",
            "isolate": bool(meta.get("isolate")),
            "glottocode": meta.get("glottocode") or "",
            "macroarea": meta.get("macroarea") or "",
            "source": rec["source"],
            "license": rec["license"],
            "script": rec.get("script"),
            "romanization_scheme": scheme,
            "romanized_chars": len(rom),
            "text": rom,
            "source_key": key,
        }
        prev = by_iso.get(iso)
        if prev is None or entry["romanized_chars"] > prev["romanized_chars"]:
            by_iso[iso] = entry
    stats = {
        "n_source_records": len(records),
        "n_unique_iso_romanized": len(by_iso),
        "n_skipped": len(skipped),
        "skipped_sample": skipped[:50],
        "skipped_reasons": Counter(s["reason"].split(":")[0] for s in skipped),
        "family_counts": Counter(e["family"] for e in by_iso.values()),
    }
    return by_iso, stats


# Common ISO 639-1 → 639-3 for Tatoeba / Wikipedia codes.
def _load_iso1_to3() -> dict[str, str]:
    builtin = {
        "en": "eng", "es": "spa", "de": "deu", "fr": "fra", "it": "ita", "pt": "por",
        "ru": "rus", "zh": "zho", "ja": "jpn", "ko": "kor", "ar": "ara", "hi": "hin",
        "bn": "ben", "pa": "pan", "gu": "guj", "ta": "tam", "te": "tel", "ml": "mal",
        "kn": "kan", "mr": "mar", "ur": "urd", "fa": "fas", "tr": "tur", "vi": "vie",
        "th": "tha", "id": "ind", "ms": "msa", "nl": "nld", "pl": "pol", "uk": "ukr",
        "cs": "ces", "sk": "slk", "ro": "ron", "hu": "hun", "fi": "fin", "sv": "swe",
        "da": "dan", "no": "nor", "nb": "nob", "nn": "nno", "el": "ell", "he": "heb",
        "yi": "yid", "ga": "gle", "cy": "cym", "gd": "gla", "eu": "eus", "ca": "cat",
        "gl": "glg", "sw": "swh", "zu": "zul", "xh": "xho", "af": "afr",
        "sq": "sqi", "sr": "srp", "hr": "hrv", "bs": "bos", "sl": "slv", "bg": "bul",
        "mk": "mkd", "et": "est", "lv": "lav", "lt": "lit", "is": "isl", "fo": "fao",
        "mt": "mlt", "hy": "hye", "ka": "kat", "az": "aze", "kk": "kaz", "uz": "uzb",
        "ky": "kir", "tg": "tgk", "mn": "mon", "my": "mya", "km": "khm", "lo": "lao",
        "ne": "nep", "si": "sin", "am": "amh", "ti": "tir", "so": "som", "ha": "hau",
        "yo": "yor", "ig": "ibo", "rw": "kin", "rn": "run", "lg": "lug", "ny": "nya",
        "sn": "sna", "st": "sot", "tn": "tsn", "ts": "tso", "ve": "ven", "ss": "ssw",
        "la": "lat", "eo": "epo", "ia": "ina", "io": "ido", "vo": "vol", "jv": "jav",
        "su": "sun", "tl": "tgl", "mg": "mlg", "sm": "smo",
        "to": "ton", "ty": "tah", "mi": "mri", "qu": "que", "ay": "aym",
        "gn": "grn", "bo": "bod", "dz": "dzo", "ug": "uig", "ps": "pus", "sd": "snd",
        "ku": "kur", "be": "bel", "tt": "tat", "ba": "bak", "cv": "chv",
        "ce": "che", "os": "oss", "ab": "abk", "kv": "kom",
    }
    path = Path(__file__).resolve().parents[2] / "data" / "manifests" / "iso639_1_to_3.json"
    if path.exists():
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            builtin.update({k.lower(): v.lower() for k, v in extra.items()})
        except Exception:  # noqa: BLE001
            pass
    return builtin


_iso1_to3 = _load_iso1_to3()


def assign_splits(by_iso: dict[str, dict]) -> dict:
    """Freeze train / holdout / historical using HOLDOUT_FAMILY_NAMES."""
    holdout_families = set(HOLDOUT_FAMILY_NAMES)
    holdout: dict[str, dict] = {}
    train_pool: dict[str, dict] = {}
    for iso, entry in by_iso.items():
        fam = entry["family"]
        # Entire registered holdout families + all isolates + forced Basque
        if (
            fam in holdout_families
            or entry.get("isolate")
            or iso in FORCE_HOLDOUT_ISO
            or fam == "isolate"
        ):
            holdout[iso] = entry
        else:
            train_pool[iso] = entry

    # Historical: Latin / Italian / German if present — score-only, not in train
    historical = {}
    for iso in ("lat", "ita", "deu"):
        if iso in train_pool:
            historical[iso] = train_pool.pop(iso)
        elif iso in by_iso and iso not in holdout:
            historical[iso] = by_iso[iso]

    return {
        "holdout_families": sorted(holdout_families),
        "holdout": holdout,
        "train_pool": train_pool,
        "historical": historical,
    }


def stratified_train_subset(
    train_pool: dict[str, dict],
    n: int = MIN_TRAIN_SUBSET,
    seed: int = DATA_SEED,
) -> list[str]:
    """≥400 languages from all non-held-out families; proportional round-robin."""
    rng = np.random.default_rng(seed)
    by_fam: dict[str, list[str]] = defaultdict(list)
    for iso, e in train_pool.items():
        by_fam[e["family"]].append(iso)
    for fam in by_fam:
        rng.shuffle(by_fam[fam])
    selected: list[str] = []
    # Ensure every family contributes at least one if possible
    fams = sorted(by_fam.keys())
    idx = {f: 0 for f in fams}
    while len(selected) < min(n, len(train_pool)):
        progress = False
        for f in fams:
            i = idx[f]
            if i < len(by_fam[f]):
                selected.append(by_fam[f][i])
                idx[f] = i + 1
                progress = True
                if len(selected) >= n:
                    break
        if not progress:
            break
    return selected


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SignalModelV16(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb: int = MODEL_EMB,
        hidden: int = MODEL_HIDDEN,
        layers: int = MODEL_LAYERS,
        n_feats: int = 4,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.feat_proj = nn.Linear(n_feats, emb)
        self.lstm = nn.LSTM(
            emb, hidden, num_layers=layers, batch_first=True, bidirectional=True
        )
        self.mask_head = nn.Linear(hidden * 2, 1)
        self.recon_head = nn.Linear(hidden * 2, vocab_size)
        self.world_head = nn.Linear(hidden * 2, 4)

    def forward(self, x: torch.Tensor, feats: torch.Tensor):
        h, _ = self.lstm(self.emb(x) + self.feat_proj(feats))
        return (
            self.mask_head(h).squeeze(-1),
            self.recon_head(h),
            self.world_head(h.mean(1)),
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model_v16(
    train: list[dict],
    val: list[dict],
    vocab: dict[str, int],
    device: str,
    updates: int = MAX_UPDATES_DEFAULT,
) -> tuple[SignalModelV16, dict]:
    torch.manual_seed(MODEL_SEED)
    model = SignalModelV16(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    history = []
    best_score = -1.0
    best_state = None
    rng = np.random.default_rng(MODEL_SEED)
    by_world = {w: [i for i, s in enumerate(train) if s["world"] == w] for w in range(4)}
    for w, idxs in by_world.items():
        if not idxs:
            by_world[w] = list(range(len(train)))
    val_c = [s for s in val if s["world"] == WORLD_C] or val
    vocab_size = len(vocab)
    model.train()
    t0 = time.time()
    for step in range(1, updates + 1):
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
        mask_logit, _, world_logit = model(x, feats)
        loss_ctc = ctc_deletion_loss(mask_logit, x, batch["ctc_targets"], vocab_size)
        bce = nn.BCEWithLogitsLoss(reduction="none")(mask_logit, mask.float())
        null_w = torch.where(mask == 0, torch.full_like(bce, NULL_LOSS_WEIGHT), torch.ones_like(bce))
        loss_bce = (bce * null_w).mean()
        pred_null = torch.sigmoid(-mask_logit).mean()
        loss_rate = (pred_null - PRIMARY_FILLER_RATE).pow(2)
        loss_world = nn.CrossEntropyLoss()(world_logit, world)
        loss = (
            ALIGN_CTC_WEIGHT * loss_ctc
            + ALIGN_MASK_BCE_WEIGHT * loss_bce
            + ALIGN_RATE_WEIGHT * loss_rate
            + ALIGN_WORLD_WEIGHT * loss_world
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == updates:
            model.eval()
            with torch.no_grad():
                probs = predict_mask_probs(model, val_c[:400], vocab, device)
                masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
                rank_model = fit_rank_bigram(
                    [s["text"] for s in train if s["world"] == WORLD_B][:200]
                    or [s["text"] for s in train[:200]]
                )
                metrics = evaluate_masks(val_c[:400], masks, rank_model)
                metrics["null_f1"] = null_f1_from_metrics(metrics)
                metrics["step"] = step
                metrics["loss"] = float(loss.detach().cpu())
                history.append(metrics)
                score = float(metrics["recon_acc"])
                if score >= best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "param_count": model.param_count(),
        "best_val_recon_acc": best_score,
        "updates": updates,
        "seconds": time.time() - t0,
        "device": device,
        "emb": MODEL_EMB,
        "hidden": MODEL_HIDDEN,
        "layers": MODEL_LAYERS,
        "decoder": "exact_count_keep_round((1-0.30)*L)",
        "history": history,
    }
    return model, summary


# ---------------------------------------------------------------------------
# Pass rule
# ---------------------------------------------------------------------------


def language_beats_random(neural: dict, random_b: dict) -> bool:
    return float(neural["recon_acc"]) > float(random_b["recon_acc"])


def apply_pass_rule_v16(per_lang: dict[str, dict]) -> dict:
    passers = [lang for lang, row in per_lang.items() if row.get("beats_matched_random")]
    passer_fams = {per_lang[lang]["family"] for lang in passers}
    non_ie_passers = [lang for lang in passers if not is_indo_european(per_lang[lang]["family"])]
    non_ie_frac = (len(non_ie_passers) / len(passers)) if passers else 0.0

    macro_recon = float(np.mean([row["neural"]["recon_acc"] for row in per_lang.values()]))
    macro_rand = float(
        np.mean([row["baselines"]["matched_random"]["recon_acc"] for row in per_lang.values()])
    )
    macro_null_rec = float(np.mean([row["neural"]["null_recall"] for row in per_lang.values()]))
    macro_null_prec = float(np.mean([row["neural"]["null_precision"] for row in per_lang.values()]))
    macro_pred_null = float(np.mean([row["neural"]["pred_null_rate"] for row in per_lang.values()]))
    macro_vocab = float(
        np.mean([row["baselines"]["vocab_filter"]["recon_acc"] for row in per_lang.values()])
    )

    reasons = []
    ok = True
    if len(passers) < PASS16_MIN_LANGS_BEAT_RANDOM:
        ok = False
        reasons.append(f"only {len(passers)} langs beat matched-random (need ≥{PASS16_MIN_LANGS_BEAT_RANDOM})")
    if len(passer_fams) < PASS16_MIN_FAMILIES_IN_PASSERS:
        ok = False
        reasons.append(f"passer families {len(passer_fams)} < {PASS16_MIN_FAMILIES_IN_PASSERS}")
    if non_ie_frac < PASS16_MIN_NON_IE_PASSER_FRAC:
        ok = False
        reasons.append(f"non-IE passer frac {non_ie_frac:.3f} < {PASS16_MIN_NON_IE_PASSER_FRAC}")
    if not (macro_recon > macro_rand):
        ok = False
        reasons.append(f"macro recon {macro_recon:.4f} not > macro rand {macro_rand:.4f}")
    if macro_null_rec < PASS16_NULL_RECALL_MIN:
        ok = False
        reasons.append(f"macro null_recall {macro_null_rec:.4f} < {PASS16_NULL_RECALL_MIN}")
    if macro_null_prec < PASS16_NULL_PRECISION_MIN:
        ok = False
        reasons.append(f"macro null_precision {macro_null_prec:.4f} < {PASS16_NULL_PRECISION_MIN}")
    if not (PASS16_PRED_NULL_RATE_MIN <= macro_pred_null <= PASS16_PRED_NULL_RATE_MAX):
        ok = False
        reasons.append(f"macro pred_null_rate {macro_pred_null:.4f} out of band")
    if macro_vocab >= macro_recon:
        ok = False
        reasons.append(f"vocab-filter macro recon {macro_vocab:.4f} >= model {macro_recon:.4f} (cheat)")

    return {
        "passed": ok,
        "n_holdout_scored": len(per_lang),
        "n_beat_matched_random": len(passers),
        "passer_families": sorted(passer_fams),
        "n_passer_families": len(passer_fams),
        "non_ie_passer_fraction": non_ie_frac,
        "n_non_ie_passers": len(non_ie_passers),
        "macro_recon_acc": macro_recon,
        "macro_matched_random_recon_acc": macro_rand,
        "macro_null_recall": macro_null_rec,
        "macro_null_precision": macro_null_prec,
        "macro_pred_null_rate": macro_pred_null,
        "macro_vocab_filter_recon_acc": macro_vocab,
        "reasons": reasons,
        "thresholds": {
            "min_langs_beat_random": PASS16_MIN_LANGS_BEAT_RANDOM,
            "min_families_in_passers": PASS16_MIN_FAMILIES_IN_PASSERS,
            "min_non_ie_passer_frac": PASS16_MIN_NON_IE_PASSER_FRAC,
            "null_recall_min": PASS16_NULL_RECALL_MIN,
            "null_precision_min": PASS16_NULL_PRECISION_MIN,
            "pred_null_rate_band": [PASS16_PRED_NULL_RATE_MIN, PASS16_PRED_NULL_RATE_MAX],
            "vocab_cheat": "vocab_filter macro recon >= model macro recon ⇒ FAIL",
        },
    }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def generate_train_val(
    texts: dict[str, str],
    train_langs: list[str],
    n_train: int,
    n_val: int,
    seed: int,
    filler_families: tuple[str, ...] = EASY_FILLER_FAMILIES,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    world_p = np.array([0.10, 0.15, 0.60, 0.15], dtype=float)
    train, val = [], []

    def build(n: int, store: list[dict]) -> None:
        for _ in range(n):
            world = int(rng.choice(4, p=world_p))
            lang = str(rng.choice(train_langs))
            piece = chunks_from_text(texts[lang], rng, 1)[0]
            alphabet = "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
            sample = make_sample(
                piece, rng, world, PRIMARY_FILLER_RATE, alphabet, filler_families=filler_families
            )
            sample["language"] = lang
            store.append(sample)

    build(n_train, train)
    build(n_val, val)
    return train, val


def generate_holdout_for_lang(
    text: str,
    language: str,
    n: int,
    seed: int,
    filler_families: tuple[str, ...] = EASY_FILLER_FAMILIES,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
    pieces = chunks_from_text(text, rng, n)
    out = []
    for piece in pieces:
        sample = make_sample(
            piece, rng, WORLD_C, PRIMARY_FILLER_RATE, alphabet, filler_families=filler_families
        )
        sample["language"] = language
        out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Acquire-only / main
# ---------------------------------------------------------------------------


def acquire_phase(root: Path, wiki_fill: bool = True) -> dict:
    raw = root / "data" / "raw" / "exp0016_corpora"
    family_map = load_glottolog_family_map(raw / "wiki" / "glottolog_languages.csv")

    records: dict[str, dict] = {}
    records.update(parse_udhr(raw / "udhr" / "txt"))
    records.update(parse_flores(raw / "flores" / "flores200_dataset" / "dev"))

    tatoeba_csv = raw / "tatoeba" / "sentences.csv"
    if tatoeba_csv.exists():
        records.update(parse_tatoeba(tatoeba_csv))

    # Salvage EXP-0015 multilang texts as extras
    salvage = root / "data" / "raw" / "multilang_corpora"
    if salvage.exists():
        for path in salvage.glob("*.romanized.txt"):
            iso_guess = path.name.split(".")[0]
            # map common names
            name_to_iso = {
                "english": "eng", "spanish": "spa", "german": "deu", "russian": "rus",
                "greek": "ell", "irish": "gle", "arabic": "ara", "turkish": "tur",
                "indonesian": "ind", "swahili": "swh", "vietnamese": "vie", "hindi": "hin",
                "finnish": "fin", "hungarian": "hun", "basque": "eus", "hebrew": "heb",
                "tagalog": "tgl", "latin": "lat", "italian_historical": "ita",
            }
            iso = name_to_iso.get(iso_guess, iso_guess[:3])
            text = path.read_text(encoding="utf-8", errors="replace")
            records[f"salvage_{iso_guess}"] = {
                "iso6393": iso,
                "name": iso_guess,
                "script": "Latn",
                "source": "exp0015_salvage",
                "license": "see exp0015_corpora.json",
                "text": text,
                "bcp47": iso,
            }

    by_iso, stats = merge_and_romanize(records, family_map)

    # Wikipedia fill toward 1000
    wiki_added = 0
    wiki_failed = 0
    if wiki_fill and len(by_iso) < MIN_POOL_LANGUAGES:
        editions = json.loads((raw / "wiki" / "wikipedia_editions.json").read_text())
        wiki_dir = raw / "wiki" / "extracts"
        wiki_dir.mkdir(parents=True, exist_ok=True)
        for ed in editions:
            if len(by_iso) >= MIN_POOL_LANGUAGES:
                break
            code = (ed.get("code") or "").lower()
            iso = _iso1_to3.get(code, code if len(code) == 3 else "")
            if not iso or iso in by_iso:
                continue
            dest = wiki_dir / f"{code}.txt"
            if dest.exists() and dest.stat().st_size > 500:
                text = dest.read_text(encoding="utf-8", errors="replace")
            else:
                titles = ["Wikipedia:About", "Main Page"]
                if ed.get("name"):
                    titles = [ed["name"], "Wikipedia"] + titles
                text = fetch_wikipedia_extract(code, titles[:3], dest)
                if not text:
                    wiki_failed += 1
                    continue
            rom, scheme, ok = romanize_auto(text, "Latn")
            if not ok or len(rom) < MIN_CHARS:
                rom, scheme, ok = romanize_auto(text, None)
            if not ok or len(rom) < MIN_CHARS:
                wiki_failed += 1
                continue
            meta = family_map.get(iso, {"family": "unknown", "name": iso, "isolate": False})
            by_iso[iso] = {
                "iso6393": iso,
                "name": ed.get("name") or iso,
                "family": meta.get("family") or "unknown",
                "isolate": bool(meta.get("isolate")),
                "glottocode": meta.get("glottocode") or "",
                "macroarea": meta.get("macroarea") or "",
                "source": "wikipedia_extract",
                "license": "CC BY-SA 4.0 (Wikipedia)",
                "script": "auto",
                "romanization_scheme": scheme,
                "romanized_chars": len(rom),
                "text": rom,
                "source_key": f"wiki_{code}",
            }
            wiki_added += 1

    splits = assign_splits(by_iso)
    train_subset = stratified_train_subset(splits["train_pool"], n=MIN_TRAIN_SUBSET, seed=DATA_SEED)

    # Persist romanized texts (gitignored under data/raw)
    clean_root = raw / "romanized"
    clean_root.mkdir(parents=True, exist_ok=True)
    lang_manifest = {}
    for iso, e in by_iso.items():
        p = clean_root / f"{iso}.txt"
        p.write_text(e["text"], encoding="utf-8")
        lang_manifest[iso] = {
            **{k: v for k, v in e.items() if k != "text"},
            "romanized_sha256": digest(p),
            "role": (
                "holdout"
                if iso in splits["holdout"]
                else "historical"
                if iso in splits["historical"]
                else "train_pool"
            ),
        }

    stop_reason = None
    if len(by_iso) < MIN_POOL_LANGUAGES:
        stop_reason = (
            f"Legal open sources yielded {len(by_iso)} romanized languages after UDHR+FLORES+"
            f"Tatoeba+Wikipedia fill (wiki_added={wiki_added}, wiki_failed={wiki_failed}); "
            f"stopped below {MIN_POOL_LANGUAGES} rather than inventing text."
        )

    manifest = {
        "experiment": "EXP-0016",
        "romanization": {**ROMANIZATION_DOC, "auto": "detect script → table or latin fold; CJK skipped"},
        "n_languages_pool": len(by_iso),
        "n_holdout": len(splits["holdout"]),
        "n_train_pool": len(splits["train_pool"]),
        "n_historical": len(splits["historical"]),
        "n_train_subset": len(train_subset),
        "holdout_families": splits["holdout_families"],
        "n_holdout_families_present": len({e["family"] for e in splits["holdout"].values()}),
        "train_subset_isos": train_subset,
        "train_subset_rule": (
            f"stratified round-robin across all non-held-out families; "
            f"seed={DATA_SEED}; target≥{MIN_TRAIN_SUBSET}; score full holdout set"
        ),
        "family_counts_pool": dict(Counter(e["family"] for e in by_iso.values())),
        "family_counts_holdout": dict(Counter(e["family"] for e in splits["holdout"].values())),
        "family_counts_train_pool": dict(Counter(e["family"] for e in splits["train_pool"].values())),
        "acquisition_stats": {
            **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
            "wiki_added": wiki_added,
            "wiki_failed": wiki_failed,
            "stop_reason": stop_reason,
        },
        "languages": lang_manifest,
        "decoder": "exact_count_keep_round((1-0.30)*L)",
        "pass_rule_summary": {
            "min_langs_beat_random": PASS16_MIN_LANGS_BEAT_RANDOM,
            "min_families_in_passers": PASS16_MIN_FAMILIES_IN_PASSERS,
            "min_non_ie_passer_frac": PASS16_MIN_NON_IE_PASSER_FRAC,
            "macro_null_gates": True,
            "vocab_cheat_macro": True,
        },
        "seeds": {"data": DATA_SEED, "model": MODEL_SEED, "holdout": HOLDOUT_SEED},
        "note": "Bulk text gitignored; this is provenance only. Not a decipherment.",
    }
    write_json(root / "data" / "manifests" / "exp0016_corpora.json", manifest)
    # Compact split file without per-lang blob for quick load
    write_json(
        root / "data" / "manifests" / "exp0016_splits.json",
        {
            "holdout_isos": sorted(splits["holdout"]),
            "train_pool_isos": sorted(splits["train_pool"]),
            "train_subset_isos": train_subset,
            "historical_isos": sorted(splits["historical"]),
            "holdout_families": splits["holdout_families"],
            "n_pool": len(by_iso),
            "stop_reason": stop_reason,
        },
    )
    return {
        "by_iso": by_iso,
        "splits": splits,
        "train_subset": train_subset,
        "manifest": manifest,
        "stop_reason": stop_reason,
    }


def run_experiment(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    filler_families = (
        parse_filler_families(args.filler_families) if args.filler_families else EASY_FILLER_FAMILIES
    )
    acq = acquire_phase(root, wiki_fill=not args.no_wiki_fill)
    by_iso = acq["by_iso"]
    splits = acq["splits"]
    train_subset = acq["train_subset"]

    if args.acquire_only:
        return {
            "n_pool": len(by_iso),
            "n_holdout": len(splits["holdout"]),
            "n_train_pool": len(splits["train_pool"]),
            "n_train_subset": len(train_subset),
            "n_holdout_families_present": len({e["family"] for e in splits["holdout"].values()}),
            "holdout_family_counts": dict(
                Counter(e["family"] for e in splits["holdout"].values())
            ),
            "stop_reason": acq.get("stop_reason"),
            "meets_min_pool": len(by_iso) >= MIN_POOL_LANGUAGES,
            "meets_min_holdout": len(splits["holdout"]) >= MIN_HOLDOUT_LANGUAGES,
        }

    if len(splits["holdout"]) < MIN_HOLDOUT_LANGUAGES:
        raise RuntimeError(
            f"Need ≥{MIN_HOLDOUT_LANGUAGES} holdout languages; got {len(splits['holdout'])}. "
            f"Pool={len(by_iso)}. {acq.get('stop_reason')}"
        )
    if len({e['family'] for e in splits['holdout'].values()}) < MIN_HOLDOUT_FAMILIES:
        raise RuntimeError("Fewer than 8 holdout families present in acquired data")

    texts = {iso: e["text"] for iso, e in by_iso.items()}
    train_langs = list(train_subset)
    train, val = generate_train_val(
        texts, train_langs, n_train=args.n_train, n_val=args.n_val, seed=DATA_SEED,
        filler_families=filler_families,
    )

    proc = root / "data" / "processed" / "exp0016"
    out_root = root / "outputs" / "EXP-0016"
    res_root = root / "results" / "EXP-0016"
    for p in (proc, out_root, res_root):
        p.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> str:
        path = proc / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "world": r["world"],
                            "language": r.get("language"),
                            "text": r["text"],
                            "mask": r["mask"],
                            "ciphered": r.get("ciphered", ""),
                            "filler_family": r.get("filler_family", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return digest(path)

    digests = {"train": dump("train.jsonl", train), "validation": dump("validation.jsonl", val)}
    holdouts: dict[str, list[dict]] = {}
    for i, iso in enumerate(sorted(splits["holdout"])):
        rows = generate_holdout_for_lang(
            texts[iso], iso, n=args.n_holdout, seed=HOLDOUT_SEED + 17 * (i + 1),
            filler_families=filler_families,
        )
        holdouts[iso] = rows
        digests[f"holdout_{iso}"] = dump(f"holdout_{iso}.jsonl", rows)

    historical_rows: dict[str, list[dict]] = {}
    for i, iso in enumerate(sorted(splits["historical"])):
        rows = generate_holdout_for_lang(
            texts[iso], iso, n=min(40, args.n_holdout), seed=HOLDOUT_SEED + 9000 + i,
            filler_families=filler_families,
        )
        historical_rows[iso] = rows
        digests[f"historical_{iso}"] = dump(f"historical_{iso}.jsonl", rows)

    write_json(
        root / "data" / "manifests" / "exp0016_data.json",
        {
            "experiment": "EXP-0016",
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "holdout_seed": HOLDOUT_SEED,
            "filler_rate": PRIMARY_FILLER_RATE,
            "filler_families": list(filler_families),
            "n_train": len(train),
            "n_val": len(val),
            "n_holdout_langs": len(holdouts),
            "n_holdout_per_lang": args.n_holdout,
            "train_subset_isos": train_langs,
            "holdout_isos": sorted(holdouts),
            "historical_isos": sorted(historical_rows),
            "derived_sha256": digests,
            "decoder": "exact_count_keep_round((1-0.30)*L)",
            "model_plan": {
                "emb": MODEL_EMB,
                "hidden": MODEL_HIDDEN,
                "layers": MODEL_LAYERS,
                "estimated_params": "~8-12M depending on vocab",
            },
        },
    )

    if args.prereg_only:
        return {"status": "prereg_data_ready", "n_holdout": len(holdouts), "n_train_subset": len(train_langs)}

    vocab = build_vocab()
    device = resolve_device(args.device)
    # Estimate params before train
    est = SignalModelV16(len(vocab)).param_count()
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400] or [s["text"] for s in train[:400]]
    )
    model, train_summary = train_model_v16(train, val, vocab, device, updates=args.updates)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "summary": {k: v for k, v in train_summary.items() if k != "history"},
            "decoder": "exact_count",
        },
        out_root / "model.pt",
    )

    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))

    per_lang: dict[str, dict] = {}
    for iso, rows in holdouts.items():
        probs = predict_mask_probs(model, rows, vocab, device)
        neural_masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
        neural = evaluate_masks(rows, neural_masks, rank_model)
        neural["null_f1"] = null_f1_from_metrics(neural)
        majority = majority_baseline(rows, rank_model)
        random_b = matched_random_baseline(rows, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)
        vocab_b = vocab_filter_baseline(rows, train_token_vocab, rank_model)
        beats = language_beats_random(neural, random_b)
        fam = splits["holdout"][iso]["family"]
        per_lang[iso] = {
            "family": fam,
            "indo_european": is_indo_european(fam),
            "name": splits["holdout"][iso].get("name"),
            "neural": neural,
            "baselines": {
                "majority": majority,
                "matched_random": random_b,
                "vocab_filter": vocab_b,
            },
            "beats_matched_random": beats,
        }

    # Family-level macros
    by_fam: dict[str, list[str]] = defaultdict(list)
    for iso, row in per_lang.items():
        by_fam[row["family"]].append(iso)
    family_table = {}
    for fam, isos in sorted(by_fam.items()):
        family_table[fam] = {
            "n": len(isos),
            "n_beat_random": sum(1 for i in isos if per_lang[i]["beats_matched_random"]),
            "macro_recon": float(np.mean([per_lang[i]["neural"]["recon_acc"] for i in isos])),
            "macro_rand": float(
                np.mean([per_lang[i]["baselines"]["matched_random"]["recon_acc"] for i in isos])
            ),
            "indo_european": is_indo_european(fam),
        }

    historical_scores = {}
    for iso, rows in historical_rows.items():
        probs = predict_mask_probs(model, rows, vocab, device)
        masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
        neural = evaluate_masks(rows, masks, rank_model)
        random_b = matched_random_baseline(rows, rank_model)
        historical_scores[iso] = {
            "family": by_iso[iso]["family"],
            "neural": neural,
            "matched_random": random_b,
            "note": "Score-only historical transfer; not used in pass decision",
        }

    decision = apply_pass_rule_v16(per_lang)
    report = {
        "experiment": "EXP-0016",
        "environment": environment(),
        "param_count": train_summary["param_count"],
        "param_count_estimate_before_train": est,
        "train_summary": {k: v for k, v in train_summary.items() if k != "history"},
        "n_pool": len(by_iso),
        "n_train_subset": len(train_langs),
        "n_holdout": len(per_lang),
        "stop_reason": acq.get("stop_reason"),
        "holdout_families": sorted(by_fam.keys()),
        "family_table": family_table,
        "per_language": per_lang,
        "historical": historical_scores,
        "decision": decision,
        "decoder": "exact_count_keep_round((1-0.30)*L)",
        "filler_families": list(filler_families),
        "data_digests": digests,
        "voynich_label_free": None,
        "simplifications": [
            "EXP-0015 superseded as session bar; thousand-language scale",
            "Entire families held out of training",
            "CJK/unromanizable skipped and counted",
            "Train subset stratified ≥400; full holdout scored",
            "Not a decipherment",
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
    # Compact CSV-like table for notebook
    compact = {
        iso: {
            "family": row["family"],
            "recon": row["neural"]["recon_acc"],
            "rand": row["baselines"]["matched_random"]["recon_acc"],
            "beats": row["beats_matched_random"],
            "null_rec": row["neural"]["null_recall"],
            "null_prec": row["neural"]["null_precision"],
            "pred_null": row["neural"]["pred_null_rate"],
        }
        for iso, row in per_lang.items()
    }
    write_json(res_root / "per_language_compact.json", compact)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train", type=int, default=N_TRAIN_DEFAULT)
    parser.add_argument("--n-val", type=int, default=N_VAL_DEFAULT)
    parser.add_argument("--n-holdout", type=int, default=N_HOLDOUT_PER_LANG)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES_DEFAULT)
    parser.add_argument("--filler-families", default="random_char,periodic")
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--prereg-only", action="store_true")
    parser.add_argument("--no-wiki-fill", action="store_true")
    args = parser.parse_args()
    report = run_experiment(args)
    if "decision" in report:
        print(
            json.dumps(
                {
                    "experiment": report["experiment"],
                    "passed": report["decision"]["passed"],
                    "param_count": report["param_count"],
                    "n_pool": report["n_pool"],
                    "n_holdout": report["n_holdout"],
                    "decision": report["decision"],
                    "family_table": report["family_table"],
                    "stop_reason": report.get("stop_reason"),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2)[:8000])


if __name__ == "__main__":
    main()
