"""EXP-0015 multi-language latent recovery with exact-count decode.

Preregistration: docs/experiments/EXP-0015.md. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
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
    DATA_SEED,
    EASY_FILLER_FAMILIES,
    FINNISH_SEED,
    MATCHED_RANDOM_SEEDS,
    MODEL_SEED,
    NULL_LOSS_WEIGHT,
    PRIMARY_FILLER_RATE,
    SEQ_LEN,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    WORLD_D,
    batchify,
    build_vocab,
    classical_hsmm_null_mask,
    chunks_from_text,
    clean_plaintext,
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
from voynich.romanize import ROMANIZATION_DOC, romanize
from voynich.runtime import digest, environment, resolve_device, write_json

# ---------------------------------------------------------------------------
# Frozen EXP-0015 constants (do not alter after seeing scores)
# ---------------------------------------------------------------------------

PASS15_NULL_RECALL_MIN = 0.50
PASS15_NULL_PRECISION_MIN = 0.50
PASS15_PRED_NULL_RATE_MIN = 0.15
PASS15_PRED_NULL_RATE_MAX = 0.45

# Model size target: ~2.5M params on MPS within ~1–2h.
MODEL_EMB = 128
MODEL_HIDDEN = 256
MODEL_LAYERS = 2
MAX_UPDATES_DEFAULT = 3000
N_TRAIN_DEFAULT = 6000
N_VAL_DEFAULT = 600
N_HOLDOUT_PER_LANG = 200

# Language catalog. role: train | holdout | historical
# script: see romanize.romanize
LANGUAGE_CATALOG: dict[str, dict] = {
    # ---- training ----
    "english": {
        "family": "indo_european_germanic",
        "role": "train",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 11,
        "title": "Alice's Adventures in Wonderland",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "spanish": {
        "family": "indo_european_romance",
        "role": "train",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 2000,
        "title": "Don Quijote",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "german": {
        "family": "indo_european_germanic",
        "role": "train",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 2229,
        "title": "Faust: Der Tragödie erster Teil",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "russian": {
        "family": "indo_european_slavic",
        "role": "train",
        "script": "cyrillic",
        "source": "project_gutenberg",
        "pg_id": 16527,
        "title": "1001 задач для умственного счета (Russian; romanized)",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "greek": {
        "family": "indo_european_hellenic",
        "role": "train",
        "script": "greek",
        "source": "wikipedia",
        "wiki_lang": "el",
        "wiki_titles": ["Ελληνική_γλώσσα", "Αθήνα", "Ελλάδα", "Αρχαία_Ελλάδα"],
        "title": "Greek Wikipedia extracts (romanized classicizing map)",
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "irish": {
        "family": "indo_european_celtic",
        "role": "train",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "ga",
        "wiki_titles": ["Éire", "Gaeilge", "Baile_Átha_Cliath", "Stair_na_hÉireann"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "arabic": {
        "family": "semitic",
        "role": "train",
        "script": "arabic",
        "source": "wikipedia",
        "wiki_lang": "ar",
        "wiki_titles": ["العربية", "مصر", "القاهرة", "الإسلام"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "turkish": {
        "family": "turkic",
        "role": "train",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "tr",
        "wiki_titles": ["Türkiye", "Türkçe", "İstanbul", "Osmanlı_İmparatorluğu"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "indonesian": {
        "family": "austronesian",
        "role": "train",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "id",
        "wiki_titles": ["Indonesia", "Bahasa_Indonesia", "Jakarta", "Sejarah_Indonesia"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "swahili": {
        "family": "niger_congo",
        "role": "train",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "sw",
        "wiki_titles": ["Kiswahili", "Tanzania", "Kenya", "Afrika"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "vietnamese": {
        "family": "austroasiatic",
        "role": "train",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "vi",
        "wiki_titles": ["Việt_Nam", "Tiếng_Việt", "Hà_Nội", "Lịch_sử_Việt_Nam"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "hindi": {
        "family": "indo_aryan",
        "role": "train",
        "script": "devanagari",
        "source": "wikipedia",
        "wiki_lang": "hi",
        "wiki_titles": ["भारत", "हिन्दी", "दिल्ली", "भारतीय_इतिहास"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    # ---- holdouts ----
    "finnish": {
        "family": "uralic",
        "role": "holdout",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 7000,
        "title": "Kalevala",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "hungarian": {
        "family": "uralic",
        "role": "holdout",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 66560,
        "title": "Megtörténtek és megtörténhetők",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "basque": {
        "family": "isolate",
        "role": "holdout",
        "script": "latin",
        "source": "wikipedia",
        "wiki_lang": "eu",
        "wiki_titles": ["Euskal_Herria", "Euskara", "Bilbo", "Euskal_Herriko_historia"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "hebrew": {
        "family": "semitic",
        "role": "holdout",
        "script": "hebrew",
        "source": "wikipedia",
        "wiki_lang": "he",
        "wiki_titles": ["עברית", "ישראל", "ירושלים", "היסטוריה_של_עם_ישראל"],
        "license": "CC BY-SA 4.0 (Wikipedia extracts); attribution in manifest",
    },
    "tagalog": {
        "family": "austronesian",
        "role": "holdout",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 20228,
        "title": "Noli Me Tangere (Tagalog)",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    "latin": {
        "family": "indo_european_italic",
        "role": "holdout",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 218,
        "title": "De Bello Gallico (Latin holdout; not in train)",
        "license": "Project Gutenberg License; public domain in the USA",
    },
    # ---- historical transfer (score only) ----
    "italian_historical": {
        "family": "indo_european_romance",
        "role": "historical",
        "script": "latin",
        "source": "project_gutenberg",
        "pg_id": 1000,
        "title": "La Divina Commedia (early-modern Italian domain transfer)",
        "license": "Project Gutenberg License; public domain in the USA",
    },
}

IE_FAMILY_PREFIXES = ("indo_european", "indo_aryan")


def is_indo_european(family: str) -> bool:
    return family.startswith("indo_european") or family == "indo_aryan"


# ---------------------------------------------------------------------------
# Exact-count decoder (frozen)
# ---------------------------------------------------------------------------


def exact_count_masks(
    probs: list[np.ndarray],
    filler_rate: float = PRIMARY_FILLER_RATE,
) -> list[np.ndarray]:
    """Keep top round((1-rate)*L) positions by keep-prob; ties → lower index."""
    out = []
    for p in probs:
        p = np.asarray(p, dtype=float).reshape(-1)
        length = len(p)
        n_keep = int(round((1.0 - filler_rate) * length))
        n_keep = max(0, min(length, n_keep))
        order = np.argsort(-p, kind="stable")
        mask = np.zeros(length, dtype=int)
        if n_keep:
            mask[order[:n_keep]] = 1
        out.append(mask)
    return out


# ---------------------------------------------------------------------------
# Larger BiLSTM
# ---------------------------------------------------------------------------


class SignalModelV15(nn.Module):
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


# ---------------------------------------------------------------------------
# Corpus download
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 90, retries: int = 4) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "solving-voynich/EXP-0015 (local research agent; "
                        "contact: repo github.com/r1khilt/solving-voynich)"
                    ),
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_err = exc
            # Back off on rate limits / transient errors.
            if exc.code in {429, 503, 502} and attempt + 1 < retries:
                time.sleep(2.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"GET failed: {url}: {last_err}")


def download_pg(pg_id: int, dest: Path) -> dict:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
        f"https://www.gutenberg.org/ebooks/{pg_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}-0.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
    ]
    last_err = None
    raw = None
    used = None
    for url in urls:
        try:
            raw = _http_get(url)
            if len(raw) < 500:
                raise RuntimeError("too short")
            used = url
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if raw is None:
        raise RuntimeError(f"PG {pg_id} failed: {last_err}")
    dest.write_text(raw, encoding="utf-8")
    return {"url_used": used, "raw_bytes": dest.stat().st_size, "raw_sha256": digest(dest)}


def download_wikipedia_extracts(lang: str, titles: list[str], dest: Path) -> dict:
    chunks = []
    used = []
    for title in titles:
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
            time.sleep(1.2)  # polite rate limit
            payload = json.loads(_http_get(url))
            pages = payload.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("missing") is not None:
                    continue
                extract = page.get("extract") or ""
                if len(extract) > 200:
                    chunks.append(extract)
                    used.append(
                        {
                            "title": title,
                            "resolved": page.get("title"),
                            "pageid": page.get("pageid"),
                            "chars": len(extract),
                        }
                    )
        except Exception:  # noqa: BLE001
            continue
    if not chunks:
        raise RuntimeError(f"Wikipedia extracts failed for {lang}: {titles}")
    text = "\n\n".join(chunks)
    dest.write_text(text, encoding="utf-8")
    return {
        "url_api": f"https://{lang}.wikipedia.org/w/api.php",
        "pages": used,
        "raw_bytes": dest.stat().st_size,
        "raw_sha256": digest(dest),
    }


def acquire_corpora(raw_root: Path) -> dict:
    """Download and romanize all catalog languages that succeed."""
    raw_root.mkdir(parents=True, exist_ok=True)
    acquired: dict[str, str] = {}
    manifest = {
        "romanization": ROMANIZATION_DOC,
        "languages": {},
        "skipped": {},
        "note": "Bulk text stays local/gitignored; this is provenance only.",
    }
    for lang, meta in LANGUAGE_CATALOG.items():
        raw_path = raw_root / f"{lang}.raw.txt"
        clean_path = raw_root / f"{lang}.romanized.txt"
        entry = {**meta}
        try:
            if clean_path.exists() and clean_path.stat().st_size > 2000:
                text = clean_path.read_text(encoding="utf-8")
                entry["status"] = "cached"
                entry["romanized_sha256"] = digest(clean_path)
                entry["romanized_chars"] = len(text)
            else:
                if meta["source"] == "project_gutenberg":
                    info = download_pg(int(meta["pg_id"]), raw_path)
                elif meta["source"] == "wikipedia":
                    info = download_wikipedia_extracts(
                        meta["wiki_lang"], list(meta["wiki_titles"]), raw_path
                    )
                else:
                    raise RuntimeError(f"Unknown source {meta['source']}")
                raw = raw_path.read_text(encoding="utf-8", errors="replace")
                # Strip PG boilerplate when present, then romanize.
                stripped = clean_plaintext(raw) if meta["script"] == "latin" and meta["source"] == "project_gutenberg" else raw
                # For non-latin PG, still strip markers then romanize from original-ish text.
                if meta["source"] == "project_gutenberg" and meta["script"] != "latin":
                    stripped = raw
                    for pat in (
                        r"\*\*\*\s*START OF.+?\*\*\*",
                        r"\*\*\*\s*END OF.+?\*\*\*",
                    ):
                        import re as _re

                        m = _re.search(pat, stripped, _re.I | _re.S)
                        if m and "START" in pat.upper():
                            stripped = stripped[m.end() :]
                        elif m:
                            stripped = stripped[: m.start()]
                text = romanize(stripped, meta["script"])
                if len(text) < 1500:
                    raise RuntimeError(f"romanized text too short ({len(text)})")
                clean_path.write_text(text, encoding="utf-8")
                entry["download"] = info
                entry["status"] = "downloaded"
                entry["romanized_sha256"] = digest(clean_path)
                entry["romanized_chars"] = len(text)
            # Reject near-empty alphabet collapse
            letters = set(ch for ch in text if ch.isalpha())
            if len(letters) < 10:
                raise RuntimeError(f"too few distinct letters after romanize: {sorted(letters)}")
            acquired[lang] = text
            manifest["languages"][lang] = entry
        except Exception as exc:  # noqa: BLE001
            manifest["skipped"][lang] = {"reason": str(exc), **meta}
    return {"texts": acquired, "manifest": manifest}


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def generate_multilang_dataset(
    corpora: dict[str, str],
    train_langs: list[str],
    n_train: int,
    n_val: int,
    seed: int,
    filler_families: tuple[str, ...] = EASY_FILLER_FAMILIES,
    filler_rate: float = PRIMARY_FILLER_RATE,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    world_p = np.array([0.10, 0.15, 0.60, 0.15], dtype=float)
    train, val = [], []

    def build(n: int, store: list[dict]) -> None:
        for _ in range(n):
            world = int(rng.choice(4, p=world_p))
            lang = str(rng.choice(train_langs))
            piece = chunks_from_text(corpora[lang], rng, 1)[0]
            alphabet = "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
            sample = make_sample(
                piece, rng, world, filler_rate, alphabet, filler_families=filler_families
            )
            sample["language"] = lang
            store.append(sample)

    build(n_train, train)
    build(n_val, val)
    return train, val


def generate_language_holdout(
    text: str,
    language: str,
    n: int,
    seed: int,
    filler_families: tuple[str, ...] = EASY_FILLER_FAMILIES,
    filler_rate: float = PRIMARY_FILLER_RATE,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
    pieces = chunks_from_text(text, rng, n)
    out = []
    for piece in pieces:
        sample = make_sample(
            piece, rng, WORLD_C, filler_rate, alphabet, filler_families=filler_families
        )
        sample["language"] = language
        out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model_v15(
    train: list[dict],
    val: list[dict],
    vocab: dict[str, int],
    device: str,
    updates: int = MAX_UPDATES_DEFAULT,
) -> tuple[SignalModelV15, dict]:
    torch.manual_seed(MODEL_SEED)
    model = SignalModelV15(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
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
        mask_logit, recon_logit, world_logit = model(x, feats)
        loss_ctc = ctc_deletion_loss(mask_logit, x, batch["ctc_targets"], vocab_size)
        # Light auxiliary losses (same spirit as EXP-0013).
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
        if step % 100 == 0 or step == updates:
            model.eval()
            with torch.no_grad():
                probs = predict_mask_probs(model, val_c, vocab, device)
                masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
                # Rank model placeholder for val checkpoint: use zeros gain ok via evaluate
                rank_model = fit_rank_bigram([s["text"] for s in train if s["world"] == WORLD_B][:200] or [s["text"] for s in train[:200]])
                metrics = evaluate_masks(val_c, masks, rank_model)
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


def language_passes(neural: dict, random_b: dict, vocab_b: dict) -> dict:
    reasons = []
    ok = True
    nr = float(neural.get("null_recall", 0.0))
    np_ = float(neural.get("null_precision", 0.0))
    pr = float(neural.get("pred_null_rate", 0.0))
    if nr < PASS15_NULL_RECALL_MIN:
        ok = False
        reasons.append(f"null_recall {nr:.4f} < {PASS15_NULL_RECALL_MIN}")
    if np_ < PASS15_NULL_PRECISION_MIN:
        ok = False
        reasons.append(f"null_precision {np_:.4f} < {PASS15_NULL_PRECISION_MIN}")
    if not (PASS15_PRED_NULL_RATE_MIN <= pr <= PASS15_PRED_NULL_RATE_MAX):
        ok = False
        reasons.append(f"pred_null_rate {pr:.4f} out of band")
    if neural["recon_acc"] <= random_b["recon_acc"]:
        ok = False
        reasons.append(
            f"recon_acc {neural['recon_acc']:.4f} not > matched_random {random_b['recon_acc']:.4f}"
        )
    if vocab_b["recon_acc"] >= neural["recon_acc"]:
        ok = False
        reasons.append(
            f"vocab_filter recon {vocab_b['recon_acc']:.4f} >= model {neural['recon_acc']:.4f} (cheat surface)"
        )
    return {"pass": ok, "reasons": reasons}


def apply_pass_rule_v15(per_lang: dict[str, dict], catalog: dict[str, dict]) -> dict:
    passed_langs = [lang for lang, row in per_lang.items() if row["decision"]["pass"]]
    non_ie_passed = [
        lang
        for lang in passed_langs
        if not is_indo_european(catalog[lang]["family"])
    ]
    macro_recon = float(np.mean([row["neural"]["recon_acc"] for row in per_lang.values()]))
    macro_rand = float(
        np.mean([row["baselines"]["matched_random"]["recon_acc"] for row in per_lang.values()])
    )
    run_pass = (
        len(passed_langs) >= 4
        and len(non_ie_passed) >= 1
        and macro_recon > macro_rand
    )
    reasons = []
    if len(passed_langs) < 4:
        reasons.append(f"only {len(passed_langs)} languages passed (need ≥4)")
    if len(non_ie_passed) < 1:
        reasons.append("no non-Indo-European holdout passed")
    if not (macro_recon > macro_rand):
        reasons.append(f"macro recon {macro_recon:.4f} not > macro matched_random {macro_rand:.4f}")
    return {
        "passed": run_pass,
        "n_languages_scored": len(per_lang),
        "languages_passed": passed_langs,
        "non_ie_passed": non_ie_passed,
        "macro_recon_acc": macro_recon,
        "macro_matched_random_recon_acc": macro_rand,
        "reasons": reasons,
        "thresholds": {
            "null_recall_min": PASS15_NULL_RECALL_MIN,
            "null_precision_min": PASS15_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS15_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS15_PRED_NULL_RATE_MAX,
            "min_languages_pass": 4,
            "require_non_ie": True,
            "macro_recon_vs_matched_random": "strictly_greater",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    filler_families = parse_filler_families(args.filler_families) if args.filler_families else EASY_FILLER_FAMILIES
    raw_root = root / "data" / "raw" / "multilang_corpora"
    proc_root = root / "data" / "processed" / "exp0015"
    out_root = root / "outputs" / "EXP-0015"
    res_root = root / "results" / "EXP-0015"
    for p in (proc_root, out_root, res_root, root / "data" / "manifests"):
        p.mkdir(parents=True, exist_ok=True)

    acquired = acquire_corpora(raw_root)
    texts = acquired["texts"]
    write_json(root / "data" / "manifests" / "exp0015_corpora.json", acquired["manifest"])

    train_langs = sorted(
        lang
        for lang, meta in LANGUAGE_CATALOG.items()
        if meta["role"] == "train" and lang in texts
    )
    holdout_langs = sorted(
        lang
        for lang, meta in LANGUAGE_CATALOG.items()
        if meta["role"] == "holdout" and lang in texts
    )
    historical_langs = sorted(
        lang
        for lang, meta in LANGUAGE_CATALOG.items()
        if meta["role"] == "historical" and lang in texts
    )
    if len(train_langs) < 10:
        raise RuntimeError(
            f"Need ≥10 training languages; acquired {len(train_langs)}: {train_langs}; "
            f"skipped={list(acquired['manifest']['skipped'])}"
        )
    if len(holdout_langs) < 4:
        raise RuntimeError(
            f"Need ≥4 holdout languages; acquired {len(holdout_langs)}: {holdout_langs}"
        )

    if args.download_only:
        return {
            "train_langs": train_langs,
            "holdout_langs": holdout_langs,
            "historical_langs": historical_langs,
            "skipped": acquired["manifest"]["skipped"],
        }

    train, val = generate_multilang_dataset(
        {k: texts[k] for k in train_langs},
        train_langs,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=DATA_SEED,
        filler_families=filler_families,
    )

    def dump(name: str, rows: list[dict]) -> str:
        path = proc_root / name
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

    digests = {
        "train": dump("train.jsonl", train),
        "validation": dump("validation.jsonl", val),
    }
    holdouts: dict[str, list[dict]] = {}
    for i, lang in enumerate(holdout_langs):
        rows = generate_language_holdout(
            texts[lang],
            lang,
            n=args.n_holdout,
            seed=FINNISH_SEED + 17 * (i + 1),
            filler_families=filler_families,
        )
        holdouts[lang] = rows
        digests[f"holdout_{lang}"] = dump(f"holdout_{lang}.jsonl", rows)
    historical: dict[str, list[dict]] = {}
    for i, lang in enumerate(historical_langs):
        rows = generate_language_holdout(
            texts[lang],
            lang,
            n=min(100, args.n_holdout),
            seed=FINNISH_SEED + 1000 + i,
            filler_families=filler_families,
        )
        historical[lang] = rows
        digests[f"historical_{lang}"] = dump(f"historical_{lang}.jsonl", rows)

    write_json(
        root / "data" / "manifests" / "exp0015_data.json",
        {
            "experiment": "EXP-0015",
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "holdout_seed_base": FINNISH_SEED,
            "filler_rate": PRIMARY_FILLER_RATE,
            "filler_families": list(filler_families),
            "train_languages": train_langs,
            "holdout_languages": holdout_langs,
            "historical_languages": historical_langs,
            "seq_len": SEQ_LEN,
            "n_train": len(train),
            "n_val": len(val),
            "n_holdout_per_lang": args.n_holdout,
            "derived_sha256": digests,
            "decoder": "exact_count_keep_round((1-0.30)*L)",
            "pass_rule": {
                "null_recall_min": PASS15_NULL_RECALL_MIN,
                "null_precision_min": PASS15_NULL_PRECISION_MIN,
                "pred_null_rate_band": [PASS15_PRED_NULL_RATE_MIN, PASS15_PRED_NULL_RATE_MAX],
                "recon_vs_matched_random": "strictly_greater_per_language",
                "vocab_cheat": "vocab_filter.recon_acc >= model.recon_acc ⇒ language FAIL",
                "run_pass": "≥4 holdouts incl. ≥1 non-IE AND macro recon > macro matched_random",
            },
            "latin_in_training": False,
            "note": "Latin reserved as IE holdout; EXP-0014 Finnish-only confirmation not scored",
        },
    )

    if args.prereg_only:
        return {
            "train_langs": train_langs,
            "holdout_langs": holdout_langs,
            "historical_langs": historical_langs,
            "data_manifest": "data/manifests/exp0015_data.json",
        }

    vocab = build_vocab()
    device = resolve_device(args.device)
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400]
        or [s["text"] for s in train[:400]]
    )
    model, train_summary = train_model_v15(
        train, val, vocab, device, updates=args.updates
    )
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
            import re as _re

            train_token_vocab.update(_re.findall(r"\S+", s["text"]))

    per_lang: dict[str, dict] = {}
    for lang, rows in holdouts.items():
        probs = predict_mask_probs(model, rows, vocab, device)
        neural_masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
        classical_masks = [
            classical_hsmm_null_mask(s["text"], PRIMARY_FILLER_RATE) for s in rows
        ]
        neural = evaluate_masks(rows, neural_masks, rank_model)
        classical = evaluate_masks(rows, classical_masks, rank_model)
        neural["null_f1"] = null_f1_from_metrics(neural)
        classical["null_f1"] = null_f1_from_metrics(classical)
        majority = majority_baseline(rows, rank_model)
        random_b = matched_random_baseline(rows, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)
        vocab_b = vocab_filter_baseline(rows, train_token_vocab, rank_model)
        decision = language_passes(neural, random_b, vocab_b)
        row = {
            "family": LANGUAGE_CATALOG[lang]["family"],
            "indo_european": is_indo_european(LANGUAGE_CATALOG[lang]["family"]),
            "neural": neural,
            "classical": classical,
            "baselines": {
                "majority": majority,
                "matched_random": random_b,
                "vocab_filter": vocab_b,
            },
            "decision": decision,
        }
        if lang == "finnish":
            row["blindness"] = (
                "confirmatory_for_exact_count_decode: EXP-0013 exploratory Finnish "
                "exact-count ~0.218 was peeked; Finnish language still held out of training"
            )
        per_lang[lang] = row

    historical_scores: dict[str, dict] = {}
    for lang, rows in historical.items():
        probs = predict_mask_probs(model, rows, vocab, device)
        neural_masks = exact_count_masks(probs, PRIMARY_FILLER_RATE)
        neural = evaluate_masks(rows, neural_masks, rank_model)
        neural["null_f1"] = null_f1_from_metrics(neural)
        random_b = matched_random_baseline(rows, rank_model)
        historical_scores[lang] = {
            "family": LANGUAGE_CATALOG[lang]["family"],
            "neural": neural,
            "matched_random": random_b,
            "note": "Score-only historical transfer; not used in pass decision",
        }

    decision = apply_pass_rule_v15(per_lang, LANGUAGE_CATALOG)
    report = {
        "experiment": "EXP-0015",
        "environment": environment(),
        "train_languages": train_langs,
        "holdout_languages": holdout_langs,
        "historical_languages": historical_langs,
        "skipped_languages": acquired["manifest"]["skipped"],
        "filler_families": list(filler_families),
        "decoder": "exact_count_keep_round((1-0.30)*L)",
        "train_summary": {k: v for k, v in train_summary.items() if k != "history"},
        "train_history_tail": train_summary["history"][-5:],
        "param_count": train_summary["param_count"],
        "per_language": per_lang,
        "historical": historical_scores,
        "decision": decision,
        "data_digests": digests,
        "voynich_label_free": None,
        "simplifications": [
            "All plaintext romanized to a-z + space before ciphering",
            "Latin excluded from training (IE holdout)",
            "Mandarin/Japanese/Korean omitted unless romanized sources acquired",
            "Exact-count decoder frozen; not threshold-calibrated on holdouts",
            "Historical Italian scored but not used to pick the winner",
            "Not a decipherment",
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
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
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prereg-only", action="store_true", help="Acquire data + write manifests; no train/score")
    args = parser.parse_args()
    report = run_experiment(args)
    # Compact stdout
    if "decision" in report:
        table = {
            lang: {
                "recon": row["neural"]["recon_acc"],
                "rand": row["baselines"]["matched_random"]["recon_acc"],
                "pass": row["decision"]["pass"],
                "family": row["family"],
            }
            for lang, row in report["per_language"].items()
        }
        print(
            json.dumps(
                {
                    "experiment": report["experiment"],
                    "passed": report["decision"]["passed"],
                    "param_count": report["param_count"],
                    "train_languages": report["train_languages"],
                    "holdout_languages": report["holdout_languages"],
                    "decision": report["decision"],
                    "per_language": table,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
