"""EXP-0018: preregistered null-rate grid on ZL3b validation (select/confirm).

Uses frozen EXP-0016 checkpoint. Does not open the ZL3b test split.
Not a decipherment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from voynich.latent_recovery import (
    SEQ_LEN,
    bigram_bits,
    classical_hsmm_null_mask,
    fit_rank_bigram,
    pred_bits_of_selected,
    predict_mask_probs,
    rank_bucket_sequence,
)
from voynich.mass_lang_recovery import SignalModelV16
from voynich.multilang_recovery import exact_count_masks
from voynich.runtime import digest, environment, resolve_device, write_json

EXPERIMENT_ID = "EXP-0018"
SPLIT_SEED = 4018
MATCHED_RANDOM_SEEDS = 20
MATCHED_RANDOM_SEED_BASE = 9011
BEATS_RANDOM_MARGIN = 0.05
RATE_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
RATE_REFERENCE = (0.00,)
EXPECTED_CKPT_SHA256 = "434d84f80f405f375dee5a2108da667b22e4f622dd3688e2111e89975856a359"


def _strip_pua(text: str) -> str:
    return re.sub(r"[\ue000-\uf8ff]", "", text)


def load_pages(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        rows.append((page["page_id"], _strip_pua(page["text"])))
    return rows


def make_windows(page_id: str, text: str, win: int = SEQ_LEN) -> list[dict]:
    out = []
    if len(text) < 32:
        return out
    chunks = [text] if len(text) <= win else []
    if not chunks:
        for i in range(0, len(text), win):
            chunk = text[i : i + win]
            if len(chunk) >= 32:
                chunks.append(chunk)
    for chunk in chunks:
        padded = chunk + (" " * (win - len(chunk))) if len(chunk) < win else chunk[:win]
        out.append(
            {
                "page_id": page_id,
                "text": padded,
                "raw_len": len(chunk),
                "mask": [1] * win,
                "ciphered": padded,
            }
        )
    return out


def evaluate_structure(
    eval_samples: list[dict],
    masks: list[np.ndarray],
    rank_model: dict,
    *,
    null_rate_assumption: float,
    n_rand_seeds: int = MATCHED_RANDOM_SEEDS,
    collect_kept: bool = False,
) -> dict:
    gains, rand_means, null_rates, beats = [], [], [], []
    kept_stats = []
    for s, pred in zip(eval_samples, masks):
        text = s["text"]
        pred = np.asarray(pred, dtype=int)[: len(text)]
        if len(pred) < len(text):
            pred = np.pad(pred, (0, len(text) - len(pred)))
        full_bits = bigram_bits(rank_bucket_sequence(text), rank_model)
        sel_bits = pred_bits_of_selected(text, pred, rank_model)
        gain = full_bits - sel_bits
        n_null = int((pred == 0).sum())
        rand_gains = []
        rand_kept = []
        for seed in range(n_rand_seeds):
            r = np.random.default_rng(MATCHED_RANDOM_SEED_BASE + seed)
            m = np.ones(len(text), dtype=int)
            if n_null:
                m[r.choice(len(text), size=min(n_null, len(text)), replace=False)] = 0
            rb = pred_bits_of_selected(text, m, rank_model)
            rand_gains.append(full_bits - rb)
            if collect_kept:
                rand_kept.append("".join(ch for ch, keep in zip(text, m) if keep == 1))
        rg = float(np.mean(rand_gains))
        gains.append(gain)
        rand_means.append(rg)
        null_rates.append(float(1.0 - pred.mean()) if len(pred) else 0.0)
        beats.append(gain > rg + BEATS_RANDOM_MARGIN)
        if collect_kept:
            kept = "".join(ch for ch, keep in zip(text, pred) if keep == 1)
            kept_stats.append(
                {
                    "kept": kept,
                    "rand_kept_mean_len": float(np.mean([len(x) for x in rand_kept])) if rand_kept else 0.0,
                    "full_len": len(text),
                }
            )
    mean_gain = float(np.mean(gains)) if gains else 0.0
    mean_random = float(np.mean(rand_means)) if rand_means else 0.0
    out = {
        "n_windows": len(eval_samples),
        "mean_gain": mean_gain,
        "mean_random_gain": mean_random,
        "delta_vs_random": mean_gain - mean_random,
        "mean_pred_null_rate": float(np.mean(null_rates)) if null_rates else 0.0,
        "fraction_beats_random": float(np.mean(beats)) if beats else 0.0,
        "null_rate_assumption": float(null_rate_assumption),
    }
    if collect_kept and kept_stats:
        neural_kept = "".join(row["kept"] for row in kept_stats)
        # Aggregate surface stats on concatenated kept streams (descriptive only).
        words = neural_kept.split()
        adj = 0
        for a, b in zip(words, words[1:]):
            if a == b:
                adj += 1
        adj_rate = adj / max(1, len(words) - 1) if len(words) > 1 else 0.0
        chars = [c for c in neural_kept if c != "\n"]
        uni = Counter(chars)
        bi = Counter(zip(chars, chars[1:]))
        h2 = 0.0
        if chars:
            total = sum(uni.values())
            for (a, b), c in bi.items():
                p_ab = c / max(1, len(chars) - 1)
                p_a = uni[a] / total
                if p_ab > 0 and p_a > 0:
                    h2 -= p_ab * math.log2(p_ab / p_a)
        out["kept_stream_aggregate"] = {
            "n_chars": len(neural_kept),
            "mean_kept_len": float(np.mean([len(r["kept"]) for r in kept_stats])),
            "mean_full_len": float(np.mean([r["full_len"] for r in kept_stats])),
            "mean_rand_kept_len": float(np.mean([r["rand_kept_mean_len"] for r in kept_stats])),
            "adjacent_word_repeat_rate": float(adj_rate),
            "char_h2_bits": float(h2),
        }
    return out


def select_rate(table: list[dict]) -> dict:
    """Choose among candidates only (exclude reference rate 0)."""
    candidates = [row for row in table if row["role"] == "candidate"]
    if not candidates:
        raise RuntimeError("no candidate rates")
    best = sorted(
        candidates,
        key=lambda r: (
            -r["neural"]["delta_vs_random"],
            -r["neural"]["fraction_beats_random"],
            r["null_rate"],
        ),
    )[0]
    return best


def apply_pass_rule(confirm_neural: dict) -> dict:
    mean_ok = confirm_neural["mean_gain"] > confirm_neural["mean_random_gain"]
    frac_ok = confirm_neural["fraction_beats_random"] > 0.50
    passed = bool(mean_ok and frac_ok)
    reasons = []
    if not mean_ok:
        reasons.append(
            f"mean_gain {confirm_neural['mean_gain']:.6f} not > mean_random_gain "
            f"{confirm_neural['mean_random_gain']:.6f}"
        )
    if not frac_ok:
        reasons.append(
            f"fraction_beats_random {confirm_neural['fraction_beats_random']:.4f} not > 0.50"
        )
    return {
        "passed": passed,
        "reasons": reasons if not passed else ["confirm mean_gain > mean_random_gain and fraction_beats_random > 0.50"],
        "criteria": {
            "mean_gain_gt_mean_random_gain": mean_ok,
            "fraction_beats_random_gt_0_50": frac_ok,
        },
    }


def score_half(
    samples: list[dict],
    probs: list[np.ndarray],
    rank_model: dict,
    rates: list[float],
    *,
    include_classical: bool,
    collect_kept_for_rate: float | None = None,
) -> list[dict]:
    eval_samples = [{"text": s["text"][: s["raw_len"]]} for s in samples]
    trimmed = [np.asarray(p)[: s["raw_len"]] for p, s in zip(probs, samples)]
    rows = []
    for rate in rates:
        role = "reference" if abs(rate) < 1e-12 else "candidate"
        neural_masks = exact_count_masks(trimmed, rate)
        neural = evaluate_structure(
            eval_samples,
            neural_masks,
            rank_model,
            null_rate_assumption=rate,
            collect_kept=(collect_kept_for_rate is not None and abs(rate - collect_kept_for_rate) < 1e-12),
        )
        row = {"null_rate": float(rate), "role": role, "neural": neural, "classical": None}
        if include_classical:
            classical_masks = [classical_hsmm_null_mask(s["text"], rate) for s in eval_samples]
            row["classical"] = evaluate_structure(
                eval_samples,
                classical_masks,
                rank_model,
                null_rate_assumption=rate,
            )
        rows.append(row)
    return rows


def run(root: Path, device: str) -> dict:
    split_path = root / "data" / "manifests" / "exp0018_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split["split_seed"] == SPLIT_SEED

    val_path = root / "data" / "processed" / "zl3b" / "validation.jsonl"
    train_fit_path = root / "data" / "processed" / "zl3b" / "train.jsonl"
    # HARD GUARD: never score or fit on test.
    test_path = root / "data" / "processed" / "zl3b" / "test.jsonl"
    assert test_path.exists()
    assert val_path.resolve() != test_path.resolve()
    assert train_fit_path.resolve() != test_path.resolve()

    pages = load_pages(val_path)
    page_ids = [p for p, _ in pages]
    assert page_ids == split["validation_page_ids_in_file_order"]
    select_set = set(split["select_page_ids"])
    confirm_set = set(split["confirm_page_ids"])
    assert select_set.isdisjoint(confirm_set)
    assert select_set | confirm_set == set(page_ids)

    train_pages = [_strip_pua(json.loads(line)["text"]) for line in train_fit_path.read_text().splitlines() if line.strip()]
    rank_model = fit_rank_bigram(train_pages)

    ckpt_path = root / "outputs" / "EXP-0016" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt_path}; retrain required")
    ckpt_sha = digest(ckpt_path)
    if ckpt_sha != EXPECTED_CKPT_SHA256:
        raise RuntimeError(f"checkpoint sha mismatch: {ckpt_sha} != {EXPECTED_CKPT_SHA256}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    device = resolve_device(device)
    model = SignalModelV16(len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    select_samples: list[dict] = []
    confirm_samples: list[dict] = []
    for pid, text in pages:
        wins = make_windows(pid, text)
        if pid in select_set:
            select_samples.extend(wins)
        else:
            confirm_samples.extend(wins)

    select_probs = predict_mask_probs(model, select_samples, vocab, device)
    rates_all = list(RATE_REFERENCE) + list(RATE_CANDIDATES)
    select_table = score_half(
        select_samples,
        select_probs,
        rank_model,
        rates_all,
        include_classical=True,
    )
    chosen = select_rate(select_table)
    selected_rate = float(chosen["null_rate"])

    # Confirm half: ONLY the selected rate (plus optional classical at that rate).
    confirm_probs = predict_mask_probs(model, confirm_samples, vocab, device)
    confirm_rows = score_half(
        confirm_samples,
        confirm_probs,
        rank_model,
        [selected_rate],
        include_classical=True,
        collect_kept_for_rate=selected_rate,
    )
    confirm_neural = confirm_rows[0]["neural"]
    decision = apply_pass_rule(confirm_neural)

    out_root = root / "results" / EXPERIMENT_ID
    out_root.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": EXPERIMENT_ID,
        "environment": environment(),
        "preregistration": "docs/experiments/EXP-0018.md",
        "split_manifest": "data/manifests/exp0018_split.json",
        "split_seed": SPLIT_SEED,
        "select_page_ids": split["select_page_ids"],
        "confirm_page_ids": split["confirm_page_ids"],
        "test_split_scored": False,
        "test_split_opened": False,
        "checkpoint": {
            "path": "outputs/EXP-0016/model.pt",
            "sha256": ckpt_sha,
            "param_count": model.param_count(),
            "retrained": False,
        },
        "segmentation": {
            "units": "zl3b_transcribed_characters",
            "pua_stripped": True,
            "lowercased": False,
            "window": SEQ_LEN,
            "min_chunk": 32,
        },
        "bits_gain_definition": {
            "formula": "full_bits - selected_bits",
            "rank_model": "fit_rank_bigram on zl3b train only",
            "beats_random_margin": BEATS_RANDOM_MARGIN,
            "matched_random_seeds": MATCHED_RANDOM_SEEDS,
            "matched_random_seed_base": MATCHED_RANDOM_SEED_BASE,
        },
        "decoder": "exact_count_keep_round((1-r)*L)",
        "select_half": {
            "label": "selection_not_confirmation",
            "n_pages": len(select_set),
            "n_windows": len(select_samples),
            "rate_table": select_table,
            "selected_rate": selected_rate,
            "selection_rule": "max neural delta_vs_random; tie-break higher fraction_beats_random then lower rate; rate 0 excluded",
            "selected_row": chosen,
        },
        "confirm_half": {
            "label": "confirmation",
            "n_pages": len(confirm_set),
            "n_windows": len(confirm_samples),
            "selected_rate": selected_rate,
            "neural": confirm_neural,
            "classical": confirm_rows[0]["classical"],
        },
        "decision": decision,
        "note": (
            "Confirm PASS is rate-specific structure gain on held-aside validation pages only. "
            "Not a decipherment. Not a null-component proof. Synthetic EXP-0016 knew true deletion count."
        ),
    }
    write_json(out_root / "results.json", report)
    write_json(
        out_root / "decision.json",
        {
            "experiment": EXPERIMENT_ID,
            "passed": decision["passed"],
            "selected_rate": selected_rate,
            "confirm": {
                "mean_gain": confirm_neural["mean_gain"],
                "mean_random_gain": confirm_neural["mean_random_gain"],
                "fraction_beats_random": confirm_neural["fraction_beats_random"],
            },
            "reasons": decision["reasons"],
            "criteria": decision["criteria"],
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run(args.root.resolve(), args.device)
    d = report["decision"]
    c = report["confirm_half"]["neural"]
    print(
        json.dumps(
            {
                "selected_rate": report["select_half"]["selected_rate"],
                "select_best_delta": report["select_half"]["selected_row"]["neural"]["delta_vs_random"],
                "confirm_mean_gain": c["mean_gain"],
                "confirm_mean_random_gain": c["mean_random_gain"],
                "confirm_fraction_beats_random": c["fraction_beats_random"],
                "passed": d["passed"],
                "reasons": d["reasons"],
                "test_scored": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
