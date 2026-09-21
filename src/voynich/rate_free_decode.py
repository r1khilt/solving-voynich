"""EXP-0021: rate-free threshold keep/delete on frozen EXP-0016 checkpoint.

Does not open the ZL3b test split. Not a decipherment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from voynich.latent_recovery import (
    SEQ_LEN,
    bigram_bits,
    fit_rank_bigram,
    pred_bits_of_selected,
    predict_mask_probs,
    rank_bucket_sequence,
)
from voynich.mass_lang_recovery import SignalModelV16
from voynich.null_rate_grid import load_pages, make_windows
from voynich.runtime import digest, environment, resolve_device, write_json

EXPERIMENT_ID = "EXP-0021"
SPLIT_SEED = 4018
MATCHED_RANDOM_SEEDS = 20
MATCHED_RANDOM_SEED_BASE = 9011
BEATS_RANDOM_MARGIN = 0.05
SIGNAL_THRESHOLD = 0.5
REJECTED_RATE_MEDIAN_LO = 0.65
REJECTED_RATE_MEDIAN_HI = 0.75
MIN_RATE_STD = 0.05
EXPECTED_CKPT_SHA256 = "434d84f80f405f375dee5a2108da667b22e4f622dd3688e2111e89975856a359"


def threshold_masks(probs: list[np.ndarray], threshold: float = SIGNAL_THRESHOLD) -> list[np.ndarray]:
    """Keep position i iff P(signal)_i >= threshold. Rate-free; count is model-implied."""
    return [(np.asarray(p) >= threshold).astype(int) for p in probs]


def implied_null_rates(masks: list[np.ndarray], lengths: list[int]) -> np.ndarray:
    rates = []
    for m, L in zip(masks, lengths):
        pred = np.asarray(m, dtype=int)[:L]
        if L <= 0:
            rates.append(0.0)
        else:
            rates.append(float(1.0 - pred.mean()))
    return np.asarray(rates, dtype=float)


def rate_distribution(rates: np.ndarray) -> dict:
    if rates.size == 0:
        return {
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "std": 0.0,
            "n_windows": 0,
        }
    return {
        "min": float(np.min(rates)),
        "median": float(np.median(rates)),
        "mean": float(np.mean(rates)),
        "max": float(np.max(rates)),
        "std": float(np.std(rates, ddof=0)),
        "n_windows": int(rates.size),
    }


def evaluate_rate_free(
    eval_samples: list[dict],
    masks: list[np.ndarray],
    rank_model: dict,
    *,
    n_rand_seeds: int = MATCHED_RANDOM_SEEDS,
    collect_kept: bool = False,
) -> dict:
    gains, rand_means, beats = [], [], []
    lengths = [len(s["text"]) for s in eval_samples]
    rates = implied_null_rates(masks, lengths)
    kept_stats = []
    for s, pred, L in zip(eval_samples, masks, lengths):
        text = s["text"]
        pred = np.asarray(pred, dtype=int)[:L]
        if len(pred) < L:
            pred = np.pad(pred, (0, L - len(pred)))
        full_bits = bigram_bits(rank_bucket_sequence(text), rank_model)
        sel_bits = pred_bits_of_selected(text, pred, rank_model)
        gain = full_bits - sel_bits
        n_null = int((pred == 0).sum())
        rand_gains = []
        rand_kept = []
        for seed in range(n_rand_seeds):
            r = np.random.default_rng(MATCHED_RANDOM_SEED_BASE + seed)
            m = np.ones(L, dtype=int)
            if n_null:
                m[r.choice(L, size=min(n_null, L), replace=False)] = 0
            rb = pred_bits_of_selected(text, m, rank_model)
            rand_gains.append(full_bits - rb)
            if collect_kept:
                rand_kept.append("".join(ch for ch, keep in zip(text, m) if keep == 1))
        rg = float(np.mean(rand_gains))
        gains.append(gain)
        rand_means.append(rg)
        beats.append(gain > rg + BEATS_RANDOM_MARGIN)
        if collect_kept:
            kept = "".join(ch for ch, keep in zip(text, pred) if keep == 1)
            kept_stats.append(
                {
                    "kept": kept,
                    "full": text,
                    "rand_kept_mean_len": float(np.mean([len(x) for x in rand_kept])) if rand_kept else 0.0,
                    "full_len": L,
                }
            )
    mean_gain = float(np.mean(gains)) if gains else 0.0
    mean_random = float(np.mean(rand_means)) if rand_means else 0.0
    dist = rate_distribution(rates)
    out = {
        "n_windows": len(eval_samples),
        "mean_gain": mean_gain,
        "mean_random_gain": mean_random,
        "delta_vs_random": mean_gain - mean_random,
        "delete_nothing_gain": 0.0,
        "fraction_beats_random": float(np.mean(beats)) if beats else 0.0,
        "implied_null_rate_distribution": dist,
        "decode_rule": f"keep iff P(signal) >= {SIGNAL_THRESHOLD}",
        "signal_threshold": SIGNAL_THRESHOLD,
    }
    if collect_kept and kept_stats:
        neural_kept = "".join(row["kept"] for row in kept_stats)
        full_concat = "".join(row["full"] for row in kept_stats)

        def _stream_stats(s: str) -> dict:
            words = s.split()
            adj = 0
            for a, b in zip(words, words[1:]):
                if a == b:
                    adj += 1
            adj_rate = adj / max(1, len(words) - 1) if len(words) > 1 else 0.0
            chars = [c for c in s if c != "\n"]
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
            return {
                "n_chars": len(s),
                "adjacent_word_repeat_rate": float(adj_rate),
                "char_h2_bits": float(h2),
            }

        out["kept_stream_aggregate"] = {
            **_stream_stats(neural_kept),
            "mean_kept_len": float(np.mean([len(r["kept"]) for r in kept_stats])),
            "mean_full_len": float(np.mean([r["full_len"] for r in kept_stats])),
            "mean_rand_kept_len": float(np.mean([r["rand_kept_mean_len"] for r in kept_stats])),
            "full_stream": _stream_stats(full_concat),
        }
    return out


def apply_pass_rule(confirm: dict) -> dict:
    dist = confirm["implied_null_rate_distribution"]
    mean_gain = confirm["mean_gain"]
    mean_random = confirm["mean_random_gain"]
    frac = confirm["fraction_beats_random"]
    rate_std = dist["std"]
    rate_median = dist["median"]

    gain_pos = mean_gain > 0.0
    gain_vs_rand = mean_gain > mean_random
    frac_ok = frac > 0.50
    std_ok = rate_std > MIN_RATE_STD
    median_ok = not (REJECTED_RATE_MEDIAN_LO <= rate_median <= REJECTED_RATE_MEDIAN_HI)
    rate_ok = std_ok and median_ok

    passed = bool(gain_pos and gain_vs_rand and frac_ok and rate_ok)
    reasons = []
    if not gain_pos:
        reasons.append(f"mean_gain {mean_gain:.6f} not > 0 (kept not more predictable than full text)")
    if not gain_vs_rand:
        reasons.append(f"mean_gain {mean_gain:.6f} not > mean_random_gain {mean_random:.6f}")
    if not frac_ok:
        reasons.append(f"fraction_beats_random {frac:.4f} not > 0.50")
    if not std_ok:
        reasons.append(f"implied_null_rate std {rate_std:.6f} not > {MIN_RATE_STD}")
    if not median_ok:
        reasons.append(
            f"median implied_null_rate {rate_median:.6f} inside rejected "
            f"[{REJECTED_RATE_MEDIAN_LO}, {REJECTED_RATE_MEDIAN_HI}] neighborhood"
        )
    return {
        "passed": passed,
        "reasons": reasons
        if not passed
        else [
            "confirm: mean_gain>0, mean_gain>mean_random, fraction_beats_random>0.50, "
            "rate std>0.05 and median outside [0.65,0.75]"
        ],
        "criteria": {
            "mean_gain_gt_0": gain_pos,
            "mean_gain_gt_mean_random_gain": gain_vs_rand,
            "fraction_beats_random_gt_0_50": frac_ok,
            "implied_null_rate_std_gt_0_05": std_ok,
            "median_implied_null_rate_outside_0_65_0_75": median_ok,
        },
    }


def run(root: Path, device: str) -> dict:
    split_path = root / "data" / "manifests" / "exp0021_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split["split_seed"] == SPLIT_SEED
    legacy = json.loads((root / "data" / "manifests" / "exp0018_split.json").read_text(encoding="utf-8"))
    assert split["select_page_ids"] == legacy["select_page_ids"]
    assert split["confirm_page_ids"] == legacy["confirm_page_ids"]

    val_path = root / "data" / "processed" / "zl3b" / "validation.jsonl"
    train_fit_path = root / "data" / "processed" / "zl3b" / "train.jsonl"
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

    from voynich.null_rate_grid import _strip_pua

    train_pages = [
        _strip_pua(json.loads(line)["text"])
        for line in train_fit_path.read_text().splitlines()
        if line.strip()
    ]
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
    confirm_probs = predict_mask_probs(model, confirm_samples, vocab, device)

    select_eval = [{"text": s["text"][: s["raw_len"]]} for s in select_samples]
    confirm_eval = [{"text": s["text"][: s["raw_len"]]} for s in confirm_samples]
    select_masks = threshold_masks(
        [np.asarray(p)[: s["raw_len"]] for p, s in zip(select_probs, select_samples)]
    )
    confirm_masks = threshold_masks(
        [np.asarray(p)[: s["raw_len"]] for p, s in zip(confirm_probs, confirm_samples)]
    )

    select_neural = evaluate_rate_free(select_eval, select_masks, rank_model, collect_kept=False)
    confirm_neural = evaluate_rate_free(confirm_eval, confirm_masks, rank_model, collect_kept=True)
    decision = apply_pass_rule(confirm_neural)

    out_root = root / "results" / EXPERIMENT_ID
    out_root.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": EXPERIMENT_ID,
        "environment": environment(),
        "preregistration": "docs/experiments/EXP-0021.md",
        "split_manifest": "data/manifests/exp0021_split.json",
        "reuses_exp0018_split": True,
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
            "positive_means": "kept stream more predictable than full text",
            "rank_model": "fit_rank_bigram on zl3b train only",
            "beats_random_margin": BEATS_RANDOM_MARGIN,
            "matched_random_seeds": MATCHED_RANDOM_SEEDS,
            "matched_random_seed_base": MATCHED_RANDOM_SEED_BASE,
            "delete_nothing_gain": 0.0,
        },
        "decode_rule": {
            "name": "threshold_keep_p_signal_ge_0.5",
            "keep_when": "P(signal) >= 0.5",
            "threshold": SIGNAL_THRESHOLD,
            "global_exact_count_rate": None,
            "tuned_on_voynich": False,
        },
        "select_half": {
            "label": "diagnostic_only_not_confirmation",
            "n_pages": len(select_set),
            "n_windows": len(select_samples),
            "neural": select_neural,
        },
        "confirm_half": {
            "label": "confirmation",
            "n_pages": len(confirm_set),
            "n_windows": len(confirm_samples),
            "neural": confirm_neural,
        },
        "decision": decision,
        "note": (
            "Confirm PASS is rate-free structure gain on held-aside validation pages only. "
            "Not a decipherment. Not a null-component proof. "
            "Negative gain is damage; less-negative than random is not a null layer."
        ),
    }
    write_json(out_root / "results.json", report)
    write_json(
        out_root / "decision.json",
        {
            "experiment": EXPERIMENT_ID,
            "passed": decision["passed"],
            "decode_rule": "keep iff P(signal) >= 0.5",
            "confirm": {
                "mean_gain": confirm_neural["mean_gain"],
                "mean_random_gain": confirm_neural["mean_random_gain"],
                "delete_nothing_gain": 0.0,
                "fraction_beats_random": confirm_neural["fraction_beats_random"],
                "implied_null_rate_distribution": confirm_neural["implied_null_rate_distribution"],
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
    s = report["select_half"]["neural"]
    print(
        json.dumps(
            {
                "decode_rule": report["decode_rule"],
                "select_implied_rates": s["implied_null_rate_distribution"],
                "confirm_implied_rates": c["implied_null_rate_distribution"],
                "confirm_mean_gain": c["mean_gain"],
                "confirm_mean_random_gain": c["mean_random_gain"],
                "confirm_delete_nothing_gain": 0.0,
                "confirm_fraction_beats_random": c["fraction_beats_random"],
                "passed": d["passed"],
                "reasons": d["reasons"],
                "criteria": d["criteria"],
                "test_scored": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
