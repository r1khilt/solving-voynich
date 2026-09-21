"""Decode-only exact-count / joint keep-run on frozen EXP-0013 weights.

EXP-0014 / 0014b / 0014c. No encoder retrain. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from voynich.latent_recovery import (
    EASY_FILLER_FAMILIES,
    FINNISH_SEED,
    MATCHED_RANDOM_SEEDS,
    PASS13_NULL_PRECISION_MIN,
    PASS13_NULL_RECALL_MIN,
    PASS13_PRED_NULL_RATE_MAX,
    PASS13_PRED_NULL_RATE_MIN,
    PASS13_VOCAB_F1_MAX,
    PRIMARY_FILLER_RATE,
    TinySignalModel,
    WORLD_B,
    WORLD_C,
    apply_pass_rule_v13,
    classical_hsmm_null_mask,
    evaluate_masks,
    fit_rank_bigram,
    generate_finnish_holdout,
    matched_random_baseline,
    majority_baseline,
    null_f1_from_metrics,
    predict_mask_probs,
    vocab_filter_baseline,
)
from voynich.runtime import digest, environment, resolve_device, write_json

# Frozen EXP-0014 gate (EXP-0013 recorded matched-random; do not recompute as the gate).
FROZEN_MATCHED_RANDOM_RECON = 0.2043264147237504
EXP0013_HOLDOUT_SHA256 = "31984a91eac2d4fecc44b716805dba11c939eb2c8a0b951090e683e7a3b93740"
EXP0013_TRAIN_SHA256 = "8c996b2183b1b1784a6e3945d845f2e1ec4eec65393ba0ab0b75c655361bc432"
EXP0013_VAL_SHA256 = "0d516e073ebd16d1a39a688a6a041900f94b72e4d10dacfc46c6eda8887d65ad"
STAY_KEEP = 0.88
STAY_DELETE = 0.72


def exact_count_keep_masks(
    probs: list[np.ndarray],
    filler_rate: float = PRIMARY_FILLER_RATE,
) -> list[np.ndarray]:
    """Keep top round((1-rate)*L) by keep-prob; ties → lower index."""
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


def joint_keep_run_masks(
    probs: list[np.ndarray],
    filler_rate: float = PRIMARY_FILLER_RATE,
    stay_keep: float = STAY_KEEP,
    stay_delete: float = STAY_DELETE,
) -> list[np.ndarray]:
    """Exact n_keep plus sticky keep/delete transitions (EXP-0014c)."""
    log_stay_k = math.log(stay_keep)
    log_sw_kd = math.log(1.0 - stay_keep)
    log_stay_d = math.log(stay_delete)
    log_sw_dk = math.log(1.0 - stay_delete)
    log_start_k = math.log(1.0 - filler_rate)
    log_start_d = math.log(filler_rate)
    neg = -1.0e30
    out = []
    for raw in probs:
        p = np.clip(np.asarray(raw, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
        length = int(p.size)
        n_keep = int(round((1.0 - filler_rate) * length))
        n_keep = max(0, min(length, n_keep))
        log_k = np.log(p)
        log_d = np.log(1.0 - p)
        nk = n_keep + 1
        dp_k = np.full((length, nk), neg)
        dp_d = np.full((length, nk), neg)
        ptr_k = np.zeros((length, nk), dtype=np.int8)
        ptr_d = np.zeros((length, nk), dtype=np.int8)
        if n_keep >= 1:
            dp_k[0, 1] = log_start_k + log_k[0]
        dp_d[0, 0] = log_start_d + log_d[0]
        for i in range(1, length):
            for kept in range(nk):
                if kept >= 1:
                    from_k = dp_k[i - 1, kept - 1] + log_stay_k
                    from_d = dp_d[i - 1, kept - 1] + log_sw_dk
                    if from_k >= from_d:
                        dp_k[i, kept] = from_k + log_k[i]
                        ptr_k[i, kept] = 1
                    else:
                        dp_k[i, kept] = from_d + log_k[i]
                        ptr_k[i, kept] = 0
                from_d = dp_d[i - 1, kept] + log_stay_d
                from_k = dp_k[i - 1, kept] + log_sw_kd
                if from_k >= from_d:
                    dp_d[i, kept] = from_k + log_d[i]
                    ptr_d[i, kept] = 1
                else:
                    dp_d[i, kept] = from_d + log_d[i]
                    ptr_d[i, kept] = 0
        last = length - 1
        use_keep = dp_k[last, n_keep] >= dp_d[last, n_keep]
        mask = np.zeros(length, dtype=int)
        kept = n_keep
        for i in range(last, -1, -1):
            if use_keep:
                mask[i] = 1
                prev_keep = bool(ptr_k[i, kept])
                kept -= 1
            else:
                mask[i] = 0
                prev_keep = bool(ptr_d[i, kept])
            use_keep = prev_keep
        out.append(mask)
    return out


def load_jsonl_samples(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["mask"] = list(row["mask"])
            rows.append(row)
    return rows


def apply_pass_rule_v14(neural: dict, vocab_b: dict, frozen_random_recon: float = FROZEN_MATCHED_RANDOM_RECON) -> dict:
    """EXP-0014 confirmation rule: neural only; recon vs frozen 0013 matched-random."""
    reasons = []
    ok = True
    nr = float(neural.get("null_recall", 0.0))
    np_ = float(neural.get("null_precision", 0.0))
    pr = float(neural.get("pred_null_rate", 0.0))
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
    if neural["recon_acc"] <= frozen_random_recon:
        ok = False
        reasons.append(
            f"recon_acc {neural['recon_acc']:.4f} not > frozen_matched_random "
            f"{frozen_random_recon:.4f}"
        )
    if vocab_b["mask_f1"] > PASS13_VOCAB_F1_MAX:
        ok = False
        reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS13_VOCAB_F1_MAX} (cheat surface)")
    if not reasons:
        reasons.append("all EXP-0014 gates cleared")
    return {
        "passed": ok,
        "winner": "neural" if ok else None,
        "rule": "EXP-0014",
        "neural_check": {
            "model": "neural",
            "pass": ok,
            "reasons": reasons,
            "metrics": neural,
            "null_f1": null_f1_from_metrics(neural),
        },
        "thresholds": {
            "null_recall_min": PASS13_NULL_RECALL_MIN,
            "null_precision_min": PASS13_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
            "recon_vs_frozen_matched_random": frozen_random_recon,
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        },
    }


def load_frozen_exp0013(root: Path, device: str) -> tuple[TinySignalModel, dict, dict]:
    ckpt_path = root / "outputs" / "EXP-0013" / "model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = payload["vocab"]
    model = TinySignalModel(len(vocab)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, vocab, payload


def finnish_holdout_for_eval(
    root: Path,
    filler_families: tuple[str, ...],
    require_exp0013_digest: bool,
) -> tuple[list[dict], str, bool]:
    """Load frozen EXP-0013 holdout when families match; else regenerate."""
    proc = root / "data" / "processed" / "exp0013"
    hold_path = proc / "finnish_holdout.jsonl"
    easy = tuple(filler_families) == tuple(EASY_FILLER_FAMILIES)
    if easy and hold_path.is_file() and digest(hold_path) == EXP0013_HOLDOUT_SHA256:
        return load_jsonl_samples(hold_path), digest(hold_path), False
    if require_exp0013_digest and easy:
        raise RuntimeError(
            f"EXP-0013 holdout digest mismatch or missing at {hold_path}; "
            "refusing to silently substitute another set for EXP-0014"
        )
    from voynich.latent_recovery import clean_plaintext

    raw = root / "data" / "raw" / "latent_corpora" / "finnish.clean.txt"
    if not raw.is_file():
        raw = root / "data" / "raw" / "latent_corpora" / "finnish.txt"
    text = clean_plaintext(raw.read_text(encoding="utf-8", errors="replace"))
    rows = generate_finnish_holdout(text, n=300, seed=FINNISH_SEED, filler_families=filler_families)
    return rows, "regenerated", True


def run_decode_only(
    root: Path,
    experiment_id: str,
    decoder: str,
    device: str,
    filler_families: tuple[str, ...],
) -> dict:
    device = resolve_device(device)
    model, vocab, ckpt = load_frozen_exp0013(root, device)
    train_path = root / "data" / "processed" / "exp0013" / "train.jsonl"
    if not train_path.is_file() or digest(train_path) != EXP0013_TRAIN_SHA256:
        raise RuntimeError("EXP-0013 train.jsonl missing or digest mismatch; needed for vocab-filter/rank model")
    train = load_jsonl_samples(train_path)
    require_digest = experiment_id.upper() in {"EXP-0014", "EXP0014"}
    holdout, hold_digest, regenerated = finnish_holdout_for_eval(
        root, filler_families, require_exp0013_digest=require_digest
    )
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400] or [s["text"] for s in train[:400]]
    )
    probs = predict_mask_probs(model, holdout, vocab, device)
    if decoder == "exact_count":
        neural_preds = exact_count_keep_masks(probs, PRIMARY_FILLER_RATE)
    elif decoder == "joint_keep_run":
        neural_preds = joint_keep_run_masks(probs, PRIMARY_FILLER_RATE)
    else:
        raise ValueError(decoder)
    classical_preds = [classical_hsmm_null_mask(s["text"], PRIMARY_FILLER_RATE) for s in holdout]
    neural = evaluate_masks(holdout, neural_preds, rank_model)
    classical = evaluate_masks(holdout, classical_preds, rank_model)
    neural["null_f1"] = null_f1_from_metrics(neural)
    classical["null_f1"] = null_f1_from_metrics(classical)
    gold = evaluate_masks(holdout, [np.array(s["mask"], dtype=int) for s in holdout], rank_model)
    neural["teacher_forced_recon_acc"] = gold["recon_acc"]
    majority = majority_baseline(holdout, rank_model)
    random_b = matched_random_baseline(holdout, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)
    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)
    decision = apply_pass_rule_v14(neural, vocab_b)
    v13_note = apply_pass_rule_v13(neural, classical, majority, random_b, vocab_b)
    slug = experiment_id.lower().replace("-", "")
    res_root = root / "results" / experiment_id
    res_root.mkdir(parents=True, exist_ok=True)
    man_path = root / "data" / "manifests" / f"{slug}_data.json"
    write_json(
        man_path,
        {
            "experiment": experiment_id,
            "decoder": decoder,
            "checkpoint": "outputs/EXP-0013/model.pt",
            "holdout_digest": hold_digest,
            "holdout_regenerated": regenerated,
            "filler_families": list(filler_families),
            "frozen_matched_random_recon": FROZEN_MATCHED_RANDOM_RECON,
            "pass_rule_source": "EXP-0014",
        },
    )
    report = {
        "experiment": experiment_id,
        "parent_pass_rule": "EXP-0014",
        "decoder": decoder,
        "filler_families": list(filler_families),
        "environment": environment(),
        "checkpoint_param_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_summary": {k: v for k, v in ckpt.get("summary", {}).items() if k != "history"},
        "holdout_language": "finnish",
        "holdout_digest": hold_digest,
        "holdout_regenerated": regenerated,
        "filler_rate": PRIMARY_FILLER_RATE,
        "neural": neural,
        "classical": classical,
        "baselines": {
            "majority": majority,
            "matched_random_recomputed": random_b,
            "matched_random_frozen_gate": FROZEN_MATCHED_RANDOM_RECON,
            "vocab_filter": vocab_b,
        },
        "decision": decision,
        "transparency_exp0013_rule_on_recomputed_random": {
            "passed": v13_note["passed"],
            "neural_reasons": v13_note["neural_check"]["reasons"],
        },
        "voynich_label_free": None,
        "simplifications": [
            "Decode-only on frozen EXP-0013 weights; no retrain",
            "Primary recon gate is frozen EXP-0013 matched-random 0.2043264147237504",
            "Recomputed matched-random reported for transparency only",
            "ZL3b not scored",
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--decoder", choices=("exact_count", "joint_keep_run"), required=True)
    parser.add_argument(
        "--filler-families",
        default="random_char,periodic",
        help="Comma-separated filler families for holdout generation",
    )
    args = parser.parse_args()
    families = tuple(x.strip() for x in args.filler_families.split(",") if x.strip())
    report = run_decode_only(args.root, args.experiment_id, args.decoder, args.device, families)
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "decoder": report["decoder"],
                "passed": report["decision"]["passed"],
                "neural_recon_acc": report["neural"]["recon_acc"],
                "frozen_random_gate": FROZEN_MATCHED_RANDOM_RECON,
                "recomputed_random": report["baselines"]["matched_random_recomputed"]["recon_acc"],
                "null_recall": report["neural"]["null_recall"],
                "null_precision": report["neural"]["null_precision"],
                "pred_null_rate": report["neural"]["pred_null_rate"],
                "reasons": report["decision"]["neural_check"]["reasons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
