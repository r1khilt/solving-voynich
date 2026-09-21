"""Typed decoder-program search (EXP-0017). Synthetic method test; not a decipherment."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from voynich.exact_count_decode import (
    EXP0013_HOLDOUT_SHA256,
    EXP0013_TRAIN_SHA256,
    EXP0013_VAL_SHA256,
    exact_count_keep_masks,
    joint_keep_run_masks,
    load_frozen_exp0013,
    load_jsonl_samples,
)
from voynich.latent_recovery import (
    MATCHED_RANDOM_SEEDS,
    PASS13_NULL_PRECISION_MIN,
    PASS13_NULL_RECALL_MIN,
    PASS13_PRED_NULL_RATE_MAX,
    PASS13_PRED_NULL_RATE_MIN,
    PASS13_VOCAB_F1_MAX,
    PRIMARY_FILLER_RATE,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    copy_aware_features,
    evaluate_masks,
    fit_rank_bigram,
    matched_random_baseline,
    null_f1_from_metrics,
    predict_mask_probs,
    recon_accuracy,
    vocab_filter_baseline,
)
from voynich.runtime import digest, environment, resolve_device, write_json

LENGTH_PENALTY = 0.02
SEARCH_SEED = 4017
MAX_CANDIDATES = 200
OPS = (
    "exact_count_neural",
    "joint_keep_run_neural",
    "exact_count_anti_copy",
    "exact_count_anti_periodic",
    "delete_copy_tokens",
    "homophone_collapse",
    "mono_sub_rank",
)
MASK_OPS = {
    "exact_count_neural",
    "joint_keep_run_neural",
    "exact_count_anti_copy",
    "exact_count_anti_periodic",
    "delete_copy_tokens",
}
STRING_OPS = {"homophone_collapse", "mono_sub_rank"}


def readable(program: tuple[str, ...]) -> str:
    return " |> ".join(program)


def _edit_dist1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            j += 1
            if diffs > 1:
                return False
    return diffs + (len(b) - j) <= 1


def period_hit_scores(text: str) -> np.ndarray:
    n = len(text)
    hit = np.zeros(n, dtype=np.float64)
    for period in (3, 4, 5, 6, 7, 8):
        for i in range(period, n):
            if text[i] != " " and text[i] == text[i - period]:
                hit[i] += 1.0
    return hit


def _token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def delete_copy_token_mask(text: str) -> np.ndarray:
    n = len(text)
    mask = np.ones(n, dtype=int)
    seen: list[str] = []
    for start, end, tok in _token_spans(text):
        if any(_edit_dist1(tok, prev) for prev in seen):
            mask[start:end] = 0
        seen.append(tok)
    return mask


def anti_copy_keep_scores(text: str) -> np.ndarray:
    feats = copy_aware_features(text)
    return -(feats[:, 1].astype(np.float64) + feats[:, 2].astype(np.float64))


def _exact_from_scores(scores: np.ndarray) -> np.ndarray:
    return exact_count_keep_masks([scores], PRIMARY_FILLER_RATE)[0]


def apply_mask_ops(
    text: str,
    program: tuple[str, ...],
    neural_p: np.ndarray | None,
) -> np.ndarray:
    mask = np.ones(len(text), dtype=int)
    applied = False
    for op in program:
        if op not in MASK_OPS:
            continue
        applied = True
        if op == "exact_count_neural":
            if neural_p is None:
                raise ValueError("exact_count_neural needs keep-probs")
            new = exact_count_keep_masks([neural_p[: len(text)]], PRIMARY_FILLER_RATE)[0]
        elif op == "joint_keep_run_neural":
            if neural_p is None:
                raise ValueError("joint_keep_run_neural needs keep-probs")
            new = joint_keep_run_masks([neural_p[: len(text)]], PRIMARY_FILLER_RATE)[0]
        elif op == "exact_count_anti_copy":
            new = _exact_from_scores(anti_copy_keep_scores(text))
        elif op == "exact_count_anti_periodic":
            new = _exact_from_scores(-period_hit_scores(text))
        elif op == "delete_copy_tokens":
            new = delete_copy_token_mask(text)
        else:
            continue
        mask = (mask & new[: len(mask)]).astype(int)
    if not applied:
        mask = _exact_from_scores(anti_copy_keep_scores(text))
    return mask


def homophone_collapse_string(kept: str) -> str:
    letters = [ch for ch in kept if ch != " "]
    if not letters:
        return kept
    counts = Counter(letters)
    ranked = [ch for ch, _ in counts.most_common()]
    n_buckets = 4
    members: dict[int, list[str]] = {}
    bucket_of: dict[str, int] = {}
    for i, ch in enumerate(ranked):
        b = min(n_buckets - 1, i * n_buckets // max(1, len(ranked)))
        bucket_of[ch] = b
        members.setdefault(b, []).append(ch)
    bucket_rep = {ch: members[bucket_of[ch]][0] for ch in ranked}
    return "".join(bucket_rep.get(ch, ch) if ch != " " else " " for ch in kept)


def mono_sub_rank_string(kept: str, english_rank: list[str]) -> str:
    letters = [ch for ch in kept if ch != " "]
    if not letters or not english_rank:
        return kept
    counts = Counter(letters)
    src_rank = [ch for ch, _ in counts.most_common()]
    mapping = {}
    for i, ch in enumerate(src_rank):
        mapping[ch] = english_rank[min(i, len(english_rank) - 1)]
    return "".join(mapping.get(ch, ch) if ch != " " else " " for ch in kept)


def recover_string(
    text: str,
    mask: np.ndarray,
    program: tuple[str, ...],
    english_rank: list[str],
) -> str:
    kept = "".join(ch for ch, m in zip(text, mask) if m == 1)
    for op in program:
        if op == "homophone_collapse":
            kept = homophone_collapse_string(kept)
        elif op == "mono_sub_rank":
            kept = mono_sub_rank_string(kept, english_rank)
    return kept


def prefix_length_recon(pred: str, true_ciphered: str) -> float:
    if not pred and not true_ciphered:
        return 1.0
    text = pred
    mask = np.ones(len(text), dtype=int)
    return recon_accuracy(true_ciphered, text, mask)


def english_rank_from_train(train: list[dict]) -> list[str]:
    blob = "".join(s["text"] for s in train if s["world"] == WORLD_A)
    counts = Counter(ch for ch in blob if ch != " ")
    return [ch for ch, _ in counts.most_common()]


def score_program_on_samples(
    samples: list[dict],
    program: tuple[str, ...],
    probs: list[np.ndarray] | None,
    rank_model: dict,
    english_rank: list[str],
) -> dict:
    masks = []
    recons = []
    for i, s in enumerate(samples):
        p = None if probs is None else np.asarray(probs[i]).reshape(-1)
        mask = apply_mask_ops(s["text"], program, p)
        masks.append(mask)
        true_c = s.get("ciphered") or "".join(ch for ch, m in zip(s["text"], s["mask"]) if m == 1)
        recovered = recover_string(s["text"], mask, program, english_rank)
        recons.append(prefix_length_recon(recovered, true_c))
    metrics = evaluate_masks(samples, masks, rank_model)
    metrics["recon_acc"] = float(np.mean(recons))
    metrics["null_f1"] = null_f1_from_metrics(metrics)
    metrics["n_ops"] = len(program)
    metrics["program"] = readable(program)
    metrics["selection_score"] = metrics["recon_acc"] - LENGTH_PENALTY * (len(program) - 1)
    return metrics


def val_gates_ok(m: dict) -> bool:
    return (
        m["null_recall"] >= PASS13_NULL_RECALL_MIN
        and m["null_precision"] >= PASS13_NULL_PRECISION_MIN
        and PASS13_PRED_NULL_RATE_MIN <= m["pred_null_rate"] <= PASS13_PRED_NULL_RATE_MAX
    )


def enumerate_programs() -> list[tuple[str, ...]]:
    length1 = [(op,) for op in OPS]
    length2 = [p for p in itertools.permutations(OPS, 2)]
    length3 = [p for p in itertools.permutations(OPS, 3)]
    rng = np.random.default_rng(SEARCH_SEED)
    order = np.arange(len(length3))
    rng.shuffle(order)
    n_keep3 = max(0, MAX_CANDIDATES - len(length1) - len(length2))
    chosen3 = [length3[int(i)] for i in order[:n_keep3]]
    return length1 + length2 + chosen3


def random_programs(n: int, length: int, seed: int) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(tuple(rng.choice(OPS, size=length, replace=False)))
    return out


def run_search(root: Path, device: str) -> dict:
    device = resolve_device(device)
    train_path = root / "data" / "processed" / "exp0013" / "train.jsonl"
    val_path = root / "data" / "processed" / "exp0013" / "validation.jsonl"
    hold_path = root / "data" / "processed" / "exp0013" / "finnish_holdout.jsonl"
    if digest(train_path) != EXP0013_TRAIN_SHA256:
        raise RuntimeError("train digest mismatch")
    if digest(val_path) != EXP0013_VAL_SHA256:
        raise RuntimeError("validation digest mismatch")
    if digest(hold_path) != EXP0013_HOLDOUT_SHA256:
        raise RuntimeError("holdout digest mismatch")
    train = load_jsonl_samples(train_path)
    val = [s for s in load_jsonl_samples(val_path) if s["world"] == WORLD_C]
    holdout = load_jsonl_samples(hold_path)
    english_rank = english_rank_from_train(train)
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400] or [s["text"] for s in train[:400]]
    )
    model, vocab, _ckpt = load_frozen_exp0013(root, device)
    val_probs = predict_mask_probs(model, val, vocab, device)
    candidates = enumerate_programs()
    val_rows = []
    feasible = []
    for prog in candidates:
        metrics = score_program_on_samples(val, prog, val_probs, rank_model, english_rank)
        row = {
            "program": readable(prog),
            "n_ops": len(prog),
            "recon_acc": metrics["recon_acc"],
            "selection_score": metrics["selection_score"],
            "null_recall": metrics["null_recall"],
            "null_precision": metrics["null_precision"],
            "pred_null_rate": metrics["pred_null_rate"],
            "feasible": val_gates_ok(metrics),
        }
        val_rows.append(row)
        if row["feasible"]:
            feasible.append((prog, metrics))
    if not feasible:
        winner = None
        winner_metrics = None
    else:
        feasible.sort(
            key=lambda item: (
                -item[1]["selection_score"],
                item[1]["n_ops"],
                readable(item[0]),
            )
        )
        winner, winner_metrics = feasible[0]
    random_ctrl = []
    if winner is not None:
        for prog in random_programs(20, len(winner), SEARCH_SEED + 1):
            m = score_program_on_samples(val, prog, val_probs, rank_model, english_rank)
            random_ctrl.append(m["selection_score"])
    random_ctrl_mean = float(np.mean(random_ctrl)) if random_ctrl else None

    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))

    hold_probs = predict_mask_probs(model, holdout, vocab, device)
    if winner is None:
        hold_metrics = None
        decision = {
            "passed": False,
            "winner_program": None,
            "reasons": ["no validation-feasible program"],
            "rule": "EXP-0017",
        }
    else:
        hold_metrics = score_program_on_samples(holdout, winner, hold_probs, rank_model, english_rank)
        random_b = matched_random_baseline(holdout, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)
        vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)
        reasons = []
        ok = True
        if hold_metrics["null_recall"] < PASS13_NULL_RECALL_MIN:
            ok = False
            reasons.append(f"null_recall {hold_metrics['null_recall']:.4f} < {PASS13_NULL_RECALL_MIN}")
        if hold_metrics["null_precision"] < PASS13_NULL_PRECISION_MIN:
            ok = False
            reasons.append(f"null_precision {hold_metrics['null_precision']:.4f} < {PASS13_NULL_PRECISION_MIN}")
        pr = hold_metrics["pred_null_rate"]
        if not (PASS13_PRED_NULL_RATE_MIN <= pr <= PASS13_PRED_NULL_RATE_MAX):
            ok = False
            reasons.append(f"pred_null_rate {pr:.4f} out of band")
        if hold_metrics["recon_acc"] <= random_b["recon_acc"]:
            ok = False
            reasons.append(
                f"recon_acc {hold_metrics['recon_acc']:.4f} not > matched_random {random_b['recon_acc']:.4f}"
            )
        if hold_metrics["recon_acc"] <= vocab_b["recon_acc"]:
            ok = False
            reasons.append(
                f"recon_acc {hold_metrics['recon_acc']:.4f} not > vocab_filter {vocab_b['recon_acc']:.4f}"
            )
        if vocab_b["mask_f1"] > PASS13_VOCAB_F1_MAX:
            ok = False
            reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS13_VOCAB_F1_MAX}")
        if random_ctrl_mean is not None and winner_metrics["selection_score"] <= random_ctrl_mean:
            ok = False
            reasons.append(
                f"val selection_score {winner_metrics['selection_score']:.4f} not > "
                f"random-program mean {random_ctrl_mean:.4f}"
            )
        if not reasons:
            reasons.append("all EXP-0017 gates cleared")
        decision = {
            "passed": ok,
            "winner_program": readable(winner),
            "reasons": reasons,
            "rule": "EXP-0017",
            "holdout_metrics": hold_metrics,
            "val_winner_metrics": {
                "recon_acc": winner_metrics["recon_acc"],
                "selection_score": winner_metrics["selection_score"],
                "null_recall": winner_metrics["null_recall"],
                "null_precision": winner_metrics["null_precision"],
                "pred_null_rate": winner_metrics["pred_null_rate"],
            },
            "baselines": {
                "matched_random_recon": random_b["recon_acc"],
                "vocab_filter_recon": vocab_b["recon_acc"],
                "vocab_filter_mask_f1": vocab_b["mask_f1"],
                "random_program_val_score_mean": random_ctrl_mean,
            },
            "thresholds": {
                "length_penalty": LENGTH_PENALTY,
                "null_recall_min": PASS13_NULL_RECALL_MIN,
                "null_precision_min": PASS13_NULL_PRECISION_MIN,
                "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
                "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
                "vocab_f1_max": PASS13_VOCAB_F1_MAX,
            },
        }

    res_root = root / "results" / "EXP-0017"
    res_root.mkdir(parents=True, exist_ok=True)
    compact_val = sorted(val_rows, key=lambda r: -r["selection_score"])[:30]
    report = {
        "experiment": "EXP-0017",
        "environment": environment(),
        "n_candidates": len(candidates),
        "n_val_world_c": len(val),
        "n_feasible_val": len(feasible),
        "winner_program": None if winner is None else readable(winner),
        "top_val_programs": compact_val,
        "decision": decision,
        "holdout_metrics": hold_metrics,
        "voynich_label_free": None,
        "simplifications": [
            "Typed catalog search only; no LLM proposer; no EVA/Latin fluency score",
            "Finnish holdout unused during search",
            "String-only programs inherit exact_count_anti_copy as the deletion hypothesis",
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
    write_json(
        root / "data" / "manifests" / "exp0017_data.json",
        {
            "experiment": "EXP-0017",
            "train_digest": EXP0013_TRAIN_SHA256,
            "val_digest": EXP0013_VAL_SHA256,
            "holdout_digest": EXP0013_HOLDOUT_SHA256,
            "ops": list(OPS),
            "length_penalty": LENGTH_PENALTY,
            "search_seed": SEARCH_SEED,
            "max_candidates": MAX_CANDIDATES,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_search(args.root, args.device)
    print(
        json.dumps(
            {
                "experiment": "EXP-0017",
                "passed": report["decision"]["passed"],
                "winner_program": report["winner_program"],
                "n_candidates": report["n_candidates"],
                "n_feasible_val": report["n_feasible_val"],
                "reasons": report["decision"].get("reasons"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
