"""EXP-0026 — ciphertext-only typed inverse on the EXP-0023 copy_mutate split.

Select program on train (masks for selection metrics only).
Decode holdout from ciphertext (+ optional neural keep-probs) only.
Synthetic method test. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch

from voynich.copy_mutate_transfer import apply_exp0014_gates
from voynich.decoder_programs import (
    LENGTH_PENALTY,
    MASK_OPS,
    english_rank_from_train,
    readable,
    recover_string,
    val_gates_ok,
)
from voynich.exact_count_decode import (
    FROZEN_MATCHED_RANDOM_RECON,
    exact_count_keep_masks,
    load_jsonl_samples,
)
from voynich.latent_recovery import (
    PASS13_NULL_PRECISION_MIN,
    PASS13_NULL_RECALL_MIN,
    PASS13_PRED_NULL_RATE_MAX,
    PASS13_PRED_NULL_RATE_MIN,
    PASS13_VOCAB_F1_MAX,
    PRIMARY_FILLER_RATE,
    TinySignalModel,
    WORLD_B,
    WORLD_C,
    copy_aware_features,
    evaluate_masks,
    fit_rank_bigram,
    predict_mask_probs,
    vocab_filter_baseline,
)
from voynich.runtime import environment, resolve_device, write_json

EXPERIMENT_ID = "EXP-0026"
PARENT_EXP = "EXP-0023"
SEARCH_SEED = 4026
MAX_CANDIDATES = 200
SELECT_TRAIN_N = 400
EXPECTED_HOLDOUT_SHA = "cd72bf0fca93f14eebb274c548072cb58a23bd04a7f629c8984d76a888ac59b4"
EXPECTED_TRAIN_SHA = "2c824915231d69f98f41120829140525b03839a19518d02973ac68f654f6e08e"
EXP0023_WINNER_RECON = 0.17585403660739804
EXP0024_ORACLE_RECON = 1.0
CKPT_REL = "outputs/EXP-0023/model.pt"

# Typed family: EXP-0017 classes + repetition/splice detector.
OPS = (
    "exact_count_neural",
    "joint_keep_run_neural",
    "exact_count_anti_copy",
    "exact_count_anti_periodic",
    "delete_copy_tokens",
    "repetition_splice_detector",
    "homophone_collapse",
    "mono_sub_rank",
)
MASK_OPS_LOCAL = MASK_OPS | {"repetition_splice_detector"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repetition_splice_keep_scores(text: str, window: int = 12) -> np.ndarray:
    """Human-readable copy/splice detector keep scores (ciphertext only).

    keep_score = 0.5*is_space − 3*exact_recent_char_copy − 2*token_edit1_copy
    − 1*digram_splice_flag. Exact-count then keeps top round(0.70 L).
    """
    n = len(text)
    feats = copy_aware_features(text)
    scores = 0.5 * feats[:, 0] - 3.0 * feats[:, 1] - 2.0 * feats[:, 2]
    # Splice flag: current digram with previous non-space also appears earlier non-adjacent.
    recent_digrams: list[str] = []
    prev_ns: str | None = None
    for i, ch in enumerate(text):
        if ch == " ":
            prev_ns = None
            continue
        if prev_ns is not None:
            dig = prev_ns + ch
            if dig in recent_digrams[:-1]:
                scores[i] -= 1.0
            recent_digrams.append(dig)
            if len(recent_digrams) > window:
                recent_digrams.pop(0)
        prev_ns = ch
    return scores.astype(np.float64)


def apply_mask_ops_ct(
    text: str,
    program: tuple[str, ...],
    neural_p: np.ndarray | None,
) -> np.ndarray:
    """Ciphertext-only mask ops (no generator mask). Extends EXP-0017 with splice detector."""
    from voynich.decoder_programs import (
        anti_copy_keep_scores,
        delete_copy_token_mask,
        period_hit_scores,
    )
    from voynich.exact_count_decode import joint_keep_run_masks

    mask = np.ones(len(text), dtype=int)
    applied = False
    for op in program:
        if op not in MASK_OPS_LOCAL:
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
            new = exact_count_keep_masks(
                [anti_copy_keep_scores(text)], PRIMARY_FILLER_RATE
            )[0]
        elif op == "exact_count_anti_periodic":
            new = exact_count_keep_masks(
                [-period_hit_scores(text)], PRIMARY_FILLER_RATE
            )[0]
        elif op == "delete_copy_tokens":
            new = delete_copy_token_mask(text)
        elif op == "repetition_splice_detector":
            new = exact_count_keep_masks(
                [repetition_splice_keep_scores(text)], PRIMARY_FILLER_RATE
            )[0]
        else:
            continue
        mask = (mask & new[: len(mask)]).astype(int)
    if not applied:
        mask = exact_count_keep_masks(
            [repetition_splice_keep_scores(text)], PRIMARY_FILLER_RATE
        )[0]
    return mask


def score_program_ct(
    samples: list[dict],
    program: tuple[str, ...],
    probs: list[np.ndarray] | None,
    rank_model: dict,
    english_rank: list[str],
) -> dict:
    masks = []
    recons = []
    from voynich.decoder_programs import prefix_length_recon
    from voynich.latent_recovery import null_f1_from_metrics

    for i, s in enumerate(samples):
        p = None if probs is None else np.asarray(probs[i]).reshape(-1)
        # Decode path: text + optional neural probs only — never sample["mask"].
        mask = apply_mask_ops_ct(s["text"], program, p)
        masks.append(mask)
        true_c = s.get("ciphered") or "".join(
            ch for ch, m in zip(s["text"], s["mask"]) if m == 1
        )
        recovered = recover_string(s["text"], mask, program, english_rank)
        recons.append(prefix_length_recon(recovered, true_c))
    metrics = evaluate_masks(samples, masks, rank_model)
    metrics["recon_acc"] = float(np.mean(recons))
    metrics["null_f1"] = null_f1_from_metrics(metrics)
    metrics["n_ops"] = len(program)
    metrics["program"] = readable(program)
    metrics["selection_score"] = metrics["recon_acc"] - LENGTH_PENALTY * (len(program) - 1)
    metrics["used_eval_mask"] = False
    return metrics


def enumerate_programs(seed: int = SEARCH_SEED) -> list[tuple[str, ...]]:
    length1 = [(op,) for op in OPS]
    length2 = [p for p in itertools.permutations(OPS, 2)]
    length3 = [p for p in itertools.permutations(OPS, 3)]
    rng = np.random.default_rng(seed)
    order = np.arange(len(length3))
    rng.shuffle(order)
    n_keep3 = max(0, MAX_CANDIDATES - len(length1) - len(length2))
    chosen3 = [length3[int(i)] for i in order[:n_keep3]]
    return length1 + length2 + chosen3


def load_exp0023_model(root: Path, device: str) -> tuple[TinySignalModel, dict]:
    ckpt_path = root / CKPT_REL
    if not ckpt_path.is_file():
        raise RuntimeError(f"Missing checkpoint {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = blob["vocab"]
    model = TinySignalModel(len(vocab)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, vocab


def assert_no_mask_in_decode_path() -> None:
    """Static sanity: apply_mask_ops_ct source must not reference sample masks."""
    import inspect

    src = inspect.getsource(apply_mask_ops_ct)
    if "sample" in src and "mask" in src:
        # function takes text only; ok
        pass
    if "s[\"mask\"]" in src or "s['mask']" in src:
        raise RuntimeError("decode path must not read sample masks")


def run_experiment(root: Path, device: str) -> dict:
    assert_no_mask_in_decode_path()
    device = resolve_device(device)
    proc = root / "data" / "processed" / "exp0023"
    hold_path = proc / "finnish_holdout.jsonl"
    train_path = proc / "train.jsonl"
    res_root = root / "results" / EXPERIMENT_ID
    man_path = root / "data" / "manifests" / "exp0026_data.json"
    res_root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)

    hold_sha = file_sha256(hold_path)
    train_sha = file_sha256(train_path)
    if hold_sha != EXPECTED_HOLDOUT_SHA:
        raise RuntimeError(f"Holdout digest mismatch: {hold_sha}")
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"Train digest mismatch: {train_sha}")

    train = load_jsonl_samples(train_path)
    holdout = load_jsonl_samples(hold_path)
    select = [s for s in train if s["world"] == WORLD_C][:SELECT_TRAIN_N]
    if len(select) < 50:
        raise RuntimeError(f"Too few WORLD_C train samples for selection: {len(select)}")

    model, vocab = load_exp0023_model(root, device)
    english_rank = english_rank_from_train(train)
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400]
        or [s["text"] for s in train[:400]]
    )
    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)

    select_probs = predict_mask_probs(model, select, vocab, device)
    hold_probs = predict_mask_probs(model, holdout, vocab, device)

    candidates = enumerate_programs(SEARCH_SEED)
    select_rows = []
    feasible: list[tuple[tuple[str, ...], dict]] = []
    for prog in candidates:
        # Selection metrics may use train gold masks via evaluate_masks / recon vs ciphered.
        metrics = score_program_ct(select, prog, select_probs, rank_model, english_rank)
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
        select_rows.append(row)
        if row["feasible"]:
            feasible.append((prog, metrics))

    if not feasible:
        winner = None
        winner_select = None
        hold_metrics = None
        primary_ok = False
        reasons = ["no train-selection-feasible program"]
        mode = "ciphertext_only_inverse_missed"
    else:
        feasible.sort(
            key=lambda item: (
                -item[1]["selection_score"],
                item[1]["n_ops"],
                readable(item[0]),
            )
        )
        winner, winner_select = feasible[0]
        # Holdout decode: ciphertext + neural probs only.
        hold_metrics = score_program_ct(
            holdout, winner, hold_probs, rank_model, english_rank
        )
        primary_ok, reasons = apply_exp0014_gates(hold_metrics, vocab_b)
        mode = (
            "ciphertext_only_inverse"
            if primary_ok
            else "ciphertext_only_inverse_missed"
        )
        if primary_ok:
            reasons = ["all frozen EXP-0014 gates cleared on holdout"]

    decision = {
        "passed": bool(primary_ok),
        "mode": mode,
        "winner_program": readable(winner) if winner else None,
        "winner_program_text": (
            f"ciphertext_only: {readable(winner)}" if winner else None
        ),
        "reasons": reasons,
        "rule": EXPERIMENT_ID,
        "parent_pass_rule": "EXP-0014",
        "used_eval_generator_mask": False,
        "selection": {
            "split": "train_world_C",
            "n": len(select),
            "used_train_masks_for_metrics_only": True,
            "winner_select_metrics": (
                {
                    "recon_acc": winner_select["recon_acc"],
                    "selection_score": winner_select["selection_score"],
                    "null_recall": winner_select["null_recall"],
                    "null_precision": winner_select["null_precision"],
                    "pred_null_rate": winner_select["pred_null_rate"],
                }
                if winner_select
                else None
            ),
        },
        "holdout_metrics": hold_metrics,
        "comparisons": {
            "exp0024_oracle_recon": EXP0024_ORACLE_RECON,
            "exp0023_winner_recon": EXP0023_WINNER_RECON,
            "test_recon": None if hold_metrics is None else hold_metrics["recon_acc"],
            "frozen_bar": FROZEN_MATCHED_RANDOM_RECON,
        },
        "thresholds": {
            "null_recall_min": PASS13_NULL_RECALL_MIN,
            "null_precision_min": PASS13_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
            "recon_vs_frozen_matched_random": FROZEN_MATCHED_RANDOM_RECON,
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
            "length_penalty": LENGTH_PENALTY,
        },
        "controls": {
            "vocab_filter_recon": vocab_b["recon_acc"],
            "vocab_filter_mask_f1": vocab_b["mask_f1"],
        },
        "non_claim": (
            "Synthetic ciphertext-only inverse attempt only; not a manuscript reading "
            "or decipherment."
        ),
    }

    results = {
        "experiment": EXPERIMENT_ID,
        "question": (
            "Can a typed ciphertext-only decoder, selected on train masks then applied "
            "without eval masks, clear frozen EXP-0014 gates on the EXP-0023 holdout?"
        ),
        "linked": ["EXP-0023", "EXP-0024", "EXP-0014", "EXP-0017"],
        "environment": environment(),
        "digests": {"train": train_sha, "finnish_holdout": hold_sha},
        "checkpoint": CKPT_REL,
        "ops": list(OPS),
        "n_candidates": len(candidates),
        "n_feasible_on_select": len(feasible) if winner is not None or feasible else 0,
        "select_top": sorted(select_rows, key=lambda r: -r["selection_score"])[:15],
        "decision": decision,
        "non_claims": [
            "Not a manuscript reading",
            "Not a Voynich decipherment",
            "Synthetic channel only",
        ],
    }
    # fix n_feasible when winner is None but feasible empty
    if winner is None:
        results["n_feasible_on_select"] = 0

    write_json(res_root / "results.json", results)
    write_json(res_root / "decision.json", decision)
    manifest = {
        "experiment": EXPERIMENT_ID,
        "parent": PARENT_EXP,
        "search_seed": SEARCH_SEED,
        "select_train_n": SELECT_TRAIN_N,
        "ops": list(OPS),
        "derived_sha256": {"train": train_sha, "finnish_holdout": hold_sha},
        "checkpoint": CKPT_REL,
        "frozen_matched_random_recon": FROZEN_MATCHED_RANDOM_RECON,
        "passed": decision["passed"],
        "mode": decision["mode"],
        "winner_program": decision["winner_program"],
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(res_root),
                "passed": decision["passed"],
                "mode": decision["mode"],
                "winner": decision["winner_program"],
                "test_recon": decision["comparisons"]["test_recon"],
                "oracle_recon": EXP0024_ORACLE_RECON,
                "exp0023_winner_recon": EXP0023_WINNER_RECON,
            },
            indent=2,
        )
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)
    run_experiment(args.root.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
