"""EXP-0024 — oracle inverse of the EXP-0023 copy_mutate channel.

Synthetic method diagnosis only. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from voynich.copy_mutate_transfer import FILLER_FAMILIES, apply_exp0014_gates
from voynich.exact_count_decode import FROZEN_MATCHED_RANDOM_RECON, load_jsonl_samples
from voynich.latent_recovery import (
    DATA_SEED,
    FINNISH_SEED,
    PASS13_NULL_PRECISION_MIN,
    PASS13_NULL_RECALL_MIN,
    PASS13_PRED_NULL_RATE_MAX,
    PASS13_PRED_NULL_RATE_MIN,
    PASS13_VOCAB_F1_MAX,
    PRIMARY_FILLER_RATE,
    SEQ_LEN,
    WORLD_B,
    WORLD_C,
    evaluate_masks,
    fit_rank_bigram,
    vocab_filter_baseline,
)
from voynich.runtime import digest, environment, write_json

EXPERIMENT_ID = "EXP-0024"
PARENT_EXP = "EXP-0023"
ORACLE_NAME = "oracle_gold_mask_delete"
EXPECTED_HOLDOUT_SHA = "cd72bf0fca93f14eebb274c548072cb58a23bd04a7f629c8984d76a888ac59b4"
EXPECTED_TRAIN_SHA = "2c824915231d69f98f41120829140525b03839a19518d02973ac68f654f6e08e"
EXP0023_WINNER_RECON = 0.17585403660739804


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_inverse_masks(samples: list[dict]) -> tuple[list[np.ndarray] | None, str | None]:
    """Known inverse of insert_nulls: delete stored null positions.

    Uses the generator's recorded mask from synthetic supervision. Ciphertext alone
    does not uniquely determine which glyphs are inserted (mutated) copies.
    """
    masks: list[np.ndarray] = []
    for i, sample in enumerate(samples):
        if "mask" not in sample or sample["mask"] is None:
            return None, (
                f"sample {i} missing stored mask; channel not invertible from "
                "ciphertext alone without generator trace"
            )
        mask = np.asarray(sample["mask"], dtype=int)
        if mask.size == 0:
            return None, f"sample {i} has empty mask"
        if not set(np.unique(mask)).issubset({0, 1}):
            return None, f"sample {i} mask is not binary"
        if len(mask) != len(sample["text"]):
            return None, (
                f"sample {i} mask length {len(mask)} != text length {len(sample['text'])}"
            )
        masks.append(mask.copy())
    return masks, None


def decide_mode(oracle_ok: bool, invertible: bool) -> tuple[bool, str, list[str]]:
    """Frozen before scores in docs/experiments/EXP-0024.md."""
    if not invertible:
        return False, "channel_not_invertible", [
            "generator mask/trace not stored; map not invertible from ciphertext"
        ]
    # EXP-0023 winner already failed; this id only diagnoses oracle vs gates.
    if oracle_ok:
        return False, "search_missed_inverse", [
            "oracle inverse clears frozen EXP-0014 gates while EXP-0023 typed winner failed",
            "typed family/search cannot find a decoder the metric can see",
        ]
    return False, "gate_blind_to_copy_inverse", [
        "oracle inverse of the registered channel also fails the frozen EXP-0014 gates",
        "this metric cannot certify the true copy_mutate/null-insertion channel",
    ]


def run_experiment(root: Path) -> dict:
    proc = root / "data" / "processed" / "exp0023"
    hold_path = proc / "finnish_holdout.jsonl"
    train_path = proc / "train.jsonl"
    res_root = root / "results" / EXPERIMENT_ID
    man_path = root / "data" / "manifests" / "exp0024_data.json"
    res_root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)

    if not hold_path.is_file():
        raise RuntimeError(f"Missing EXP-0023 holdout: {hold_path}")
    if not train_path.is_file():
        raise RuntimeError(f"Missing EXP-0023 train: {train_path}")

    hold_sha = file_sha256(hold_path)
    train_sha = file_sha256(train_path)
    if hold_sha != EXPECTED_HOLDOUT_SHA:
        raise RuntimeError(
            f"Holdout digest {hold_sha} != registered EXP-0023 {EXPECTED_HOLDOUT_SHA}"
        )
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(
            f"Train digest {train_sha} != registered EXP-0023 {EXPECTED_TRAIN_SHA}"
        )

    holdout = load_jsonl_samples(hold_path)
    train = load_jsonl_samples(train_path)

    # Confirm EXP-0023 decision on disk before relying on the reported winner recon.
    parent_decision = json.loads(
        (root / "results" / PARENT_EXP / "decision.json").read_text(encoding="utf-8")
    )
    parent_recon = float(parent_decision["holdout_metrics"]["recon_acc"])
    parent_passed = bool(parent_decision["passed"])
    if abs(parent_recon - EXP0023_WINNER_RECON) > 1e-12 or parent_passed:
        raise RuntimeError(
            f"EXP-0023 decision.json unexpected: passed={parent_passed} recon={parent_recon}"
        )

    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400]
        or [s["text"] for s in train[:400]]
    )
    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)

    pred_masks, invert_err = oracle_inverse_masks(holdout)
    if pred_masks is None:
        passed, mode, reasons = decide_mode(False, invertible=False)
        oracle_metrics = None
        gate_ok = False
        gate_reasons = [invert_err or "not invertible"]
        used_generator_trace = False
    else:
        used_generator_trace = True
        oracle_metrics = evaluate_masks(holdout, pred_masks, rank_model)
        oracle_metrics["program"] = ORACLE_NAME
        oracle_metrics["null_f1"] = (
            0.0
            if oracle_metrics["null_precision"] + oracle_metrics["null_recall"] == 0
            else float(
                2
                * oracle_metrics["null_precision"]
                * oracle_metrics["null_recall"]
                / (oracle_metrics["null_precision"] + oracle_metrics["null_recall"])
            )
        )
        gate_ok, gate_reasons = apply_exp0014_gates(oracle_metrics, vocab_b)
        passed, mode, reasons = decide_mode(gate_ok, invertible=True)
        if gate_ok:
            reasons = list(reasons)
        else:
            reasons = list(reasons) + list(gate_reasons)

    decision = {
        "passed": passed,
        "fail_mode": mode,
        "oracle_program": ORACLE_NAME,
        "oracle_passed_exp0014_gates": gate_ok,
        "used_generator_trace": used_generator_trace,
        "generator_trace_kind": "stored_binary_mask_from_insert_nulls" if used_generator_trace else None,
        "trace_rationale": (
            "Ciphertext alone does not uniquely identify inserted null positions "
            "(copy_mutate nulls may be exact or mutated copies of nearby signal). "
            "Oracle therefore applies the stored mask from EXP-0023 synthetic jsonl."
        ),
        "reasons": reasons,
        "gate_reasons": gate_reasons,
        "rule": EXPERIMENT_ID,
        "parent_pass_rule": "EXP-0014",
        "parent_experiment": PARENT_EXP,
        "parent_winner_failed": True,
        "holdout_metrics": oracle_metrics,
        "exp0023_winner_recon": parent_recon,
        "thresholds": {
            "null_recall_min": PASS13_NULL_RECALL_MIN,
            "null_precision_min": PASS13_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
            "recon_vs_frozen_matched_random": FROZEN_MATCHED_RANDOM_RECON,
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        },
        "controls": {
            "vocab_filter_recon": vocab_b["recon_acc"],
            "vocab_filter_mask_f1": vocab_b["mask_f1"],
        },
        "non_claim": "Synthetic oracle diagnosis only; not a manuscript reading or decipherment.",
    }

    results = {
        "experiment": EXPERIMENT_ID,
        "environment": environment(),
        "parent_experiment": PARENT_EXP,
        "filler_families": list(FILLER_FAMILIES),
        "filler_rate": PRIMARY_FILLER_RATE,
        "data_seed": DATA_SEED,
        "finnish_seed": FINNISH_SEED,
        "seq_len": SEQ_LEN,
        "n_holdout": len(holdout),
        "holdout_sha256": hold_sha,
        "train_sha256": train_sha,
        "oracle": ORACLE_NAME,
        "used_generator_trace": used_generator_trace,
        "oracle_metrics": oracle_metrics,
        "exp0023_winner_recon": parent_recon,
        "vocab_filter": {
            "recon_acc": vocab_b["recon_acc"],
            "mask_f1": vocab_b["mask_f1"],
        },
        "decision": {
            "passed": passed,
            "fail_mode": mode,
            "oracle_passed_exp0014_gates": gate_ok,
        },
        "notes": [
            "Reuses EXP-0023 processed split; no retrain; no typed search",
            "Synthetic copy_mutate channel diagnosis only; ZL3b unscored",
        ],
    }

    manifest = {
        "experiment": EXPERIMENT_ID,
        "parent_experiment": PARENT_EXP,
        "data_seed": DATA_SEED,
        "finnish_seed": FINNISH_SEED,
        "filler_rate": PRIMARY_FILLER_RATE,
        "filler_families": list(FILLER_FAMILIES),
        "seq_len": SEQ_LEN,
        "n_holdout": len(holdout),
        "derived_sha256": {
            "finnish_holdout": hold_sha,
            "train": train_sha,
        },
        "source_paths": {
            "finnish_holdout": "data/processed/exp0023/finnish_holdout.jsonl",
            "train": "data/processed/exp0023/train.jsonl",
        },
        "oracle": ORACLE_NAME,
        "frozen_matched_random_recon": FROZEN_MATCHED_RANDOM_RECON,
        "pass_rule_source": "EXP-0014 gates; FAIL modes search_missed_inverse / gate_blind_to_copy_inverse / channel_not_invertible",
        "used_generator_trace": used_generator_trace,
    }

    write_json(res_root / "results.json", results)
    write_json(res_root / "decision.json", decision)
    write_json(man_path, manifest)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    decision = run_experiment(args.root.resolve())
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
