"""EXP-0023 — typed decoder transfer on copy_mutate (retrain + search).

Synthetic method test only. Not a Voynich decipherment.
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

from voynich.decoder_programs import (
    LENGTH_PENALTY,
    OPS,
    english_rank_from_train,
    readable,
    score_program_on_samples,
    val_gates_ok,
)
from voynich.exact_count_decode import (
    FROZEN_MATCHED_RANDOM_RECON,
    load_frozen_exp0013,
)
from voynich.latent_recovery import (
    DATA_SEED,
    FINNISH_SEED,
    MATCHED_RANDOM_SEEDS,
    MAX_UPDATES,
    MODEL_SEED,
    PASS13_NULL_PRECISION_MIN,
    PASS13_NULL_RECALL_MIN,
    PASS13_PRED_NULL_RATE_MAX,
    PASS13_PRED_NULL_RATE_MIN,
    PASS13_VOCAB_F1_MAX,
    PRIMARY_FILLER_RATE,
    SEQ_LEN,
    WORLD_B,
    WORLD_C,
    build_vocab,
    download_corpora,
    fit_rank_bigram,
    generate_dataset,
    generate_finnish_holdout,
    matched_random_baseline,
    predict_mask_probs,
    train_model,
    vocab_filter_baseline,
)
from voynich.runtime import digest, environment, resolve_device, write_json

EXPERIMENT_ID = "EXP-0023"
FILLER_FAMILIES = ("random_char", "periodic", "copy_mutate")
SEARCH_SEED = 4023
MAX_CANDIDATES = 200
N_TRAIN = 4000
N_VAL = 400
N_HOLDOUT = 300
UPDATES = MAX_UPDATES
NO_RETRAIN_PROGRAM = ("exact_count_neural",)


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


def random_programs(n: int, length: int, seed: int) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(tuple(rng.choice(OPS, size=length, replace=False)))
    return out


def dump_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "world": row["world"],
                        "language": row.get("language"),
                        "text": row["text"],
                        "mask": row["mask"],
                        "ciphered": row.get("ciphered", ""),
                        "filler_family": row.get("filler_family", ""),
                        "filler_rate": row.get("filler_rate", 0),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return digest(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_exp0014_gates(metrics: dict, vocab_b: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    if metrics["null_recall"] < PASS13_NULL_RECALL_MIN:
        ok = False
        reasons.append(f"null_recall {metrics['null_recall']:.4f} < {PASS13_NULL_RECALL_MIN}")
    if metrics["null_precision"] < PASS13_NULL_PRECISION_MIN:
        ok = False
        reasons.append(f"null_precision {metrics['null_precision']:.4f} < {PASS13_NULL_PRECISION_MIN}")
    pr = metrics["pred_null_rate"]
    if not (PASS13_PRED_NULL_RATE_MIN <= pr <= PASS13_PRED_NULL_RATE_MAX):
        ok = False
        reasons.append(f"pred_null_rate {pr:.4f} out of band")
    if metrics["recon_acc"] <= FROZEN_MATCHED_RANDOM_RECON:
        ok = False
        reasons.append(
            f"recon_acc {metrics['recon_acc']:.4f} not > frozen_matched_random "
            f"{FROZEN_MATCHED_RANDOM_RECON:.4f}"
        )
    if vocab_b["mask_f1"] > PASS13_VOCAB_F1_MAX:
        ok = False
        reasons.append(f"vocab_filter_f1 {vocab_b['mask_f1']:.4f} > {PASS13_VOCAB_F1_MAX}")
    return ok, reasons


def search_typed(
    val: list[dict],
    probs: list[np.ndarray],
    rank_model: dict,
    english_rank: list[str],
) -> tuple[tuple[str, ...] | None, dict | None, list[dict], float | None]:
    candidates = enumerate_programs(SEARCH_SEED)
    val_rows = []
    feasible: list[tuple[tuple[str, ...], dict]] = []
    for prog in candidates:
        metrics = score_program_on_samples(val, prog, probs, rank_model, english_rank)
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
        return None, None, val_rows, None
    feasible.sort(
        key=lambda item: (
            -item[1]["selection_score"],
            item[1]["n_ops"],
            readable(item[0]),
        )
    )
    winner, winner_metrics = feasible[0]
    random_ctrl = []
    for prog in random_programs(20, len(winner), SEARCH_SEED + 1):
        m = score_program_on_samples(val, prog, probs, rank_model, english_rank)
        random_ctrl.append(m["selection_score"])
    return winner, winner_metrics, val_rows, float(np.mean(random_ctrl))


def run_experiment(root: Path, device: str) -> dict:
    device = resolve_device(device)
    proc_root = root / "data" / "processed" / "exp0023"
    out_root = root / "outputs" / EXPERIMENT_ID
    res_root = root / "results" / EXPERIMENT_ID
    for path in (proc_root, out_root, res_root, root / "data" / "manifests"):
        path.mkdir(parents=True, exist_ok=True)

    downloaded = download_corpora(root / "data" / "raw" / "latent_corpora")
    texts = downloaded["texts"]
    for required in ("english", "latin", "finnish"):
        if required not in texts or len(texts[required]) < 5000:
            raise RuntimeError(f"Corpus {required} missing or too short")

    train, val = generate_dataset(
        {"english": texts["english"], "latin": texts["latin"]},
        n_train=N_TRAIN,
        n_val=N_VAL,
        seed=DATA_SEED,
        filler_rate=PRIMARY_FILLER_RATE,
        filler_families=FILLER_FAMILIES,
    )
    holdout = generate_finnish_holdout(
        texts["finnish"],
        n=N_HOLDOUT,
        seed=FINNISH_SEED,
        filler_families=FILLER_FAMILIES,
    )
    digests = {
        "train": dump_jsonl(proc_root / "train.jsonl", train),
        "validation": dump_jsonl(proc_root / "validation.jsonl", val),
        "finnish_holdout": dump_jsonl(proc_root / "finnish_holdout.jsonl", holdout),
    }

    vocab = build_vocab()
    model, train_summary = train_model(
        train,
        val,
        vocab,
        device,
        updates=UPDATES,
        balanced_null_loss=False,
        alignment_ctc=True,
    )
    ckpt_path = out_root / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "summary": train_summary,
            "signal_threshold": train_summary.get("signal_threshold", 0.5),
            "experiment": EXPERIMENT_ID,
            "filler_families": list(FILLER_FAMILIES),
        },
        ckpt_path,
    )
    ckpt_sha = file_sha256(ckpt_path)
    param_count = int(sum(p.numel() for p in model.parameters()))

    english_rank = english_rank_from_train(train)
    rank_model = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400] or [s["text"] for s in train[:400]]
    )
    val_c = [s for s in val if s["world"] == WORLD_C]
    val_probs = predict_mask_probs(model, val_c, vocab, device)
    winner, winner_metrics, val_rows, random_ctrl_mean = search_typed(
        val_c, val_probs, rank_model, english_rank
    )

    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)
    random_b = matched_random_baseline(holdout, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)

    hold_probs = predict_mask_probs(model, holdout, vocab, device)
    if winner is None:
        hold_metrics = None
        primary_ok = False
        reasons = ["no validation-feasible program"]
    else:
        hold_metrics = score_program_on_samples(
            holdout, winner, hold_probs, rank_model, english_rank
        )
        primary_ok, reasons = apply_exp0014_gates(hold_metrics, vocab_b)
        if random_ctrl_mean is not None and winner_metrics["selection_score"] <= random_ctrl_mean:
            primary_ok = False
            reasons.append(
                f"val selection_score {winner_metrics['selection_score']:.4f} not > "
                f"random-program mean {random_ctrl_mean:.4f}"
            )
        if primary_ok:
            reasons = ["all EXP-0023 gates cleared"]

    # No-retrain control: frozen EXP-0013 + exact_count_neural on the same holdout.
    frozen_model, frozen_vocab, _ = load_frozen_exp0013(root, device)
    no_retrain_probs = predict_mask_probs(frozen_model, holdout, frozen_vocab, device)
    no_retrain_metrics = score_program_on_samples(
        holdout, NO_RETRAIN_PROGRAM, no_retrain_probs, rank_model, english_rank
    )
    no_retrain_ok, no_retrain_reasons = apply_exp0014_gates(no_retrain_metrics, vocab_b)

    decision = {
        "passed": primary_ok,
        "winner_program": None if winner is None else readable(winner),
        "reasons": reasons,
        "rule": "EXP-0023",
        "parent_pass_rule": "EXP-0014",
        "holdout_metrics": hold_metrics,
        "val_winner_metrics": None
        if winner_metrics is None
        else {
            "recon_acc": winner_metrics["recon_acc"],
            "selection_score": winner_metrics["selection_score"],
            "null_recall": winner_metrics["null_recall"],
            "null_precision": winner_metrics["null_precision"],
            "pred_null_rate": winner_metrics["pred_null_rate"],
        },
        "controls": {
            "matched_random_recon": random_b["recon_acc"],
            "matched_random_frozen_gate": FROZEN_MATCHED_RANDOM_RECON,
            "vocab_filter_recon": vocab_b["recon_acc"],
            "vocab_filter_mask_f1": vocab_b["mask_f1"],
            "random_program_val_score_mean": random_ctrl_mean,
            "no_retrain": {
                "program": readable(NO_RETRAIN_PROGRAM),
                "checkpoint": "outputs/EXP-0013/model.pt",
                "passed_exp0014_gates": no_retrain_ok,
                "reasons": no_retrain_reasons,
                "metrics": no_retrain_metrics,
            },
        },
        "thresholds": {
            "length_penalty": LENGTH_PENALTY,
            "null_recall_min": PASS13_NULL_RECALL_MIN,
            "null_precision_min": PASS13_NULL_PRECISION_MIN,
            "pred_null_rate_min": PASS13_PRED_NULL_RATE_MIN,
            "pred_null_rate_max": PASS13_PRED_NULL_RATE_MAX,
            "recon_vs_frozen_matched_random": FROZEN_MATCHED_RANDOM_RECON,
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        },
    }

    write_json(
        root / "data" / "manifests" / "exp0023_data.json",
        {
            "experiment": EXPERIMENT_ID,
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "finnish_seed": FINNISH_SEED,
            "search_seed": SEARCH_SEED,
            "filler_rate": PRIMARY_FILLER_RATE,
            "filler_families": list(FILLER_FAMILIES),
            "seq_len": SEQ_LEN,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_holdout": N_HOLDOUT,
            "updates": UPDATES,
            "derived_sha256": digests,
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_path": "outputs/EXP-0023/model.pt",
            "param_count": param_count,
            "ops": list(OPS),
            "length_penalty": LENGTH_PENALTY,
            "max_candidates": MAX_CANDIDATES,
            "frozen_matched_random_recon": FROZEN_MATCHED_RANDOM_RECON,
            "pass_rule_source": "EXP-0014 gates + EXP-0017 typed search",
            "copy_lag_scope": "excluded; separate channel (see EXP-0020)",
        },
    )

    report = {
        "experiment": EXPERIMENT_ID,
        "environment": environment(),
        "filler_families": list(FILLER_FAMILIES),
        "param_count": param_count,
        "checkpoint_sha256": ckpt_sha,
        "train_summary": {k: v for k, v in train_summary.items() if k != "history"},
        "n_candidates": len(enumerate_programs(SEARCH_SEED)),
        "n_val_world_c": len(val_c),
        "n_feasible_val": sum(1 for r in val_rows if r["feasible"]),
        "winner_program": decision["winner_program"],
        "top_val_programs": sorted(val_rows, key=lambda r: -r["selection_score"])[:30],
        "decision": decision,
        "holdout_metrics": hold_metrics,
        "no_retrain_control": decision["controls"]["no_retrain"],
        "matched_random": random_b,
        "vocab_filter": vocab_b,
        "derived_sha256": digests,
        "voynich_label_free": None,
        "simplifications": [
            "Synthetic copy_mutate transfer only; ZL3b unscored",
            "Primary gate is frozen EXP-0014 recon 0.2043264147237504",
            "No-retrain control uses frozen EXP-0013 + exact_count_neural on same holdout",
            "copy_lag excluded from this filler set",
            "No paid LLM; no EVA fluency score",
        ],
    }
    write_json(res_root / "results.json", report)
    write_json(res_root / "decision.json", decision)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_experiment(args.root, args.device)
    ctrl = report["decision"]["controls"]["no_retrain"]
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_ID,
                "passed": report["decision"]["passed"],
                "winner_program": report["winner_program"],
                "holdout_recon": None
                if report["holdout_metrics"] is None
                else report["holdout_metrics"]["recon_acc"],
                "frozen_gate": FROZEN_MATCHED_RANDOM_RECON,
                "no_retrain_recon": ctrl["metrics"]["recon_acc"],
                "no_retrain_passed_gates": ctrl["passed_exp0014_gates"],
                "reasons": report["decision"]["reasons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
