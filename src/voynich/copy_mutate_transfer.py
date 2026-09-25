"""Typed decoder transfer on copy_mutate (historical EXP-0023 and corrected reruns).

Synthetic method test only. Not a Voynich decipherment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import signal
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
COPY_ONLY_MARGIN = 0.05
COPY_ONLY_N_MIN = 60
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


def apply_copy_only_gate(metrics: dict | None, random_control: dict | None, n: int) -> tuple[bool, list[str]]:
    if n < COPY_ONLY_N_MIN:
        return False, [f"only {n} copy_mutate holdout samples (<{COPY_ONLY_N_MIN})"]
    if metrics is None or random_control is None:
        return False, ["copy-only metrics unavailable"]
    floor = random_control["recon_acc"] + COPY_ONLY_MARGIN
    if metrics["recon_acc"] <= floor:
        return False, [f"copy-only recon {metrics['recon_acc']:.4f} not > matched-random + 0.05 ({floor:.4f})"]
    return True, []


def search_typed(
    val: list[dict],
    probs: list[np.ndarray],
    rank_model: dict,
    english_rank: list[str],
    search_seed: int = SEARCH_SEED,
) -> tuple[tuple[str, ...] | None, dict | None, list[dict], float | None]:
    candidates = enumerate_programs(search_seed)
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
    for prog in random_programs(20, len(winner), search_seed + 1):
        m = score_program_on_samples(val, prog, probs, rank_model, english_rank)
        random_ctrl.append(m["selection_score"])
    return winner, winner_metrics, val_rows, float(np.mean(random_ctrl))


def run_experiment(
    root: Path,
    device: str,
    *,
    experiment_id: str,
    data_seed: int,
    finnish_seed: int,
    search_seed: int,
) -> dict:
    if experiment_id == EXPERIMENT_ID:
        raise ValueError("EXP-0023 is an immutable historical run; use a new experiment ID")
    if not re.fullmatch(r"EXP-\d{4}", experiment_id):
        raise ValueError("experiment_id must have form EXP-0000")
    device = resolve_device(device)
    proc_root = root / "data" / "processed" / experiment_id.lower().replace("-", "")
    out_root = root / "outputs" / experiment_id
    res_root = root / "results" / experiment_id
    manifest_path = root / "data" / "manifests" / f"{experiment_id.lower().replace('-', '')}_data.json"
    if manifest_path.exists() or any(path.exists() for path in (proc_root, out_root, res_root)):
        raise FileExistsError(f"{experiment_id} output already exists; refusing to overwrite")
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
        seed=data_seed,
        filler_rate=PRIMARY_FILLER_RATE,
        filler_families=FILLER_FAMILIES,
    )
    holdout = generate_finnish_holdout(
        texts["finnish"],
        n=N_HOLDOUT,
        seed=finnish_seed,
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
            "experiment": experiment_id,
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
        val_c, val_probs, rank_model, english_rank, search_seed=search_seed
    )

    train_token_vocab: set[str] = set()
    for s in train:
        if s["world"] in (WORLD_B, WORLD_C):
            train_token_vocab.update(re.findall(r"\S+", s["text"]))
    vocab_b = vocab_filter_baseline(holdout, train_token_vocab, rank_model)
    random_b = matched_random_baseline(holdout, rank_model, n_seeds=MATCHED_RANDOM_SEEDS)

    hold_probs = predict_mask_probs(model, holdout, vocab, device)
    copy_indices = [i for i, sample in enumerate(holdout) if sample["filler_family"] == "copy_mutate"]
    copy_samples = [holdout[i] for i in copy_indices]
    copy_probs = [hold_probs[i] for i in copy_indices]
    copy_random = matched_random_baseline(
        copy_samples, rank_model, n_seeds=MATCHED_RANDOM_SEEDS
    ) if copy_samples else None
    if winner is None:
        hold_metrics = None
        copy_metrics = None
        primary_ok = False
        reasons = ["no validation-feasible program"]
    else:
        hold_metrics = score_program_on_samples(
            holdout, winner, hold_probs, rank_model, english_rank
        )
        copy_metrics = score_program_on_samples(
            copy_samples, winner, copy_probs, rank_model, english_rank
        ) if copy_samples else None
        primary_ok, reasons = apply_exp0014_gates(hold_metrics, vocab_b)
        copy_ok, copy_reasons = apply_copy_only_gate(copy_metrics, copy_random, len(copy_samples))
        primary_ok = primary_ok and copy_ok
        reasons.extend(copy_reasons)
        if random_ctrl_mean is not None and winner_metrics["selection_score"] <= random_ctrl_mean:
            primary_ok = False
            reasons.append(
                f"val selection_score {winner_metrics['selection_score']:.4f} not > "
                f"random-program mean {random_ctrl_mean:.4f}"
            )
        if primary_ok:
            reasons = [f"all {experiment_id} gates cleared"]

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
        "rule": experiment_id,
        "parent_pass_rule": "EXP-0014",
        "holdout_metrics": hold_metrics,
        "copy_only_metrics": copy_metrics,
        "n_copy_holdout": len(copy_samples),
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
            "copy_only_matched_random_recon": None if copy_random is None else copy_random["recon_acc"],
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
            "copy_only_recon_margin_over_matched_random": COPY_ONLY_MARGIN,
            "copy_only_n_min": COPY_ONLY_N_MIN,
            "vocab_f1_max": PASS13_VOCAB_F1_MAX,
        },
    }

    write_json(
        manifest_path,
        {
            "experiment": experiment_id,
            "runner_sha256": file_sha256(Path(__file__)),
            "generator_sha256": file_sha256(Path(generate_dataset.__code__.co_filename)),
            "data_seed": data_seed,
            "model_seed": MODEL_SEED,
            "finnish_seed": finnish_seed,
            "search_seed": search_seed,
            "filler_rate": PRIMARY_FILLER_RATE,
            "filler_families": list(FILLER_FAMILIES),
            "seq_len": SEQ_LEN,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_holdout": N_HOLDOUT,
            "updates": UPDATES,
            "derived_sha256": digests,
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_path": f"outputs/{experiment_id}/model.pt",
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
        "experiment": experiment_id,
        "runner_sha256": file_sha256(Path(__file__)),
        "generator_sha256": file_sha256(Path(generate_dataset.__code__.co_filename)),
        "environment": environment(),
        "filler_families": list(FILLER_FAMILIES),
        "param_count": param_count,
        "checkpoint_sha256": ckpt_sha,
        "train_summary": {k: v for k, v in train_summary.items() if k != "history"},
        "n_candidates": len(enumerate_programs(search_seed)),
        "n_val_world_c": len(val_c),
        "n_feasible_val": sum(1 for r in val_rows if r["feasible"]),
        "winner_program": decision["winner_program"],
        "top_val_programs": sorted(val_rows, key=lambda r: -r["selection_score"])[:30],
        "decision": decision,
        "holdout_metrics": hold_metrics,
        "no_retrain_control": decision["controls"]["no_retrain"],
        "matched_random": random_b,
        "copy_only_matched_random": copy_random,
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
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--finnish-seed", type=int, required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")

    def wall_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"experiment exceeded {args.max_wall_seconds} seconds")

    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(args.max_wall_seconds)
    report = run_experiment(
        args.root,
        args.device,
        experiment_id=args.experiment_id,
        data_seed=args.data_seed,
        finnish_seed=args.finnish_seed,
        search_seed=args.search_seed,
    )
    ctrl = report["decision"]["controls"]["no_retrain"]
    print(
        json.dumps(
            {
                "experiment": args.experiment_id,
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
