"""Post-result, exploratory family breakdown of the frozen EXP-0030 decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from voynich.decoder_programs import english_rank_from_train, score_program_on_samples
from voynich.exact_count_decode import load_frozen_exp0013
from voynich.latent_recovery import (
    WORLD_B,
    TinySignalModel,
    fit_rank_bigram,
    matched_random_baseline,
    predict_mask_probs,
)
from voynich.runtime import digest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    proc = root / "data" / "processed" / "exp0030"

    def load(split: str) -> list[dict]:
        return [json.loads(line) for line in (proc / f"{split}.jsonl").read_text().splitlines()]

    train, holdout = load("train"), load("finnish_holdout")
    ckpt_path = root / "outputs" / "EXP-0030" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    model = TinySignalModel(len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()
    frozen, frozen_vocab, _ = load_frozen_exp0013(root, "cpu")
    rank = fit_rank_bigram(
        [s["text"] for s in train if s["world"] == WORLD_B][:400]
        or [s["text"] for s in train[:400]]
    )
    english_rank = english_rank_from_train(train)
    families = []
    for family in ("random_char", "periodic", "copy_mutate"):
        rows = [s for s in holdout if s["filler_family"] == family]
        new_probs = predict_mask_probs(model, rows, vocab, "cpu")
        old_probs = predict_mask_probs(frozen, rows, frozen_vocab, "cpu")
        new = score_program_on_samples(rows, ("exact_count_neural",), new_probs, rank, english_rank)
        old = score_program_on_samples(rows, ("exact_count_neural",), old_probs, rank, english_rank)
        random = matched_random_baseline(rows, rank, n_seeds=20)
        families.append(
            {
                "family": family,
                "n": len(rows),
                "retrained_recon": new["recon_acc"],
                "frozen_easy_recon": old["recon_acc"],
                "matched_random_recon": random["recon_acc"],
                "retrained_null_precision": new["null_precision"],
                "retrained_null_recall": new["null_recall"],
            }
        )
    primary = json.loads((root / "results" / "EXP-0030" / "decision.json").read_text())
    weighted = sum(x["n"] * x["retrained_recon"] for x in families) / sum(x["n"] for x in families)
    if abs(weighted - primary["holdout_metrics"]["recon_acc"]) > 1e-12:
        raise AssertionError("family breakdown does not reconstruct primary mean")
    report = {
        "experiment": "EXP-0030",
        "status": "exploratory_post_result",
        "holdout_sha256": digest(proc / "finnish_holdout.jsonl"),
        "checkpoint_sha256": digest(ckpt_path),
        "decision_sha256": digest(root / "results" / "EXP-0030" / "decision.json"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "families": families,
        "weighted_retrained_recon": weighted,
    }
    write_json(root / "results" / "EXP-0030" / "family_diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
