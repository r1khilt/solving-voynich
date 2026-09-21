"""EXP-0025: generator-adversary with harder iid falsifier (HYP-005 / R3 redesign).

Same search and held-out rule as EXP-0019; iid control is iid_uniform_fixedlen
instead of train-unigram iid_char. Not a decipherment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from voynich.generator_adversary import (
    DEFAULT_GENOME,
    FLOORS,
    Genome,
    HELDOUT_METRICS,
    N_ELITE,
    N_GENERATIONS,
    OPT_METRICS,
    POP_SIZE,
    fit_bigram_and_lengths,
    load_pages,
    metric_verdicts,
    metrics_from_pages,
    page_shapes,
    run_search,
)

EXPERIMENT_ID = "EXP-0025"
SEED = 4025
IID_WORD_LEN = 8
IID_NAME = "iid_uniform_fixedlen"


def generate_iid_uniform_fixedlen(
    rng: np.random.Generator,
    shapes: list[dict],
    word_len: int = IID_WORD_LEN,
) -> list[dict]:
    """Harder negative control: fixed-length uniform a–z words (not train unigram)."""
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    pages = []
    for shape in shapes:
        lines = []
        for n_words in shape["line_word_counts"]:
            words = []
            for _ in range(max(n_words, 1)):
                chars = "".join(
                    alphabet[int(rng.integers(len(alphabet)))] for _ in range(word_len)
                )
                words.append(chars)
            if n_words:
                lines.append(" ".join(words[:n_words]))
            else:
                lines.append(words[0])
        pages.append(
            {
                "page_id": f"iid-uf-{shape['page_id']}",
                "L": shape["L"],
                "text": "\n".join(lines),
            }
        )
    return pages


def decide_pass(
    winner_opt_n_match: int,
    winner_heldout_n_separate: int,
    iid_opt_n_separate: int,
) -> dict:
    reasons = []
    if iid_opt_n_separate < 3:
        mode = "iid_control_failed"
        reasons.append(
            f"{IID_NAME} separated on only {iid_opt_n_separate}/4 optimizable metrics "
            "(need ≥3)"
        )
        passed = False
        hyp005 = "untested"
    elif winner_opt_n_match < 3:
        mode = "surface_unmatchable"
        reasons.append(
            f"winner matched only {winner_opt_n_match}/4 optimizable metrics (need ≥3); "
            "generator family cannot force selected surface match"
        )
        passed = False
        hyp005 = "open_not_killed"
    elif winner_heldout_n_separate < 2:
        mode = "heldout_also_matched"
        reasons.append(
            f"winner separated on only {winner_heldout_n_separate}/3 held-out "
            "diagnostics (need ≥2)"
        )
        passed = False
        hyp005 = "open_compatible"
    else:
        mode = "surface_match_heldout_fail"
        reasons.append("all EXP-0025 gates cleared")
        passed = True
        hyp005 = "open_compatible"
    return {
        "passed": passed,
        "mode": mode,
        "hyp005_status": hyp005,
        "reasons": reasons,
        "rule": EXPERIMENT_ID,
        "gates": {
            "optimizable_match_min": 3,
            "heldout_separate_min": 2,
            "iid_optimizable_separate_min": 3,
            "winner_optimizable_n_match": winner_opt_n_match,
            "winner_heldout_n_separate": winner_heldout_n_separate,
            "iid_optimizable_n_separate": iid_opt_n_separate,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    data_root = args.data_root or (root / "data" / "processed" / "zl3b")
    out_dir = root / "results" / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train = load_pages(data_root / "train.jsonl")
    validation = load_pages(data_root / "validation.jsonl")
    test_path = data_root / "test.jsonl"
    test_pages = sum(1 for _ in test_path.open()) if test_path.exists() else 0

    train_m = metrics_from_pages(train)
    val_m = metrics_from_pages(validation)
    shapes = page_shapes(validation)

    train_a = [p for p in train if p["L"] == "A"]
    train_b = [p for p in train if p["L"] == "B"]
    style_models = {}
    for label, subset in (("A", train_a), ("B", train_b)):
        bigrams, starts, lengths = fit_bigram_and_lengths(subset)
        style_models[label] = {"bigrams": bigrams, "starts": starts, "lengths": lengths}
    n_ab = len(train_a) + len(train_b)
    default_style_probs = {"A": len(train_a) / n_ab, "B": len(train_b) / n_ab}

    iid_rng = np.random.default_rng(args.seed + 17)
    iid_pages = generate_iid_uniform_fixedlen(iid_rng, shapes, IID_WORD_LEN)
    iid_m = metrics_from_pages(iid_pages)
    iid_opt = metric_verdicts(train_m, val_m, iid_m, OPT_METRICS)
    iid_held = metric_verdicts(train_m, val_m, iid_m, HELDOUT_METRICS)

    search = run_search(
        train_m, val_m, shapes, style_models, default_style_probs, seed=args.seed
    )
    winner = search["winner"]
    heldout_n_separate = sum(
        1 for v in winner["heldout"]["per_metric"].values() if v["verdict"] == "separate"
    )
    iid_opt_n_separate = iid_opt["n_metrics"] - iid_opt["n_match"]

    decision = decide_pass(
        winner_opt_n_match=winner["optimizable"]["n_match"],
        winner_heldout_n_separate=heldout_n_separate,
        iid_opt_n_separate=iid_opt_n_separate,
    )
    decision["winner_genome"] = winner["genome"]
    decision["optimizable"] = winner["optimizable"]
    decision["heldout"] = winner["heldout"]
    decision["iid_optimizable"] = iid_opt
    decision["iid_control"] = {
        "name": IID_NAME,
        "word_len": IID_WORD_LEN,
        "alphabet": "uniform_a_z",
        "vs_exp0019": "replaces train-unigram iid_char",
    }
    decision["thresholds"] = {
        "floors": FLOORS,
        "optimizable_metrics": list(OPT_METRICS),
        "heldout_metrics": list(HELDOUT_METRICS),
        "match_rule": "abs(G-R) <= max(|T-R|, floor)",
        "population": POP_SIZE,
        "generations": N_GENERATIONS,
        "seed": args.seed,
        "iid_word_len": IID_WORD_LEN,
    }

    # If iid failed, strip held-out interpretation from decision payload note.
    if decision["mode"] == "iid_control_failed":
        decision["heldout_interpretation"] = "not_interpreted"
    else:
        decision["heldout_interpretation"] = "preregistered_rule_applied"

    elapsed = time.perf_counter() - t0
    results = {
        "experiment": EXPERIMENT_ID,
        "question": (
            "With harder iid_uniform_fixedlen control, can evolutionary copy-mutate "
            "search force ≥3/4 optimizable ZL3b surface match and then separate on "
            "≥2/3 held-out diagnostics?"
        ),
        "linked": ["HYP-005", "R3", "EXP-0019", "NB-0022"],
        "parent_fail": {
            "experiment": "EXP-0019",
            "mode": "iid_control_failed",
            "iid_was": "iid_char_train_unigram",
        },
        "seed": args.seed,
        "data_root": str(data_root),
        "test_pages_present_unscored": test_pages,
        "elapsed_seconds": elapsed,
        "reference_metrics": {"train": train_m, "validation": val_m},
        "iid_control": {
            "name": IID_NAME,
            "word_len": IID_WORD_LEN,
            "role": "harder_negative_control",
            "optimizable": iid_opt,
            "heldout_audit_only": iid_held,
            "metrics": {k: iid_m[k] for k in list(OPT_METRICS) + list(HELDOUT_METRICS)},
        },
        "search": {
            "population": POP_SIZE,
            "generations_max": N_GENERATIONS,
            "elite": N_ELITE,
            "n_unique_genomes_scored": search["n_unique_genomes_scored"],
            "history": search["history"],
            "defaults_genome": DEFAULT_GENOME,
        },
        "defaults_reference": search["defaults_reference"],
        "winner": winner,
        "decision": decision,
        "non_claims": [
            "Not a decipherment",
            "Not a language ID",
            "Surface-stat match alone is not a HYP-005 proof",
        ],
    }

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    results_s = _sanitize(results)
    decision_s = _sanitize(decision)
    (out_dir / "results.json").write_text(json.dumps(results_s, indent=2) + "\n")
    (out_dir / "decision.json").write_text(json.dumps(decision_s, indent=2) + "\n")

    manifest = {
        "experiment": EXPERIMENT_ID,
        "seed": args.seed,
        "data_root": str(data_root),
        "train_pages": len(train),
        "validation_pages": len(validation),
        "test_pages_unscored": test_pages,
        "iid_control": IID_NAME,
        "iid_word_len": IID_WORD_LEN,
        "optimizable_metrics": list(OPT_METRICS),
        "heldout_metrics": list(HELDOUT_METRICS),
        "floors": FLOORS,
        "passed": decision["passed"],
        "mode": decision["mode"],
        "hyp005_status": decision["hyp005_status"],
    }
    man_path = root / "data" / "manifests" / "exp0025_data.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "wrote": str(out_dir),
                "passed": decision["passed"],
                "mode": decision["mode"],
                "hyp005_status": decision["hyp005_status"],
                "winner_opt_match": winner["optimizable"]["n_match"],
                "winner_heldout_separate": heldout_n_separate,
                "iid_opt_separate": iid_opt_n_separate,
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
