"""One frozen Portuguese and iid evaluation of hard versus soft targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
import torch

from voynich.exact_count_decode import exact_count_keep_masks
from voynich.exp0032_model import ContextMaskModel, arrays, canonical_token_ids, probabilities
from voynich.latent_recovery import CIPHER_POOL, copy_aware_features, recon_accuracy
from voynich.runtime import write_json


EXPERIMENT = "EXP-0035"
DATA_MANIFEST_SHA = "ca8e3e6c7dfab5df74f750b6434b9905629f2f629a9c886e8f780935e406ec44"
BASELINE_FREEZE_SHA = "69a8000b6f7310a9a330fbf3fa89d614fd71004794c928e6a874a8668ea07fbb"
SEEDS = (320101, 320202)
CONDITIONS = ("hard", "uniform", "weighted")
FAMILIES = ("random_char", "periodic", "copy_mutate")
RANDOM_SEEDS = tuple(range(350400, 350420))
BOOTSTRAP_SEED = 350350
BOOTSTRAP_DRAWS = 2000
PROBES = (0, 74, 149, 224, 299, 449, 599)
MAX_SECONDS = 300


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(mask: np.ndarray) -> str:
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="big").tobytes().hex()


def score(rows: list[dict], masks: np.ndarray) -> dict:
    tp = fp = fn = exact = 0
    recon = []
    for row, mask in zip(rows, masks, strict=True):
        gold = np.asarray(row["mask"], dtype=np.uint8)
        tp += int(((gold == 0) & (mask == 0)).sum())
        fp += int(((gold == 1) & (mask == 0)).sum())
        fn += int(((gold == 0) & (mask == 1)).sum())
        exact += int(np.array_equal(gold, mask))
        recon.append(recon_accuracy(row["ciphered"], row["text"], mask))
    return {"n": len(rows), "null_precision": tp / max(tp + fp, 1),
            "null_recall": tp / max(tp + fn, 1), "null_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "recon_acc": float(np.mean(recon)), "exact_mask_rate": exact / len(rows)}


def random_baseline(rows: list[dict]) -> dict:
    reports = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        masks = np.asarray([exact_count_keep_masks([rng.random(128)])[0] for _ in rows], dtype=np.uint8)
        reports.append({family: score([r for r in rows if r["filler_family"] == family],
                                      np.asarray([m for r, m in zip(rows, masks, strict=True) if r["filler_family"] == family], dtype=np.uint8))
                        for family in sorted({r["filler_family"] for r in rows})})
    return {family: {metric: float(np.mean([entry[family][metric] for entry in reports]))
                     for metric in ("null_precision", "null_recall", "null_f1", "recon_acc", "exact_mask_rate")}
            for family in reports[0]}


def micro_null_f1(truth: np.ndarray, predicted: np.ndarray, selected: np.ndarray) -> float:
    a, b = truth[selected], predicted[selected]
    tp = int(((a == 0) & (b == 0)).sum())
    fp = int(((a == 1) & (b == 0)).sum())
    fn = int(((a == 0) & (b == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def paired_gain(rows: list[dict], predictions: dict[str, np.ndarray], left: str, right: str) -> tuple[float, list[float]]:
    ids = np.asarray([i for i, row in enumerate(rows) if row["filler_family"] == "copy_mutate"], dtype=int)
    gold = np.asarray([rows[i]["mask"] for i in ids], dtype=np.uint8)
    selected = {name: masks[ids] for name, masks in predictions.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))

    def difference(indices: np.ndarray) -> float:
        return float(np.mean([micro_null_f1(gold, selected[f"{left}_{seed}"], indices)
                              - micro_null_f1(gold, selected[f"{right}_{seed}"], indices)
                              for seed in SEEDS]))

    observed = difference(np.arange(len(ids)))
    interval = np.quantile([difference(indices) for indices in draws], [0.025, 0.975]).tolist()
    return observed, interval


def check_invariance(rows: list[dict], model: ContextMaskModel, device: str) -> float:
    pool = list(CIPHER_POOL)
    permutation = np.random.default_rng(350300).permutation(len(pool))
    mapping = {glyph: pool[int(permutation[i])] for i, glyph in enumerate(pool)}
    sample = rows[:16]
    renamed = [{**row, "text": "".join(mapping.get(ch, ch) for ch in row["text"])} for row in sample]
    for original, changed in zip(sample, renamed, strict=True):
        if canonical_token_ids(original["text"]) != canonical_token_ids(changed["text"]):
            raise AssertionError("canonical encoding not rename invariant")
        if not np.array_equal(copy_aware_features(original["text"]), copy_aware_features(changed["text"])):
            raise AssertionError("copy side features not rename invariant")
    first = probabilities(model, arrays(sample, "canonical", device), list(range(len(sample))))
    second = probabilities(model, arrays(renamed, "canonical", device), list(range(len(sample))))
    return float(np.max(np.abs(first - second)))


def evaluate(root: Path, device: str) -> dict:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(6)
    manifest_path = root / "data/manifests/exp0035_data.json"
    audit_path = root / "results/EXP-0035/data_audit.json"
    if digest(manifest_path) != DATA_MANIFEST_SHA:
        raise AssertionError("EXP-0035 source manifest drift")
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["manifest_sha256"] != digest(manifest_path):
        raise AssertionError("data audit gate failed")
    freeze_path = root / "results/EXP-0032/training_freeze.json"
    if digest(freeze_path) != BASELINE_FREEZE_SHA:
        raise AssertionError("hard baseline freeze changed")
    files = {name: root / f"data/processed/exp0035/{name}.jsonl" for name in ("portuguese_holdout", "iid_control")}
    for name, path in files.items():
        if digest(path) != manifest["derived_sha256"][name]:
            raise AssertionError(f"{name} data checksum changed")
    groups = {name: [json.loads(line) for line in path.read_text().splitlines()] for name, path in files.items()}
    if len(groups["portuguese_holdout"]) != 600 or len(groups["iid_control"]) != 600:
        raise AssertionError("held-out population size drift")
    baselines = {name: random_baseline(group) for name, group in groups.items()}
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}
    parameter_counts = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            name = f"{condition}_{seed}"
            if condition == "hard":
                report_path = root / f"results/EXP-0032/train_canonical_{seed}.json"
                checkpoint_path = root / f"outputs/EXP-0032/canonical_{seed}.pt"
            else:
                report_path = root / f"results/EXP-0035/train_{condition}_{seed}.json"
                checkpoint_path = root / f"outputs/EXP-0035/{condition}_{seed}.pt"
            training = json.loads(report_path.read_text())
            if training["checkpoint_sha256"] != digest(checkpoint_path):
                raise AssertionError(f"checkpoint digest drift: {name}")
            if condition != "hard" and training["data_manifest_sha256"] != DATA_MANIFEST_SHA:
                raise AssertionError(f"soft training data mismatch: {name}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["seed"] != seed or (condition != "hard" and checkpoint["variant"] != condition):
                raise AssertionError(f"checkpoint identity mismatch: {name}")
            model = ContextMaskModel().to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            parameter_counts.add(sum(p.numel() for p in model.parameters()))
            invariance = check_invariance(groups["portuguese_holdout"], model, device)
            model_report = {"checkpoint_sha256": digest(checkpoint_path),
                            "selected_step": training["best_step"],
                            "validation_copy_null_f1": training["best_validation_copy_null_f1"],
                            "invariance_max_probability_error": invariance}
            for group_name, rows in groups.items():
                data = arrays(rows, "canonical", device)
                probs = probabilities(model, data, list(range(len(rows))))
                masks = np.asarray(exact_count_keep_masks([p for p in probs]), dtype=np.uint8)
                if group_name == "portuguese_holdout":
                    predictions[name] = masks
                model_report[group_name] = {
                    "by_family": {family: score([r for r in rows if r["filler_family"] == family],
                                                np.asarray([m for r, m in zip(rows, masks, strict=True) if r["filler_family"] == family], dtype=np.uint8))
                                  for family in sorted({r["filler_family"] for r in rows})},
                    "packed_masks": [pack(mask) for mask in masks],
                    "probability_probes": [{"row_index": i, "probabilities": probs[i].astype(float).tolist()} for i in PROBES],
                }
            reports[name] = model_report
    if parameter_counts != {10_746_629}:
        raise AssertionError("model capacity mismatch")
    weighted_hard, primary_ci = paired_gain(groups["portuguese_holdout"], predictions, "weighted", "hard")
    uniform_hard, uniform_ci = paired_gain(groups["portuguese_holdout"], predictions, "uniform", "hard")
    weighted_uniform, weighted_uniform_ci = paired_gain(groups["portuguese_holdout"], predictions, "weighted", "uniform")
    periodic_ok = all(reports[f"{condition}_{seed}"]["portuguese_holdout"]["by_family"]["periodic"]["null_f1"] >= 0.70
                      for condition in CONDITIONS for seed in SEEDS)
    iid_random = baselines["iid_control"]["iid_indistinguishable"]["null_f1"]
    iid_ok = all(reports[f"{condition}_{seed}"]["iid_control"]["by_family"]["iid_indistinguishable"]["null_f1"] <= iid_random + 0.05
                 for condition in CONDITIONS for seed in SEEDS)
    invariant = all(reports[f"{condition}_{seed}"]["invariance_max_probability_error"] <= 1e-6
                    for condition in CONDITIONS for seed in SEEDS)
    copy_recon = {condition: float(np.mean([reports[f"{condition}_{seed}"]["portuguese_holdout"]["by_family"]["copy_mutate"]["recon_acc"]
                                            for seed in SEEDS])) for condition in CONDITIONS}
    success = bool(periodic_ok and iid_ok and invariant and weighted_hard >= 0.03 and primary_ci[0] > 0.01
                   and copy_recon["weighted"] - copy_recon["hard"] >= 0.03)
    decision = "INCONCLUSIVE_OR_INVALID" if not (periodic_ok and iid_ok and invariant) else ("SUPPORTED" if success else "NOT_SUPPORTED")
    return {"experiment": EXPERIMENT, "status": "prospective_portuguese_source_holdout",
            "device": device, "torch": str(torch.__version__),
            "data_manifest_sha256": DATA_MANIFEST_SHA, "data_audit_sha256": digest(audit_path),
            "baseline_freeze_sha256": BASELINE_FREEZE_SHA, "evaluator_sha256": digest(Path(__file__)),
            "parameter_count": next(iter(parameter_counts)),
            "random_seeds": RANDOM_SEEDS, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "baselines": baselines, "models": reports,
            "paired_copy_f1_weighted_minus_hard": weighted_hard, "paired_copy_f1_weighted_minus_hard_ci95": primary_ci,
            "paired_copy_f1_uniform_minus_hard": uniform_hard, "paired_copy_f1_uniform_minus_hard_ci95": uniform_ci,
            "paired_copy_f1_weighted_minus_uniform": weighted_uniform, "paired_copy_f1_weighted_minus_uniform_ci95": weighted_uniform_ci,
            "copy_mean_reconstruction": copy_recon,
            "periodic_competence": periodic_ok, "iid_no_leak": iid_ok, "rename_invariance": invariant,
            "registered_success": success, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    output = args.root / "results/EXP-0035/evaluation.json"
    if output.exists():
        raise FileExistsError("EXP-0035 evaluation output already exists")

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError("EXP-0035 evaluation exceeded 300 seconds")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    try:
        report = evaluate(args.root, args.device)
    finally:
        signal.alarm(0)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in (
        "parameter_count", "paired_copy_f1_weighted_minus_hard", "paired_copy_f1_weighted_minus_hard_ci95",
        "paired_copy_f1_uniform_minus_hard", "paired_copy_f1_weighted_minus_uniform",
        "copy_mean_reconstruction", "periodic_competence", "iid_no_leak", "rename_invariance",
        "registered_success", "decision")}, indent=2))


if __name__ == "__main__":
    main()
