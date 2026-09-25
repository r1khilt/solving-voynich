"""One frozen held-out evaluation of the four EXP-0032 checkpoints."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch

from voynich.exact_count_decode import exact_count_keep_masks
from voynich.exp0032_model import (
    SEEDS,
    VARIANTS,
    ContextMaskModel,
    arrays,
    canonical_token_ids,
    digest,
    probabilities,
)
from voynich.latent_recovery import CIPHER_POOL, copy_aware_features, recon_accuracy
from voynich.runtime import write_json


FAMILIES = ("random_char", "periodic", "copy_mutate")
BOOTSTRAP_SEED = 320320
BOOTSTRAP_DRAWS = 2000
RANDOM_SEEDS = tuple(range(320400, 320420))
MAX_SECONDS = 300


def unpackable_mask(mask: np.ndarray) -> str:
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="big").tobytes().hex()


def counts(true: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    return (
        int(((true == 0) & (pred == 0)).sum()),
        int(((true == 1) & (pred == 0)).sum()),
        int(((true == 0) & (pred == 1)).sum()),
    )


def score(rows: list[dict], masks: list[np.ndarray]) -> dict:
    tp = fp = fn = 0
    recons = []
    exact = 0
    for row, pred in zip(rows, masks, strict=True):
        true = np.asarray(row["mask"])
        a, b, c = counts(true, pred)
        tp += a
        fp += b
        fn += c
        recons.append(recon_accuracy(row["ciphered"], row["text"], pred))
        exact += int(np.array_equal(true, pred))
    return {
        "n": len(rows),
        "null_precision": tp / max(tp + fp, 1),
        "null_recall": tp / max(tp + fn, 1),
        "null_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "recon_acc": float(np.mean(recons)),
        "exact_mask_rate": exact / len(rows),
    }


def matched_random(rows: list[dict]) -> dict:
    by_seed = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        masks = [exact_count_keep_masks([rng.random(128)])[0] for _ in rows]
        by_seed.append({
            family: score([r for r in rows if r["filler_family"] == family], [m for r, m in zip(rows, masks, strict=True) if r["filler_family"] == family])
            for family in sorted({r["filler_family"] for r in rows})
        })
    return {
        family: {metric: float(np.mean([entry[family][metric] for entry in by_seed])) for metric in ("null_precision", "null_recall", "null_f1", "recon_acc", "exact_mask_rate")}
        for family in by_seed[0]
    }


def micro_f1_from_subset(true_masks: np.ndarray, predictions: np.ndarray, selected: np.ndarray) -> float:
    true = true_masks[selected]
    pred = predictions[selected]
    tp = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 1) & (pred == 0)).sum())
    fn = int(((true == 0) & (pred == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def bootstrap_copy_gain(rows: list[dict], predictions: dict[str, np.ndarray]) -> tuple[float, list[float]]:
    copy_ids = np.asarray([i for i, row in enumerate(rows) if row["filler_family"] == "copy_mutate"], dtype=int)
    true = np.asarray([rows[int(i)]["mask"] for i in copy_ids], dtype=np.uint8)
    pred = {name: masks[copy_ids] for name, masks in predictions.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(copy_ids), size=(BOOTSTRAP_DRAWS, len(copy_ids)))
    differences = []
    for chosen in draws:
        gain = np.mean([
            micro_f1_from_subset(true, pred[f"canonical_{seed}"], chosen)
            - micro_f1_from_subset(true, pred[f"raw_{seed}"], chosen)
            for seed in SEEDS
        ])
        differences.append(gain)
    observed = float(np.mean([
        micro_f1_from_subset(true, pred[f"canonical_{seed}"], np.arange(len(copy_ids)))
        - micro_f1_from_subset(true, pred[f"raw_{seed}"], np.arange(len(copy_ids)))
        for seed in SEEDS
    ]))
    return observed, np.quantile(differences, [0.025, 0.975]).astype(float).tolist()


def check_canonical_invariance(rows: list[dict], model: ContextMaskModel, device: str) -> float:
    pool = list(CIPHER_POOL)
    permutation = np.random.default_rng(320300).permutation(len(pool))
    mapping = {ch: pool[int(permutation[i])] for i, ch in enumerate(pool)}
    renamed = ["".join(mapping.get(ch, ch) for ch in row["text"]) for row in rows]
    for row, changed in zip(rows, renamed, strict=True):
        if canonical_token_ids(row["text"]) != canonical_token_ids(changed):
            raise AssertionError("canonical input changed under bijection")
        if not np.array_equal(copy_aware_features(row["text"]), copy_aware_features(changed)):
            raise AssertionError("invariant side features changed under bijection")
    selected = list(range(min(16, len(rows))))
    original_rows = [rows[i] for i in selected]
    changed_rows = [{**rows[i], "text": renamed[i]} for i in selected]
    original_data = arrays(original_rows, "canonical", device)
    changed_data = arrays(changed_rows, "canonical", device)
    orig = probabilities(model, original_data, list(range(len(selected))))
    changed = probabilities(model, changed_data, list(range(len(selected))))
    return float(np.max(np.abs(orig - changed)))


def evaluate(root: Path, device: str) -> dict:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(6)
    manifest_path = root / "data/manifests/exp0032_data.json"
    source_audit = json.loads((root / "results/EXP-0032/data_audit.json").read_text())
    if not source_audit["passed"] or source_audit["manifest_sha256"] != digest(manifest_path):
        raise AssertionError("data audit gate failed")
    manifest = json.loads(manifest_path.read_text())
    paths = {name: root / f"data/processed/exp0032/{name}.jsonl" for name in ("polish_holdout", "iid_control")}
    for name, path in paths.items():
        if digest(path) != manifest["derived_sha256"][name]:
            raise AssertionError(f"{name} data changed")
    rows = {name: [json.loads(line) for line in path.read_text().splitlines()] for name, path in paths.items()}
    if len(rows["polish_holdout"]) != 600 or len(rows["iid_control"]) != 600:
        raise AssertionError("evaluation populations changed")
    baseline = {name: matched_random(group) for name, group in rows.items()}
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}
    params = set()
    canonical_invariance = []
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f"{variant}_{seed}"
            train_path = root / f"results/EXP-0032/train_{name}.json"
            train_report = json.loads(train_path.read_text())
            ckpt_path = root / f"outputs/EXP-0032/{name}.pt"
            if train_report["checkpoint_sha256"] != digest(ckpt_path) or train_report["manifest_sha256"] != digest(manifest_path):
                raise AssertionError(f"training checkpoint link invalid: {name}")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if checkpoint["variant"] != variant or checkpoint["seed"] != seed:
                raise AssertionError("wrong checkpoint identity")
            model = ContextMaskModel().to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            params.add(sum(p.numel() for p in model.parameters()))
            if variant == "canonical":
                canonical_invariance.append(check_canonical_invariance(rows["polish_holdout"], model, device))
            model_report = {"checkpoint_sha256": digest(ckpt_path), "best_step": train_report["best_step"], "validation_copy_null_f1": train_report["best_validation_copy_null_f1"]}
            for split, group in rows.items():
                data = arrays(group, variant, device)
                probs = probabilities(model, data, list(range(len(group))))
                masks = np.asarray(exact_count_keep_masks([p for p in probs]), dtype=np.uint8)
                if split == "polish_holdout":
                    predictions[name] = masks
                model_report[split] = {
                    "by_family": {
                        family: score([r for r in group if r["filler_family"] == family], [m for r, m in zip(group, masks, strict=True) if r["filler_family"] == family])
                        for family in sorted({r["filler_family"] for r in group})
                    },
                    "packed_masks": [unpackable_mask(mask) for mask in masks],
                    "probability_probes": [{"row_index": i, "probabilities": probs[i].astype(float).tolist()} for i in (0, 74, 149, 224, 299, 449, 599)],
                }
            reports[name] = model_report
    if len(params) != 1:
        raise AssertionError("raw/canonical model capacities differ")
    paired_gain, ci = bootstrap_copy_gain(rows["polish_holdout"], predictions)
    periodic_competence = all(reports[f"{variant}_{seed}"]["polish_holdout"]["by_family"]["periodic"]["null_f1"] >= 0.70 for seed in SEEDS for variant in VARIANTS)
    iid_baseline = baseline["iid_control"]["iid_indistinguishable"]["null_f1"]
    iid_no_leak = all(reports[f"{variant}_{seed}"]["iid_control"]["by_family"]["iid_indistinguishable"]["null_f1"] <= iid_baseline + 0.05 for seed in SEEDS for variant in VARIANTS)
    copy_random_recon = baseline["polish_holdout"]["copy_mutate"]["recon_acc"]
    canonical_copy_recon = float(np.mean([reports[f"canonical_{seed}"]["polish_holdout"]["by_family"]["copy_mutate"]["recon_acc"] for seed in SEEDS]))
    invariant = max(canonical_invariance) <= 1e-6
    success = bool(periodic_competence and iid_no_leak and invariant and paired_gain >= 0.05 and ci[0] > 0.02 and canonical_copy_recon > copy_random_recon + 0.05)
    decision = "INCONCLUSIVE_OR_INVALID" if not (periodic_competence and iid_no_leak and invariant) else ("SUPPORTED" if success else "NOT_SUPPORTED")
    return {
        "experiment": "EXP-0032",
        "status": "exploratory_new_polish_source",
        "device": device,
        "torch": str(torch.__version__),
        "manifest_sha256": digest(manifest_path),
        "data_audit_sha256": digest(root / "results/EXP-0032/data_audit.json"),
        "evaluator_sha256": digest(Path(__file__)),
        "parameter_count": next(iter(params)),
        "random_seeds": RANDOM_SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "baselines": baseline,
        "models": reports,
        "paired_copy_null_f1_gain": paired_gain,
        "paired_copy_null_f1_gain_ci95": ci,
        "canonical_invariance_max_probability_error": max(canonical_invariance),
        "periodic_competence": periodic_competence,
        "iid_no_leak": iid_no_leak,
        "invariant": invariant,
        "canonical_copy_mean_recon": canonical_copy_recon,
        "copy_random_mean_recon": copy_random_recon,
        "registered_success": success,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    output = args.root / "results/EXP-0032/evaluation.json"
    if output.exists():
        raise FileExistsError("evaluation output exists; refusing overwrite")
    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"EXP-0032 evaluation exceeded {MAX_SECONDS} seconds")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    result = evaluate(args.root, args.device)
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ("parameter_count", "paired_copy_null_f1_gain", "paired_copy_null_f1_gain_ci95", "canonical_invariance_max_probability_error", "periodic_competence", "iid_no_leak", "canonical_copy_mean_recon", "copy_random_mean_recon", "registered_success", "decision")}, indent=2))


if __name__ == "__main__":
    main()
