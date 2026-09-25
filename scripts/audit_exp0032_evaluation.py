"""Independent packed-mask arithmetic and selected direct-model replay for EXP-0032."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from voynich.exp0032_model import ContextMaskModel
from voynich.latent_recovery import CIPHER_POOL, build_vocab, copy_aware_features
from voynich.runtime import write_json


SEEDS = (320101, 320202)
VARIANTS = ("raw", "canonical")
FAMILIES = ("random_char", "periodic", "copy_mutate")
PROBES = (0, 74, 149, 224, 299, 449, 599)
RANDOM_SEEDS = tuple(range(320400, 320420))
BOOT_SEED = 320320
BOOT_DRAWS = 2000
VOCAB = build_vocab()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual: float, expected: float, tol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=tol):
        raise AssertionError(f"{actual} != {expected}")


def independent_ids(text: str, variant: str) -> list[int]:
    if variant == "raw":
        return [VOCAB.get(ch, VOCAB["<unk>"]) for ch in text]
    seen: dict[str, int] = {}
    output = []
    for char in text:
        if char in CIPHER_POOL:
            seen.setdefault(char, len(seen))
            output.append(VOCAB[CIPHER_POOL[seen[char]]])
        else:
            output.append(VOCAB.get(char, VOCAB["<unk>"]))
    return output


def unpack_masks(strings: list[str]) -> np.ndarray:
    output = []
    for value in strings:
        packed = bytes.fromhex(value)
        if len(packed) != 16:
            raise AssertionError("packed mask must contain exactly 128 bits")
        output.append(np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big"))
    return np.asarray(output, dtype=np.uint8)


def independent_recon(row: dict, prediction: np.ndarray) -> float:
    observed = "".join(char for char, keep in zip(row["text"], prediction, strict=True) if keep)
    target = row["ciphered"]
    common = min(len(observed), len(target))
    if common == 0:
        return float(observed == target)
    matches = sum(a == b for a, b in zip(observed[:common], target[:common], strict=True))
    return matches / max(len(observed), len(target), 1)


def independent_score(rows: list[dict], predictions: np.ndarray) -> dict:
    if len(rows) != len(predictions):
        raise AssertionError("score population mismatch")
    true = np.asarray([row["mask"] for row in rows], dtype=np.uint8)
    tp = int(((true == 0) & (predictions == 0)).sum())
    fp = int(((true == 1) & (predictions == 0)).sum())
    fn = int(((true == 0) & (predictions == 1)).sum())
    return {
        "n": len(rows),
        "null_precision": tp / max(tp + fp, 1),
        "null_recall": tp / max(tp + fn, 1),
        "null_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "recon_acc": float(np.mean([independent_recon(row, mask) for row, mask in zip(rows, predictions, strict=True)])),
        "exact_mask_rate": float(np.mean(np.all(true == predictions, axis=1))),
    }


def audit(root: Path) -> dict:
    result_path = root / "results/EXP-0032/evaluation.json"
    result = json.loads(result_path.read_text())
    manifest_path = root / "data/manifests/exp0032_data.json"
    if result["experiment"] != "EXP-0032" or result["manifest_sha256"] != digest(manifest_path):
        raise AssertionError("experiment or manifest link mismatch")
    source_audit_path = root / "results/EXP-0032/data_audit.json"
    if result["data_audit_sha256"] != digest(source_audit_path) or not json.loads(source_audit_path.read_text())["passed"]:
        raise AssertionError("source audit link mismatch")
    if result["evaluator_sha256"] != digest(root / "scripts/evaluate_exp0032.py"):
        raise AssertionError("evaluator hash mismatch")
    if result["bootstrap_seed"] != BOOT_SEED or result["bootstrap_draws"] != BOOT_DRAWS or tuple(result["random_seeds"]) != RANDOM_SEEDS:
        raise AssertionError("frozen seeds changed")
    manifest = json.loads(manifest_path.read_text())
    rows = {}
    for split in ("polish_holdout", "iid_control"):
        path = root / f"data/processed/exp0032/{split}.jsonl"
        if digest(path) != manifest["derived_sha256"][split]:
            raise AssertionError("evaluation data hash mismatch")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[split]) != 600:
            raise AssertionError("wrong evaluation size")
    predictions: dict[str, np.ndarray] = {}
    max_probe_error = 0.0
    max_invariance_error = 0.0
    param_counts = set()
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f"{variant}_{seed}"
            saved = result["models"][name]
            checkpoint_path = root / f"outputs/EXP-0032/{name}.pt"
            report_path = root / f"results/EXP-0032/train_{name}.json"
            report = json.loads(report_path.read_text())
            if digest(checkpoint_path) != saved["checkpoint_sha256"] or digest(checkpoint_path) != report["checkpoint_sha256"]:
                raise AssertionError("checkpoint hash mismatch")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if ckpt["runner_sha256"] != digest(root / "src/voynich/exp0032_model.py") or report["runner_sha256"] != ckpt["runner_sha256"]:
                raise AssertionError("training runner hash mismatch")
            model = ContextMaskModel().eval()
            model.load_state_dict(ckpt["model"])
            param_counts.add(sum(p.numel() for p in model.parameters()))
            if ckpt["best_step"] != saved["best_step"] or report["best_step"] != saved["best_step"]:
                raise AssertionError("selection step mismatch")
            if ckpt["validation_copy_null_f1"] != saved["validation_copy_null_f1"] or report["best_validation_copy_null_f1"] != saved["validation_copy_null_f1"]:
                raise AssertionError("validation selection score mismatch")
            for split, source in rows.items():
                split_report = saved[split]
                masks = unpack_masks(split_report["packed_masks"])
                if len(masks) != len(source) or not np.all(masks.sum(axis=1) == 90):
                    raise AssertionError("exact-count mask invariant failed")
                if split == "polish_holdout":
                    predictions[name] = masks
                for family in sorted({row["filler_family"] for row in source}):
                    subset = [i for i, row in enumerate(source) if row["filler_family"] == family]
                    computed = independent_score([source[i] for i in subset], masks[subset])
                    for metric, value in computed.items():
                        check(split_report["by_family"][family][metric], value)
                if {(probe["row_index"]) for probe in split_report["probability_probes"]} != set(PROBES):
                    raise AssertionError("probe roster mismatch")
                with torch.no_grad():
                    for probe in split_report["probability_probes"]:
                        index = probe["row_index"]
                        text = source[index]["text"]
                        x = torch.tensor([independent_ids(text, variant)], dtype=torch.long)
                        features = torch.tensor(copy_aware_features(text)[None], dtype=torch.float32)
                        logits, _ = model(x, features)
                        p = logits.sigmoid().numpy()[0]
                        error = float(np.max(np.abs(p - probe["probabilities"])))
                        max_probe_error = max(max_probe_error, error)
                        if error > 1e-5:
                            raise AssertionError(f"direct model probability replay mismatch {name}/{split}/{index}")
            if variant == "canonical":
                pool = list(CIPHER_POOL)
                order = np.random.default_rng(320300).permutation(len(pool))
                translation = str.maketrans({char: pool[int(order[i])] for i, char in enumerate(pool)})
                with torch.no_grad():
                    for source in rows["polish_holdout"]:
                        changed = source["text"].translate(translation)
                        if independent_ids(source["text"], "canonical") != independent_ids(changed, "canonical"):
                            raise AssertionError("canonical IDs changed under rename")
                        if not np.array_equal(copy_aware_features(source["text"]), copy_aware_features(changed)):
                            raise AssertionError("canonical side features changed under rename")
                    for source in rows["polish_holdout"][:16]:
                        original = source["text"]
                        changed = original.translate(translation)
                        x0 = torch.tensor([independent_ids(original, "canonical")], dtype=torch.long)
                        x1 = torch.tensor([independent_ids(changed, "canonical")], dtype=torch.long)
                        f0 = torch.tensor(copy_aware_features(original)[None], dtype=torch.float32)
                        f1 = torch.tensor(copy_aware_features(changed)[None], dtype=torch.float32)
                        p0 = model(x0, f0)[0].sigmoid()
                        p1 = model(x1, f1)[0].sigmoid()
                        max_invariance_error = max(max_invariance_error, float((p0 - p1).abs().max().item()))
    if param_counts != {result["parameter_count"]} or len(param_counts) != 1:
        raise AssertionError("capacity matching failed")

    baseline = {}
    for split, source in rows.items():
        individual = []
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            masks = []
            for _ in source:
                scores = rng.random(128)
                top = np.argsort(-scores, kind="stable")[:90]
                mask = np.zeros(128, dtype=np.uint8)
                mask[top] = 1
                masks.append(mask)
            masks = np.asarray(masks)
            individual.append({family: independent_score([source[i] for i, row in enumerate(source) if row["filler_family"] == family], masks[[i for i, row in enumerate(source) if row["filler_family"] == family]]) for family in sorted({r["filler_family"] for r in source})})
        baseline[split] = {family: {metric: float(np.mean([item[family][metric] for item in individual])) for metric in ("null_precision", "null_recall", "null_f1", "recon_acc", "exact_mask_rate")} for family in individual[0]}
        for family, metrics in baseline[split].items():
            for metric, value in metrics.items():
                check(result["baselines"][split][family][metric], value)

    copy = [i for i, row in enumerate(rows["polish_holdout"]) if row["filler_family"] == "copy_mutate"]
    truth = np.asarray([rows["polish_holdout"][i]["mask"] for i in copy], dtype=np.uint8)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(copy), size=(BOOT_DRAWS, len(copy)))

    def f1(true: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> float:
        y, p = true[indices], mask[indices]
        tp = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 1) & (p == 0)).sum())
        fn = int(((y == 0) & (p == 1)).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1)

    pred = {name: masks[copy] for name, masks in predictions.items()}
    all_indices = np.arange(len(copy))
    observed = float(np.mean([f1(truth, pred[f"canonical_{seed}"], all_indices) - f1(truth, pred[f"raw_{seed}"], all_indices) for seed in SEEDS]))
    boots = [np.mean([f1(truth, pred[f"canonical_{seed}"], selected) - f1(truth, pred[f"raw_{seed}"], selected) for seed in SEEDS]) for selected in draws]
    ci = np.quantile(boots, [0.025, 0.975])
    check(result["paired_copy_null_f1_gain"], observed)
    for actual, expected in zip(result["paired_copy_null_f1_gain_ci95"], ci, strict=True):
        check(actual, expected)
    periodic = all(result["models"][f"{variant}_{seed}"]["polish_holdout"]["by_family"]["periodic"]["null_f1"] >= 0.70 for variant in VARIANTS for seed in SEEDS)
    iid_floor = baseline["iid_control"]["iid_indistinguishable"]["null_f1"]
    iid = all(result["models"][f"{variant}_{seed}"]["iid_control"]["by_family"]["iid_indistinguishable"]["null_f1"] <= iid_floor + 0.05 for variant in VARIANTS for seed in SEEDS)
    copy_random = baseline["polish_holdout"]["copy_mutate"]["recon_acc"]
    copy_recon = float(np.mean([result["models"][f"canonical_{seed}"]["polish_holdout"]["by_family"]["copy_mutate"]["recon_acc"] for seed in SEEDS]))
    check(result["canonical_invariance_max_probability_error"], max_invariance_error, 1e-6)
    invariant = max_invariance_error <= 1e-6
    success = bool(periodic and iid and invariant and observed >= 0.05 and ci[0] > 0.02 and copy_recon > copy_random + 0.05)
    decision = "INCONCLUSIVE_OR_INVALID" if not (periodic and iid and invariant) else ("SUPPORTED" if success else "NOT_SUPPORTED")
    for key, value in (("periodic_competence", periodic), ("iid_no_leak", iid), ("invariant", invariant), ("registered_success", success)):
        if result[key] != value:
            raise AssertionError(f"decision mismatch: {key}")
    check(result["canonical_copy_mean_recon"], copy_recon)
    check(result["copy_random_mean_recon"], copy_random)
    if result["decision"] != decision:
        raise AssertionError("final decision label mismatch")
    return {
        "experiment": "EXP-0032",
        "passed": True,
        "evaluation_sha256": digest(result_path),
        "auditor_sha256": digest(Path(__file__)),
        "models_checked": len(SEEDS) * len(VARIANTS),
        "packed_masks_checked": 4 * 2 * 600,
        "probability_probes_replayed": 4 * 2 * len(PROBES),
        "max_probe_abs_error": max_probe_error,
        "max_direct_invariance_error": max_invariance_error,
        "paired_copy_null_f1_gain": observed,
        "paired_copy_null_f1_gain_ci95": ci.astype(float).tolist(),
        "registered_success": success,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    path = args.root / "results/EXP-0032/evaluation_audit.json"
    if path.exists():
        raise FileExistsError("evaluation audit already exists; refusing overwrite")
    report = audit(args.root)
    write_json(path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
