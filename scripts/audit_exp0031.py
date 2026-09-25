"""Independent arithmetic and selected direct-inference replay for EXP-0031."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from voynich.latent_recovery import CIPHER_POOL, TinySignalModel, copy_aware_features
from voynich.runtime import write_json


SEED = 310031
BOOT_SEED = 310131
REPS = (0, 1, 17, 31)
PROBE_ROWS = (0, 74, 149, 224, 299)
FAMILIES = ("random_char", "periodic", "copy_mutate")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tol: float = 1e-7) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=tol):
        raise AssertionError(f"{actual} != {expected} (tolerance {tol})")


def relabel_independently(text: str, row_index: int, replicate: int) -> str:
    if replicate == 0:
        return text
    symbols = list(CIPHER_POOL)
    permuted = np.random.default_rng(SEED + 1000 * row_index + replicate).permutation(symbols)
    translate = str.maketrans({source: str(target) for source, target in zip(symbols, permuted, strict=True)})
    return text.translate(translate)


def aggregate(entries: list[dict], kind: str) -> dict[str, float]:
    n = len(entries)
    counts = [r[f"{kind}_null_counts"] for r in entries]
    tp = sum(c["null_tp"] for c in counts)
    fp = sum(c["null_fp"] for c in counts)
    fn = sum(c["null_fn"] for c in counts)
    pred_null = sum(c["pred_nulls"] for c in counts)
    return {
        "n": n,
        "recon_acc": float(np.mean([r[f"{kind}_recon"] for r in entries])),
        "null_precision": tp / max(tp + fp, 1),
        "null_recall": tp / max(tp + fn, 1),
        "pred_null_rate": pred_null / (128 * n),
    }


def independent_recon(source: dict, pred_mask: np.ndarray) -> float:
    predicted = "".join(ch for ch, keep in zip(source["text"], pred_mask, strict=True) if keep)
    target = source["ciphered"]
    common = min(len(predicted), len(target))
    if common == 0:
        return float(predicted == target)
    correct = sum(a == b for a, b in zip(predicted[:common], target[:common], strict=True))
    return correct / max(len(predicted), len(target), 1)


def audit(root: Path) -> dict:
    result_path = root / "results/EXP-0031/results.json"
    result = json.loads(result_path.read_text())
    ckpt_path = root / "outputs/EXP-0030/model.pt"
    holdout_path = root / "data/processed/exp0030/finnish_holdout.jsonl"
    runner_path = root / "src/voynich/symbol_renaming_assay.py"
    for name, path in (
        ("checkpoint_sha256", ckpt_path),
        ("holdout_sha256", holdout_path),
        ("runner_sha256", runner_path),
    ):
        if result[name] != digest(path):
            raise AssertionError(f"{name} mismatch")
    if result["experiment"] != "EXP-0031" or result["status"] != "exploratory_exposed_holdout":
        raise AssertionError("wrong experiment identity/status")
    if result["n_rows"] != 300 or result["n_relabel_per_row"] != 31:
        raise AssertionError("wrong population or intervention count")
    if result["permutation_seed"] != SEED or result["bootstrap_seed"] != BOOT_SEED:
        raise AssertionError("seed mismatch")
    if result["bootstrap_draws"] != 2000:
        raise AssertionError("bootstrap count mismatch")
    holdout = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    rows = result["rows"]
    if len(holdout) != 300 or len(rows) != 300:
        raise AssertionError("row count mismatch")
    for i, (source, row) in enumerate(zip(holdout, rows, strict=True)):
        if row["row_index"] != i or row["family"] != source["filler_family"]:
            raise AssertionError(f"row identity mismatch: {i}")
        for kind in ("original", "ensemble"):
            c = row[f"{kind}_null_counts"]
            packed = bytes.fromhex(row[f"{kind}_mask_hex"])
            if len(packed) != 16:
                raise AssertionError("packed mask length mismatch")
            pred = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
            if not np.all(np.isin(pred, (0, 1))):
                raise AssertionError("invalid mask values")
            true = np.asarray(source["mask"], dtype=np.uint8)
            if len(true) != len(pred):
                raise AssertionError("mask/source length mismatch")
            replay_counts = {
                "null_tp": int(np.count_nonzero((true == 0) & (pred == 0))),
                "null_fp": int(np.count_nonzero((true == 1) & (pred == 0))),
                "null_fn": int(np.count_nonzero((true == 0) & (pred == 1))),
                "pred_nulls": int(np.count_nonzero(pred == 0)),
            }
            if c != replay_counts:
                raise AssertionError("per-row null counts do not match packed mask")
            close(row[f"{kind}_recon"], independent_recon(source, pred), 1e-10)
            if set(c) != {"null_tp", "null_fp", "null_fn", "pred_nulls"}:
                raise AssertionError("count keys mismatch")
            if c["null_tp"] + c["null_fp"] != c["pred_nulls"]:
                raise AssertionError("predicted-null count mismatch")
            if c["null_tp"] + c["null_fn"] != source["mask"].count(0):
                raise AssertionError("true-null count mismatch")
            if not 0 <= row[f"{kind}_recon"] <= 1:
                raise AssertionError("reconstruction out of range")
        if not 0 <= row["mean_keep_mask_flip_fraction"] <= 1:
            raise AssertionError("flip fraction out of range")
    for kind in ("original", "ensemble"):
        recomputed = aggregate(rows, kind)
        recorded = result[f"mixed_{kind}"]
        for key, value in recomputed.items():
            close(recorded[key], value)
        for family in FAMILIES:
            subset = [r for r in rows if r["family"] == family]
            recomputed = aggregate(subset, kind)
            recorded = result["families"][family][kind]
            for key, value in recomputed.items():
                close(recorded[key], value)
            for metric in (
                "mean_probability_abs_drift",
                "mean_keep_mask_flip_fraction",
                "mean_embedding_relative_l2",
                "mean_hidden_relative_l2",
            ):
                close(
                    result["families"][family][metric],
                    float(np.mean([r[metric] for r in subset])),
                )
    close(result["mixed_original"]["recon_acc"], 0.18838171635900422, 1e-10)
    close(result["families"]["copy_mutate"]["original"]["recon_acc"], 0.16651924193353831, 1e-10)
    flips = np.array([r["mean_keep_mask_flip_fraction"] for r in rows if r["family"] == "copy_mutate"])
    indices = np.random.default_rng(BOOT_SEED).integers(0, len(flips), size=(2000, len(flips)))
    ci = np.quantile(flips[indices].mean(axis=1), [0.025, 0.975])
    for actual, expected in zip(result["copy_flip_bootstrap_95"], ci, strict=True):
        close(actual, float(expected), 1e-10)
    valid = result["identity_max_probability_abs_diff"] <= 1e-6 and result["all_copy_features_invariant"]
    if result["valid"] != valid:
        raise AssertionError("validity decision mismatch")
    material = valid and float(flips.mean()) > 0.05 and float(ci[0]) > 0.02
    copy_orig = result["families"]["copy_mutate"]["original"]["recon_acc"]
    copy_ensemble = result["families"]["copy_mutate"]["ensemble"]["recon_acc"]
    benefit = valid and copy_ensemble > copy_orig + 0.02 and copy_ensemble > 0.2074623649767012
    if result["material_symmetry_violation"] != material or result["group_averaging_benefit"] != benefit:
        raise AssertionError("registered decision mismatch")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = TinySignalModel(len(ckpt["vocab"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    probes = result["probes"]
    if {(p["row_index"], p["replicate"]) for p in probes} != {
        (i, rep) for i in PROBE_ROWS for rep in REPS
    } or len(probes) != len(PROBE_ROWS) * len(REPS):
        raise AssertionError("probe roster mismatch")
    max_error = 0.0
    with torch.no_grad():
        for probe in probes:
            i, rep = probe["row_index"], probe["replicate"]
            original = holdout[i]["text"]
            renamed = relabel_independently(original, i, rep)
            if digest_text(renamed) != probe["text_sha256"]:
                raise AssertionError("probe text mismatch")
            if not np.array_equal(copy_aware_features(original), copy_aware_features(renamed)):
                raise AssertionError("probe feature invariance failed")
            token_ids = [ckpt["vocab"].get(ch, ckpt["vocab"]["<unk>"]) for ch in renamed]
            x = torch.tensor([token_ids], dtype=torch.long)
            f = torch.tensor(copy_aware_features(renamed)[None], dtype=torch.float32)
            logits, _, _ = model(x, f)
            direct = torch.sigmoid(logits).numpy()[0]
            error = float(np.max(np.abs(direct - probe["probabilities"])))
            max_error = max(max_error, error)
            if error > 1e-6:
                raise AssertionError(f"probe model replay mismatch: {i}/{rep} {error}")
    return {
        "experiment": "EXP-0031",
        "passed": True,
        "results_sha256": digest(result_path),
        "auditor_sha256": digest(Path(__file__)),
        "n_rows_checked": 300,
        "n_probes_replayed": len(probes),
        "max_probe_probability_abs_error": max_error,
        "copy_flip_bootstrap_95": ci.astype(float).tolist(),
        "material_symmetry_violation": material,
        "group_averaging_benefit": benefit,
    }


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    output = args.root / "results/EXP-0031/audit.json"
    if output.exists():
        raise FileExistsError(f"{output} already exists; refusing to overwrite")
    report = audit(args.root)
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
