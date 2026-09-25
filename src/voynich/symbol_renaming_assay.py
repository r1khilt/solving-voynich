"""EXP-0031: frozen-model symbol-renaming intervention on EXP-0030 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch

from voynich.decoder_programs import apply_mask_ops
from voynich.latent_recovery import (
    BATCH_SIZE,
    CIPHER_POOL,
    TinySignalModel,
    copy_aware_features,
    encode,
    recon_accuracy,
)
from voynich.runtime import write_json


EXPERIMENT_ID = "EXP-0031"
PERMUTATION_SEED = 310031
BOOTSTRAP_SEED = 310131
N_RELABEL = 31
BOOTSTRAP_DRAWS = 2000
MAX_WALL_SECONDS = 300
POOL = tuple(CIPHER_POOL)
CHECKPOINT_SHA256 = "b1fe842a1320a65b5923446910efec2d273ef62dac97bff44a7f6a8825bebda2"
HOLDOUT_SHA256 = "138f158484ff5e3859fad113f96910d61be224dc44d6b9ab16084d0edce014d7"
COPY_RANDOM_RECON = 0.2074623649767012
BASE_MIXED_RECON = 0.18838171635900422
BASE_COPY_RECON = 0.16651924193353831
PROBE_INDICES = (0, 74, 149, 224, 299)
PROBE_REPS = (0, 1, 17, 31)
PROGRAM = ("exact_count_neural",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relabel_text(text: str, row_index: int, replicate: int) -> str:
    if replicate == 0:
        return text
    if replicate < 0 or replicate > N_RELABEL:
        raise ValueError("replicate outside frozen range")
    rng = np.random.default_rng(PERMUTATION_SEED + 1000 * row_index + replicate)
    shuffled = rng.permutation(len(POOL))
    mapping = {ch: POOL[int(shuffled[i])] for i, ch in enumerate(POOL)}
    if any(ch != " " and ch not in mapping for ch in text):
        raise ValueError("text contains non-cipher non-space character")
    return "".join(mapping.get(ch, ch) for ch in text)


@torch.no_grad()
def forward_details(
    model: TinySignalModel,
    vocab: dict[str, int],
    texts: list[str],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.tensor([encode(s, vocab) for s in texts], dtype=torch.long)
    feat = torch.tensor(features, dtype=torch.float32)
    emb = model.emb(x)
    h, _ = model.lstm(emb + model.feat_proj(feat))
    probs = model.mask_head(h).squeeze(-1).sigmoid()
    return emb.numpy(), h.numpy(), probs.numpy()


def infer_all(
    model: TinySignalModel,
    vocab: dict[str, int],
    texts: list[str],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb, hidden, probs = [], [], []
    for start in range(0, len(texts), BATCH_SIZE):
        e, h, p = forward_details(
            model,
            vocab,
            texts[start : start + BATCH_SIZE],
            features[start : start + BATCH_SIZE],
        )
        emb.append(e)
        hidden.append(h)
        probs.append(p)
    return np.concatenate(emb), np.concatenate(hidden), np.concatenate(probs)


def mask_metrics(rows: list[dict], masks: list[np.ndarray]) -> dict:
    recons = []
    null_tp = null_fp = null_fn = pred_nulls = 0
    for row, mask in zip(rows, masks, strict=True):
        true = np.asarray(row["mask"], dtype=int)
        pred = np.asarray(mask, dtype=int)
        recons.append(recon_accuracy(row["ciphered"], row["text"], pred))
        null_tp += int(((true == 0) & (pred == 0)).sum())
        null_fp += int(((true == 1) & (pred == 0)).sum())
        null_fn += int(((true == 0) & (pred == 1)).sum())
        pred_nulls += int((pred == 0).sum())
    return {
        "n": len(rows),
        "recon_acc": float(np.mean(recons)),
        "null_precision": null_tp / max(null_tp + null_fp, 1),
        "null_recall": null_tp / max(null_tp + null_fn, 1),
        "pred_null_rate": pred_nulls / max(sum(len(r["text"]) for r in rows), 1),
        "per_row_recon": recons,
    }


def null_counts(row: dict, mask: np.ndarray) -> dict[str, int]:
    true = np.asarray(row["mask"], dtype=int)
    pred = np.asarray(mask, dtype=int)
    return {
        "null_tp": int(((true == 0) & (pred == 0)).sum()),
        "null_fp": int(((true == 1) & (pred == 0)).sum()),
        "null_fn": int(((true == 0) & (pred == 1)).sum()),
        "pred_nulls": int((pred == 0).sum()),
    }


def packed_mask(mask: np.ndarray) -> str:
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="big").tobytes().hex()


def summarize_family(rows: list[dict], summaries: list[dict], family: str) -> dict:
    indices = [i for i, row in enumerate(rows) if row["filler_family"] == family]
    subset = [summaries[i] for i in indices]
    return {
        "n": len(indices),
        "mean_probability_abs_drift": float(np.mean([r["mean_probability_abs_drift"] for r in subset])),
        "mean_keep_mask_flip_fraction": float(np.mean([r["mean_keep_mask_flip_fraction"] for r in subset])),
        "mean_embedding_relative_l2": float(np.mean([r["mean_embedding_relative_l2"] for r in subset])),
        "mean_hidden_relative_l2": float(np.mean([r["mean_hidden_relative_l2"] for r in subset])),
    }


def run(root: Path) -> dict:
    torch.set_num_threads(6)
    ckpt_path = root / "outputs" / "EXP-0030" / "model.pt"
    holdout_path = root / "data" / "processed" / "exp0030" / "finnish_holdout.jsonl"
    audit_path = root / "results" / "EXP-0030" / "source_oracle_audit.json"
    parent_decision_path = root / "results" / "EXP-0030" / "decision.json"
    if sha256(ckpt_path) != CHECKPOINT_SHA256 or sha256(holdout_path) != HOLDOUT_SHA256:
        raise ValueError("frozen checkpoint or holdout digest mismatch")
    parent_audit = json.loads(audit_path.read_text())
    if not parent_audit["passed"]:
        raise ValueError("parent true-source audit did not pass")
    parent_decision = json.loads(parent_decision_path.read_text())
    if parent_decision["passed"]:
        raise ValueError("EXP-0030 parent decision unexpectedly passed")
    rows = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    if len(rows) != 300 or {r["world"] for r in rows} != {2}:
        raise ValueError("holdout population drift")
    if {family: sum(r["filler_family"] == family for r in rows) for family in
        ("random_char", "periodic", "copy_mutate")} != {
            "random_char": 94, "periodic": 107, "copy_mutate": 99
        }:
        raise ValueError("holdout filler-family counts drift")
    if len(POOL) != 90 or len(set(POOL)) != 90 or any(c.isspace() for c in POOL):
        raise ValueError("cipher pool drift")
    if any(any(ch not in CIPHER_POOL + " " for ch in row["text"]) for row in rows):
        raise ValueError("holdout contains non-cipher character")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    model = TinySignalModel(len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()
    texts = [row["text"] for row in rows]
    base_features = np.stack([copy_aware_features(s) for s in texts])
    base_emb, base_hidden, base_prob = infer_all(model, vocab, texts, base_features)
    _, _, identity_prob = infer_all(model, vocab, texts, base_features)
    identity_max_abs = float(np.max(np.abs(identity_prob - base_prob)))
    base_masks = [apply_mask_ops(texts[i], PROGRAM, base_prob[i]) for i in range(len(rows))]
    orig_mixed = mask_metrics(rows, base_masks)
    copy_ids = [i for i, row in enumerate(rows) if row["filler_family"] == "copy_mutate"]
    orig_copy = mask_metrics([rows[i] for i in copy_ids], [base_masks[i] for i in copy_ids])
    if abs(orig_mixed["recon_acc"] - BASE_MIXED_RECON) > 1e-10:
        raise ValueError("baseline mixed score does not reproduce EXP-0030")
    if abs(orig_copy["recon_acc"] - BASE_COPY_RECON) > 1e-10:
        raise ValueError("baseline copy score does not reproduce EXP-0030")

    n = len(rows)
    prob_sum = base_prob.astype(np.float64).copy()
    drift = np.zeros((n, 4), dtype=np.float64)
    probes = []
    for i in PROBE_INDICES:
        probes.append({
            "row_index": i,
            "replicate": 0,
            "text_sha256": hashlib.sha256(texts[i].encode()).hexdigest(),
            "probabilities": base_prob[i].astype(float).tolist(),
        })
    tasks = [(i, r, relabel_text(texts[i], i, r)) for i in range(n) for r in range(1, N_RELABEL + 1)]
    for start in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[start : start + BATCH_SIZE]
        transformed = [item[2] for item in batch]
        features = np.stack([copy_aware_features(s) for s in transformed])
        for j, (i, _, _) in enumerate(batch):
            if not np.array_equal(features[j], base_features[i]):
                raise AssertionError("copy-aware features changed under symbol bijection")
        emb, hidden, prob = forward_details(model, vocab, transformed, features)
        for j, (i, r, renamed) in enumerate(batch):
            p = prob[j]
            prob_sum[i] += p
            renamed_mask = apply_mask_ops(renamed, PROGRAM, p)
            drift[i, 0] += float(np.mean(np.abs(p - base_prob[i])))
            drift[i, 1] += float(np.mean(renamed_mask != base_masks[i]))
            drift[i, 2] += float(np.linalg.norm(emb[j] - base_emb[i]) / max(np.linalg.norm(base_emb[i]), 1e-12))
            drift[i, 3] += float(np.linalg.norm(hidden[j] - base_hidden[i]) / max(np.linalg.norm(base_hidden[i]), 1e-12))
            if i in PROBE_INDICES and r in PROBE_REPS:
                probes.append({
                    "row_index": i,
                    "replicate": r,
                    "text_sha256": hashlib.sha256(renamed.encode()).hexdigest(),
                    "probabilities": p.astype(float).tolist(),
                })
    drift /= N_RELABEL
    ensemble_prob = prob_sum / (N_RELABEL + 1)
    ensemble_masks = [apply_mask_ops(texts[i], PROGRAM, ensemble_prob[i]) for i in range(n)]
    ensemble_mixed = mask_metrics(rows, ensemble_masks)
    ensemble_copy = mask_metrics([rows[i] for i in copy_ids], [ensemble_masks[i] for i in copy_ids])

    summaries = []
    for i, row in enumerate(rows):
        summaries.append({
            "row_index": i,
            "family": row["filler_family"],
            "mean_probability_abs_drift": float(drift[i, 0]),
            "mean_keep_mask_flip_fraction": float(drift[i, 1]),
            "mean_embedding_relative_l2": float(drift[i, 2]),
            "mean_hidden_relative_l2": float(drift[i, 3]),
            "original_recon": float(orig_mixed["per_row_recon"][i]),
            "ensemble_recon": float(ensemble_mixed["per_row_recon"][i]),
            "original_mask_hex": packed_mask(base_masks[i]),
            "ensemble_mask_hex": packed_mask(ensemble_masks[i]),
            "original_null_counts": null_counts(row, base_masks[i]),
            "ensemble_null_counts": null_counts(row, ensemble_masks[i]),
        })
    families = {}
    for family in ("random_char", "periodic", "copy_mutate"):
        indices = [i for i, row in enumerate(rows) if row["filler_family"] == family]
        families[family] = summarize_family(rows, summaries, family)
        families[family]["original"] = {
            k: v for k, v in mask_metrics([rows[i] for i in indices], [base_masks[i] for i in indices]).items()
            if k != "per_row_recon"
        }
        families[family]["ensemble"] = {
            k: v for k, v in mask_metrics([rows[i] for i in indices], [ensemble_masks[i] for i in indices]).items()
            if k != "per_row_recon"
        }
    copy_flips = np.asarray([summaries[i]["mean_keep_mask_flip_fraction"] for i in copy_ids])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.mean(copy_flips[rng.integers(0, len(copy_flips), size=(BOOTSTRAP_DRAWS, len(copy_flips)))], axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).astype(float).tolist()
    valid = identity_max_abs <= 1e-6
    material = valid and families["copy_mutate"]["mean_keep_mask_flip_fraction"] > 0.05 and ci[0] > 0.02
    benefit = valid and ensemble_copy["recon_acc"] > orig_copy["recon_acc"] + 0.02 and ensemble_copy["recon_acc"] > COPY_RANDOM_RECON
    probes.sort(key=lambda item: (item["row_index"], item["replicate"]))
    return {
        "experiment": EXPERIMENT_ID,
        "status": "exploratory_exposed_holdout",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "n_rows": n,
        "n_relabel_per_row": N_RELABEL,
        "permutation_seed": PERMUTATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "parent_oracle_audit_sha256": sha256(audit_path),
        "parent_decision_sha256": sha256(parent_decision_path),
        "runner_sha256": sha256(Path(__file__)),
        "identity_max_probability_abs_diff": identity_max_abs,
        "all_copy_features_invariant": True,
        "valid": valid,
        "copy_flip_bootstrap_95": ci,
        "material_symmetry_violation": material,
        "group_averaging_benefit": benefit,
        "thresholds": {
            "material_copy_flip_mean_gt": 0.05,
            "material_copy_flip_ci_lower_gt": 0.02,
            "group_averaging_copy_recon_gain_gt": 0.02,
            "group_averaging_copy_recon_gt": COPY_RANDOM_RECON,
        },
        "mixed_original": {k: v for k, v in orig_mixed.items() if k != "per_row_recon"},
        "mixed_ensemble": {k: v for k, v in ensemble_mixed.items() if k != "per_row_recon"},
        "families": families,
        "rows": summaries,
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    def wall_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"{EXPERIMENT_ID} exceeded {MAX_WALL_SECONDS} seconds")

    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(MAX_WALL_SECONDS)
    started = time.monotonic()
    output = args.root / "results" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"{output} already exists; refusing to overwrite")
    result = run(args.root)
    result["elapsed_seconds"] = time.monotonic() - started
    output.mkdir(parents=True)
    write_json(output / "results.json", result)
    print(json.dumps({
        "experiment": EXPERIMENT_ID,
        "valid": result["valid"],
        "copy_flip": result["families"]["copy_mutate"]["mean_keep_mask_flip_fraction"],
        "copy_flip_ci": result["copy_flip_bootstrap_95"],
        "material_symmetry_violation": result["material_symmetry_violation"],
        "copy_ensemble_recon": result["families"]["copy_mutate"]["ensemble"]["recon_acc"],
        "group_averaging_benefit": result["group_averaging_benefit"],
    }, indent=2))


if __name__ == "__main__":
    main()
