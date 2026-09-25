"""Independently verify and freeze EXP-0035 validation-selected weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from voynich.exact_count_decode import exact_count_keep_masks
from voynich.exp0032_model import ContextMaskModel, arrays
from voynich.runtime import write_json


SEEDS = (320101, 320202)
VARIANTS = ("uniform", "weighted")
MANIFEST_SHA = "ca8e3e6c7dfab5df74f750b6434b9905629f2f629a9c886e8f780935e406ec44"
BASELINE_FREEZE_SHA = "69a8000b6f7310a9a330fbf3fa89d614fd71004794c928e6a874a8668ea07fbb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_score(rows: list[dict], checkpoint: dict) -> float:
    torch.set_num_threads(6)
    model = ContextMaskModel().cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = arrays(rows, "canonical", "cpu")
    probs = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            logits, _ = model(data["x"][start:start + 32], data["features"][start:start + 32])
            probs.extend(torch.sigmoid(logits).numpy())
    masks = np.asarray(exact_count_keep_masks([row for row in probs]), dtype=np.uint8)
    gold = np.asarray([row["mask"] for row in rows], dtype=np.uint8)
    tp = int(np.count_nonzero((gold == 0) & (masks == 0)))
    fp = int(np.count_nonzero((gold == 1) & (masks == 0)))
    fn = int(np.count_nonzero((gold == 0) & (masks == 1)))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def verify(root: Path) -> dict:
    manifest_path = root / "data/manifests/exp0035_data.json"
    if digest(manifest_path) != MANIFEST_SHA:
        raise AssertionError("data manifest drift")
    data_audit_path = root / "results/EXP-0035/data_audit.json"
    data_audit = json.loads(data_audit_path.read_text())
    if not data_audit["passed"] or data_audit["manifest_sha256"] != MANIFEST_SHA:
        raise AssertionError("data audit gate invalid")
    if digest(root / "results/EXP-0032/training_freeze.json") != BASELINE_FREEZE_SHA:
        raise AssertionError("hard baseline freeze drift")
    manifest = json.loads(manifest_path.read_text())
    val_path = root / "data/processed/exp0032/validation.jsonl"
    if digest(val_path) != manifest["prior_validation_sha256"]:
        raise AssertionError("original validation data drift")
    rows = [json.loads(line) for line in val_path.read_text().splitlines()]
    copy_rows = [row for row in rows if row["world"] == 2 and row["filler_family"] == "copy_mutate"]
    if len(copy_rows) != 221:
        raise AssertionError("validation copy population drift")
    runner_hash = digest(root / "src/voynich/exp0035_model.py")
    entries = {}
    elapsed = 0.0
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"{variant}_{seed}"
            path = root / f"results/EXP-0035/train_{name}.json"
            checkpoint_path = root / f"outputs/EXP-0035/{name}.pt"
            report = json.loads(path.read_text())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if report["experiment"] != "EXP-0035" or report["variant"] != variant or report["seed"] != seed:
                raise AssertionError(f"training report identity drift: {name}")
            if checkpoint["experiment"] != "EXP-0035" or checkpoint["variant"] != variant or checkpoint["seed"] != seed:
                raise AssertionError(f"training checkpoint identity drift: {name}")
            if report["runner_sha256"] != runner_hash or checkpoint["runner_sha256"] != runner_hash:
                raise AssertionError(f"runner hash drift: {name}")
            if report["data_manifest_sha256"] != MANIFEST_SHA or checkpoint["data_manifest_sha256"] != MANIFEST_SHA:
                raise AssertionError(f"manifest hash drift: {name}")
            if report["target_sha256"] != manifest["derived_sha256"]["train_soft_targets"]:
                raise AssertionError(f"target hash drift: {name}")
            if report["checkpoint_sha256"] != digest(checkpoint_path):
                raise AssertionError(f"checkpoint byte hash drift: {name}")
            if report["parameter_count"] != 10_746_629 or report["steps"] != 3000 or report["validation_copy_n"] != len(copy_rows):
                raise AssertionError(f"training schedule/population drift: {name}")
            if not 0 < report["elapsed_seconds"] <= 600:
                raise AssertionError(f"training cap invalid: {name}")
            history = report["history"]
            if [entry["step"] for entry in history] != list(range(250, 3001, 250)):
                raise AssertionError(f"validation history steps invalid: {name}")
            chosen = next(entry for entry in history if entry["validation_copy_null_f1"] == max(h["validation_copy_null_f1"] for h in history))
            if chosen["step"] != report["best_step"] or chosen["step"] != checkpoint["best_step"]:
                raise AssertionError(f"checkpoint not first validation maximum: {name}")
            if abs(chosen["validation_copy_null_f1"] - report["best_validation_copy_null_f1"]) > 1e-12:
                raise AssertionError(f"reported best F1 drift: {name}")
            direct = replay_score(copy_rows, checkpoint)
            if abs(direct - chosen["validation_copy_null_f1"]) > 1e-8:
                raise AssertionError(f"CPU direct selected validation score drift: {name}: {direct} vs {chosen['validation_copy_null_f1']}")
            entries[name] = {"report_sha256": digest(path), "checkpoint_sha256": digest(checkpoint_path),
                             "best_step": chosen["step"], "validation_copy_null_f1": direct,
                             "elapsed_seconds": report["elapsed_seconds"], "parameter_count": report["parameter_count"]}
            elapsed += report["elapsed_seconds"]
    return {"experiment": "EXP-0035", "passed": True, "data_manifest_sha256": MANIFEST_SHA,
            "data_audit_sha256": digest(data_audit_path), "baseline_freeze_sha256": BASELINE_FREEZE_SHA,
            "auditor_sha256": digest(Path(__file__)), "validation_copy_rows": len(copy_rows),
            "total_training_elapsed_seconds": elapsed, "new_models": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0035/training_freeze.json"
    if out.exists():
        raise FileExistsError("EXP-0035 training freeze already exists")
    result = verify(args.root)
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
