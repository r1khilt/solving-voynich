"""Separate mask-arithmetic, control, bootstrap and model-probe audit for EXP-0035."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
import torch

from voynich.exp0032_model import ContextMaskModel, arrays, canonical_token_ids
from voynich.latent_recovery import CIPHER_POOL, copy_aware_features
from voynich.runtime import write_json


SEEDS = (320101, 320202)
CONDITIONS = ("hard", "uniform", "weighted")
FAMILIES = ("copy_mutate", "periodic", "random_char")
RANDOM_SEEDS = tuple(range(350400, 350420))
BOOTSTRAP_SEED = 350350
BOOTSTRAP_DRAWS = 2000
PROBES = (0, 74, 149, 224, 299, 449, 599)
MAX_SECONDS = 300


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def unpack(mask_hex: str) -> np.ndarray:
    raw = bytes.fromhex(mask_hex)
    if len(raw) != 16:
        raise AssertionError("packed mask byte count invalid")
    mask = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    if len(mask) != 128 or int(mask.sum()) != 90:
        raise AssertionError("packed mask keep count invalid")
    return mask


def row_reconstruction(row: dict, mask: np.ndarray) -> float:
    predicted = "".join(row["text"][i] for i in range(128) if mask[i])
    gold = row["ciphered"]
    common = min(len(predicted), len(gold))
    if common == 0:
        return float(predicted == gold)
    correct = sum(predicted[i] == gold[i] for i in range(common))
    return (correct / common) * (common / max(len(predicted), len(gold)))


def family_score(rows: list[dict], masks: np.ndarray) -> dict:
    tp = fp = fn = exact = 0
    recon = []
    for row, mask in zip(rows, masks, strict=True):
        gold = np.asarray(row["mask"], dtype=np.uint8)
        tp += int(np.count_nonzero((gold == 0) & (mask == 0)))
        fp += int(np.count_nonzero((gold == 1) & (mask == 0)))
        fn += int(np.count_nonzero((gold == 0) & (mask == 1)))
        exact += int(np.array_equal(gold, mask))
        recon.append(row_reconstruction(row, mask))
    return {"n": len(rows), "null_precision": tp / max(tp + fp, 1),
            "null_recall": tp / max(tp + fn, 1), "null_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "recon_acc": float(np.mean(recon)), "exact_mask_rate": exact / len(rows)}


def compare_scores(actual: dict, expected: dict) -> None:
    if actual["n"] != expected["n"]:
        raise AssertionError("family size mismatch")
    for metric in ("null_precision", "null_recall", "null_f1", "recon_acc", "exact_mask_rate"):
        near(actual[metric], expected[metric])


def random_scores(rows: list[dict]) -> dict:
    reports = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        masks = []
        for _ in rows:
            random_values = rng.random(128)
            selected = np.argsort(random_values, kind="stable")[-90:]
            mask = np.zeros(128, dtype=np.uint8)
            mask[selected] = 1
            masks.append(mask)
        reports.append({family: family_score([row for row in rows if row["filler_family"] == family],
                                             np.asarray([mask for row, mask in zip(rows, masks, strict=True) if row["filler_family"] == family]))
                        for family in {row["filler_family"] for row in rows}})
    return {family: {metric: float(np.mean([report[family][metric] for report in reports]))
                     for metric in ("null_precision", "null_recall", "null_f1", "recon_acc", "exact_mask_rate")}
            for family in reports[0]}


def micro_f1(true: np.ndarray, pred: np.ndarray, indices: np.ndarray) -> float:
    actual, selected = true[indices], pred[indices]
    tp = int(np.count_nonzero((actual == 0) & (selected == 0)))
    fp = int(np.count_nonzero((actual == 1) & (selected == 0)))
    fn = int(np.count_nonzero((actual == 0) & (selected == 1)))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def paired(rows: list[dict], masks: dict[str, np.ndarray], left: str, right: str) -> tuple[float, list[float]]:
    copy_ids = np.asarray([i for i, row in enumerate(rows) if row["filler_family"] == "copy_mutate"], dtype=int)
    gold = np.asarray([rows[i]["mask"] for i in copy_ids], dtype=np.uint8)
    a = [masks[f"{left}_{seed}"][copy_ids] for seed in SEEDS]
    b = [masks[f"{right}_{seed}"][copy_ids] for seed in SEEDS]

    def gain(indices: np.ndarray) -> float:
        return float(np.mean([micro_f1(gold, x, indices) - micro_f1(gold, y, indices) for x, y in zip(a, b, strict=True)]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(copy_ids), size=(BOOTSTRAP_DRAWS, len(copy_ids)))
    return gain(np.arange(len(copy_ids))), np.quantile([gain(draw) for draw in draws], [0.025, 0.975]).tolist()


def direct_probe(model: ContextMaskModel, row: dict) -> np.ndarray:
    data = arrays([row], "canonical", "cpu")
    with torch.no_grad():
        logits, _ = model(data["x"], data["features"])
        return torch.sigmoid(logits)[0].detach().numpy()


def check_model(root: Path, name: str, report: dict, groups: dict[str, list[dict]]) -> float:
    condition, seed_text = name.rsplit("_", 1)
    seed = int(seed_text)
    if condition == "hard":
        report_path = root / f"results/EXP-0032/train_canonical_{seed}.json"
        ckpt_path = root / f"outputs/EXP-0032/canonical_{seed}.pt"
    else:
        report_path = root / f"results/EXP-0035/train_{condition}_{seed}.json"
        ckpt_path = root / f"outputs/EXP-0035/{condition}_{seed}.pt"
    train = json.loads(report_path.read_text())
    if train["checkpoint_sha256"] != digest(ckpt_path) or report["checkpoint_sha256"] != digest(ckpt_path):
        raise AssertionError("checkpoint hash mismatch")
    if report["selected_step"] != train["best_step"] or report["validation_copy_null_f1"] != train["best_validation_copy_null_f1"]:
        raise AssertionError("validation freeze mismatch")
    if condition != "hard":
        scores = [row["validation_copy_null_f1"] for row in train["history"]]
        selected = next(row for row in train["history"] if row["validation_copy_null_f1"] == max(scores))
        if selected["step"] != train["best_step"]:
            raise AssertionError("soft checkpoint not first validation maximum")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = ContextMaskModel().cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    maximum_error = 0.0
    for group_name, rows in groups.items():
        probes = report[group_name]["probability_probes"]
        if [entry["row_index"] for entry in probes] != list(PROBES):
            raise AssertionError("probability probe indices drift")
        for entry in probes:
            direct = direct_probe(model, rows[entry["row_index"]])
            error = float(np.max(np.abs(direct - np.asarray(entry["probabilities"]))))
            maximum_error = max(maximum_error, error)
            if error > 1e-6:
                raise AssertionError(f"direct probability probe mismatch: {name}/{group_name}")
    sample = groups["portuguese_holdout"][:16]
    pool = list(CIPHER_POOL)
    permutation = np.random.default_rng(350300).permutation(len(pool))
    mapping = {ch: pool[int(permutation[i])] for i, ch in enumerate(pool)}
    renamed = [{**row, "text": "".join(mapping.get(ch, ch) for ch in row["text"])} for row in sample]
    for original, changed in zip(sample, renamed, strict=True):
        if canonical_token_ids(original["text"]) != canonical_token_ids(changed["text"]) or not np.array_equal(copy_aware_features(original["text"]), copy_aware_features(changed["text"])):
            raise AssertionError("rename-invariant inputs changed")
    with torch.no_grad():
        first = torch.sigmoid(model(*[arrays(sample, "canonical", "cpu")[k] for k in ("x", "features")])[0]).numpy()
        second = torch.sigmoid(model(*[arrays(renamed, "canonical", "cpu")[k] for k in ("x", "features")])[0]).numpy()
    invariance_error = float(np.max(np.abs(first - second)))
    near(invariance_error, report["invariance_max_probability_error"], 1e-6)
    if invariance_error > 1e-6:
        raise AssertionError("direct invariance control failed")
    return maximum_error


def audit(root: Path) -> dict:
    result_path = root / "results/EXP-0035/evaluation.json"
    result = json.loads(result_path.read_text())
    if result["experiment"] != "EXP-0035" or result["evaluator_sha256"] != digest(root / "scripts/evaluate_exp0035.py"):
        raise AssertionError("experiment/evaluator identity drift")
    if result["data_manifest_sha256"] != digest(root / "data/manifests/exp0035_data.json"):
        raise AssertionError("data source drift")
    groups = {name: [json.loads(line) for line in (root / f"data/processed/exp0035/{name}.jsonl").read_text().splitlines()]
              for name in ("portuguese_holdout", "iid_control")}
    if len(groups["portuguese_holdout"]) != 600 or len(groups["iid_control"]) != 600:
        raise AssertionError("evaluation population drift")
    if tuple(result["random_seeds"]) != RANDOM_SEEDS or result["bootstrap_seed"] != BOOTSTRAP_SEED or result["bootstrap_draws"] != BOOTSTRAP_DRAWS:
        raise AssertionError("frozen control seed drift")
    for group_name, rows in groups.items():
        expected = random_scores(rows)
        for family, score in expected.items():
            for metric, value in score.items():
                near(result["baselines"][group_name][family][metric], value)
    predictions = {}
    max_probe_error = 0.0
    for condition in CONDITIONS:
        for seed in SEEDS:
            name = f"{condition}_{seed}"
            report = result["models"][name]
            max_probe_error = max(max_probe_error, check_model(root, name, report, groups))
            for group_name, rows in groups.items():
                masks = np.asarray([unpack(value) for value in report[group_name]["packed_masks"]])
                if masks.shape != (600, 128):
                    raise AssertionError("packed model mask count drift")
                if group_name == "portuguese_holdout":
                    predictions[name] = masks
                for family in {row["filler_family"] for row in rows}:
                    part = [r for r in rows if r["filler_family"] == family]
                    selected = masks[[r["filler_family"] == family for r in rows]]
                    compare_scores(report[group_name]["by_family"][family], family_score(part, selected))
    for label, left, right in (("weighted_minus_hard", "weighted", "hard"),
                               ("uniform_minus_hard", "uniform", "hard"),
                               ("weighted_minus_uniform", "weighted", "uniform")):
        gain, ci = paired(groups["portuguese_holdout"], predictions, left, right)
        near(result[f"paired_copy_f1_{label}"], gain)
        for observed, expected in zip(result[f"paired_copy_f1_{label}_ci95"], ci, strict=True):
            near(observed, expected)
    periodic = all(result["models"][f"{condition}_{seed}"]["portuguese_holdout"]["by_family"]["periodic"]["null_f1"] >= 0.70
                   for condition in CONDITIONS for seed in SEEDS)
    iid_random = result["baselines"]["iid_control"]["iid_indistinguishable"]["null_f1"]
    iid = all(result["models"][f"{condition}_{seed}"]["iid_control"]["by_family"]["iid_indistinguishable"]["null_f1"] <= iid_random + 0.05
              for condition in CONDITIONS for seed in SEEDS)
    invariant = all(result["models"][f"{condition}_{seed}"]["invariance_max_probability_error"] <= 1e-6
                    for condition in CONDITIONS for seed in SEEDS)
    recon = {condition: float(np.mean([result["models"][f"{condition}_{seed}"]["portuguese_holdout"]["by_family"]["copy_mutate"]["recon_acc"] for seed in SEEDS]))
             for condition in CONDITIONS}
    for condition in CONDITIONS:
        near(recon[condition], result["copy_mean_reconstruction"][condition])
    success = periodic and iid and invariant and result["paired_copy_f1_weighted_minus_hard"] >= 0.03 and result["paired_copy_f1_weighted_minus_hard_ci95"][0] > 0.01 and recon["weighted"] - recon["hard"] >= 0.03
    decision = "INCONCLUSIVE_OR_INVALID" if not (periodic and iid and invariant) else ("SUPPORTED" if success else "NOT_SUPPORTED")
    if result["periodic_competence"] != periodic or result["iid_no_leak"] != iid or result["rename_invariance"] != invariant or result["registered_success"] != success or result["decision"] != decision:
        raise AssertionError("registered decision/control mismatch")
    return {"experiment": "EXP-0035", "passed": True, "evaluation_sha256": digest(result_path),
            "auditor_sha256": digest(Path(__file__)), "rows_replayed": sum(map(len, groups.values())),
            "packed_masks_rescored": len(CONDITIONS) * len(SEEDS) * 1200,
            "direct_probability_probes": len(CONDITIONS) * len(SEEDS) * len(groups) * len(PROBES),
            "max_direct_probability_error": max_probe_error, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0035/evaluation_audit.json"
    if out.exists():
        raise FileExistsError("EXP-0035 evaluation audit already exists")

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError("EXP-0035 evaluation audit exceeded 300 seconds")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    try:
        report = audit(args.root)
    finally:
        signal.alarm(0)
    write_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
