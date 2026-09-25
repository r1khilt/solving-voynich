"""EXP-0042: form/history/layout residual prediction beyond previous-last glyph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np

from .boundary_cross_transcription import (
    GC_SHA,
    SPLIT_SHA,
    ZL_TRAIN_SHA,
    ZL_VALID_SHA,
    fit,
    load_gc,
    load_zl,
    sha,
)
from .readable_edge_calibration import source_window, wrap


EXPERIMENT = "EXP-0042"
EXP39_SHA = "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329"
EXP41_SHA = "7a1f1e66b35561ade184b52a91fc2e346331605098998bc646f1d353b46159cd"
EXP41_MODULE_SHA = "1aa0701f484e7fabbde47a8b1eaf627a86c7675e59d009d056b72561e601f11c"
FAMILIES = ("form", "history", "layout", "all")
KAPPAS = (5, 20, 80)
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
LOG2 = math.log(2)


@dataclass(frozen=True)
class Row:
    leaf: str
    context: tuple[str, str]
    role: str
    previous: tuple[str, ...]
    before: tuple[str, ...] | None
    index: int
    run_length: int
    target: str


def rows(groups: list[dict]) -> list[Row]:
    output = []
    for group in groups:
        words = group["words"]
        for index in range(len(words) - 1):
            role = "first" if index == 0 else "last" if index == len(words) - 2 else "middle"
            output.append(Row(group["leaf"], tuple(group["context"]), role,
                              tuple(words[index]), tuple(words[index - 1]) if index else None,
                              index, len(words), words[index + 1][0]))
    return output


def keys(row: Row, family: str) -> tuple[tuple, ...]:
    previous = row.previous
    before = row.before
    edge = (("edge_last", previous[-1]),
            ("edge_last_role", (previous[-1], row.role)),
            ("edge_last_context", (previous[-1], row.context)))
    form = (("first", previous[0]), ("length", min(len(previous), 8)),
            ("prefix2", previous[:2]), ("prefix3", previous[:3]),
            ("suffix2", previous[-2:]), ("suffix3", previous[-3:]),
            ("whole", previous))
    history = (("before_first", before[0] if before else "<start>"),
               ("before_last", before[-1] if before else "<start>"),
               ("before_suffix2", before[-2:] if before else ("<start>",)))
    layout = (("position_quartile", min(3, 4 * row.index // max(row.run_length - 1, 1))),
              ("run_length", min(row.run_length, 12)))
    if family == "edge":
        return edge
    if family == "form":
        return edge + form
    if family == "history":
        return edge + history
    if family == "layout":
        return edge + layout
    if family == "all":
        return edge + form + history + layout
    raise ValueError(f"Unknown feature family {family}")


def _target(raw: str, alphabet: set[str]) -> str:
    return raw if raw in alphabet else "<unk>"


def fit_counts(groups: list[dict]) -> dict:
    baseline = fit(groups)
    classes = sorted(baseline["alphabet"]) + ["<unk>"]
    index = {symbol: i for i, symbol in enumerate(classes)}
    observed = rows(groups)
    q_counts = np.ones(len(classes), dtype=np.float64)
    feature_counts: dict[tuple, np.ndarray] = {}
    for row in observed:
        target_index = index[row.target]
        q_counts[target_index] += 1
        for feature in keys(row, "all"):
            counts = feature_counts.setdefault(feature, np.zeros(len(classes), dtype=np.float64))
            counts[target_index] += 1
    q = q_counts / q_counts.sum()
    return {"baseline": baseline, "classes": classes, "index": index,
            "q": q, "feature_counts": feature_counts, "train_pairs": len(observed)}


def probability_matrices(observed: list[Row], model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = model["baseline"]
    classes = model["classes"]
    base_log = np.zeros((len(observed), len(classes)), dtype=np.float64)
    context_log = np.zeros_like(base_log)
    labels = np.zeros(len(observed), dtype=np.int64)
    for row_index, row in enumerate(observed):
        role = row.role
        context_key = (row.context, role)
        last_key = (row.context, role, "last", row.previous[-1])
        glob = np.array([(baseline["global_counts"][role][symbol] + 1) /
                         (baseline["global_n"][role] + baseline["n_symbols"])
                         for symbol in classes])
        context = np.array([(baseline["context_counts"][context_key][symbol] + 50 * p) /
                            (baseline["context_n"][context_key] + 50)
                            for symbol, p in zip(classes, glob)])
        last = np.array([(baseline["feature_counts"][last_key][symbol] + 20 * p) /
                         (baseline["feature_n"][last_key] + 20)
                         for symbol, p in zip(classes, context)])
        base_log[row_index] = np.log(last)
        context_log[row_index] = np.log(context)
        labels[row_index] = model["index"][_target(row.target, baseline["alphabet"])]
    return base_log, context_log, labels


def ratio_table(model: dict, kappa: int) -> dict[tuple, np.ndarray]:
    q = model["q"]
    log_q = np.log(q)
    return {key: np.log((count + kappa * q) / (count.sum() + kappa)) - log_q
            for key, count in model["feature_counts"].items()}


def feature_matrix(observed: list[Row], model: dict, family: str, kappa: int) -> np.ndarray:
    table = ratio_table(model, kappa)
    matrix = np.zeros((len(observed), len(model["classes"])), dtype=np.float64)
    for row_index, row in enumerate(observed):
        feature_keys = keys(row, family)
        for feature in feature_keys:
            contribution = table.get(feature)
            if contribution is not None:
                matrix[row_index] += contribution
        matrix[row_index] /= len(feature_keys)
    return matrix


def residual_bits(base_log: np.ndarray, labels: np.ndarray,
                  extra: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0:
        return np.zeros(len(labels), dtype=np.float64)
    logits = base_log + alpha * extra
    peak = logits.max(axis=1)
    normalizer = peak + np.log(np.exp(logits - peak[:, None]).sum(axis=1))
    selected = logits[np.arange(len(labels)), labels] - normalizer
    return (selected - base_log[np.arange(len(labels)), labels]) / LOG2


def internal_split(groups: list[dict]) -> tuple[list[dict], list[dict]]:
    fitting, development = [], []
    for group in groups:
        value = int.from_bytes(hashlib.sha256(group["leaf"].encode("utf-8")).digest()[:4], "big")
        (development if value % 5 == 0 else fitting).append(group)
    if not fitting or not development:
        raise ValueError("Empty internal leaf split")
    return fitting, development


def select(groups: list[dict], families: tuple[str, ...]) -> tuple[dict, list[dict]]:
    fitting, development = internal_split(groups)
    model = fit_counts(fitting)
    observed = rows(development)
    base_log, _, labels = probability_matrices(observed, model)
    grid = []
    best = {"family": families[0], "kappa": 80, "alpha": 0.0,
            "dev_gain_bits_per_pair": 0.0}
    for kappa in KAPPAS:
        for family in families:
            extra = feature_matrix(observed, model, family, kappa)
            for alpha in ALPHAS:
                gain = float(residual_bits(base_log, labels, extra, alpha).mean())
                candidate = {"family": family, "kappa": kappa, "alpha": alpha,
                             "dev_gain_bits_per_pair": gain}
                grid.append(candidate)
                if gain > best["dev_gain_bits_per_pair"] + 1e-12:
                    best = candidate
                elif abs(gain - best["dev_gain_bits_per_pair"]) <= 1e-12 and gain > 0:
                    old_order = (best["alpha"], -best["kappa"], families.index(best["family"]))
                    new_order = (alpha, -kappa, families.index(family))
                    if new_order < old_order:
                        best = candidate
    if best["dev_gain_bits_per_pair"] <= 0.005:
        best = {"family": families[0], "kappa": 80, "alpha": 0.0,
                "dev_gain_bits_per_pair": 0.0}
    return best, grid


def leaf_bootstrap(by_leaf: dict[str, dict], seed: int) -> list[float]:
    labels = sorted(by_leaf)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        sample = rng.choices(labels, k=len(labels))
        total = sum(by_leaf[leaf]["bits_sum"] for leaf in sample)
        pairs = sum(by_leaf[leaf]["pairs"] for leaf in sample)
        draws.append(total / pairs)
    draws.sort()
    return [draws[49], draws[1949]]


def evaluate_view(train: list[dict], valid: list[dict],
                  bootstrap_seed: int | None) -> dict:
    edge_choice, edge_grid = select(train, ("edge",))
    richer_choice, richer_grid = select(train, FAMILIES)
    model = fit_counts(train)
    observed = rows(valid)
    base_log, context_log, labels = probability_matrices(observed, model)
    edge_extra = feature_matrix(observed, model, "edge", edge_choice["kappa"])
    richer_extra = feature_matrix(observed, model, richer_choice["family"],
                                  richer_choice["kappa"])
    edge_delta = residual_bits(base_log, labels, edge_extra, edge_choice["alpha"])
    richer_delta = residual_bits(base_log, labels, richer_extra, richer_choice["alpha"])
    difference = richer_delta - edge_delta
    base_gain = (base_log[np.arange(len(labels)), labels] -
                 context_log[np.arange(len(labels)), labels]) / LOG2
    by_leaf = defaultdict(lambda: {"pairs": 0, "bits_sum": 0.0})
    for row, bits in zip(observed, difference):
        by_leaf[row.leaf]["pairs"] += 1
        by_leaf[row.leaf]["bits_sum"] += float(bits)
    output = {"edge_selection": edge_choice, "edge_development_grid": edge_grid,
              "richer_selection": richer_choice, "richer_development_grid": richer_grid,
              "train_groups": len(train), "train_pairs": model["train_pairs"],
              "validation_groups": len(valid), "validation_pairs": len(observed),
              "alphabet_size_train_plus_unk": len(model["classes"]),
              "baseline_last_gain_bits_per_pair": float(base_gain.mean()),
              "edge_over_p0_bits_per_pair": float(edge_delta.mean()),
              "richer_over_p0_bits_per_pair": float(richer_delta.mean()),
              "richer_minus_edge_bits_per_pair": float(difference.mean()),
              "by_leaf": dict(sorted(by_leaf.items()))}
    if bootstrap_seed is not None:
        output["physical_leaf_bootstrap95"] = leaf_bootstrap(output["by_leaf"], bootstrap_seed)
    return output


def synthetic(skeleton: list[dict], seed: int, persistence: float) -> list[dict]:
    state_rng = random.Random(seed)
    middle_rng = random.Random(seed + 42_000_000)
    output = []
    for group in skeleton:
        state = state_rng.choice(("x", "y"))
        new_words = []
        for index in range(len(group["words"])):
            if index:
                if persistence == 0:
                    state = state_rng.choice(("x", "y"))
                elif state_rng.random() > persistence:
                    state = "y" if state == "x" else "x"
            new_words.append((state, *(middle_rng.choice("abcdefgh") for _ in range(6)), "z"))
        output.append({**group, "words": new_words})
    return output


def load_views(root: Path) -> dict[str, tuple[list[dict], list[dict]]]:
    inputs = (root / "data/raw/v101/GC2a-n.txt", root / "data/manifests/zl3b_split.json",
              root / "data/processed/zl3b/train.jsonl",
              root / "data/processed/zl3b/validation.jsonl")
    if [sha(path) for path in inputs] != [GC_SHA, SPLIT_SHA, ZL_TRAIN_SHA, ZL_VALID_SHA]:
        raise ValueError("EXP-0039 source drift")
    if sha(root / "results/EXP-0039/results.json") != EXP39_SHA:
        raise ValueError("EXP-0039 baseline result drift")
    if sha(root / "results/EXP-0041/results.json") != EXP41_SHA:
        raise ValueError("EXP-0041 control result drift")
    from . import readable_edge_calibration as source_module

    if sha(Path(source_module.__file__)) != EXP41_MODULE_SHA:
        raise ValueError("EXP-0041 source helper drift")
    split = json.loads(inputs[1].read_text())["leaf_assignments"]
    all_gc, _, ids = load_gc(inputs[0], split)
    gc_train = [group for group in all_gc if split[group["leaf"]] == "train"]
    gc_valid = [group for group in all_gc if split[group["leaf"]] == "validation"]
    zl_train, _ = load_zl(inputs[2], ids)
    zl_valid, _ = load_zl(inputs[3], ids)
    n_train = sum(len(g["words"]) for g in zl_train)
    n_valid = sum(len(g["words"]) for g in zl_valid)
    controls = json.loads((root / "results/EXP-0041/results.json").read_text())[
        "source_window_sha256"]
    views = {"zl_basic_matched": (zl_train, zl_valid), "gc_v101": (gc_train, gc_valid)}
    for language in ("english", "italian_historical"):
        words, window_hash = source_window(root, language, n_train + n_valid)
        if window_hash != controls[language]:
            raise ValueError("EXP-0041 source-window drift")
        views[language] = (wrap(words[:n_train], zl_train), wrap(words[n_train:], zl_valid))
    for replicate in range(3):
        train_seed = 420001 + replicate
        valid_seed = 420101 + replicate
        views[f"synthetic_positive_{replicate}"] = (
            synthetic(zl_train, train_seed, 0.85), synthetic(zl_valid, valid_seed, 0.85))
        views[f"synthetic_negative_{replicate}"] = (
            synthetic(zl_train, train_seed, 0.0), synthetic(zl_valid, valid_seed, 0.0))
    return views


def run(root: Path) -> dict:
    started = time.monotonic()
    views = load_views(root)
    reference = json.loads((root / "results/EXP-0039/results.json").read_text())["views"]
    results = {}
    for name, (train, valid) in views.items():
        bootstrap_seed = {"zl_basic_matched": 420600, "gc_v101": 420700}.get(name)
        results[name] = evaluate_view(train, valid, bootstrap_seed)
        if name in reference and not math.isclose(
                results[name]["baseline_last_gain_bits_per_pair"],
                reference[name]["real_mean"]["last"], rel_tol=0, abs_tol=1e-10):
            raise AssertionError(f"EXP-0039 baseline mismatch: {name}")
    positives = [results[f"synthetic_positive_{i}"]["richer_minus_edge_bits_per_pair"]
                 for i in range(3)]
    negatives = [results[f"synthetic_negative_{i}"]["richer_minus_edge_bits_per_pair"]
                 for i in range(3)]
    synthetic_calibration = all(value >= 0.10 for value in positives) and all(
        abs(value) <= 0.03 for value in negatives)
    readable_sensitivity = any(results[name]["richer_minus_edge_bits_per_pair"] >= 0.02
                               for name in ("english", "italian_historical"))
    richer = synthetic_calibration and readable_sensitivity and all(
        results[name]["richer_minus_edge_bits_per_pair"] >= 0.02 and
        results[name]["physical_leaf_bootstrap95"][0] > 0
        for name in ("zl_basic_matched", "gc_v101"))
    if time.monotonic() - started > 600:
        raise TimeoutError("EXP-0042 runner wall cap exceeded")
    return {"experiment": EXPERIMENT, "status": "complete",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "numpy": np.__version__, "wall_seconds": time.monotonic() - started,
            "inputs_sha256": {"gc": GC_SHA, "split": SPLIT_SHA,
                              "zl_train": ZL_TRAIN_SHA, "zl_validation": ZL_VALID_SHA,
                              "exp0039_result": EXP39_SHA, "exp0041_result": EXP41_SHA,
                              "exp0041_module": EXP41_MODULE_SHA},
            "synthetic_calibration": synthetic_calibration,
            "readable_sensitivity": readable_sensitivity,
            "richer_context_signal": richer,
            "views": results,
            "limits": "Repeatedly exposed manuscript validation; source-specific low-data residual; no semantic or historical inference."}


def main() -> None:
    root = Path.cwd()
    report = run(root)
    path = root / "results/EXP-0042/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"experiment": EXPERIMENT, "wall_seconds": report["wall_seconds"],
                      "synthetic_calibration": report["synthetic_calibration"],
                      "readable_sensitivity": report["readable_sensitivity"],
                      "richer_context_signal": report["richer_context_signal"],
                      "views": {name: {"gain": row["richer_minus_edge_bits_per_pair"],
                                       "edge_selection": row["edge_selection"],
                                       "richer_selection": row["richer_selection"]}
                                for name, row in report["views"].items()}}, indent=2))


if __name__ == "__main__":
    main()
