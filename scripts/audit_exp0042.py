"""Independent raw-source, model-grid and held-out replay of EXP-0042."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from scripts.audit_exp0039 import gc_source, model_fit, zl_source
from scripts.audit_exp0041 import load_sources


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "gc": "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f",
    "split": "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e",
    "zl_train": "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4",
    "zl_validation": "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3",
    "exp0039_result": "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329",
    "exp0041_result": "7a1f1e66b35561ade184b52a91fc2e346331605098998bc646f1d353b46159cd",
    "exp0041_module": "1aa0701f484e7fabbde47a8b1eaf627a86c7675e59d009d056b72561e601f11c",
}
FAMILIES = ("form", "history", "layout", "all")
KAPPAS = (5, 20, 80)
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
ROLE_NAMES = ("first", "middle", "last")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (actual, expected)


def events(groups: list[tuple]) -> list[tuple]:
    output = []
    for leaf, _, context, words in groups:
        for position in range(len(words) - 1):
            role = 0 if position == 0 else 2 if position == len(words) - 2 else 1
            output.append((leaf, context, role, words[position],
                           words[position - 1] if position else None,
                           position, len(words), words[position + 1][0]))
    return output


def feature_keys(event: tuple, family: str) -> tuple[tuple, ...]:
    _, context, role, word, older, position, width, _ = event
    edge = (("edge_last", word[-1]),
            ("edge_last_role", (word[-1], ROLE_NAMES[role])),
            ("edge_last_context", (word[-1], context)))
    form = (("first", word[0]), ("length", min(8, len(word))),
            ("prefix2", tuple(word[:2])), ("prefix3", tuple(word[:3])),
            ("suffix2", tuple(word[-2:])), ("suffix3", tuple(word[-3:])),
            ("whole", tuple(word)))
    history = (("before_first", older[0] if older else "<start>"),
               ("before_last", older[-1] if older else "<start>"),
               ("before_suffix2", tuple(older[-2:]) if older else ("<start>",)))
    layout = (("position_quartile", min(3, 4 * position // max(width - 1, 1))),
              ("run_length", min(width, 12)))
    return {"edge": edge, "form": edge + form, "history": edge + history,
            "layout": edge + layout, "all": edge + form + history + layout}[family]


def data_fit(groups: list[tuple]) -> dict:
    base_model = model_fit(groups)
    alphabet = base_model[0]
    symbols = sorted(alphabet) + ["<unk>"]
    locations = {symbol: index for index, symbol in enumerate(symbols)}
    q_counts = np.ones(len(symbols), dtype=np.float64)
    counts: dict[tuple, np.ndarray] = {}
    for event in events(groups):
        outcome = locations[event[-1]]
        q_counts[outcome] += 1
        for key in feature_keys(event, "all"):
            if key not in counts:
                counts[key] = np.zeros(len(symbols), dtype=np.float64)
            counts[key][outcome] += 1
    return {"base": base_model, "symbols": symbols, "locations": locations,
            "q": q_counts / q_counts.sum(), "counts": counts}


def base_matrices(observed: list[tuple], model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphabet, global_, global_n, context_, context_n, conditioned, conditioned_n = model["base"]
    symbols = model["symbols"]
    base_log = np.empty((len(observed), len(symbols)), dtype=np.float64)
    context_log = np.empty_like(base_log)
    targets = np.empty(len(observed), dtype=np.int64)
    for index, event in enumerate(observed):
        _, ctx, role, word, _, _, _, outcome = event
        context_key = (ctx, role)
        conditioned_key = (ctx, role, 0, word[-1])
        global_distribution = np.array(
            [(global_[role][symbol] + 1) / (global_n[role] + len(symbols))
             for symbol in symbols], dtype=np.float64)
        context_distribution = np.array(
            [(context_[context_key][symbol] + 50 * value) / (context_n[context_key] + 50)
             for symbol, value in zip(symbols, global_distribution)], dtype=np.float64)
        edge_distribution = np.array(
            [(conditioned[conditioned_key][symbol] + 20 * value) /
             (conditioned_n[conditioned_key] + 20)
             for symbol, value in zip(symbols, context_distribution)], dtype=np.float64)
        base_log[index] = np.log(edge_distribution)
        context_log[index] = np.log(context_distribution)
        targets[index] = model["locations"][outcome if outcome in alphabet else "<unk>"]
    return base_log, context_log, targets


def effect_matrix(observed: list[tuple], model: dict, family: str, kappa: int) -> np.ndarray:
    q = model["q"]
    ratios = {key: np.log((counts + kappa * q) / (counts.sum() + kappa) / q)
              for key, counts in model["counts"].items()}
    result = np.zeros((len(observed), len(q)), dtype=np.float64)
    for i, event in enumerate(observed):
        features = feature_keys(event, family)
        result[i] = sum((ratios[key] for key in features if key in ratios),
                        np.zeros(len(q), dtype=np.float64)) / len(features)
    return result


def gain_vector(base: np.ndarray, target: np.ndarray,
                effects: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0:
        return np.zeros(len(target), dtype=np.float64)
    logits = base + alpha * effects
    normalized = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
    return (normalized[np.arange(len(target)), target] -
            base[np.arange(len(target)), target]) / math.log(2)


def select(groups: list[tuple], families: tuple[str, ...]) -> tuple[dict, list[dict]]:
    fit_groups, dev_groups = [], []
    for group in groups:
        code = hashlib.sha256(group[0].encode()).digest()
        (dev_groups if int.from_bytes(code[:4], "big") % 5 == 0 else fit_groups).append(group)
    assert fit_groups and dev_groups
    model = data_fit(fit_groups)
    dev_events = events(dev_groups)
    base, _, target = base_matrices(dev_events, model)
    grid = []
    best = {"family": families[0], "kappa": 80, "alpha": 0.0,
            "dev_gain_bits_per_pair": 0.0}
    for kappa in KAPPAS:
        for family in families:
            effect = effect_matrix(dev_events, model, family, kappa)
            for alpha in ALPHAS:
                value = float(gain_vector(base, target, effect, alpha).mean())
                candidate = {"family": family, "kappa": kappa, "alpha": alpha,
                             "dev_gain_bits_per_pair": value}
                grid.append(candidate)
                improvement = value - best["dev_gain_bits_per_pair"]
                if improvement > 1e-12:
                    best = candidate
                elif abs(improvement) <= 1e-12 and value > 0:
                    old_rank = (best["alpha"], -best["kappa"], families.index(best["family"]))
                    new_rank = (alpha, -kappa, families.index(family))
                    if new_rank < old_rank:
                        best = candidate
    if best["dev_gain_bits_per_pair"] <= .005:
        best = {"family": families[0], "kappa": 80, "alpha": 0.0,
                "dev_gain_bits_per_pair": 0.0}
    return best, grid


def synthetic(skeleton: list[tuple], seed: int, persistence: float) -> list[tuple]:
    states = random.Random(seed)
    middles = random.Random(seed + 42_000_000)
    transformed = []
    for leaf, locus, context, words in skeleton:
        state = states.choice(("x", "y"))
        stream = []
        for index in range(len(words)):
            if index > 0:
                if persistence == 0:
                    state = states.choice(("x", "y"))
                elif states.random() > persistence:
                    state = "y" if state == "x" else "x"
            middle = tuple(middles.choice("abcdefgh") for _ in range(6))
            stream.append((state, *middle, "z"))
        transformed.append((leaf, locus, context, stream))
    return transformed


def wrap(words: list[tuple[str, ...]], template: list[tuple]) -> list[tuple]:
    output = []
    cursor = 0
    for leaf, locus, context, source in template:
        end = cursor + len(source)
        output.append((leaf, locus, context, words[cursor:end]))
        cursor = end
    assert cursor == len(words)
    return output


def all_views() -> dict[str, tuple[list[tuple], list[tuple]]]:
    paths = {"gc": ROOT / "data/raw/v101/GC2a-n.txt",
             "split": ROOT / "data/manifests/zl3b_split.json",
             "zl_train": ROOT / "data/processed/zl3b/train.jsonl",
             "zl_validation": ROOT / "data/processed/zl3b/validation.jsonl",
             "exp0039_result": ROOT / "results/EXP-0039/results.json",
             "exp0041_result": ROOT / "results/EXP-0041/results.json",
             "exp0041_module": ROOT / "src/voynich/readable_edge_calibration.py"}
    assert {name: digest(path) for name, path in paths.items()} == INPUTS
    split = json.loads(paths["split"].read_text())["leaf_assignments"]
    all_gc, _, ids = gc_source(paths["gc"], split)
    gc_train = [row for row in all_gc if split[row[0]] == "train"]
    gc_valid = [row for row in all_gc if split[row[0]] == "validation"]
    zl_train, _ = zl_source(paths["zl_train"], ids)
    zl_valid, _ = zl_source(paths["zl_validation"], ids)
    train_words = sum(len(row[3]) for row in zl_train)
    valid_words = sum(len(row[3]) for row in zl_valid)
    plain, hashes = load_sources(train_words + valid_words)
    assert hashes == json.loads(paths["exp0041_result"].read_text())["source_window_sha256"]
    views = {"zl_basic_matched": (zl_train, zl_valid), "gc_v101": (gc_train, gc_valid)}
    for language in ("english", "italian_historical"):
        words = plain[language]
        views[language] = (wrap(words[:train_words], zl_train),
                           wrap(words[train_words:], zl_valid))
    for i in range(3):
        views[f"synthetic_positive_{i}"] = (
            synthetic(zl_train, 420001 + i, .85), synthetic(zl_valid, 420101 + i, .85))
        views[f"synthetic_negative_{i}"] = (
            synthetic(zl_train, 420001 + i, 0), synthetic(zl_valid, 420101 + i, 0))
    return views


def replay(name: str, train: list[tuple], held: list[tuple], saved: dict,
           reference: dict) -> None:
    edge, edge_grid = select(train, ("edge",))
    rich, rich_grid = select(train, FAMILIES)
    for title, actual_choice, actual_grid in (
        ("edge", edge, edge_grid), ("richer", rich, rich_grid)
    ):
        archived_choice = saved[f"{title}_selection"]
        assert {key: actual_choice[key] for key in ("family", "kappa", "alpha")} == {
            key: archived_choice[key] for key in ("family", "kappa", "alpha")}
        close(actual_choice["dev_gain_bits_per_pair"], archived_choice["dev_gain_bits_per_pair"])
        archived_grid = saved[f"{title}_development_grid"]
        assert len(actual_grid) == len(archived_grid)
        for actual, archived in zip(actual_grid, archived_grid):
            assert {key: actual[key] for key in ("family", "kappa", "alpha")} == {
                key: archived[key] for key in ("family", "kappa", "alpha")}
            close(actual["dev_gain_bits_per_pair"], archived["dev_gain_bits_per_pair"])
    model = data_fit(train)
    observed = events(held)
    base, context, labels = base_matrices(observed, model)
    edge_effect = effect_matrix(observed, model, "edge", edge["kappa"])
    rich_effect = effect_matrix(observed, model, rich["family"], rich["kappa"])
    edge_bits = gain_vector(base, labels, edge_effect, edge["alpha"])
    rich_bits = gain_vector(base, labels, rich_effect, rich["alpha"])
    differences = rich_bits - edge_bits
    baseline = (base[np.arange(len(labels)), labels] -
                context[np.arange(len(labels)), labels]) / math.log(2)
    assert len(train) == saved["train_groups"]
    assert len(held) == saved["validation_groups"]
    assert len(events(train)) == saved["train_pairs"]
    assert len(observed) == saved["validation_pairs"]
    assert len(model["symbols"]) == saved["alphabet_size_train_plus_unk"]
    close(float(baseline.mean()), saved["baseline_last_gain_bits_per_pair"])
    if reference:
        close(float(baseline.mean()), reference["real_mean"]["last"])
    close(float(edge_bits.mean()), saved["edge_over_p0_bits_per_pair"])
    close(float(rich_bits.mean()), saved["richer_over_p0_bits_per_pair"])
    close(float(differences.mean()), saved["richer_minus_edge_bits_per_pair"])
    by_leaf = defaultdict(lambda: [0, 0.0])
    for event, bits in zip(observed, differences):
        item = by_leaf[event[0]]
        item[0] += 1
        item[1] += float(bits)
    assert set(by_leaf) == set(saved["by_leaf"])
    for leaf, (pairs, total) in by_leaf.items():
        assert pairs == saved["by_leaf"][leaf]["pairs"]
        close(total, saved["by_leaf"][leaf]["bits_sum"])
    if name in {"zl_basic_matched", "gc_v101"}:
        rng = random.Random(420600 if name == "zl_basic_matched" else 420700)
        leaves = sorted(by_leaf)
        draws = []
        for _ in range(2000):
            sample = rng.choices(leaves, k=len(leaves))
            draws.append(sum(by_leaf[leaf][1] for leaf in sample) /
                         sum(by_leaf[leaf][0] for leaf in sample))
        draws.sort()
        for actual, archived in zip((draws[49], draws[1949]),
                                    saved["physical_leaf_bootstrap95"]):
            close(actual, archived)
    else:
        assert "physical_leaf_bootstrap95" not in saved
    print(name, "independent full-grid/source/score replay pass", flush=True)


def main() -> None:
    started = time.monotonic()
    path = ROOT / "results/EXP-0042/results.json"
    report = json.loads(path.read_text())
    assert report["source_sha256"] == digest(ROOT / "src/voynich/context_residual.py")
    assert report["inputs_sha256"] == INPUTS
    views = all_views()
    assert set(views) == set(report["views"]) and len(views) == 10
    reference = json.loads((ROOT / "results/EXP-0039/results.json").read_text())["views"]
    for name, (train, held) in views.items():
        replay(name, train, held, report["views"][name], reference.get(name, {}))
    positives = [report["views"][f"synthetic_positive_{i}"]["richer_minus_edge_bits_per_pair"]
                 for i in range(3)]
    negatives = [report["views"][f"synthetic_negative_{i}"]["richer_minus_edge_bits_per_pair"]
                 for i in range(3)]
    calibration = all(value >= .10 for value in positives) and all(abs(value) <= .03
                                                                   for value in negatives)
    readable = any(report["views"][name]["richer_minus_edge_bits_per_pair"] >= .02
                   for name in ("english", "italian_historical"))
    stronger = calibration and readable and all(
        report["views"][name]["richer_minus_edge_bits_per_pair"] >= .02 and
        report["views"][name]["physical_leaf_bootstrap95"][0] > 0
        for name in ("zl_basic_matched", "gc_v101"))
    assert (calibration, readable, stronger) == (
        report["synthetic_calibration"], report["readable_sensitivity"],
        report["richer_context_signal"])
    elapsed = time.monotonic() - started
    if elapsed > 600:
        raise TimeoutError("EXP-0042 audit wall cap exceeded")
    audit = {"experiment": "EXP-0042", "audit": "pass", "views": 10,
             "grids_replayed": 10, "wall_seconds": elapsed,
             "result_sha256": digest(path), "richer_context_signal": stronger}
    destination = ROOT / "results/EXP-0042/audit.json"
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
