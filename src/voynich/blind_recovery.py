"""EXP-0008: text-only predictive partitions, with truth opened only after freezing.

The generators know their rules. The LM and abstraction fitting functions do not
receive those rules, latent labels, oracle distributions, or latent state counts.
This is exploratory predictive-state abstraction, not a cipher translation tool.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ModelConfig, VoynichTransformer
from .runtime import digest, environment, resolve_device, write_json

ALPHABET = 12
FAMILIES = ("cycle_null", "branch_null", "copy_lag", "iid")
CLUSTER_COUNTS = (1, 2, 4, 8, 16, 32)
HORIZONS = (1, 2, 3)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()).hexdigest()


def keyed_hmm(family, permutation):
    """Return edge-emission probabilities; used only in generation/final truth."""
    if family == "cycle_null":
        states = 3
    elif family == "branch_null":
        states = 4
    elif family == "rrxor":
        states = 5
    else:
        raise ValueError(f"Not an HMM family: {family}")
    edge = np.zeros((ALPHABET, states, states), dtype=np.float64)
    if family == "rrxor":
        for bit in range(2):
            for token in np.asarray(permutation)[bit * 6:(bit + 1) * 6]:
                edge[token, 0, 1 + bit] += 0.5 / 6
                for first in range(2):
                    edge[token, 1 + first, 3 + (first ^ bit)] += 0.5 / 6
                edge[token, 3 + bit, 0] += 1 / 6
        prior = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    else:
        width = ALPHABET // states
        for state in range(states):
            edge[:, state, state] += 0.3 / ALPHABET
            moves = [(1, 1.0)] if family == "cycle_null" else [(1, 0.75), (2, 0.25)]
            for step, probability in moves:
                next_state = (state + step) % states
                tokens = np.asarray(permutation)[next_state * width:(next_state + 1) * width]
                edge[tokens, state, next_state] += 0.7 * probability / width
        prior = np.full(states, 1 / states)
    if not np.allclose(edge.sum(axis=(0, 2)), 1):
        raise AssertionError("Invalid edge-emission process")
    return edge, prior


def sample_process(family, permutation, n, length, rng, *, truth_at=None):
    """Independent strings and optional sealed final diagnostics, never fit inputs."""
    visible = np.empty((n, length), dtype=np.uint8)
    state_at = None
    edge = None
    if family in {"cycle_null", "branch_null", "rrxor"}:
        edge, prior = keyed_hmm(family, permutation)
        states = rng.choice(len(prior), n, p=prior)
        table = edge.transpose(1, 0, 2).reshape(len(prior), -1).cumsum(axis=1)
        for t in range(length):
            chosen = (rng.random(n)[:, None] > table[states]).sum(axis=1)
            visible[:, t], states = chosen // len(prior), chosen % len(prior)
            if t == truth_at:
                state_at = states.copy()
    elif family == "copy_lag":
        for t in range(length):
            visible[:, t] = rng.integers(ALPHABET, size=n)
            if t >= 8:
                copied = rng.random(n) < 0.8
                visible[copied, t] = visible[copied, t - 8]
    elif family == "iid":
        visible[:] = rng.integers(ALPHABET, size=(n, length))
    else:
        raise ValueError(family)
    if truth_at is None:
        return visible, None
    if truth_at + 3 >= length:
        raise ValueError("Three held-out future observations required")
    if edge is not None:
        posterior = np.tile(prior, (n, 1))
        for t in range(truth_at + 1):
            posterior = np.einsum("ni,nij->nj", posterior, edge[visible[:, t]])
            posterior /= posterior.sum(axis=1, keepdims=True)
        forecast = []
        belief = posterior.copy()
        transition = edge.sum(axis=0)
        for _ in HORIZONS:
            forecast.append(belief @ edge.sum(axis=2).T)
            belief = belief @ transition
        label = state_at
        map_label = posterior.argmax(axis=1)
        label_kind = "actual_hidden_state"
    elif family == "copy_lag":
        forecast = np.full((3, n, ALPHABET), 0.2 / ALPHABET)
        for h in HORIZONS:
            forecast[h - 1, np.arange(n), visible[:, truth_at + h - 8]] += 0.8
        label = visible[:, truth_at + 1 - 8].astype(int)
        map_label = label.copy()
        label_kind = "next_copy_source_symbol_not_complete_hidden_state"
    else:
        forecast = np.full((3, n, ALPHABET), 1 / ALPHABET)
        # An independently sampled latent label is deliberately unidentifiable.
        label = rng.integers(7, size=n)
        map_label = label.copy()
        label_kind = "independent_unrecoverable_label"
    return visible, {
        "label": np.asarray(label), "map_label": map_label,
        "oracle": np.stack(forecast, axis=1),
        "nuisance": rng.integers(2, size=n), "label_kind": label_kind,
    }


def prepare(root, config):
    """Generate fixed independent streams; visible/truth files are separate."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Refusing to replace data: {root}")
    root.mkdir(parents=True)
    master = np.random.SeedSequence(config["data_seed"])
    children = iter(master.spawn(200))
    keys = {}
    for family in (*FAMILIES, "rrxor"):
        for key in range(10 if family != "rrxor" else 4):
            permutation = np.random.default_rng(next(children)).permutation(ALPHABET)
            keys[f"{family}-key{key}"] = {"family": family, "key": key,
                                              "permutation": permutation.tolist()}
    # Never disclose key tables or process metadata to fitting functions.
    write_json(root / "sealed_generator_keys.json", keys)
    train, validation = [], []
    streams = np.random.SeedSequence(config["data_seed"] + 1).spawn(len(keys) * 5)
    stream_index = 0
    datasets = []
    complete_string_hashes = set()
    for identity, key in keys.items():
        family, number, permutation = key["family"], key["key"], key["permutation"]
        for split, count in (("train", config["train_per_key"]),
                             ("lm_validation", config["lm_validation_per_key"])):
            rng = np.random.default_rng(streams[stream_index])
            stream_index += 1
            if family != "rrxor" and number < 8:
                data, _ = sample_process(family, permutation, count, config["context"] + 1, rng)
                (train if split == "train" else validation).append(data)
                for row in data:
                    row_digest = hashlib.sha256(row.tobytes()).digest()
                    if row_digest in complete_string_hashes:
                        raise AssertionError("Duplicate independent complete LM strings")
                    complete_string_hashes.add(row_digest)
        diagnostic = family == "rrxor" or number in {0, 1, 8, 9}
        pools = {}
        for split, count in (("fit", config["fit_contexts"]), ("validation", config["validation_contexts"]),
                             ("test", config["test_contexts"])):
            rng = np.random.default_rng(streams[stream_index])
            stream_index += 1
            if not diagnostic:
                continue
            length = config["analysis_context"]
            data, truth = sample_process(family, permutation, count, length + 3, rng,
                                         truth_at=length - 1 if split == "test" else None)
            for row in data[:, :length]:
                row_digest = hashlib.sha256(row.tobytes()).digest()
                if row_digest in complete_string_hashes:
                    raise AssertionError("Duplicate complete diagnostic context")
                complete_string_hashes.add(row_digest)
            file = root / f"{identity}-{split}-visible.npz"
            np.savez_compressed(file, x=data[:, :length], future=data[:, length:length + 3])
            pools[split] = {"file": file.name, "sha256": digest(file), "contexts": count}
            if truth is not None:
                file = root / f"{identity}-test-SEALED-TRUTH.npz"
                np.savez_compressed(file, **truth)
                pools["sealed_truth"] = {"file": file.name, "sha256": digest(file)}
        if diagnostic:
            datasets.append({"id": identity, "family": family,
                             "transfer": "novel_family" if family == "rrxor" else
                                         "fresh_stream_control" if family in {"iid", "copy_lag"} else
                                         "unseen_key" if number >= 8 else "seen_key",
                             "key_status": "permutation_invariant_control" if family in {"iid", "copy_lag"}
                                           else "unseen_encoding" if family == "rrxor" or number >= 8
                                           else "seen_encoding",
                             "pools": pools})
    np.save(root / "train_visible.npy", np.concatenate(train))
    np.save(root / "validation_visible.npy", np.concatenate(validation))
    manifest = {"config": config, "datasets": datasets,
                "files": {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
                "truth_policy": "Only test truth is saved; all fits accept visible arrays only.",
                "exact_full_context_duplicate_audit": "passed; no near-duplicate/subsequence audit"}
    write_json(root / "manifest.json", manifest)
    return manifest


def model_config(size, context):
    shapes = {"compact": (128, 2, 352), "large": (256, 4, 704)}
    width, layers, feedforward = shapes[size]
    return ModelConfig(vocab_size=ALPHABET + 1, pad_id=0, d_model=width, n_layers=layers,
                       n_heads=4, d_ff=feedforward, context_length=context, dropout=0.0)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def memory_record(device):
    result = {"peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return result


def ensure_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Registered EXP-0008 wall-clock deadline reached")


@torch.no_grad()
def lm_validation(model, visible, device, batch):
    model.eval()
    losses = []
    for start in range(0, len(visible), batch):
        data = torch.as_tensor(visible[start:start + batch].astype(np.int64) + 1, device=device)
        losses.append(float(F.cross_entropy(model(data[:, :-1]).logits.transpose(1, 2), data[:, 1:]).cpu()))
    return float(np.mean(losses)) / math.log(2)


def train_predictor(train_visible, validation_visible, cfg, settings, seed, output, device, deadline):
    """Only visible strings cross this interface; no generator metadata argument."""
    torch.manual_seed(seed)
    model = VoynichTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.01)
    rng = np.random.default_rng(seed + 8800)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    best = float("inf")
    started = time.monotonic()
    for step in range(1, settings["updates"] + 1):
        ensure_time(deadline)
        model.train()
        rows = rng.integers(len(train_visible), size=settings["batch_size"])
        data = torch.as_tensor(train_visible[rows].astype(np.int64) + 1, device=device)
        logits = model(data[:, :-1]).logits
        loss = F.cross_entropy(logits.transpose(1, 2), data[:, 1:])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite language-model objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % settings["eval_every"] == 0 or step == settings["updates"]:
            score = lm_validation(model, validation_visible, device, settings["batch_size"])
            history.append({"step": step, "validation_bits": score,
                            "elapsed_seconds": time.monotonic() - started, **memory_record(device)})
            state = {"config": cfg.to_dict(), "state_dict": {k: v.cpu().clone()
                     for k, v in model.state_dict().items()}, "step": step, "validation_bits": score}
            torch.save(state, output / "last.pt")
            if score < best:
                best = score
                torch.save(state, output / "best.pt")
            write_json(output / "history.json", history)
            print(json.dumps({"run": output.name, **history[-1]}), flush=True)
    synchronize(device)
    result = {"seed": seed, "config": cfg.to_dict(), "parameters": model.parameter_count,
              "history": history, "best_sha256": digest(output / "best.pt"),
              "elapsed_seconds": time.monotonic() - started}
    write_json(output / "summary.json", result)
    del optimizer, model
    if device == "mps":
        torch.mps.empty_cache()
    return result


@torch.no_grad()
def features(model, visible, device, batch=32):
    """Both representations are frozen-model functions of preceding text only."""
    model.eval()
    residuals, forecasts = [], []
    site = f"blocks.{model.config.n_layers - 1}.resid_post"
    for start in range(0, len(visible), batch):
        x = torch.as_tensor(visible[start:start + batch].astype(np.int64) + 1, device=device)
        output = model(x, cache_names=[site])
        residuals.append(output.cache[site][:, -1].cpu().numpy())
        forecasts.append(output.logits[:, -1, 1:].softmax(-1).cpu().numpy())
    return {"residual": np.concatenate(residuals), "forecast": np.concatenate(forecasts)}


def assign(values, centers):
    distances = (values ** 2).sum(axis=1, keepdims=True) + (centers ** 2).sum(axis=1) - 2 * values @ centers.T
    return distances.argmin(axis=1)


def kmeans(values, count, seed, iterations=30):
    """Fixed unsupervised initialization/iterations; no labels or forecasts used."""
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(len(values), count, replace=False)].copy()
    previous = None
    for _ in range(iterations):
        labels = assign(values, centers)
        if previous is not None and np.array_equal(previous, labels):
            break
        for k in range(count):
            member = values[labels == k]
            if len(member):
                centers[k] = member.mean(axis=0)
        previous = labels
    return centers


def emission_table(labels, future, count):
    table = np.ones((count, len(HORIZONS), ALPHABET), dtype=np.float64)
    for h in range(len(HORIZONS)):
        np.add.at(table[:, h], (labels, future[:, h]), 1)
    return table / table.sum(axis=-1, keepdims=True)


def observed_losses(probabilities, future):
    selected = np.take_along_axis(probabilities, future[:, :, None], axis=-1)[..., 0]
    return -np.log2(np.maximum(selected, 1e-12))


@dataclass
class Partition:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    centers: np.ndarray
    emissions: np.ndarray

    def labels(self, values):
        return assign(((values - self.mean) / self.scale) @ self.projection, self.centers)

    def predict(self, values):
        return self.emissions[self.labels(values)]

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)


def fit_partition(fit_features, fit_future, validation_features, validation_future, seed):
    """Blind interface: observed histories/features and actual future text only.

    The candidate count grid and representation rank do not depend on a true
    state count. Selection predicts three observed future symbols, not oracles.
    """
    mean = fit_features.mean(axis=0)
    scale = fit_features.std(axis=0).clip(1e-5)
    standardized = (fit_features - mean) / scale
    covariance = standardized.T @ standardized / len(standardized)
    _, eigenvectors = np.linalg.eigh(covariance)
    projection = eigenvectors[:, -min(16, fit_features.shape[1]):]
    values = standardized @ projection
    validation_values = ((validation_features - mean) / scale) @ projection
    candidates = []
    models = []
    for count in CLUSTER_COUNTS:
        if count > len(values):
            continue
        centers = kmeans(values, count, seed + count)
        emissions = emission_table(assign(values, centers), fit_future, count)
        loss = observed_losses(emissions[assign(validation_values, centers)], validation_future)
        candidates.append({"count": count, "validation_bits_mean_three_horizons": float(loss.mean()),
                           "per_horizon_bits": loss.mean(axis=0).tolist()})
        models.append(Partition(mean, scale, projection, centers, emissions))
    best = min(item["validation_bits_mean_three_horizons"] for item in candidates)
    selected = next(i for i, item in enumerate(candidates)
                    if item["validation_bits_mean_three_horizons"] <= best + 0.01)
    return models[selected], {"candidates": candidates, "selected_count": candidates[selected]["count"],
                              "selection": "smallest count within 0.01 bits of minimum validation mean(h1,h2,h3)"}


def surface_baselines(fit_x, fit_future, test_x):
    """Matched abstraction examples, fixed Laplace counts; no full-corpus advantage."""
    unigram = emission_table(np.zeros(len(fit_x), dtype=int), fit_future, 1)[0]
    last = emission_table(fit_x[:, -1], fit_future, ALPHABET)
    suffixes = {}
    for row, future in zip(fit_x, fit_future, strict=True):
        key = tuple(row[-4:])
        if key not in suffixes:
            suffixes[key] = np.ones((3, ALPHABET))
        suffixes[key][np.arange(3), future] += 1
    local = []
    for row in test_x:
        counts = suffixes.get(tuple(row[-4:]))
        local.append(last[row[-1]] if counts is None else counts / counts.sum(axis=1, keepdims=True))
    return {"unigram": np.tile(unigram, (len(test_x), 1, 1)),
            "last_symbol": last[test_x[:, -1]], "last_four_backoff": np.stack(local)}


def adjusted_rand(first, second):
    """Permutation-invariant agreement with chance correction, without sklearn."""
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Matching label vectors with at least two samples required")
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    def combinations(v):
        return np.sum(v * (v - 1) / 2)
    joint = combinations(table)
    rows, columns = combinations(table.sum(axis=1)), combinations(table.sum(axis=0))
    expected = rows * columns / (len(a) * (len(a) - 1) / 2)
    denominator = (rows + columns) / 2 - expected
    return 1.0 if denominator == 0 else float((joint - expected) / denominator)


def oracle_kl(oracle, predictions):
    return (oracle * np.log2(np.maximum(oracle, 1e-12) / np.maximum(predictions, 1e-12))).sum(axis=-1)


def final_diagnostics(labels, probabilities, future, truth, baseline, seed):
    """Diagnostic-only; returns numbers and cannot alter any frozen artifact."""
    rng = np.random.default_rng(seed)
    permutations = [adjusted_rand(labels, rng.permutation(truth["map_label"])) for _ in range(100)]
    losses = observed_losses(probabilities, future)
    ari = adjusted_rand(labels, truth["map_label"])
    last_ari = adjusted_rand(baseline["last_label"], truth["map_label"])
    improvement = float(baseline["last_loss"].mean() - losses.mean())
    return {
        "per_horizon_bits": losses.mean(axis=0).tolist(), "mean_three_horizon_bits": float(losses.mean()),
        "per_horizon_oracle_kl_bits": oracle_kl(truth["oracle"], probabilities).mean(axis=0).tolist(),
        "actual_label_ari": adjusted_rand(labels, truth["label"]), "posterior_map_ari": ari,
        "permuted_map_ari_99percentile": float(np.quantile(permutations, 0.99)),
        "unrecoverable_nuisance_ari": adjusted_rand(labels, truth["nuisance"]),
        "last_symbol_map_ari": last_ari, "mean_bits_gain_over_last_symbol": improvement,
        "predictive_partition_criterion": bool(
            baseline["unigram_loss"].mean() - losses.mean() >= 0.05 and ari > np.quantile(permutations, 0.99)),
        "beyond_last_symbol_criterion": bool(improvement >= 0.03 and ari - last_ari >= 0.05),
        "per_context_three_horizon_bits": losses.mean(axis=1).tolist(),
        "counts": Counter(map(int, labels)),
    }


def read_visible(path):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"x", "future"}:
            raise ValueError("Visible pool contains unexpected fields")
        return {name: saved[name].copy() for name in saved.files}


def checked_visible(root, record):
    path = Path(root) / record["file"]
    if digest(path) != record["sha256"]:
        raise AssertionError("Registered visible data changed")
    return read_visible(path)


def run(config_path, data_root, output_root, device):
    settings = json.loads(Path(config_path).read_text())
    torch.set_num_threads(2)
    device = resolve_device(device)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + settings["max_seconds"]
    provenance = environment()
    if provenance["git_dirty"]:
        raise RuntimeError("EXP-0008 requires a clean published source tree")
    write_json(output / "run_manifest.json", {"environment": provenance, "settings": settings,
               "config_sha256": digest(config_path), "device": device, "manuscript_test_scored": False,
               "paid_api_calls": 0, "phase": "data_preparation"})
    try:
        manifest = prepare(data_root, settings)
        write_json(output / "dataset_manifest.json", manifest)
        data_root = Path(data_root)
        for name in ("train_visible.npy", "validation_visible.npy"):
            if digest(data_root / name) != manifest["files"][name]:
                raise AssertionError("Registered LM data changed")
        train = np.load(data_root / "train_visible.npy", mmap_mode="r")
        validation = np.load(data_root / "validation_visible.npy", mmap_mode="r")
        runs = []
        # All LM selection finishes before any final truth file is opened.
        for size in settings["sizes"]:
            for seed in settings["seeds"]:
                identity = f"{size}-seed{seed}"
                summary = train_predictor(train, validation, model_config(size, settings["context"]),
                                          settings, seed, output / identity, device, deadline)
                runs.append({"id": identity, "size": size, "seed": seed, "summary": summary})
        frozen = []
        for run_item in runs:
            checkpoint_path = output / run_item["id"] / "best.pt"
            if digest(checkpoint_path) != run_item["summary"]["best_sha256"]:
                raise AssertionError("Selected neural checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = VoynichTransformer(ModelConfig(**checkpoint["config"])).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            for dataset in manifest["datasets"]:
                ensure_time(deadline)
                pools = {split: checked_visible(data_root, dataset["pools"][split])
                         for split in ("fit", "validation", "test")}
                representation = {split: features(model, value["x"], device, settings["analysis_batch"])
                                  for split, value in pools.items()}
                target = output / "abstractions" / run_item["id"] / dataset["id"]
                target.mkdir(parents=True)
                predictions = surface_baselines(pools["fit"]["x"], pools["fit"]["future"], pools["test"]["x"])
                # Freeze the neural model's next-token forecasts too, without using truth.
                predictions["neural_next"] = representation["test"]["forecast"]
                decisions = {}
                for method in ("residual", "forecast"):
                    partition, selection = fit_partition(
                        representation["fit"][method], pools["fit"]["future"],
                        representation["validation"][method], pools["validation"]["future"],
                        settings["extraction_seed"])
                    partition.save(target / f"{method}.npz")
                    predictions[f"{method}_probabilities"] = partition.predict(representation["test"][method])
                    predictions[f"{method}_labels"] = partition.labels(representation["test"][method])
                    decisions[method] = selection
                np.savez_compressed(target / "frozen_predictions.npz", **predictions)
                write_json(target / "decisions.json", decisions)
                frozen.append({"run": run_item["id"], "dataset": dataset["id"],
                               "files": {str(p.relative_to(output)): digest(p) for p in sorted(target.iterdir())}})
                print(json.dumps({"phase": "blind_extraction", "run": run_item["id"], "dataset": dataset["id"],
                                  "counts": {k: v["selected_count"] for k, v in decisions.items()}}), flush=True)
            del model
            if device == "mps":
                torch.mps.empty_cache()
        # This is the one-way boundary: no fitting or model selection occurs below.
        freeze = {"runs": runs, "artifacts": frozen, "truth_opened": False,
                  "data_manifest_sha256": digest(data_root / "manifest.json"),
                  "elapsed_seconds": time.monotonic() - started}
        write_json(output / "FROZEN_BEFORE_TRUTH.json", freeze)
        reports = []
        for item in frozen:
            ensure_time(deadline)
            for file, expected in item["files"].items():
                if digest(output / file) != expected:
                    raise AssertionError("Frozen analysis artifact changed")
            dataset = next(value for value in manifest["datasets"] if value["id"] == item["dataset"])
            truth_file = data_root / dataset["pools"]["sealed_truth"]["file"]
            if digest(truth_file) != dataset["pools"]["sealed_truth"]["sha256"]:
                raise AssertionError("Sealed truth changed")
            with np.load(truth_file, allow_pickle=False) as saved:
                truth = {name: saved[name] for name in saved.files}
            visible = checked_visible(data_root, dataset["pools"]["test"])
            target = output / "abstractions" / item["run"] / item["dataset"]
            with np.load(target / "frozen_predictions.npz", allow_pickle=False) as saved:
                predictions = {name: saved[name] for name in saved.files}
            baseline = {name: {"per_horizon_bits": observed_losses(predictions[name], visible["future"]).mean(axis=0).tolist(),
                               "mean_three_horizon_bits": float(observed_losses(predictions[name], visible["future"]).mean())}
                        for name in ("unigram", "last_symbol", "last_four_backoff")}
            baseline["neural_next"] = {"bits": float(-np.log2(predictions["neural_next"][np.arange(len(visible["x"])), visible["future"][:, 0]].clip(1e-12)).mean()),
                                        "oracle_kl_bits": float(oracle_kl(truth["oracle"][:, 0], predictions["neural_next"]).mean())}
            reference = {"last_label": visible["x"][:, -1],
                         "last_loss": observed_losses(predictions["last_symbol"], visible["future"]),
                         "unigram_loss": observed_losses(predictions["unigram"], visible["future"])}
            methods = {method: final_diagnostics(predictions[f"{method}_labels"],
                       predictions[f"{method}_probabilities"], visible["future"], truth, reference,
                       settings["diagnostic_seed"]) for method in ("residual", "forecast")}
            reports.append({"run": item["run"], "dataset": item["dataset"], "family": dataset["family"],
                            "transfer": dataset["transfer"], "label_kind": str(truth["label_kind"]),
                            "key_status": dataset["key_status"],
                            "decisions": json.loads((target / "decisions.json").read_text()),
                            "baselines": baseline, "methods": methods})
        final_environment = environment()
        if (final_environment["git_dirty"] or final_environment["git_commit"] != provenance["git_commit"]
                or final_environment["source_sha256"] != provenance["source_sha256"]):
            raise AssertionError("Published source changed during registered execution")
        result = {"experiment": "EXP-0008", "runs": runs, "reports": reports,
                  "source_unchanged": True,
                  "freeze_sha256": digest(output / "FROZEN_BEFORE_TRUTH.json"),
                  "elapsed_seconds": time.monotonic() - started, "status": "complete",
                  "manuscript_test_scored": False, "paid_api_calls": 0, **memory_record(device)}
        write_json(output / "report.json", result)
        print(json.dumps({"status": "complete", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "elapsed_seconds": time.monotonic() - started, **memory_record(device)})
        raise


def benchmark(device, size, batch, context, steps):
    torch.set_num_threads(2)
    device = resolve_device(device)
    torch.manual_seed(8811)
    model = VoynichTransformer(model_config(size, context)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(1, ALPHABET + 1, (batch, context), device=device)
    times = []
    for step in range(steps + 3):
        synchronize(device)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).logits.transpose(1, 2), x)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if step >= 3:
            times.append(time.monotonic() - started)
    result = {"size": size, "parameters": model.parameter_count, "batch": batch, "context": context,
              "measured_steps": steps, "seconds_per_update": float(np.mean(times)), **memory_record(device)}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", default="configs/exp0008_blind.json")
    run_parser.add_argument("--data-root", default="data/processed/exp0008-blind")
    run_parser.add_argument("--output-root", default="outputs/EXP-0008")
    run_parser.add_argument("--device", default="mps")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--device", default="mps")
    bench.add_argument("--size", choices=("compact", "large"), default="large")
    bench.add_argument("--batch", type=int, default=32)
    bench.add_argument("--context", type=int, default=256)
    bench.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.data_root, args.output_root, args.device)
    else:
        benchmark(args.device, args.size, args.batch, args.context, args.steps)


if __name__ == "__main__":
    main()
