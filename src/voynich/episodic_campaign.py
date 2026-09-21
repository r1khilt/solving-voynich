"""Bounded fresh-task training and evaluation; synthetic evidence, never plaintext.

Training accepts visible arrays only. Task descriptions and oracles are read by
the separate evaluate stage after every neural checkpoint has been frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import torch

from .episodic_data import canonicalize, make_task, oracle_joint, sample_tasks, sample_task
from .episodic_models import ModelSpec, build_model, raw_log_probs, fit_edge_hmm
from .runtime import digest, write_json

FAMILIES = ("cycle", "branch", "pair_parity", "iid")
HELDOUT = ("rr_xor", "switching")
LOG2 = math.log(2)


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def provenance():
    root = Path(__file__).resolve().parents[2]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "source_hashes": {str(p.relative_to(root)): digest(p) for p in sorted(root.glob("src/voynich/*.py"))},
    }


def resources(device):
    result = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if str(device) == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_bytes=torch.mps.recommended_max_memory())
    return result


def inputs(raw, canonical, device):
    raw = np.asarray(raw, dtype=np.int64)
    if canonical:
        tokens, counts, inverse = canonicalize(raw, 4)
        return tuple(torch.as_tensor(a.copy(), device=device) for a in (tokens, counts, inverse))
    return torch.as_tensor(raw.copy(), device=device), None, None


def log_probs(model, raw, spec, device):
    tokens, counts, inverse = inputs(raw, spec.canonical, device)
    logits = model(tokens)
    return raw_log_probs(logits, counts, inverse) if spec.canonical else logits.log_softmax(-1)


def loss_on(model, raw, spec, device, warmup):
    lp = log_probs(model, raw[:, :-1], spec, device)
    y = torch.as_tensor(np.asarray(raw[:, 1:], dtype=np.int64).copy(), device=device)
    losses = -lp.gather(-1, y[..., None]).squeeze(-1)
    return losses[:, warmup:].mean()


def prepare(config, out):
    """Generate matched visible pools. Each fresh row has its own parameter seed."""
    out = Path(out)
    if (out / "data_manifest.json").exists():
        raise FileExistsError("Prepared pools already exist; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    n = config["steps"] * config["batch_size"]
    length = config["context"] + 1
    seed = config["data_seed"]
    fixed = [make_task(f, seed + 1000 + i) for i, f in enumerate(FAMILIES * 8)]
    all_hashes = {}
    for regime in ("fixed", "fresh"):
        path = out / f"train_{regime}.npy"
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, length))
        for start in range(0, n, 512):
            stop = min(n, start + 512)
            if regime == "fresh":
                tasks = [make_task(FAMILIES[i % len(FAMILIES)], seed + 100000 + i) for i in range(start, stop)]
            else:
                tasks = [fixed[i % len(fixed)] for i in range(start, stop)]
            arr[start:stop] = sample_tasks(tasks, length, seed + 9000000 + start)
        arr.flush()
        del arr
        all_hashes[path.name] = digest(path)
        print(json.dumps({"prepared": regime, "episodes": n, "length": length}), flush=True)
    tasks_by_split = {}
    for split, offset, task_count in (("development", 20000000, config["dev_tasks_per_family"]),
                                      ("confirmation", 30000000, config["test_tasks_per_family"])):
        families = FAMILIES if split == "development" else FAMILIES + HELDOUT
        tasks = [make_task(f, seed + offset + i * 1000 + j)
                 for i, f in enumerate(families) for j in range(task_count)]
        groups = []
        rows = []
        for i, task in enumerate(tasks):
            rows.append(sample_task(task, config["sequences_per_task"], length, seed + offset + 700000 + i))
            groups.extend([i] * config["sequences_per_task"])
        path = out / f"{split}.npy"
        np.save(path, np.concatenate(rows).astype(np.uint8))
        np.save(out / f"{split}_groups.npy", np.array(groups, dtype=np.int64))
        write_json(out / f"{split}_tasks.json", tasks)
        tasks_by_split[split] = [t["task_id"] for t in tasks]
        for p in (path, out / f"{split}_groups.npy", out / f"{split}_tasks.json"):
            all_hashes[p.name] = digest(p)
    assert not set(tasks_by_split["development"]) & set(tasks_by_split["confirmation"])
    manifest = {"config": config, "sha256": all_hashes, "task_ids": tasks_by_split,
                "fixed_task_ids": [t["task_id"] for t in fixed], "fresh_parameter_seed_range": [
                    seed + 100000, seed + 100000 + n - 1], "training_reads_oracles": False,
                "independence_unit": "task parameter seed; alphabet permutations can repeat"}
    write_json(out / "data_manifest.json", manifest)
    return manifest


@torch.no_grad()
def validation(model, data, spec, device, warmup, batch_size=32):
    was_training = model.training
    model.eval()
    weighted, count = 0.0, 0
    for start in range(0, len(data), batch_size):
        raw = data[start:start + batch_size]
        weighted += float(loss_on(model, raw, spec, device, warmup)) * len(raw)
        count += len(raw)
    model.train(was_training)
    return weighted / count / LOG2


def benchmark(config, out, device):
    """Artificial random input only, no final pools or scientific selection."""
    rows = []
    for condition in config["conditions"]:
        spec = ModelSpec(**condition["model"])
        torch.manual_seed(991)
        model = build_model(spec).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        raw = np.random.default_rng(881).integers(0, 4, (config["batch_size"], config["context"] + 1))
        elapsed = []
        for i in range(8):
            sync(device)
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(model, raw, spec, device, config["warmup"])
            loss.backward()
            optimizer.step()
            sync(device)
            if i >= 3:
                elapsed.append(time.monotonic() - start)
        row = {"condition": condition["name"], "parameters": sum(p.numel() for p in model.parameters()),
               "seconds_per_update": float(np.median(elapsed)), "resources": resources(device)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer
        if device == "mps":
            torch.mps.empty_cache()
    report = {"artificial_input_only": True, "rows": rows, "config": config, "provenance": provenance()}
    write_json(Path(out) / "benchmark.json", report)
    return report


def train(config, out, device, only=None, exclude=None):
    out = Path(out)
    manifest = json.loads((out / "data_manifest.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Training config differs from frozen prepared data")
    dev = np.load(out / "development.npy", mmap_mode="r")
    for name in ("development.npy", "train_fixed.npy", "train_fresh.npy"):
        if digest(out / name) != manifest["sha256"][name]:
            raise ValueError(f"Input hash mismatch: {name}")
    campaign_start = time.monotonic()
    for condition in config["conditions"]:
        if only and condition["name"] != only:
            continue
        if exclude and condition["name"] == exclude:
            continue
        data = np.load(out / f"train_{condition['regime']}.npy", mmap_mode="r")
        for seed in config["model_seeds"]:
            run = out / "runs" / f"{condition['name']}-s{seed}"
            if (run / "summary.json").exists():
                continue
            if run.exists():
                raise FileExistsError(f"Incomplete run exists; preserve it and select a new output: {run}")
            run.mkdir(parents=True)
            spec = ModelSpec(**condition["model"])
            torch.manual_seed(seed)
            model = build_model(spec).to(device)
            initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                         weight_decay=config["weight_decay"])
            started = time.monotonic()
            info = {"condition": condition, "model_seed": seed, "spec": asdict(spec),
                    "parameters": sum(p.numel() for p in model.parameters()), "initial_hash": initial_hash,
                    "provenance": provenance(), "data_manifest_sha256": digest(out / "data_manifest.json")}
            write_json(run / "manifest.json", info)
            best, history = float("inf"), []
            for step in range(config["steps"]):
                elapsed = time.monotonic() - started
                if elapsed > config["run_seconds_cap"] or time.monotonic() - campaign_start > config["campaign_seconds_cap"]:
                    write_json(run / "stopped.json", {"reason": "registered_time_cap", "step": step})
                    raise TimeoutError("Registered compute cap reached")
                if device == "mps" and torch.mps.driver_allocated_memory() > config["driver_bytes_cap"]:
                    raise MemoryError("Registered MPS allocation guard")
                batch = data[step * config["batch_size"]:(step + 1) * config["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_on(model, batch, spec, device, config["warmup"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                ratio = (step + 1) / config["steps"]
                lr = config["learning_rate"] * min(1, (step + 1) / 50) * (0.15 + 0.85 * (1 + math.cos(math.pi * ratio)) / 2)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                if (step + 1) % config["eval_every"] == 0 or step + 1 == config["steps"]:
                    score = validation(model, dev, spec, device, config["warmup"])
                    row = {"step": step + 1, "train_bits": float(loss.detach()) / LOG2,
                           "development_bits": score, "elapsed_seconds": time.monotonic() - started,
                           "resources": resources(device)}
                    history.append(row)
                    if score < best:
                        best = score
                        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    "spec": asdict(spec), "step": step + 1}, run / "best.pt")
                    write_json(run / "history.json", history)
                    print(json.dumps({"run": run.name, **row}), flush=True)
            sync(device)
            summary = {**info, "elapsed_seconds": time.monotonic() - started,
                       "best_development_bits": best, "best_checkpoint_sha256": digest(run / "best.pt"),
                       "updates": config["steps"], "sampled_targets": config["steps"] * config["batch_size"] *
                       (config["context"] - config["warmup"]), "resources": resources(device)}
            write_json(run / "summary.json", summary)
            del model, optimizer
            if device == "mps":
                torch.mps.empty_cache()
    completion = only or ("except_" + exclude if exclude else "all")
    write_json(out / f"training_complete_{completion}.json", {"conditions": completion, "provenance": provenance()})


def load_model(run, device):
    saved = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
    spec = ModelSpec(**saved["spec"])
    model = build_model(spec).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, spec


@torch.no_grad()
def neural_joint(model, spec, prefixes, horizon, device):
    """Exact tree expansion, lexicographic next-string order, normalized by AR."""
    base = np.asarray(prefixes, dtype=np.int64)
    joint = np.ones((len(base), 1), dtype=np.float64)
    contexts = base.copy()
    for _ in range(horizon):
        probabilities = []
        for start in range(0, len(contexts), 128):
            probabilities.append(log_probs(model, contexts[start:start + 128], spec, device)[:, -1].exp().cpu().numpy())
        p = np.concatenate(probabilities)
        joint = (joint.reshape(-1, 1) * p).reshape(len(base), -1)
        contexts = np.concatenate((np.repeat(contexts, 4, axis=0), np.tile(np.arange(4), len(contexts))[:, None]), axis=1)
    return joint


def kl_bits(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return np.sum(np.where(p > 0, p * (np.log2(np.maximum(p, 1e-30)) - np.log2(np.maximum(q, 1e-30))), 0), axis=-1)


def online_baseline(sequence, order):
    tables = {}
    losses = []
    for t in range(1, len(sequence)):
        previous_key = tuple(sequence[max(0, t - 1 - order):t - 1]) if order else ()
        tables.setdefault(previous_key, np.ones(4) * 0.5)[sequence[t - 1]] += 1
        key = tuple(sequence[max(0, t - order):t]) if order else ()
        counts = tables.get(key, np.ones(4) * 0.5)
        losses.append(-math.log2(counts[sequence[t]] / counts.sum()))
    return np.array(losses)


def oracle_losses(task, sequence):
    edge, belief = np.array(task["edge"]), np.array(task["prior"])
    losses = []
    for t, x in enumerate(sequence):
        nxt = belief @ edge[x]
        probability = nxt.sum()
        if t:
            losses.append(-math.log2(max(probability, 1e-30)))
        belief = nxt / max(probability, 1e-30)
    return np.array(losses)


def paired_interval(values, seed=16012):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lo95": float(np.quantile(means, .025)),
            "hi95": float(np.quantile(means, .975)), "independent_tasks": len(values)}


def verify_frozen_runs(config, out):
    """Shared gate for every entry point that can inspect confirmation data."""
    out = Path(out)
    runs = [out / "runs" / f"{c['name']}-s{s}" for c in config["conditions"] for s in config["model_seeds"]]
    if any(not (r / "summary.json").exists() for r in runs):
        raise ValueError("Every registered training run must finish before final evaluation")
    manifest = json.loads((out / "data_manifest.json").read_text())
    if config != manifest["config"]:
        raise ValueError("Evaluation config differs from frozen prepared data")
    current_source = provenance()["source_hashes"]
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        if summary["data_manifest_sha256"] != digest(out / "data_manifest.json"):
            raise ValueError("Run used a different data manifest")
        if summary["best_checkpoint_sha256"] != digest(run / "best.pt"):
            raise ValueError("Frozen checkpoint hash mismatch")
        if summary["provenance"]["source_hashes"] != current_source:
            raise ValueError("Scientific source changed after training")
    return runs, manifest


def evaluate(config, out, device):
    out = Path(out)
    if (out / "confirmation_report.json").exists():
        raise FileExistsError("Final report exists; do not adapt/re-score the same final pool")
    runs, manifest = verify_frozen_runs(config, out)
    for name in ("confirmation.npy", "confirmation_groups.npy", "confirmation_tasks.json"):
        if digest(out / name) != manifest["sha256"][name]:
            raise ValueError("Final pool hash mismatch")
    data = np.load(out / "confirmation.npy")
    groups = np.load(out / "confirmation_groups.npy")
    tasks = json.loads((out / "confirmation_tasks.json").read_text())
    baseline = []
    for i, task in enumerate(tasks):
        rows = data[groups == i]
        baseline.append({"task_id": task["task_id"], "family": task["family"],
                         "oracle_bits": float(np.mean([oracle_losses(task, r)[config['warmup']:].mean() for r in rows])),
                         **{f"order_{k}_bits": float(np.mean([online_baseline(r, k)[config['warmup']:].mean() for r in rows]))
                            for k in (0, 1, 2)}})
    write_json(out / "confirmation_baselines.json", baseline)
    results = []
    for run in runs:
        model, spec = load_model(run, device)
        by_task = []
        for i, task in enumerate(tasks):
            rows = data[groups == i]
            score = validation(model, rows, spec, device, config["warmup"])
            prefix = rows[:config["joint_prefixes_per_task"], :config["joint_prefix_length"]]
            q = neural_joint(model, spec, prefix, config["joint_horizon"], device)
            p = np.stack([oracle_joint(task, r, config["joint_horizon"]) for r in prefix])
            q1 = q.reshape(len(prefix), 4, -1).sum(-1)
            p1 = p.reshape(len(prefix), 4, -1).sum(-1)
            kls, first = kl_bits(p, q), kl_bits(p1, q1)
            renaming_tv = []
            for permutation in (np.array([1, 2, 3, 0]), np.array([3, 2, 1, 0]), np.array([1, 0, 2, 3])):
                renamed = neural_joint(model, spec, permutation[prefix], 1, device)
                aligned = renamed[:, permutation]
                renaming_tv.append(float((.5 * np.abs(q1 - aligned).sum(-1)).mean()))
            by_task.append({**baseline[i], "model_bits": score, "joint_kl_bits": float(kls.mean()),
                            "first_kl_bits": float(first.mean()), "beyond_first_kl_bits": float((kls - first).mean()),
                            "renaming_tv_mean": float(np.mean(renaming_tv)),
                            "renaming_tv_max_of_three_means": float(max(renaming_tv)),
                            "joint_normalization_error": float(np.abs(q.sum(-1) - 1).max())})
        row = {"run": run.name, "spec": asdict(spec), "tasks": by_task,
               "mean_bits": float(np.mean([x['model_bits'] for x in by_task])),
               "mean_joint_kl_bits": float(np.mean([x['joint_kl_bits'] for x in by_task]))}
        results.append(row)
        write_json(run / "confirmation.json", row)
        print(json.dumps({k: v for k, v in row.items() if k not in {'tasks', 'spec'}}), flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    contrasts = []
    for left, right in config["contrasts"]:
        for family in (*FAMILIES, *HELDOUT, "in_family", "out_of_family"):
            diffs = []
            for i, task in enumerate(tasks):
                f = task["family"]
                if not (f == family or family == "in_family" and f in FAMILIES or family == "out_of_family" and f in HELDOUT):
                    continue
                left_scores = [r['tasks'][i]['model_bits'] for r in results if r['run'].startswith(left + '-s')]
                right_scores = [r['tasks'][i]['model_bits'] for r in results if r['run'].startswith(right + '-s')]
                diffs.append(float(np.mean(left_scores) - np.mean(right_scores)))
            interval = paired_interval(diffs)
            contrasts.append({"left_minus_right": [left, right], "family": family, **interval,
                              "registered_gain": interval['mean'] < -.02 and interval['hi95'] < 0})
    report = {"results": results, "contrasts": contrasts, "provenance": provenance(),
              "primary_units": "bits per observed synthetic symbol after prefix warmup",
              "scope": "new parameter tasks, not historical decoding; bootstrap descriptive across tasks"}
    write_json(out / "confirmation_report.json", report)
    return report


def explicit_comparison(config, out):
    out = Path(out)
    if (out / "explicit_report.json").exists():
        raise FileExistsError("Explicit confirmation report already exists")
    rows = []
    start = time.monotonic()
    _, manifest = verify_frozen_runs(config, out)
    for name in ("confirmation.npy", "confirmation_groups.npy", "confirmation_tasks.json"):
        if digest(out / name) != manifest["sha256"][name]:
            raise ValueError("Final pool hash mismatch")
    tasks = json.loads((out / "confirmation_tasks.json").read_text())
    data = np.load(out / "confirmation.npy")
    groups = np.load(out / "confirmation_groups.npy")
    for family_index, family in enumerate(FAMILIES + HELDOUT):
        selected = [(i, t) for i, t in enumerate(tasks) if t["family"] == family][:config["explicit_tasks_per_family"]]
        for task_index, (group_index, task) in enumerate(selected):
            if time.monotonic() - start > config["explicit_seconds_cap"]:
                raise TimeoutError("Explicit model registered time cap")
            a = sample_task(task, 32, config["context"], 51000 + family_index * 100 + task_index)
            b = sample_task(task, 8, config["context"], 52000 + family_index * 100 + task_index)
            c = data[groups == group_index]
            model, fit = fit_edge_hmm(a, b, states=(1, 2, 4, 8), restarts=2,
                                    iterations=config["em_iterations"], seed=61000 + family_index * 100 + task_index,
                                    condition_validation=False)
            predictions, oracle = [], []
            for seq in c[:config["joint_prefixes_per_task"]]:
                prefix = seq[:config["joint_prefix_length"]]
                predictions.append(np.asarray(model.joint_probs(model.filter(prefix), config["joint_horizon"])).reshape(-1))
                oracle.append(oracle_joint(task, prefix, config["joint_horizon"]))
            row = {"family": family, "task_id": task["task_id"], "fit": fit,
                   "test_bits": float(np.mean([-(model.log_likelihood(seq) -
                       model.log_likelihood(seq[:config["warmup"] + 1])) /
                       (len(seq) - config["warmup"] - 1) / LOG2 for seq in c])),
                   "joint_kl_bits": float(np.mean(kl_bits(oracle, predictions))),
                   "adaptation_observed_symbols": int(a.size + b.size)}
            rows.append(row)
            write_json(out / "explicit_progress.json", rows)
            print(json.dumps({k: v for k, v in row.items() if k != 'fit'}), flush=True)
    write_json(out / "explicit_report.json", {"tasks": rows, "elapsed_seconds": time.monotonic() - start,
               "scope": "same task/target subset and warmup as neural; explicit model receives additional per-task fitting observations"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "benchmark", "train", "evaluate", "explicit"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--only")
    parser.add_argument("--exclude")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no silent CPU substitution")
        torch.mps.set_per_process_memory_fraction(.70)
    config = json.loads(Path(args.config).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(config, args.out)
    elif args.action == "benchmark":
        benchmark(config, args.out, args.device)
    elif args.action == "train":
        train(config, args.out, args.device, args.only, args.exclude)
    elif args.action == "evaluate":
        evaluate(config, args.out, args.device)
    else:
        explicit_comparison(config, args.out)


if __name__ == "__main__":
    main()
