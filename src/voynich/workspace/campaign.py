"""Bounded local calibration; science records and bulk arrays stay separate."""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .tasks import build_tasks, normalize_answer, task_manifest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def answer_correct(text, answers, *, strict=False):
    value = text if strict else text.strip().strip(" .!\"'`")
    return normalize_answer(value) in {normalize_answer(a) for a in answers}


def article_windows(text, encode, *, count, length, seed):
    """One seeded, nonoverlapping article sample; no cross-article windows."""
    articles = re.split(r"(?m)^\s*= ([^=\n]+) =\s*$", text)
    eligible = []
    for i in range(1, len(articles), 2):
        title, body = articles[i], articles[i + 1]
        ids = encode(body)
        if len(ids) >= length:
            eligible.append((i // 2, title, ids))
    if len(eligible) < count:
        raise ValueError(f"Need {count} independent articles, found {len(eligible)}")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(eligible))[:count]
    records = []
    for index in chosen:
        article_id, title, ids = eligible[int(index)]
        start = int(rng.integers(0, len(ids) - length + 1))
        records.append({"article_id": article_id, "title": title, "start": start,
                        "token_ids": ids[start:start + length]})
    return records, len(eligible)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["experiment"] != "JSPACE-0001":
        raise ValueError("Wrong experiment")
    return config


def prepare(config, model):
    output, result = Path(config["output"]), Path(config["results"])
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result / "inputs.json").exists():
        raise FileExistsError("Inputs already frozen; use the existing manifest")
    forms = [prefix + word for word in config["words"] for prefix in ("", " ")]
    ids = [model.encode(form) for form in forms]
    if not all(len(row) == 1 for row in ids):
        raise ValueError("Every predeclared form must be exactly one token")
    windows, eligible = article_windows(
        Path(config["calibration_file"]).read_text(), model.encode,
        count=config["calibration_prompts"], length=config["sequence_length"], seed=config["seed"],
    )
    write_json(output / "calibration-windows.json", windows)
    source_paths = sorted(Path("src/voynich/workspace").glob("*.py"))
    source_paths += [Path("configs/jspace0001.json"), Path("docs/experiments/JSPACE-0001.md")]
    snapshot = Path(config["model_snapshot"])
    files = sorted(p for p in snapshot.iterdir() if p.is_file())
    manifest = {
        "experiment": config["experiment"], "config": config, "config_sha256": canonical_digest(config),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): digest(p) for p in source_paths},
        "model_sha256": {p.name: digest(p) for p in files},
        "corpus_sha256": digest(config["calibration_file"]),
        "corpus_source": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "corpus_license": "CC-BY-SA-3.0 / GFDL; Wikipedia-derived WikiText-2 nonraw training distribution",
        "corpus_source_pin": "content SHA-256; upstream main commit was not resolved",
        "model_source": "https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192",
        "model_license": "Apache-2.0", "eligible_articles": eligible,
        "calibration_sha256": digest(output / "calibration-windows.json"),
        "calibration_articles": [{k: v for k, v in row.items() if k != "token_ids"} for row in windows],
        "tasks": task_manifest(build_tasks(seed=config["seed"])),
        "dictionary": [{"form": form, "token_id": row[0]} for form, row in zip(forms, ids)],
        "precision": "existing four-bit values expanded to float32 in transformer blocks; float32 residuals; quantized embedding and output head",
        "MLX_ENABLE_TF32": "0",
        "qualification_sha256": digest(result / "qualification.json"),
        "official_source_sha256": {p.name: digest(p) for p in (output / "provenance").glob("official-*.py")},
        "software": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    write_json(result / "inputs.json", manifest)
    return manifest


def capability(config, model, manifest):
    output, result = Path(config["output"]), Path(config["results"])
    path = result / "pilot.json"
    if path.exists():
        raise FileExistsError("Pilot already scored")
    started, rows, cache = time.monotonic(), [], {}
    for task in build_tasks(seed=config["seed"], split="pilot"):
        if time.monotonic() - started > config["evaluation_seconds_cap"]:
            raise TimeoutError("Pilot time cap")
        prompt = task["prompt"]
        if prompt not in cache:
            ids = model.encode(prompt, chat=True)
            text, generated = model.greedy(ids, max_tokens=config["max_generated_tokens"])
            cache[prompt] = (text, generated, ids)
        text, generated, ids = cache[prompt]
        rows.append({"id": task["id"], "family": task["family"], "relation": task["relation"],
                     "prompt_group": task["prompt_group"], "pair_id": task["pair_id"],
                     "answer": text, "accepted": task["answers"],
                     "correct": answer_correct(text, task["answers"]),
                     "strict_correct": answer_correct(text, task["answers"], strict=True),
                     "generated_tokens": generated, "prompt_tokens": len(ids),
                     "truncated": len(generated) == config["max_generated_tokens"]})
        if len(rows) % 10 == 0:
            print(json.dumps({"phase": "pilot", "completed": len(rows), "seconds": time.monotonic() - started}), flush=True)
    families, relations = defaultdict(list), defaultdict(list)
    for row in rows:
        families[row["family"]].append(row["correct"])
        if row["family"] != "surface_copy":
            relations[row["relation"]].append(row["correct"])
    scores = {k: float(np.mean(v)) for k, v in families.items()}
    relation_scores = {k: float(np.mean(v)) for k, v in relations.items()}
    passed = (all(scores[k] >= config["capability_semantic_min"] for k in ("indirect_fact", "anchored_alias"))
              and scores["surface_copy"] >= config["capability_copy_min"]
              and sum(v >= .75 for v in relation_scores.values()) >= 3)
    report = {"passed": passed, "family_accuracy": scores, "relation_accuracy": relation_scores,
              "seconds": time.monotonic() - started, "rows": rows,
              "inputs_sha256": digest(result / "inputs.json"), "unique_prompts": len(cache)}
    write_json(path, report)
    write_json(output / "pilot-rendered-inputs.json", {p: value[2] for p, value in cache.items()})
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    return report


def fit(config, model, manifest):
    output, result = Path(config["output"]), Path(config["results"])
    if not json.loads((result / "pilot.json").read_text())["passed"]:
        raise ValueError("Capability gate failed; calibration not launched")
    if canonical_digest(config) != manifest["config_sha256"]:
        raise ValueError("Configuration drift")
    windows_path = output / "calibration-windows.json"
    if digest(windows_path) != manifest["calibration_sha256"]:
        raise ValueError("Calibration drift")
    windows = json.loads(windows_path.read_text())
    rows = model.output_rows([r["token_id"] for r in manifest["dictionary"]])
    shape = (len(config["layers"]), len(rows), model.width)
    path = output / "derivatives.npy"
    progress = output / "fit-progress.json"
    completed, elapsed = 0, 0.
    if path.exists():
        state = json.loads(progress.read_text())
        if state["inputs_sha256"] != digest(result / "inputs.json"):
            raise ValueError("Resume manifest drift")
        completed, elapsed = state["completed"], state["seconds"]
        values = np.lib.format.open_memmap(path, mode="r+")
        if values.shape != (len(windows), *shape):
            raise ValueError("Resume array shape mismatch")
    else:
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(windows), *shape))
    started = time.monotonic()
    state = {"completed": completed, "seconds": elapsed, "inputs_sha256": digest(result / "inputs.json")}
    write_json(progress, state)
    for index in range(completed, len(windows)):
        if elapsed + time.monotonic() - started >= config["fit_seconds_cap"]:
            break
        for start in range(0, len(rows), config["direction_batch"]):
            stop = min(start + config["direction_batch"], len(rows))
            values[index, :, start:stop] = model.jacobian_rows(
                windows[index]["token_ids"], rows[start:stop], tuple(config["layers"]), skip_first=config["skip_first"],
            )
            if model.mx.get_peak_memory() > config["memory_bytes_cap"]:
                raise MemoryError("MLX memory cap exceeded")
        completed = index + 1
        if completed % config["checkpoint_every"] == 0 or completed == len(windows):
            values.flush()
            state = {"completed": completed, "seconds": elapsed + time.monotonic() - started,
                     "peak_memory_bytes": model.mx.get_peak_memory(), "inputs_sha256": digest(result / "inputs.json")}
            write_json(progress, state)
            print(json.dumps({"phase": "fit", **state}), flush=True)
    values.flush()
    state = {"completed": completed, "seconds": elapsed + time.monotonic() - started,
             "peak_memory_bytes": model.mx.get_peak_memory(), "inputs_sha256": digest(result / "inputs.json")}
    write_json(progress, state)
    if completed != len(windows):
        write_json(result / "fit.json", {**state, "complete": False, "reason": "time_cap"})
        return
    # Distinct article halves, float64 accumulation, no task-dependent fitting.
    total = np.asarray(values).mean(axis=0, dtype=np.float64)
    first = np.asarray(values[:len(windows)//2]).mean(axis=0, dtype=np.float64)
    second = np.asarray(values[len(windows)//2:]).mean(axis=0, dtype=np.float64)
    second_moment = np.einsum("nlrd,nlrd->lr", values, values, dtype=np.float64) / len(windows)
    cosine = np.sum(first * second, axis=-1) / (np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1))
    signed_fraction = np.sum(total * total, axis=-1) / second_moment
    np.savez(output / "lens.npz", mean=total.astype(np.float32), first=first.astype(np.float32),
             second=second.astype(np.float32), output_rows=rows, second_moment=second_moment)
    report = {**state, "complete": True, "half_cosine": cosine.tolist(),
              "signed_energy_fraction": signed_fraction.tolist(),
              "lens_sha256": digest(output / "lens.npz"), "derivatives_sha256": digest(path),
              "method": "exact selected-row transport, not full-vocabulary J-space"}
    write_json(result / "fit.json", report)
    print(json.dumps({"phase": "fit_complete", **state}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "fit", "pilot-fit"))
    parser.add_argument("--config", default="configs/jspace0001.json")
    args = parser.parse_args()
    config = load_config(args.config)
    from .mlx_backend import QwenWorkspace
    model = QwenWorkspace(config["model_snapshot"], dense_transport=True)
    model.mx.set_memory_limit(config["memory_bytes_cap"])
    if args.stage == "prepare":
        manifest = prepare(config, model)
        print(json.dumps({"eligible_articles": manifest["eligible_articles"], "frozen": True}), flush=True)
        return
    manifest = json.loads((Path(config["results"]) / "inputs.json").read_text())
    qualification_path = Path(config["results"]) / "qualification.json"
    if (digest(qualification_path) != manifest["qualification_sha256"]
            or not json.loads(qualification_path.read_text())["passed"]):
        raise ValueError("Numerical qualification missing, failed, or changed")
    for name, expected in manifest["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen source drift: {name}")
    if args.stage in ("pilot", "pilot-fit"):
        report = capability(config, model, manifest)
        if not report["passed"]:
            return
    if args.stage in ("fit", "pilot-fit"):
        fit(config, model, manifest)


if __name__ == "__main__":
    main()
