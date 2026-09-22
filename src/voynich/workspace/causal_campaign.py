"""Frozen development selection and one-shot final causal confirmation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_backend import CausalWorkspace, qualify_cached
from .geometry import coordinate_swap, matched_random_delta, normalize_rows
from .tasks import build_tasks


def condition_seed(seed, task_id, layer, condition):
    return int(canonical_digest([seed, task_id, layer, condition])[:16], 16)


def meets(value, threshold):
    """Include exact decimal boundaries despite binary floating-point rounding."""
    return value is not None and (value >= threshold or abs(value-threshold) <= 1e-12)


def make_delta(condition, hidden, donor, country_rows, raw_rows, *, seed):
    swap, report = coordinate_swap(hidden, country_rows)
    if condition == "swap":
        delta = swap
    elif condition == "random":
        delta = matched_random_delta(swap, seed=seed)
    elif condition == "orthogonal":
        delta = matched_random_delta(swap, seed=seed, orthogonal_to=country_rows)
    elif condition == "raw_swap":
        delta, raw_report = coordinate_swap(hidden, raw_rows)
        report["raw_swap"] = raw_report
    elif condition == "donor":
        delta = (donor - hidden).astype(np.float32)
    elif condition == "identity":
        delta = np.zeros_like(hidden)
    else:
        raise ValueError("Unknown condition")
    report.update({"applied_norm": float(np.linalg.norm(delta)), "hidden_norm": float(np.linalg.norm(hidden)),
                   "donor_displacement_norm": float(np.linalg.norm(donor-hidden)),
                   "after_applied_coordinates": (normalize_rows(country_rows) @ (hidden+delta)).tolist()})
    return delta, report


def rates(rows):
    semantic = [r for r in rows if r["expected_effect"] == "change"]
    eligible = [r for r in semantic if r["baseline_correct"]]
    donor_eligible = [r for r in eligible if r["donor_correct"]]
    copy = [r for r in rows if r["expected_effect"] == "preserve"]

    def score(items, key):
        successes = sum(bool(r[key]) for r in items)
        return {"successes": successes, "count": len(items), "rate": successes / len(items) if items else None}

    return {"counterfactual": score(eligible, "counterfactual_correct"),
            "counterfactual_all": score(semantic, "counterfactual_correct"),
            "counterfactual_both_clean_correct": score(donor_eligible, "counterfactual_correct"),
            "copy_preserved": score(copy, "original_correct"),
            "original_semantic_retained": score(eligible, "original_correct"),
            "baseline_semantic": score(semantic, "baseline_correct"),
            "invalid_count": sum(not r["valid"] for r in rows)}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["condition"])].append(row)
    output = {}
    for (layer, condition), members in groups.items():
        report = rates(members)
        for field in ("family", "relation", "pair_id"):
            report[f"by_{field}"] = {
                value: rates([r for r in members if r[field] == value])
                for value in sorted({r[field] for r in members})
            }
        output.setdefault(str(layer), {})[condition] = report
    return output


def select_layer(summary, config):
    candidates = []
    for layer in config["layers"]:
        group = summary[str(layer)]
        copy_rate = group["swap"]["copy_preserved"]["rate"]
        swap = group["swap"]["counterfactual"]["rate"]
        controls = [group[c]["counterfactual"]["rate"] for c in ("random", "orthogonal", "raw_swap")]
        if meets(copy_rate, config["confirm_copy_preserve_min"]) and swap is not None and all(v is not None for v in controls):
            metrics = [group[c]["counterfactual"] for c in ("swap", "random", "orthogonal", "raw_swap")]
            if all('count' in metric for metric in metrics):
                if len({metric['count'] for metric in metrics}) != 1:
                    raise ValueError("Control denominators differ")
                # Subtract integer counts before dividing; mathematically tied
                # layer scores must not differ due to floating subtraction.
                score = (metrics[0]['successes']-max(m['successes'] for m in metrics[1:])) / metrics[0]['count']
            else:
                score = swap-max(controls)
            candidates.append({"layer": layer, "score": score, "swap_rate": swap,
                               "copy_rate": copy_rate, "control_rates": controls})
    if not candidates:
        return {"selected_layer": None, "reason": "No layer satisfies copy-preservation eligibility", "candidates": []}
    chosen = sorted(candidates, key=lambda c: (-c["score"], c["layer"]))[0]
    return {"selected_layer": chosen["layer"], "chosen": chosen, "candidates": candidates,
            "rule": "max development swap minus strongest registered control; earlier layer breaks ties"}


def decide(summary, layer, config):
    group = summary[str(layer)]
    swap = group["swap"]["counterfactual"]["rate"]
    advantages = {c: swap - group[c]["counterfactual"]["rate"]
                  if swap is not None and group[c]["counterfactual"]["rate"] is not None else None
                  for c in ("random", "orthogonal", "raw_swap")}
    relation_rates = {key: value["counterfactual"]["rate"] for key, value in group["swap"]["by_relation"].items()
                      if key != "copy" and value["counterfactual"]["count"]}
    copy = group["swap"]["copy_preserved"]["rate"]
    checks = {"swap": meets(swap, config["confirm_swap_min"]),
              "control_advantage": all(meets(v, config["confirm_control_advantage_min"]) for v in advantages.values()),
              "copy": meets(copy, config["confirm_copy_preserve_min"]),
              "every_relation": len(relation_rates) == 4 and all(meets(v, config["confirm_relation_min"]) for v in relation_rates.values()),
              "valid_interventions": all(g["invalid_count"] == 0 for g in group.values())}
    return {"supported": all(checks.values()), "checks": checks, "swap_rate": swap,
            "control_advantages": advantages, "relation_rates": relation_rates, "copy_rate": copy,
            "claim_scope": "restricted country-coordinate intervention in this quantized model and task panel only"}


def freeze_evaluation(config, model):
    result = Path(config["results"])
    path = result / "evaluation-inputs.json"
    if path.exists():
        raise FileExistsError("Evaluation inputs already frozen")
    fit_report = json.loads((result / "fit.json").read_text())
    if not fit_report["complete"] or digest(Path(config["output"]) / "lens.npz") != fit_report["lens_sha256"]:
        raise ValueError("Registered calibration is incomplete or changed")
    qualification = qualify_cached(model)
    write_json(result / "cached-qualification.json", qualification)
    if not qualification["passed"]:
        raise ValueError("Cached/trace execution did not qualify")
    paths = [Path("src/voynich/workspace") / name for name in
             ("__init__.py", "campaign.py", "causal_backend.py", "causal_campaign.py", "geometry.py", "mlx_backend.py", "tasks.py")]
    paths += [Path("docs/experiments/JSPACE-0001-evaluation.md")]
    manifest = {"config_sha256": canonical_digest(config), "fit_sha256": digest(result / "fit.json"),
                "inputs_sha256": digest(result / "inputs.json"), "lens_sha256": fit_report["lens_sha256"],
                "cached_qualification_sha256": digest(result / "cached-qualification.json"),
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {str(p): digest(p) for p in paths}}
    write_json(path, manifest)


def evaluate(config, model, split):
    output, result = Path(config["output"]), Path(config["results"])
    manifest = json.loads((result / "evaluation-inputs.json").read_text())
    if canonical_digest(config) != manifest["config_sha256"]:
        raise ValueError("Evaluation configuration drift")
    for path, expected in manifest["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Evaluation source drift: {path}")
    for filename, field in (("fit.json", "fit_sha256"), ("inputs.json", "inputs_sha256"),
                            ("cached-qualification.json", "cached_qualification_sha256")):
        if digest(result / filename) != manifest[field]:
            raise ValueError(f"Evaluation input drift: {filename}")
    if digest(output / "lens.npz") != manifest["lens_sha256"]:
        raise ValueError("Lens drift")
    state_path = output / "evaluation-progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is not None and state["evaluation_inputs_sha256"] != digest(result / "evaluation-inputs.json"):
        raise ValueError("Evaluation resume provenance mismatch")
    elapsed = state["seconds"] if state else 0.
    started = time.monotonic()
    if (result / f"{split}.json").exists():
        raise FileExistsError("Split already scored; no repeated final evaluation")
    selection = None
    if split == "final":
        selection = json.loads((result / "selection.json").read_text())
        if selection["evaluation_inputs_sha256"] != digest(result / "evaluation-inputs.json"):
            raise ValueError("Selection provenance mismatch")
        if selection["development_sha256"] != digest(result / "development.json"):
            raise ValueError("Development changed after selection")
        if selection["selected_layer"] is None:
            raise ValueError("No eligible development layer")
    layers = config["layers"] if selection is None else [selection["selected_layer"]]
    tasks = build_tasks(seed=config["seed"], split=split)
    records_path = output / f"{split}-observations.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {(r["id"], r["layer"], r["condition"]) for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate observation records")
    expected_keys = {(task["id"], layer, condition) for task in tasks for layer in layers
                     for condition in config["development_conditions"]}
    if not done <= expected_keys:
        raise ValueError("Observation grid has unknown records")
    baseline_dir = output / f"{split}-baselines"
    baseline_dir.mkdir(exist_ok=True)
    baselines = {}

    def baseline(prompt):
        key = canonical_digest(prompt)
        if key in baselines:
            return baselines[key]
        meta_path, array_path = baseline_dir / f"{key}.json", baseline_dir / f"{key}.npz"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["evaluation_inputs_sha256"] != digest(result / "evaluation-inputs.json"):
                raise ValueError("Baseline provenance mismatch")
            if digest(array_path) != meta["arrays_sha256"]:
                raise ValueError("Baseline activations changed")
            with np.load(array_path) as arrays:
                captures = {int(k): arrays[k] for k in arrays.files}
        else:
            ids = model.encode(prompt, chat=True)
            text, generated, captures = model.greedy_cached(ids, max_tokens=config["max_generated_tokens"], capture_layers=layers)
            np.savez(array_path, **{str(k): v for k, v in captures.items()})
            meta = {"text": text, "generated_tokens": generated, "ids": ids,
                    "arrays_sha256": digest(array_path),
                    "evaluation_inputs_sha256": digest(result / "evaluation-inputs.json")}
            write_json(meta_path, meta)
        baselines[key] = (meta, captures)
        return meta, captures

    with np.load(output / "lens.npz") as lens:
        directions = lens["mean"].reshape(len(config["layers"]), len(config["words"]), 2, model.width).mean(axis=2)
        raw = lens["output_rows"].reshape(len(config["words"]), 2, model.width).mean(axis=1)
    budget_exhausted = False
    with records_path.open("a") as handle:
        for task in tasks:
            if elapsed + time.monotonic() - started >= config["evaluation_seconds_cap"]:
                budget_exhausted = True
                break
            own, own_h = baseline(task["prompt"])
            donor, donor_h = baseline(task["donor_prompt"])
            countries = [config["words"].index(task[k]) for k in ("latent_concept", "donor_concept")]
            for layer in layers:
                country_rows = directions[config["layers"].index(layer), countries]
                for condition in config["development_conditions"]:
                    if (task["id"], layer, condition) in done:
                        continue
                    if elapsed + time.monotonic() - started >= config["evaluation_seconds_cap"]:
                        budget_exhausted = True
                        break
                    row = {k: task[k] for k in ("id", "family", "relation", "pair_id", "prompt_group", "paraphrase_group",
                                               "reuse_group", "expected_effect", "latent_concept", "donor_concept")}
                    row.update({"layer": layer, "condition": condition, "baseline": own["text"], "donor": donor["text"],
                                "baseline_correct": answer_correct(own["text"], task["answers"]),
                                "donor_correct": answer_correct(donor["text"], task["counterfactual_answers"]),
                                "accepted_original": task["answers"], "accepted_counterfactual": task["counterfactual_answers"]})
                    try:
                        delta, geometry = make_delta(condition, own_h[layer], donor_h[layer], country_rows, raw[countries],
                                                     seed=condition_seed(config["seed"], task["id"], layer, condition))
                        text, generated, _ = model.greedy_cached(
                            own["ids"], patches=[{"layer": layer, "position": len(own["ids"])-1, "delta": delta}],
                            max_tokens=config["max_generated_tokens"],
                        )
                        row.update({"valid": True, "answer": text, "generated_tokens": generated, "geometry": geometry,
                                    "truncated": len(generated) == config["max_generated_tokens"],
                                    "original_correct": answer_correct(text, task["answers"]),
                                    "counterfactual_correct": answer_correct(text, task["counterfactual_answers"])})
                        if condition == "identity" and generated != own["generated_tokens"]:
                            raise RuntimeError("Identity changed generation; abort scientific scoring")
                    except ValueError as error:
                        row.update({"valid": False, "error": str(error), "original_correct": False,
                                    "counterfactual_correct": False})
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    if model.mx.get_peak_memory() > config["memory_bytes_cap"]:
                        raise MemoryError("Evaluation memory cap exceeded")
                    rows.append(row)
                    done.add((task["id"], layer, condition))
                    write_json(state_path, {"seconds": elapsed + time.monotonic() - started,
                                           "split": split, "completed": len(rows),
                                           "evaluation_inputs_sha256": digest(result / "evaluation-inputs.json")})
                    if len(rows) % 120 == 0:
                        print(json.dumps({"phase": split, "completed": len(rows), "seconds": elapsed + time.monotonic() - started}), flush=True)
                if budget_exhausted:
                    break
            if budget_exhausted:
                break
    if budget_exhausted:
        write_json(result / f"{split}-incomplete.json", {"reason": "cumulative_evaluation_time_cap", "completed": len(rows)})
        return
    expected = len(tasks) * len(layers) * len(config["development_conditions"])
    if len(rows) != expected or done != expected_keys:
        raise ValueError("Incomplete or contaminated observation grid")
    summary = summarize(rows)
    report = {"complete": True, "split": split, "observation_count": len(rows), "summary": summary,
              "observations_sha256": digest(records_path), "evaluation_inputs_sha256": digest(result / "evaluation-inputs.json"),
              "seconds_this_invocation": time.monotonic() - started,
              "cumulative_evaluation_seconds": elapsed + time.monotonic() - started,
              "peak_memory_bytes": model.mx.get_peak_memory()}
    write_json(result / f"{split}.json", report)
    if split == "development":
        selected = select_layer(summary, config)
        selected.update({"development_sha256": digest(result / "development.json"),
                         "evaluation_inputs_sha256": digest(result / "evaluation-inputs.json")})
        write_json(result / "selection.json", selected)
        print(json.dumps(selected), flush=True)
    else:
        decision = decide(summary, layers[0], config)
        decision.update({"final_sha256": digest(result / "final.json"), "selection_sha256": digest(result / "selection.json")})
        write_json(result / "decision.json", decision)
        print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "development", "final"))
    args = parser.parse_args()
    config = load_config("configs/jspace0001.json")
    model = CausalWorkspace(config["model_snapshot"], dense_transport=True)
    model.mx.set_memory_limit(config["memory_bytes_cap"])
    if args.stage == "freeze":
        freeze_evaluation(config, model)
    else:
        evaluate(config, model, args.stage)


if __name__ == "__main__":
    main()
