"""Frozen from-scratch B-family qualification on new keys and authors."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import random
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from scripts import audit_blind_channel_dev001 as scalar
from scripts import audit_blind_channel_dev004 as independent
from scripts import summarize_blind_channel_dev003_search as search_audit
from scripts.run_blind_channel_dev001 import (
    baseline_log_likelihood, checked_artifact, digest, fit_glyph_baseline, require_frozen,
)
from scripts.run_blind_channel_dev004 import get_units, metrics, resource_report
from scripts.run_blind_channel_dev005 import trace_accounting
from voynich.finite_state_channel import Channel, SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import MarkovSource, decode_units
from voynich.higher_order_unit_refine import refine_units
from voynich.unit_channel_edit_floor import minimum_unit_edit_distance
from voynich.unit_channel_search import (
    UnitSearchConfig, _initial_units, literal_unit_pool, search_unit_channel,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-CONFIRM-001"
MANIFEST = "data/manifests/blind_channel_confirm001.json"
CORPUS = "data/manifests/blind_channel_confirmation_corpora.json"
SELECTION = "results/BLIND-CHANNEL-DEV-004/source_selection.json"
CASE_NAMES = tuple(f"B-key{k}{suffix}" for k in range(1, 9) for suffix in ("", "-shuffle"))
SOURCE_PATHS = [
    "scripts/run_blind_channel_confirm001.py", "scripts/build_blind_channel_confirm001.py",
    "scripts/prepare_blind_channel_confirmation_corpora.py", "scripts/build_blind_channel_development_corpora.py",
    "scripts/build_blind_channel_dev001.py", "scripts/run_blind_channel_dev001.py",
    "scripts/run_blind_channel_dev004.py", "scripts/run_blind_channel_dev005.py",
    "scripts/summarize_blind_channel_dev003_search.py", "scripts/audit_blind_channel_dev001.py",
    "scripts/audit_blind_channel_dev004.py", "scripts/evaluate_naibbe001.py",
    "src/voynich/finite_state_channel.py", "src/voynich/finite_state_channel_fit.py",
    "src/voynich/finite_state_channel_search.py", "src/voynich/unit_channel_search.py",
    "src/voynich/batched_unit_channel.py", "src/voynich/higher_order_unit_channel.py",
    "src/voynich/higher_order_unit_refine.py", "src/voynich/unit_channel_edit_floor.py",
    "src/voynich/unit_channel_decision.py", "src/voynich/naibbe_key_search.py",
    "tests/test_blind_channel_confirm001.py", "tests/test_blind_channel_confirmation_corpora.py",
    "tests/test_blind_channel_confirm001_independent.py",
    "docs/research/blind-channel-fresh-qualification-review.md",
    "docs/research/final-corpus-provenance-plan.md", "data/manifests/blind_channel_development_corpora.json",
    "docs/experiments/BLIND-CHANNEL-CONFIRM-001.md", SELECTION,
]


def write_new(path: Path, payload: dict, *, compressed=False) -> dict:
    relative = path.relative_to(ROOT)
    raw = (json.dumps(payload, sort_keys=True, indent=None if compressed else 2,
                      allow_nan=False) + "\n").encode()
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    if len(raw) > 200 * 1024 * 1024:
        raise ValueError("Artifact exceeds 200MiB cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(relative), "sha256": digest(raw), "bytes": len(raw)}


def selected_path(name):
    return ROOT / f"results/{EXPERIMENT}/{name}_freeze.json"


def sources():
    raw = (ROOT / SELECTION).read_bytes()
    if digest(raw) != "1a0136627f04e3b68b7e2b3d801e5d8cdebc617fc1e5f3f0fa7b7e344ab74c60":
        raise ValueError("Registered source selection changed")
    artifact = json.loads(raw)["models"]
    packed = (ROOT / artifact["path"]).read_bytes()
    if len(packed) != artifact["bytes"] or digest(packed) != artifact["sha256"]:
        raise ValueError("Source archive identity differs")
    models = json.loads(gzip.decompress(packed))["models"]
    return SourceModel.from_dict(models["order1"]), MarkovSource.from_dict(models["order3"]), models


def inputs(commit):
    manifest = json.loads((ROOT / MANIFEST).read_text())
    if (manifest["experiment"] != EXPERIMENT or set(manifest["cases"]) != set(CASE_NAMES)
            or manifest["final_author_roles"] != {"fit": "sallust", "transfer": "tacitus"}):
        raise ValueError("Registered case panel differs")
    require_frozen(manifest["pipeline_freeze"], SOURCE_PATHS)
    require_frozen(commit, [MANIFEST, CORPUS])
    if digest((ROOT / CORPUS).read_bytes()) != manifest["corpus_manifest_sha256"]:
        raise ValueError("Corpus manifest identity changed")
    for index, name in enumerate(CASE_NAMES):
        case = manifest["cases"][name]
        if case["search_seed"] != 63101 + 257 * index or case["positive"] != (index % 2 == 0):
            raise ValueError("Registered seed/case identity changed")
    return manifest, *sources()


def frequency_units(source, records, context):
    return _initial_units(source, tuple(records), context, literal_unit_pool(context),
                          0, random.Random(0))[0]


def fit_case(name, commit):
    resource.setrlimit(resource.RLIMIT_CPU, (840, 840))
    started, cpu = time.monotonic(), time.process_time()
    attempt = ROOT / f"results/{EXPERIMENT}/{name}_attempt.json"
    if name not in CASE_NAMES or selected_path(name).exists() or attempt.exists():
        raise ValueError("Invalid case or existing key")
    manifest, source1, source3, models = inputs(commit)
    write_new(attempt, {"case": name, "data_freeze": commit, "start_unix": time.time()})
    case = manifest["cases"][name]
    fit = checked_artifact(case["artifacts"]["fit"])
    context = CodingContext(**fit["context"])
    baseline = fit_glyph_baseline(fit["records"], context.glyph_alphabet)
    if baseline != scalar.baseline_from_fit(fit["records"], context.glyph_alphabet):
        raise ValueError("Independent glyph baseline differs")
    config = UnitSearchConfig(seed=case["search_seed"], restarts=16, max_sweeps=80,
                              max_seconds=300., batch_size=256, max_units=256)
    stage1 = search_unit_channel(source1, fit["records"], context, config=config).to_dict()
    if stage1["units"] is None:
        raise ValueError("Stage1 returned no supported dictionary")
    frequency = frequency_units(source1, fit["records"], context)
    if tuple(stage1["trace"][0]["units"]) != frequency:
        raise ValueError("Frequency baseline differs from first search initialization")
    first_artifact = write_new(ROOT / f"outputs/{EXPERIMENT}/{name}-stage1.json.gz", stage1, compressed=True)
    compact = {k: v for k, v in stage1.items() if k != "trace"}
    compact.update(experiment=EXPERIMENT, case_id=name, source_freeze=manifest["pipeline_freeze"],
        source_sha256=digest(json.dumps(models["order1"], sort_keys=True).encode()),
        input_sha256=case["artifacts"]["fit"]["sha256"], full_output=first_artifact,
        cpu_seconds_before_final_write=time.process_time() - cpu,
        peak_rss_bytes=resource_report(started, cpu)["peak_rss_bytes"],
        environment=resource_report(started, cpu), source_hashes={p: digest((ROOT / p).read_bytes()) for p in SOURCE_PATHS})
    accounting1 = search_audit.validate_case(compact, stage1, asdict(context))
    initial = tuple(stage1["units"])
    stage1_score = stage1["score"]
    replay1 = math.fsum(independent.infer_record(models["order1"], initial, row, context.stop_probability)["log_likelihood"]
                        for row in fit["records"])
    if abs(replay1 - stage1_score["log_likelihood"]) > 1e-7:
        raise ValueError("Independent stage1 selected likelihood differs")
    write_new(ROOT / f"outputs/{EXPERIMENT}/{name}-stage1-summary.json", compact)
    del stage1, compact
    gc.collect()
    stage2 = refine_units(source3, fit["records"], context, initial,
                          max_sweeps=20, max_seconds=300., tolerance=1e-8)
    accounting2 = trace_accounting(stage2, models["order3"], context)
    replay2 = math.fsum(independent.infer_record(models["order3"], tuple(stage2["units"]), row,
                                               context.stop_probability)["log_likelihood"] for row in fit["records"])
    if abs(replay2 - stage2["score"]["log_likelihood"]) > 1e-7:
        raise ValueError("Independent stage2 selected likelihood differs")
    second_artifact = write_new(ROOT / f"outputs/{EXPERIMENT}/{name}-stage2.json.gz", stage2, compressed=True)
    write_new(selected_path(name), {"experiment": EXPERIMENT, "case_id": name, "data_freeze": commit,
        "pipeline_freeze": manifest["pipeline_freeze"], "fit_sha256": case["artifacts"]["fit"]["sha256"],
        "source_selection_sha256": digest((ROOT / SELECTION).read_bytes()),
        "units": stage2["units"], "score": stage2["score"], "baseline": baseline,
        "frequency_units": list(frequency), "stage1_units": list(initial), "stage1_score": stage1_score,
        "stage1_accounting": accounting1, "stage2_accounting": accounting2,
        "stage1_trace": first_artifact, "stage2_trace": second_artifact,
        "stage2_stop": stage2["stop_reason"], "resources": resource_report(started, cpu),
        "independent_final_score_deltas": [abs(replay1 - stage1_score["log_likelihood"]),
                                            abs(replay2 - stage2["score"]["log_likelihood"])]})
    print(json.dumps({"case": name, "status": "complete", "resources": resource_report(started, cpu)}), flush=True)


def campaign(commit):
    manifest, *_ = inputs(commit)
    directory = ROOT / f"results/{EXPERIMENT}"
    if (directory / "started.json").exists() or any(selected_path(name).exists() for name in CASE_NAMES):
        raise FileExistsError("Campaign already attempted; no selective reruns")
    write_new(directory / "started.json", {"data_freeze": commit, "pipeline_freeze": manifest["pipeline_freeze"],
                                           "cases": list(CASE_NAMES), "workers": 2, "start_unix": time.time()})
    started = time.monotonic()

    def launch(name):
        env = dict(os.environ, PYTHONPATH=".:src", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        try:
            p = subprocess.run([sys.executable, "scripts/run_blind_channel_confirm001.py", "fit",
                                "--case", name, "--freeze", commit], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=960)
            return {"case": name, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except subprocess.TimeoutExpired:
            return {"case": name, "returncode": None, "status": "hard_parent_timeout"}

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, name): name for name in CASE_NAMES}
        for future in as_completed(futures):
            row = future.result()
            results[row["case"]] = row
            write_new(directory / f"{row['case']}_process.json", row)
            print(json.dumps({"finished": len(results), "case": row["case"], "returncode": row["returncode"]}), flush=True)
    write_new(directory / "campaign.json", {"experiment": EXPERIMENT, "data_freeze": commit,
        "results": [results[name] for name in CASE_NAMES], "wall_seconds": time.monotonic() - started,
        "status": "complete" if all(row["returncode"] == 0 for row in results.values()) else "failures_preserved"})


def inference_rows(source, raw, units, records, rho):
    rows, max_delta = [], 0.
    for record in records:
        if units is None:
            rows.append({"plaintext": None, "log_likelihood": None, "joint_log_probability": None})
            continue
        row = decode_units(source, units, record, rho).to_dict()
        reference = independent.infer_record(raw, units, record, rho)
        if (row["plaintext"] is None) != (reference["plaintext"] is None):
            raise ValueError("Independent support disagrees")
        if row["plaintext"] is not None:
            replay_joint = independent.reading_log_probability(raw, units, row["plaintext"], record, rho)
            max_delta = max(max_delta, abs(row["log_likelihood"] - reference["log_likelihood"]),
                            abs(row["joint_log_probability"] - reference["joint_log_probability"]),
                            abs(replay_joint - reference["joint_log_probability"]))
        rows.append(row)
    if max_delta > 1e-7:
        raise ValueError("Independent selected reading score differs")
    return rows, max_delta


def decision_gates(cases, complete):
    positive = [cases[name] for name in CASE_NAMES if not name.endswith("-shuffle")]
    rates = [row["learned"]["transfer"]["edits"] / row["learned"]["transfer"]["gold_characters"] for row in positive]
    oracle = [row["oracle"]["transfer"]["edits"] / row["oracle"]["transfer"]["gold_characters"] for row in positive]
    baseline = [row["frequency"]["transfer"]["edits"] / row["frequency"]["transfer"]["gold_characters"] for row in positive]
    gates = {"all_jobs_complete": complete, "each_transfer_cer_at_most_005": all(x <= .05 for x in rates),
             "macro_transfer_cer_at_most_002": math.fsum(rates) / len(rates) <= .02,
             "each_oracle_excess_at_most_002": all(x - y <= .02 for x, y in zip(rates, oracle, strict=True)),
             "each_beats_frequency": all(x < y for x, y in zip(rates, baseline, strict=True))}
    mean_baseline = math.fsum(baseline) / len(baseline)
    flags = [cases[name]["iid_diagnostic"]["flag"] for name in CASE_NAMES]
    return {"recovery_gates": gates, "recovery_pass": all(gates.values()),
        "macro_transfer_cer": math.fsum(rates) / len(rates), "macro_oracle_cer": math.fsum(oracle) / len(oracle),
        "relative_frequency_reduction": None if mean_baseline == 0 else 1 - math.fsum(rates) / len(rates) / mean_baseline,
        "iid_screen_pass": complete and all(flags[i] == (i % 2 == 0) for i in range(len(flags))),
        "semantic_rejection_qualified": False}


def evaluate(commit):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    def timeout(_s, _f):
        raise TimeoutError("Evaluation exceeded45min wall bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2700)
    started, cpu = time.monotonic(), time.process_time()
    manifest, source1, source, models = inputs(commit)
    directory = ROOT / f"results/{EXPERIMENT}"
    campaign_path = directory / "campaign.json"
    require_frozen(commit, [str(campaign_path.relative_to(ROOT)), *[
        str(selected_path(name).relative_to(ROOT)) for name in CASE_NAMES if selected_path(name).exists()], *[
        str((directory / f"{name}_process.json").relative_to(ROOT)) for name in CASE_NAMES]])
    attempt = directory / "evaluation_started.json"
    if (attempt.exists() or (directory / "evaluation.json").exists()
            or (ROOT / f"outputs/{EXPERIMENT}/predictions.json.gz").exists()):
        raise FileExistsError("Evaluation already completed; no selective repetition")
    campaign_data = json.loads(campaign_path.read_text())
    jobs = {row["case"]: row for row in campaign_data["results"]}
    if set(jobs) != set(CASE_NAMES):
        raise ValueError("Missing campaign cases")
    write_new(attempt, {"key_freeze": commit, "start_unix": time.time()})
    full, cases, max_delta, count, floor_count = {}, {}, 0., 0, 0
    for name in CASE_NAMES:
        case = manifest["cases"][name]
        fit = checked_artifact(case["artifacts"]["fit"])
        context = CodingContext(**fit["context"])
        answer = checked_artifact(case["artifacts"]["answer"])
        successful = jobs[name]["returncode"] == 0
        chosen = json.loads(selected_path(name).read_text()) if successful else None
        if chosen and (chosen["fit_sha256"] != case["artifacts"]["fit"]["sha256"]
                       or chosen["source_selection_sha256"] != digest((ROOT / SELECTION).read_bytes())):
            raise ValueError("Selected dictionary identities differ")
        arms = {"learned": None if chosen is None else tuple(chosen["units"]),
                "stage1": None if chosen is None else tuple(chosen["stage1_units"]),
                "frequency": frequency_units(source1, fit["records"], context)}
        if case["positive"]:
            arms["oracle"] = get_units(Channel.from_dict(answer["gold_channel"]), source.alphabet)
        base = fit_glyph_baseline(fit["records"], context.glyph_alphabet)
        if base != scalar.baseline_from_fit(fit["records"], context.glyph_alphabet) or (chosen and base != chosen["baseline"]):
            raise ValueError("Independent or saved iid baseline differs")
        cases[name], full[name] = {}, {}
        base_ll, glyph_lengths = {}, {}
        for arm, units in arms.items():
            cases[name][arm], full[name][arm] = {}, {}
            for split in ("fit", "transfer"):
                observed = checked_artifact(case["artifacts"][split])
                truth = answer["plaintext"][split] if case["positive"] else None
                rows, delta = inference_rows(source, models["order3"], units, observed["records"], context.stop_probability)
                count += len(rows) if units is not None else 0
                max_delta = max(max_delta, delta)
                value = metrics(rows, truth)
                if truth is not None:
                    edits = [independent.edit_distance(row["plaintext"] or "", gold) for row, gold in zip(rows, truth, strict=True)]
                    if edits != value["record_edits"]:
                        raise ValueError("Independent plaintext edits differ")
                    # Support floors apply to final learned and generating dictionaries only.
                    if arm in ("learned", "oracle"):
                        floors = []
                        for record, gold in zip(observed["records"], truth, strict=True):
                            floor = None if units is None else minimum_unit_edit_distance(source.alphabet, units, record, gold)
                            if units is not None:
                                if floor != independent.dictionary_edit_floor(list(source.alphabet), units, record, gold):
                                    raise ValueError("Independent dictionary floor differs")
                                floor_count += 1
                            floors.append(floor)
                        value["dictionary_floor"] = {"record_edits": floors, "edits": None if None in floors else sum(floors)}
                value["log_likelihood"] = math.fsum(row["log_likelihood"] for row in rows) if all(row["log_likelihood"] is not None for row in rows) else None
                value["model_bits"] = None if units is None else independent.literal_model_bits(units, asdict(context))
                value["total_bits"] = None if value["log_likelihood"] is None else value["model_bits"] - value["log_likelihood"] / math.log(2)
                if arm == "learned" and split == "fit" and chosen:
                    if abs(value["total_bits"] - chosen["score"]["total_bits"]) > 1e-7:
                        raise ValueError("Final selected objective differs")
                cases[name][arm][split], full[name][arm][split] = value, rows
                base_ll[split] = baseline_log_likelihood(base, observed["records"])
                if abs(base_ll[split] - scalar.baseline_score(base, observed["records"])) > 1e-7:
                    raise ValueError("Independent iid likelihood differs")
                glyph_lengths[split] = sum(map(len, observed["records"]))
        learned = cases[name]["learned"]
        margin = None if learned["fit"]["total_bits"] is None else base["model_bits"] - base_ll["fit"] / math.log(2) - (learned["fit"]["total_bits"] + 1)
        gain = None if learned["transfer"]["log_likelihood"] is None else (learned["transfer"]["log_likelihood"] - base_ll["transfer"]) / (math.log(2) * glyph_lengths["transfer"])
        cases[name]["iid_diagnostic"] = {"fit_margin_bits": margin, "transfer_gain_bits_per_glyph": gain,
            "flag": margin is not None and gain is not None and margin >= 32 and gain >= .05,
            "semantic_confidence": False}
    result = {"experiment": EXPERIMENT, "key_freeze": commit, "cases": cases,
        "decision": decision_gates(cases, all(row["returncode"] == 0 for row in jobs.values())),
        "audit": {"record_predictions_replayed": count, "floor_records_replayed": floor_count,
                  "maximum_score_delta": max_delta, "scope": "Root orchestration with previously independently authored numerical references"},
        "predictions": write_new(ROOT / f"outputs/{EXPERIMENT}/predictions.json.gz", full, compressed=True),
        "resources": resource_report(started, cpu)}
    write_new(directory / "evaluation.json", result)
    print(json.dumps(result["decision"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("campaign", "fit", "evaluate"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--case", choices=CASE_NAMES)
    args = parser.parse_args()
    if args.stage == "fit":
        fit_case(args.case, args.freeze)
    elif args.stage == "campaign":
        campaign(args.freeze)
    else:
        evaluate(args.freeze)


if __name__ == "__main__":
    main()
