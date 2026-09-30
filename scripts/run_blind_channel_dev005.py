"""Fixed-budget source-context dictionary refinement and post-freeze replay."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scripts import audit_blind_channel_dev004 as independent
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import (
    CASE_NAMES, MANIFEST, SELECTION, get_units, load_archive, metrics, resource_report, save_new,
)
from voynich.finite_state_channel import Channel
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import MarkovSource, decode_units
from voynich.higher_order_unit_refine import refine_units
from voynich.unit_channel_edit_floor import minimum_unit_edit_distance
from voynich.unit_channel_search import channel_from_units


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-005"
SOURCE_PATHS = ["scripts/run_blind_channel_dev005.py", "scripts/run_blind_channel_dev004.py",
    "scripts/run_blind_channel_dev001.py", "scripts/audit_blind_channel_dev004.py",
    "scripts/evaluate_naibbe001.py", "scripts/build_blind_channel_dev001.py",
    "src/voynich/higher_order_unit_channel.py", "src/voynich/higher_order_unit_refine.py",
    "src/voynich/finite_state_channel.py", "src/voynich/finite_state_channel_fit.py",
    "src/voynich/unit_channel_edit_floor.py", "src/voynich/unit_channel_search.py",
    "src/voynich/batched_unit_channel.py", "docs/experiments/BLIND-CHANNEL-DEV-005.md",
    "tests/test_higher_order_unit_refine.py", "tests/test_run_blind_channel_dev005.py", MANIFEST, SELECTION]


def selected_path(name: str) -> Path:
    return ROOT / f"results/{EXPERIMENT}/{name}_freeze.json"


def inputs(commit: str):
    require_frozen(commit, [*SOURCE_PATHS, *[
        f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json" for name in CASE_NAMES]])
    if digest((ROOT / MANIFEST).read_bytes()) != "8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea":
        raise ValueError("Original case manifest changed")
    if digest((ROOT / SELECTION).read_bytes()) != "1a0136627f04e3b68b7e2b3d801e5d8cdebc617fc1e5f3f0fa7b7e344ab74c60":
        raise ValueError("Source-only selection changed")
    selection = json.loads((ROOT / SELECTION).read_text())
    source_raw = load_archive(selection["models"])["models"][selection["primary_model"]]
    if source_raw["order"] != 3:
        raise ValueError("Registered source order changed")
    return json.loads((ROOT / MANIFEST).read_text()), MarkovSource.from_dict(source_raw), source_raw


def trace_accounting(trace: dict, source_raw: dict, context: CodingContext) -> dict:
    """Literal move/coverage/score audit; not all-neighbor likelihood replay."""
    alphabet, glyphs = source_raw["alphabet"], context.glyph_alphabet
    import itertools
    pool = tuple("".join(x) for length in range(1, context.max_emission_length + 1)
                 for x in itertools.product(glyphs, repeat=length))
    initial = tuple(trace["initial_units"])
    parent, best, best_score = initial, initial, None
    total, completed, final_certificate = 0, 0, False
    expected_parent_score = None
    if len(trace["trace"]) > trace["config"]["max_sweeps"]:
        raise ValueError("Trace exceeds registered sweep count")

    def check_score(units, score):
        if score is None:
            return
        if len(units) != len(alphabet) or any(unit not in pool for unit in units):
            raise ValueError("Trace dictionary outside registered family")
        bits = independent.literal_model_bits(tuple(units), asdict(context))
        if (score["model_bits"] != bits or not math.isfinite(score["log_likelihood"])
                or score["log_likelihood"] > 1e-10
                or abs(score["data_bits"] + score["log_likelihood"] / math.log(2)) > 1e-8
                or abs(score["total_bits"] - bits - score["data_bits"]) > 1e-8):
            raise ValueError("Trace objective arithmetic or literal cost mismatch")

    for index, sweep in enumerate(trace["trace"]):
        if sweep["sweep"] != index or tuple(sweep["parent_units"]) != parent:
            raise ValueError("Trace parent trajectory mismatch")
        if expected_parent_score is not None and sweep["parent_score"] != expected_parent_score:
            raise ValueError("Trace parent score changed between sweeps")
        check_score(parent, sweep["parent_score"])
        if best_score is None:
            best_score = sweep["parent_score"]
        expected = []
        for left in range(len(parent)):
            for right in range(left + 1, len(parent)):
                if parent[left] != parent[right]:
                    changed = list(parent)
                    changed[left], changed[right] = changed[right], changed[left]
                    expected.append(({"kind": "swap", "rows": [left, right]}, tuple(changed)))
        for row in range(len(parent)):
            for unit in pool:
                if unit != parent[row]:
                    changed = list(parent)
                    changed[row] = unit
                    expected.append(({"kind": "replace", "row": row, "unit": unit}, tuple(changed)))
        neighbors = sweep["neighbors"]
        if (sweep["expected_neighbors"] != len(expected) or len(neighbors) > len(expected)
                or sweep["complete"] != (len(neighbors) == len(expected))):
            raise ValueError("Trace neighborhood count mismatch")
        selected, selected_score = parent, sweep["parent_score"]
        for row, (move, units) in zip(neighbors, expected, strict=False):
            if row["move"] != move or tuple(row["units"]) != units:
                raise ValueError("Trace neighborhood order or move mismatch")
            check_score(units, row["score"])
            if row["score"] is not None and row["score"]["total_bits"] < selected_score["total_bits"]:
                selected, selected_score = units, row["score"]
            if row["score"] is not None and row["score"]["total_bits"] < best_score["total_bits"]:
                best, best_score = units, row["score"]
        accepted = sweep["parent_score"]["total_bits"] - selected_score["total_bits"] > trace["config"]["tolerance"]
        if (tuple(sweep["selected_units"]) != selected or sweep["selected_score"] != selected_score
                or sweep["accepted"] != accepted):
            raise ValueError("Trace selected candidate mismatch")
        total += len(neighbors)
        completed += int(sweep["complete"])
        if not sweep["complete"] or not accepted:
            if index != len(trace["trace"]) - 1:
                raise ValueError("Trace continues past stop condition")
            final_certificate = sweep["complete"] and not accepted and best == parent
            expected_stop = "local_optimum" if sweep["complete"] else "time_limit"
            if trace["stop_reason"] != expected_stop:
                raise ValueError("Trace stop reason contradicts final sweep")
        if accepted and sweep["complete"]:
            parent = selected
            expected_parent_score = selected_score
    if best_score is None:
        best_score = trace["score"]
        check_score(best, best_score)
    if (tuple(trace["units"]) != best or trace["score"] != best_score
            or trace["evaluated_neighbors"] != total or trace["completed_sweeps"] != completed
            or trace["best_is_certified_local_optimum"] != final_certificate):
        raise ValueError("Trace final retention/certificate mismatch")
    return {"neighbors_accounted": total, "complete_sweeps": completed,
            "final_local_certificate": final_certificate,
            "all_neighbor_likelihoods_independently_replayed": False}


def fit(name: str, commit: str):
    resource.setrlimit(resource.RLIMIT_CPU, (360, 360))
    start, cpu = time.monotonic(), time.process_time()
    if name not in CASE_NAMES or selected_path(name).exists():
        raise ValueError("Invalid case or selected key already exists")
    manifest, source, source_raw = inputs(commit)
    case = manifest["cases"][name]
    observed = checked_artifact(case["artifacts"]["fit"])
    context = CodingContext(**observed["context"])
    warm_path = ROOT / f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json"
    warm = json.loads(warm_path.read_text())
    if warm["input_sha256"] != case["artifacts"]["fit"]["sha256"]:
        raise ValueError("Warm-start fit identity differs")
    initial = get_units(Channel.from_dict(warm["channel"]), source.alphabet)
    trace = refine_units(source, observed["records"], context, initial, max_sweeps=20, max_seconds=300.)
    accounting = trace_accounting(trace, source_raw, context)
    # Independent backward reference replays only the final dictionary, fit only.
    replay = math.fsum(independent.infer_record(source_raw, tuple(trace["units"]), row,
                                               context.stop_probability)["log_likelihood"] for row in observed["records"])
    delta = abs(replay - trace["score"]["log_likelihood"])
    if delta > 1e-7:
        raise ValueError("Independent final fit likelihood mismatch")
    archive = save_new(ROOT / f"outputs/{EXPERIMENT}/{name}.json.gz", trace, compressed=True)
    save_new(selected_path(name), {"experiment": EXPERIMENT, "case_id": name, "source_freeze": commit,
        "source_selection_sha256": digest((ROOT / SELECTION).read_bytes()),
        "fit_sha256": case["artifacts"]["fit"]["sha256"], "warm_start_sha256": digest(warm_path.read_bytes()),
        "channel": channel_from_units(source, context, trace["units"]).to_dict(), "units": trace["units"],
        "score": trace["score"], "stop_reason": trace["stop_reason"], "trace": archive,
        "accounting": accounting, "independent_final_fit_delta": delta,
        "resources": resource_report(start, cpu)})
    print(json.dumps({"case": name, "neighbors": trace["evaluated_neighbors"], "stop": trace["stop_reason"]}))


def campaign(commit: str):
    inputs(commit)
    if any(selected_path(name).exists() for name in CASE_NAMES):
        raise FileExistsError("Refinement keys already exist; no selective rerun")
    status_path = ROOT / f"results/{EXPERIMENT}/campaign.json"
    if status_path.exists():
        raise FileExistsError("Campaign already attempted")
    start = time.monotonic()

    def launch(name):
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=".:src")
        try:
            result = subprocess.run([sys.executable, "scripts/run_blind_channel_dev005.py", "fit", "--case", name,
                                     "--freeze", commit], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=420)
            return {"case": name, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"case": name, "returncode": None, "status": "parent_wall_timeout"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(launch, CASE_NAMES))
    payload = {"experiment": EXPERIMENT, "source_freeze": commit, "workers": 2,
               "results": results, "wall_seconds": time.monotonic() - start,
               "status": "complete" if all(row["returncode"] == 0 for row in results) else "failed"}
    save_new(status_path, payload)
    print(json.dumps(payload))
    if payload["status"] != "complete":
        raise RuntimeError("Campaign failure preserved; no replacement runs")


def evaluate(commit: str):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
    def timeout(_signum, _frame):
        raise TimeoutError("Evaluation exceeded 20-minute wall bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1200)
    started, cpu = time.monotonic(), time.process_time()
    manifest, source, source_raw = inputs(commit)
    require_frozen(commit, [str(selected_path(name).relative_to(ROOT)) for name in CASE_NAMES])
    result = {"experiment": EXPERIMENT, "key_freeze": commit, "cases": {}}
    full, max_delta, record_count, floor_count, ties = {}, 0., 0, 0, 0
    for name in CASE_NAMES:
        case = manifest["cases"][name]
        chosen = json.loads(selected_path(name).read_text())
        if (chosen["fit_sha256"] != case["artifacts"]["fit"]["sha256"]
                or chosen["source_selection_sha256"] != digest((ROOT / SELECTION).read_bytes())):
            raise ValueError("Selected dictionary input identity mismatch")
        answer = checked_artifact(case["artifacts"]["answer"])
        channels = {"learned": Channel.from_dict(chosen["channel"])}
        if case["positive"]:
            channels["oracle"] = Channel.from_dict(answer["gold_channel"])
        report, predictions = {}, {}
        for arm, channel in channels.items():
            units = get_units(channel, source.alphabet)
            report[arm], predictions[arm] = {}, {}
            for split in ("fit", "transfer"):
                data = checked_artifact(case["artifacts"][split])
                context = CodingContext(**data["context"])
                truth = answer["plaintext"][split] if case["positive"] else None
                rows, floors = [], []
                for index, record in enumerate(data["records"]):
                    row = decode_units(source, units, record, channel.stop_probability).to_dict()
                    other = independent.infer_record(source_raw, units, record, channel.stop_probability)
                    if (row["plaintext"] is None) != (other["plaintext"] is None):
                        raise ValueError("Independent record support mismatch")
                    if row["plaintext"] is not None:
                        for field in ("log_likelihood", "joint_log_probability"):
                            max_delta = max(max_delta, abs(row[field] - other[field]))
                        replay_joint = independent.reading_log_probability(source_raw, units, row["plaintext"], record, channel.stop_probability)
                        max_delta = max(max_delta, abs(replay_joint - other["joint_log_probability"]))
                    ties += int(row["plaintext"] != other["plaintext"])
                    record_count += 1
                    if max_delta > 1e-7:
                        raise ValueError("Independent prediction score mismatch")
                    if truth is not None:
                        floor = minimum_unit_edit_distance(source.alphabet, units, record, truth[index])
                        if floor != independent.dictionary_edit_floor(list(source.alphabet), units, record, truth[index]):
                            raise ValueError("Independent dictionary floor mismatch")
                        floors.append(floor)
                        floor_count += 1
                    rows.append(row)
                value = metrics(rows, truth)
                if truth is not None:
                    checked_edits = [independent.edit_distance(row["plaintext"] or "", gold) for row, gold in zip(rows, truth, strict=True)]
                    if value["record_edits"] != checked_edits:
                        raise ValueError("Independent edit distance mismatch")
                    value["dictionary_floor"] = {"record_edits": floors,
                        "edits": None if any(floor is None for floor in floors) else sum(floors)}
                value["log_likelihood"] = (math.fsum(row["log_likelihood"] for row in rows)
                                           if all(row["log_likelihood"] is not None for row in rows) else None)
                value["model_bits"] = independent.literal_model_bits(units, asdict(context))
                value["total_bits"] = (value["model_bits"] - value["log_likelihood"] / math.log(2)
                                       if value["log_likelihood"] is not None else None)
                if arm == "learned" and split == "fit":
                    if abs(value["total_bits"] - chosen["score"]["total_bits"]) > 1e-7:
                        raise ValueError("Selected fit objective replay mismatch")
                report[arm][split], predictions[arm][split] = value, rows
        result["cases"][name] = report
        full[name] = predictions
    result["audit"] = {"record_predictions": record_count, "floor_records": floor_count,
                        "maximum_score_delta": max_delta, "alternative_tied_maps": ties,
                        "scope": "Root orchestration using frozen independently authored inference, edit and coding references"}
    result["resources"] = resource_report(started, cpu)
    result["predictions"] = save_new(ROOT / f"outputs/{EXPERIMENT}/predictions.json.gz", full, compressed=True)
    save_new(ROOT / f"results/{EXPERIMENT}/evaluation.json", result)
    print(json.dumps({name: {arm: row["transfer"]["edits"] for arm, row in report.items()}
                      for name, report in result["cases"].items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("campaign", "fit", "evaluate"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.case, args.freeze)
    elif args.stage == "campaign":
        campaign(args.freeze)
    else:
        evaluate(args.freeze)


if __name__ == "__main__":
    main()
