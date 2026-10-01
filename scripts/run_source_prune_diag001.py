"""One registered known-answer observational replay of closed searches."""
from __future__ import annotations

import argparse
import gzip
import json
import signal
import time
from pathlib import Path

import numpy as np

from scripts.run_source_state_systems001 import (
    ARMS, PATHS as PARENT_PATHS, ROOT, array_identity, artifact, config as parent_config,
    finite, inputs as parent_inputs, limit_resources, load_source, require_frozen,
    resource_report, save_new,
)
from voynich.source_prune_observer import NativePruneObserver, build_prune_observer, golden_path

EXP = "SOURCE-PRUNE-DIAG-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
PARENT = ROOT/"results/SOURCE-STATE-SYSTEMS-001"
PATHS = sorted(set([*PARENT_PATHS,
    "src/voynich/source_prune_observer.py", "scripts/run_source_prune_diag001.py",
    "scripts/audit_source_prune_diag001.py", "tests/test_source_prune_observer.py",
    "tests/test_source_prune_diag001.py", "docs/experiments/SOURCE-PRUNE-DIAG-001.md",
    "docs/research/source-prune-diagnosis-2026-10-01.md",
    "results/SOURCE-PRUNE-DIAG-001/native-build.json",
    "results/SOURCE-STATE-SYSTEMS-001/result.json", "results/SOURCE-STATE-SYSTEMS-001/audit.json",
    *[f"results/SOURCE-STATE-SYSTEMS-001/case{c}-{a}.json" for c in range(4) for a in ARMS]]))


def config(arm):
    # Only observation overhead allowance; original finite work/width caps unchanged.
    return {**parent_config(arm), "max_seconds": 600.}


def load_bound(spec):
    if artifact(ROOT/spec["path"]) != spec:
        raise ValueError("Closed artifact changed")
    raw = (ROOT/spec["path"]).read_bytes()
    return json.loads(gzip.decompress(raw) if spec["path"].endswith(".gz") else raw)


def prepare():
    build = build_prune_observer(BULK/"build")
    save_new(OUT/"native-build.json", {"build": build,
        "library": artifact(Path(build["library_path"])),
        "generated_cpp": artifact(Path(build["generated_cpp_path"])),
        "no_empirical_search_or_training": True, "paid_spend_usd": 0})


def inputs(freeze):
    require_frozen(freeze, PATHS)
    benchmark, parent, cases, original_build = parent_inputs(freeze)
    closed = json.loads((PARENT/"result.json").read_text())
    audit = json.loads((PARENT/"audit.json").read_text())
    if audit["status"] != "PASS_full_native_state_and_prediction_replay" or audit["result"] != artifact(PARENT/"result.json"):
        raise ValueError("Original complete search audit required")
    build = json.loads((OUT/"native-build.json").read_text())
    load_specs = [build["library"], build["generated_cpp"]]
    if any(artifact(ROOT[s["path"]]) != s for s in load_specs):
        raise ValueError("Observer build changed")
    return benchmark, parent, cases, original_build, closed, build


def validate_trace(trace, cipher, cfg):
    """Finite/monotone path accounting; never turn an alias into survival."""
    horizon = sum(map(len, cipher))
    if not trace or not all(d["processed"] for d in trace):
        raise ValueError("All known-answer states must be visited by this full replay")
    alive, last = True, -1
    for d in trace:
        layer = d["target_glyph_layer"]
        if not last < layer <= horizon or d["layer"] != layer or bool(d["alive_before"]) != alive:
            raise ValueError("Invalid path/layer accounting")
        if d["alive_before"]:
            if not d["path_present"] or d["state_log_mass"] < d["prefix_log_mass"]-1e-7:
                raise ValueError("Surviving literal prefix missing or has insufficient aggregate mass")
            if layer < horizon:
                if d["pre_rank"] < 1 or bool(d["pre_kept"]) != (d["pre_rank"] <= (cfg["guide_prewidth"] if cfg["guidance"] == "iid" else d["incoming"])):
                    raise ValueError("Preliminary beam rank disagrees")
                if d["pre_kept"] and bool(d["final_kept"]) != (d["guide_rank"] <= cfg["width"]):
                    raise ValueError("Final beam rank disagrees")
                if bool(d["alive_after"]) != bool(d["final_kept"]) or bool(d["path_expanded"]) != bool(d["alive_after"]):
                    raise ValueError("Full replay must expand every kept known-answer prefix")
        elif d["alive_after"] or d["path_present"] or d["path_expanded"]:
            raise ValueError("Dead literal prefix was revived by an alias")
        alive, last = bool(d["alive_after"]), layer
    if last != horizon:
        raise ValueError("Terminal observer absent")


def summarize_trace(trace):
    loss = next((d for d in trace if d["alive_before"] and not d["alive_after"]), None)
    terminal = trace[-1]
    cause = ("geometric_prebeam" if not loss["pre_kept"] else "final_beam") if loss else None
    return {"literal_path_survived": bool(terminal["alive_after"]), "first_loss_stage": cause,
        "first_loss": loss, "processed_gold_states": sum(d["processed"] for d in trace),
        "post_loss_structural_alias_layers": sum(bool(d["aliases"]) and not d["alive_before"] for d in trace),
        "terminal_exact_partial_key_rank": terminal["terminal_group_rank"] or None,
        "terminal_used_mapping_best_rank": terminal["terminal_used_mapping_best_rank"] or None,
        "terminal_used_mapping_returned": bool(terminal["terminal_used_mapping_returned"])}


def guard(wall, cpu, bulk_bytes):
    if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3 or bulk_bytes > 128*1024**2:
        raise MemoryError("Registered sampled2GiB host /128MiB bulk bound")


def run(freeze):
    _, parent, cases, _, closed, build = inputs(freeze)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(1800, 1600)
    wall, cpu, rows, size = time.monotonic(), time.process_time(), [], 0
    try:
        source, selected = load_source()
        identity = array_identity(source)
        if selected["counts"] != closed["source"] or identity != closed["source_arrays"] or parent["fixtures"] != closed["fixtures"]:
            raise ValueError("Frozen source/fixtures changed")
        engine = NativePruneObserver(source.probabilities, source.transitions, build["build"])
        for fixture in cases:
            for arm in ARMS:
                name = f"case{fixture['case']}-{arm}"
                baseline_spec = closed["workloads"][len(rows)]
                baseline_row = load_bound(baseline_spec)
                baseline = load_bound(baseline_row["fitted"])
                if baseline["stop_reason"] != "frontier_exhausted" or baseline_row["name"] != name:
                    raise ValueError("Only full frozen frontiers admitted")
                cell_wall, cell_cpu = time.monotonic(), time.process_time()
                control, trace = engine.observe(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], **config(arm))
                if finite(control) != baseline:
                    raise ValueError("Instrumentation changed ordinary search output")
                validate_trace(trace, fixture["cipher"], config(arm))
                targets, _ = golden_path(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], source.probabilities, source.transitions)
                states = [list(map(int, targets[d["target_glyph_layer"]])) for d in trace]
                archive = save_new(BULK/f"{name}.json.gz", {"control": finite(control), "trace": trace, "gold_states": states}, compressed=True)
                size += archive["bytes"]
                guard(wall, cpu, size)
                row = {"name": name, "case": fixture["case"], "arm": arm, "config": config(arm),
                    "original_cell": baseline_spec, "diagnostic": archive, "summary": summarize_trace(trace),
                    "ordinary_output_exact_match": True, "wall_seconds": time.monotonic()-cell_wall,
                    "cpu_seconds": time.process_time()-cell_cpu}
                save_new(OUT/f"{name}.json", row)
                rows.append(row)
                s = row["summary"]
                print(f"{name}: survival={s['literal_path_survived']}, first_loss={s['first_loss_stage']}, key_rank={s['terminal_used_mapping_best_rank']}, {row['wall_seconds']:.2f}s", flush=True)
        if len(rows) != 16 or array_identity(source) != identity:
            raise ValueError("Fixed inventory/source changed")
        save_new(OUT/"result.json", {"experiment": EXP, "freeze": freeze,
            "parent_result": artifact(PARENT/"result.json"), "parent_audit": artifact(PARENT/"audit.json"),
            "source": selected["counts"], "source_arrays": identity, "fixtures": parent["fixtures"],
            "observer_build": artifact(OUT/"native-build.json"),
            "workloads": [artifact(OUT/f"{r['name']}.json") for r in rows],
            "ordinary_output_exact_match_cells": 16, "bulk_output_bytes": size,
            "numpy_version": np.__version__, "resources": resource_report(wall, cpu),
            "paid_spend_usd": 0, "scope": "Known-answer observation of exposed synthetic searches; no new recovery qualification, gold reinsertion or neural circuit claim."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_workloads": len(rows), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.freeze:
        run(args.freeze)
    else:
        parser.error("Frozen commit required")
