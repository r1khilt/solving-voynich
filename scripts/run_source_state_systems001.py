"""Registered finite Markov-state search, equivalence and width controls."""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import numpy as np

from scripts.run_source_revise001 import (
    PATHS as PARENT_PATHS, POOL, ROOT, NativeMarginal, array_identity, artifact,
    inputs as parent_inputs, limit_resources, load_source, read_key, require_frozen,
    resource_report, save_new, score_key,
)
from voynich.native_source_state import NativeStateLattice, build_source_state_native
from voynich.unit_channel_decision import edit_distance

EXP = "SOURCE-STATE-SYSTEMS-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
ARMS = ("unmerged", "merged", "guided", "wide_guided")
PATHS = sorted(set([*PARENT_PATHS,
    "src/voynich/source_state_lattice.py", "src/voynich/native_source_state.cpp",
    "src/voynich/native_source_state.py", "scripts/run_source_state_systems001.py",
    "scripts/audit_source_state_systems001.py", "tests/test_source_state_lattice.py",
    "tests/test_native_source_state.py", "tests/test_source_state_systems001.py",
    "docs/experiments/SOURCE-STATE-SYSTEMS-001.md",
    "docs/research/source-state-lattice-2026-10-01.md",
    "results/SOURCE-STATE-SYSTEMS-001/native-build.json",
    "results/SOURCE-STATE-SYSTEMS-001/preparation-repair.json"]))


def config(arm):
    if arm not in ARMS:
        raise ValueError("Unregistered arm")
    wide = arm == "wide_guided"
    return {"glyphs": 6, "rho": 1/225, "width": 4096 if wide else 128,
        "max_expanded": 4_000_000 if wide else 500_000,
        "max_generated": 120_000_000 if wide else 20_000_000,
        "max_active": 500_000 if wide else 100_000, "max_terminals": 512,
        "merge": arm != "unmerged", "schedule": "balanced",
        "guidance": "iid" if arm in ("guided", "wide_guided") else "none",
        "guide_prewidth": 16384 if wide else 512,
        "guide_max_tables": 20_000_000 if wide else 1_000_000,
        "guide_cache_entries": 4096 if wide else 1024,
        "max_seconds": 180. if wide else 120.}


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value != -math.inf:
            raise ArithmeticError("Only zero mass may be encoded as null")
        return None
    return value


def prepare():
    build = build_source_state_native(BULK/"build")
    save_new(OUT/"native-build.json", {"build": build,
        "library": artifact(Path(build["library_path"])), "paid_spend_usd": 0,
        "no_empirical_search_or_training": True})


def inputs(freeze):
    require_frozen(freeze, PATHS)
    benchmark, parent, cases = parent_inputs(freeze)
    build = json.loads((OUT/"native-build.json").read_text())
    if artifact(ROOT/build["library"]["path"]) != build["library"]:
        raise ValueError("Compiled state library changed")
    if len(cases) != 4 or [c["case"] for c in cases] != list(range(4)):
        raise ValueError("Four exposed fixture identities required")
    return benchmark, parent, cases, build


def predict(result, cipher, case, source, native):
    """Cipher-only terminal selection/fill/reader; no gold or true length."""
    if not result["terminals"]:
        return {"status": "no_complete_terminal", "selection": None}
    chosen = result["terminals"][0]
    partial = tuple(chosen["used_key"])
    rng = np.random.default_rng(79101+case)
    fill = tuple(map(int, rng.integers(42, size=23)))
    key = tuple(POOL[fill[r] if k < 0 else k] for r, k in enumerate(partial))
    records = tuple("".join("ABCDEF"[g] for g in record) for record in cipher)
    score = score_key(native, records, key)
    if score["log_key_mass"] is None:
        raise ValueError("Literal terminal key must retain support after unseen-row fill")
    prediction = read_key(source, native, key, records, score)
    return {"status": "read", "selection": "highest_returned_partial_history_mass_then_lexical_key",
        "partial_key": partial, "fill_seed": 79101+case, "filler_indices": fill,
        "key": key, "score": score, "prediction": prediction,
        "terminal_mass_is_not_full_key_mass": True}


def diagnose(result, prediction, fixture, source):
    # Search and prediction archives must be closed before this function.
    truth = fixture["generation_source"]
    used = sorted({r for text in truth for r in text})
    gold = fixture["generation_key"]
    represented = any(all(t["used_key"][r] == gold[r] for r in used) for t in result["terminals"])
    partial_compatible = sum(all(k < 0 or k == gold[r] for r, k in enumerate(t["used_key"]))
                             for t in result["terminals"])
    row = {"true_source_letters": sum(map(len, truth)), "gold_used_rows": len(used),
        "gold_used_mapping_in_returned_terminals": represented,
        "returned_partial_keys_compatible_with_gold": partial_compatible,
        "no_claim_about_unreturned_terminals": True, "edit_errors": None,
        "record_edits": None, "exact_records": 0, "matched_gold_used_rows": None}
    if prediction["status"] == "read":
        target = ["".join(source.alphabet[r] for r in text) for text in truth]
        records = prediction["prediction"]["source_records"]
        row.update({"record_edits": [edit_distance(a, b) for a, b in zip(records, target, strict=True)],
            "exact_records": sum(a == b for a, b in zip(records, target, strict=True)),
            "matched_gold_used_rows": sum(prediction["key"][r] == POOL[gold[r]] for r in used)})
        row["edit_errors"] = sum(row["record_edits"])
    return row


def summarize(rows):
    return {arm: {"cells": sum(r["arm"] == arm for r in rows),
        "cells_with_terminals": sum(r["arm"] == arm and r["summary"]["terminal_states"] > 0 for r in rows),
        "fully_exhaustive_cells": sum(r["arm"] == arm and r["summary"]["complete_search"] for r in rows),
        "expanded": sum(r["summary"]["expanded"] for r in rows if r["arm"] == arm),
        "merged_arrivals": sum(r["summary"]["merged_arrivals"] for r in rows if r["arm"] == arm),
        "returned_gold_used_key_cells": sum(r["diagnostic"]["gold_used_mapping_in_returned_terminals"] for r in rows if r["arm"] == arm),
        "edit_errors_on_read_cells_only": sum(r["diagnostic"]["edit_errors"] or 0 for r in rows if r["arm"] == arm),
        "read_cells": sum(r["arm"] == arm and r["diagnostic"]["edit_errors"] is not None for r in rows),
        "exact_records": sum(r["diagnostic"]["exact_records"] for r in rows if r["arm"] == arm)} for arm in ARMS}


def guard(wall, cpu, bulk_bytes):
    if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3 or bulk_bytes > 256*1024**2:
        raise MemoryError("Registered sampled2GiB host /256MiB bulk bound")


def run(freeze):
    benchmark, parent, cases, build = inputs(freeze)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(3600, 3200)
    wall, cpu, rows, bulk_bytes = time.monotonic(), time.process_time(), [], 0
    try:
        source, selected = load_source()
        identity = array_identity(source)
        engine = NativeStateLattice(source.probabilities, source.transitions, build["build"])
        native = NativeMarginal(source, benchmark["build"])
        for fixture in cases:
            for arm in ARMS:
                case, name = fixture["case"], f"case{fixture['case']}-{arm}"
                cell_wall, cell_cpu = time.monotonic(), time.process_time()
                result = engine.search(fixture["cipher"], **config(arm))
                fitted = save_new(BULK/f"{name}.json.gz", finite(result), compressed=True)
                prediction = predict(result, fixture["cipher"], case, source, native)
                prediction_spec = save_new(BULK/f"{name}-prediction.json", prediction)
                bulk_bytes += fitted["bytes"]+prediction_spec["bytes"]
                guard(wall, cpu, bulk_bytes)
                diagnostic = diagnose(result, prediction, fixture, source)
                summary = finite({k: v for k, v in result.items() if k not in ("terminals", "trace")})
                row = {"name": name, "case": case, "arm": arm, "config": config(arm),
                    "fitted": fitted, "prediction": prediction_spec, "summary": summary,
                    "diagnostic": diagnostic, "cell_wall_seconds": time.monotonic()-cell_wall,
                    "cell_cpu_seconds": time.process_time()-cell_cpu}
                save_new(OUT/f"{name}.json", row)
                rows.append(row)
                print(f"{name}: {result['expanded']} expanded, {result['merged_arrivals']} merges, "
                      f"{result['terminal_states']} terminals, {result['stop_reason']}, "
                      f"edits={diagnostic['edit_errors']}, {row['cell_wall_seconds']:.2f}s", flush=True)
        if len(rows) != 16 or array_identity(source) != identity:
            raise ValueError("Fixed workload/source identity changed")
        save_new(OUT/"result.json", {"experiment": EXP, "freeze": freeze,
            "source": selected["counts"], "source_arrays": identity,
            "fixtures": parent["fixtures"], "native_build": benchmark["build"],
            "state_build": artifact(OUT/"native-build.json"),
            "workloads": [artifact(OUT/f"{r['name']}.json") for r in rows],
            "arm_totals": summarize(rows), "bulk_output_bytes": bulk_bytes,
            "numpy_version": np.__version__,
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "engineering_execution": "all16_returned_with_accounted_caps",
            "recovery_threshold_registered": False, "fresh_recovery_qualification": False,
            "scope": "Exposed synthetic systems/coverage exploration. Equal widths are not equal CPU; no historical reading or global optimum."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_workloads": len(rows),
            "resources": resource_report(wall, cpu), "no_retry": True})
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
