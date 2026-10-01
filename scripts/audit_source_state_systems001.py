"""One complete closed-artifact/state-engine replay; alternate tiny laws in tests."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import signal
import time

import numpy as np

from scripts.run_source_state_systems001 import (
    ARMS, OUT, ROOT, NativeMarginal, NativeStateLattice, array_identity,
    artifact, config, diagnose, finite, guard, inputs, limit_resources, load_source,
    predict, resource_report, save_new, summarize,
)
from voynich.native_suffix_marginal import marginal_python


def replay_config(stored, arm):
    values = config(arm)
    if stored["stop_reason"] == "time_cap":
        # The timer was checked after both pruning stages, before a WHOLE
        # glyph layer. Stop at that same expansion count, not a new time draw.
        if stored["expanded"] < 1:
            raise ValueError("Zero-expansion timed call cannot receive full prefix replay")
        values["max_expanded"] = stored["expanded"]
        values["max_seconds"] = 600.
    return values


def compare_replay(stored, actual):
    if stored["stop_reason"] == "time_cap":
        if actual["stop_reason"] != "expansion_cap":
            raise ValueError("Timed-prefix replay failed to stop at its frozen layer")
        actual = {**actual, "stop_reason": "time_cap"}
    if stored != finite(actual):
        raise ValueError("Full native layer/terminal/mass replay differs")


def audit():
    result = json.loads((OUT/"result.json").read_text())
    benchmark, parent, cases, build = inputs(result["freeze"])
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time()})
    limit_resources(3600, 3200)
    wall, cpu, rows, bulk_bytes, checked, max_delta = time.monotonic(), time.process_time(), [], 0, 0, 0.
    try:
        source, selected = load_source()
        assert selected["counts"] == result["source"] and array_identity(source) == result["source_arrays"]
        assert parent["fixtures"] == result["fixtures"] and benchmark["build"] == result["native_build"]
        assert artifact(OUT/"native-build.json") == result["state_build"]
        assert result["numpy_version"] == np.__version__
        assert result["engineering_execution"] == "all16_returned_with_accounted_caps"
        assert result["recovery_threshold_registered"] is False and result["fresh_recovery_qualification"] is False
        assert result["paid_spend_usd"] == result["resources"]["paid_spend_usd"] == 0
        assert 0 <= result["resources"]["wall_seconds"] <= 3600
        assert 0 <= result["resources"]["cpu_seconds"] <= 3200
        assert 0 < result["resources"]["peak_rss_bytes"] <= 2*1024**3
        engine = NativeStateLattice(source.probabilities, source.transitions, build["build"])
        native = NativeMarginal(source, benchmark["build"])
        for case, fixture in enumerate(cases):
            for arm in ARMS:
                name = f"case{case}-{arm}"
                spec = result["workloads"][len(rows)]
                assert spec == artifact(OUT/f"{name}.json")
                row = json.loads((ROOT/spec["path"]).read_text())
                assert (row["name"], row["case"], row["arm"], row["config"]) == (name, case, arm, config(arm))
                assert 0 <= row["cell_wall_seconds"] <= result["resources"]["wall_seconds"]
                assert 0 <= row["cell_cpu_seconds"] <= result["resources"]["cpu_seconds"]
                assert artifact(ROOT/row["fitted"]["path"]) == row["fitted"]
                assert artifact(ROOT/row["prediction"]["path"]) == row["prediction"]
                stored = json.loads(gzip.decompress((ROOT/row["fitted"]["path"]).read_bytes()))
                replay = engine.search(fixture["cipher"], **replay_config(stored, arm))
                compare_replay(stored, replay)
                assert finite({k: v for k, v in replay.items() if k not in ("terminals", "trace")}) == row["summary"] or stored["stop_reason"] == "time_cap"
                # Regardless of timer handling, directly bind all stored summary fields.
                assert {k: v for k, v in stored.items() if k not in ("terminals", "trace")} == row["summary"]
                prediction = json.loads((ROOT/row["prediction"]["path"]).read_text())
                assert finite(predict(replay, fixture["cipher"], case, source, native)) == prediction
                assert diagnose(stored, prediction, fixture, source) == row["diagnostic"]
                if prediction["status"] == "read":
                    for r, observed in enumerate(fixture["cipher"]):
                        text = "".join("ABCDEF"[g] for g in observed)
                        alternate = marginal_python(source, tuple(prediction["key"]), text, 1/225,
                            max_nodes=2_000_000, max_edges=8_000_000)
                        delta = abs(alternate.log_likelihood-prediction["score"]["record_log_likelihoods"][r])
                        assert math.isfinite(delta) and delta <= 1e-7
                        checked, max_delta = checked+1, max(max_delta, delta)
                bulk_bytes += row["fitted"]["bytes"]+row["prediction"]["bytes"]
                guard(wall, cpu, bulk_bytes)
                rows.append(row)
                print(f"Audited {name}: {stored['expanded']} expansions", flush=True)
        assert len(rows) == len(result["workloads"]) == 16
        assert summarize(rows) == result["arm_totals"] and bulk_bytes == result["bulk_output_bytes"]
        assert array_identity(source) == result["source_arrays"]
        save_new(OUT/"audit.json", {"status": "PASS_full_native_state_and_prediction_replay",
            "result": artifact(OUT/"result.json"), "workloads": 16,
            "alternate_fixed_key_record_scores": checked, "maximum_python_native_delta": max_delta,
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "same_author_and_search_implementation": True, "independent_agent_review": False,
            "independent_full_state_search": False,
            "note": "Tiny full-key/string rational enumeration tests are alternate correctness evidence; realistic state search is same native replay."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json", {"error": repr(exc), "completed_workloads": len(rows),
            "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()
