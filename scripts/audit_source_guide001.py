"""One fixed seeded replay and alternate literal/scalar audit of all 16 cells."""
import gzip
import itertools
import json
import signal
import time

import numpy as np

from scripts.run_source_guide001 import (
    ARRAY_NAMES, OUT, PARTICLES, PATHS, ROOT, SEEDS, array_identity, arrays_identity,
    artifact, comparisons, convert_finite, diagnose, fixtures, limit_resources,
    load_source, reading_bank, require_frozen, resource_report, run_guided_particles, save_new,
)


def audit():
    record = json.loads((OUT/"result.json").read_text())
    require_frozen(record["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": record["freeze"], "start_unix": time.time()})
    limit_resources(1800, 1600)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        source, selected = load_source()
        cases, rows, bulk_bytes = fixtures(source), [], record["fixtures"]["bytes"]
        if (array_identity(source) != record["source_arrays"] or selected["counts"] != record["source"]
                or artifact(ROOT/record["fixtures"]["path"]) != record["fixtures"]
                or json.loads((ROOT/record["fixtures"]["path"]).read_text())
                    != convert_finite({"fixtures": cases, "source": selected["counts"]})
                or len(record["workloads"]) != 16):
            raise ValueError("Fixed source/fixture/input grid changed")
        for spec, (fixture, seed, guidance) in zip(record["workloads"],
                itertools.product(cases, SEEDS, ("none", "iid")), strict=True):
            if artifact(ROOT/spec["path"]) != spec:
                raise ValueError("Cell metadata binding changed")
            row = json.loads((ROOT/spec["path"]).read_text())
            name = f"case{fixture['case']}-{seed}-{guidance}"
            if (row["name"], row["case"], row["seed"], row["guidance"], row["particles"]) != (
                    name, fixture["case"], seed, guidance, PARTICLES):
                raise ValueError("Fixed workload settings changed")
            for key in ("population", "prediction"):
                if artifact(ROOT/row[key]["path"]) != row[key]:
                    raise ValueError("Closed archive binding changed")
                bulk_bytes += row[key]["bytes"]
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered2GiB sampled audit guard")
            value = run_guided_particles(fixture["cipher"], source.probabilities, source.transitions,
                particles=PARTICLES, seed=seed, schedule="balanced", guidance=guidance, observe=observe)
            if value["summary"] != row["summary"] or arrays_identity(value) != row["arrays"]:
                raise ValueError("Seeded summary/array replay changed")
            with np.load(ROOT/row["population"]["path"], allow_pickle=False) as stored:
                if set(stored.files) != set(ARRAY_NAMES):
                    raise ValueError("Wrong array inventory")
                for key in ARRAY_NAMES:
                    np.testing.assert_array_equal(stored[key], value[key])
            bank, decision = reading_bank(value, fixture["cipher"], source)
            saved = json.loads(gzip.decompress((ROOT/row["prediction"]["path"]).read_bytes()))
            if (saved != convert_finite({"summary": value["summary"], "trace": value["trace"],
                    "bank": bank, "decision": decision})
                    or row["diagnostic"] != diagnose(bank, decision, fixture, source, value)
                    or row["diversity"] != convert_finite({k: v for k, v in decision.items() if not k.startswith("decision_")})):
                raise ValueError("Literal/scalar/decision/gold diagnostics changed")
            rows.append(row)
            del value, bank
            print(f"Audited {name}", flush=True)
        if (bulk_bytes != record["bulk_output_bytes"] or comparisons(rows) != record["comparison"]
                or array_identity(source) != record["source_arrays"]):
            raise ValueError("Final comparison/bulk/source accounting changed")
        save_new(OUT/"audit.json", {"status": "PASS_16_seeded_populations_literal_scalar_comparison",
            "result": artifact(OUT/"result.json"), "workloads": 16,
            "resources": resource_report(wall, cpu), "independent_agent_review": False,
            "independent_large_inference_algorithm": False, "paid_spend_usd": 0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json", {"error": repr(exc), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
