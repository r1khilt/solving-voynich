"""One bounded exact RNG/array replay plus alternate literal/scalar checks."""
import gzip
import itertools
import json
import signal
import time

import numpy as np

from scripts.run_source_particle_systems001 import (
    ARRAY_NAMES, COUNTS, OUT, PATHS, SEEDS, array_identity, arrays_identity,
    artifact, convert_finite, fixture_grid, load_source, reading_bank,
    recovery_diagnostics, require_frozen, resource_report, run_particles, save_new,
)
from scripts.run_latin_source_model001 import ROOT, limit_resources


def audit():
    record = json.loads((OUT/"result.json").read_text())
    require_frozen(record["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": record["freeze"], "start_unix": time.time()})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        source, selected = load_source()
        fixtures = fixture_grid(source)
        if (artifact(ROOT/record["fixtures"]["path"]) != record["fixtures"]
                or json.loads((ROOT/record["fixtures"]["path"]).read_text())
                    != convert_finite({"fixtures": fixtures, "source": selected["counts"]})
                or array_identity(source) != record["source_arrays"]):
            raise ValueError("Fixed source/fixture RNG/bindings differ")
        grid = list(itertools.product(fixtures, ("sequential", "balanced"), COUNTS, SEEDS))
        if len(grid) != 32 or len(record["workloads"]) != 32:
            raise ValueError("Fixed32-cell grid required")
        for spec, (fixture, schedule, count, seed) in zip(record["workloads"], grid, strict=True):
            if artifact(ROOT/spec["path"]) != spec:
                raise ValueError("Compact workload binding differs")
            row = json.loads((ROOT/spec["path"]).read_text())
            name = f"case{fixture['case']}-{schedule}-{count}-{seed}"
            if (row["name"], row["case"], row["schedule"], row["particles"], row["seed"]) != (
                    name, fixture["case"], schedule, count, seed):
                raise ValueError("Ordered workload settings differ")
            for key in ("population", "prediction"):
                if artifact(ROOT/row[key]["path"]) != row[key]:
                    raise ValueError("Closed particle archive changed")
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("2GiB sampled particle audit host guard")
            replay = run_particles(fixture["cipher"], source.probabilities, source.transitions,
                particles=count, seed=seed, schedule=schedule, observe=observe)
            if arrays_identity(replay) != row["arrays"] or replay["summary"] != row["summary"]:
                raise ValueError("All seeded particle arrays/summary must replay exactly")
            with np.load(ROOT/row["population"]["path"], allow_pickle=False) as stored:
                if set(stored.files) != set(ARRAY_NAMES):
                    raise ValueError("Exact population array inventory required")
                for key in ARRAY_NAMES:
                    np.testing.assert_array_equal(stored[key], replay[key])
            bank, decision = reading_bank(replay, fixture["cipher"], source)
            prediction = json.loads(gzip.decompress((ROOT/row["prediction"]["path"]).read_bytes()))
            if prediction != convert_finite({"summary": replay["summary"], "trace": replay["trace"],
                    "bank": bank, "decision": decision}):
                raise ValueError("Every ancestry/literal/scalar/prior/decision output must replay")
            if row["diagnostic"] != recovery_diagnostics(decision, fixture):
                raise ValueError("Exploratory synthetic diagnostic differs")
            if row["diversity"] != convert_finite({k: v for k, v in decision.items() if not k.startswith("decision_")}):
                raise ValueError("Population diversity accounting differs")
            del replay, bank
        if array_identity(source) != record["source_arrays"]:
            raise ValueError("Source arrays mutated")
        save_new(OUT/"audit.json", {"status": "PASS_32_seeded_populations_and_literal_scalar_replay",
            "result": artifact(OUT/"result.json"), "workloads": 32,
            "resources": resource_report(wall, cpu), "independent_agent_review": False,
            "independent_large_inference_algorithm": False, "paid_spend_usd": 0})
        print("All32 fixed particle populations audited", flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
