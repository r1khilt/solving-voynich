"""One finite replay of artificial systems work and alternate scalar checks."""
from __future__ import annotations

import itertools
import json
import math
import signal
import time

from scripts.benchmark_source_prefix_systems001 import (
    BULK, OUT, PATHS, DenseProvider, array_identity, convert_finite, fixtures,
    full_key_reference, load_source, rational_row, verify_dense, verify_tiny,
)
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.source_prefix_inverse import search_source_prefix


def restore_mass(value):
    if isinstance(value, dict):
        return {k: (-math.inf if v is None and k in (
            "found_log_mass", "unresolved_log_mass_upper", "evidence_log_upper") else restore_mass(v))
            for k, v in value.items()}
    if isinstance(value, list):
        return [restore_mass(v) for v in value]
    return value


def audit():
    result = json.loads((OUT/"result.json").read_text())
    require_frozen(result["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time()})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        if artifact(BULK/"workloads.jsonl") != result["trace"] or artifact(BULK/"fixtures.json") != result["fixtures"]:
            raise ValueError("Completed artificial trace/input bindings differ")
        source, selected = load_source()
        generated = fixtures(source)
        stored = json.loads((BULK/"fixtures.json").read_text())
        if stored != convert_finite({"fixtures": generated, "source": selected["counts"]}):
            raise ValueError("Fixed source/key/RNG workload generation differs")
        if selected["counts"] != result["source"] or array_identity(source) != result["array_identity"]:
            raise ValueError("Statistical source/array identity differs")
        rows = [json.loads(s) for s in (BULK/"workloads.jsonl").read_text().splitlines()]
        if len(rows) != 296 or result["tiny_calls"] != 288:
            raise ValueError("Fixed288tiny+8dense workload grid required")
        observed = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
        expected_grid = [(i, bonus, width, budget, cipher)
            for i, cipher in enumerate(itertools.product(observed, repeat=2)) for bonus in (0., 4.)
            for width, budget in ((10000, 10000), (1, 3), (3, 20), (3, 60))]
        for row, (i, bonus, width, budget, cipher) in zip(rows[:288], expected_grid, strict=True):
            if (row["type"], row["case"], row["bonus"], row["width"], row["budget"], row["cipher"]) != (
                    "tiny", i, bonus, width, budget, list(map(list, cipher))):
                raise ValueError("Ordered tiny grid differs")
            reference = full_key_reference(cipher)
            verify_tiny(restore_mass(row["result"]), reference)
            replay = search_source_prefix(cipher, lambda p: tuple(map(float, rational_row(p))), rows=2,
                glyphs=2, rho=.25, max_frontier=width, max_expanded=budget, progress_bonus=bonus)
            if convert_finite(replay) != row["result"]:
                raise ValueError("Tiny complete search/accounting replay differs")
        dense_grid = [(i, bonus) for i in range(4) for bonus in (0., 4.)]
        for row, summary, (i, bonus) in zip(rows[288:], result["dense_results"], dense_grid, strict=True):
            if (row["type"], row["case"], row["bonus"]) != ("dense", i, bonus):
                raise ValueError("Ordered dense grid differs")
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("2GiB sampled audit host guard")
            replay = search_source_prefix(generated[i]["cipher"], DenseProvider(source), max_expanded=5000,
                max_frontier=4096, max_terminals=512, progress_bonus=bonus, observe=observe)
            if convert_finite(replay) != row["result"]:
                raise ValueError("Dense candidate/work/bound replay differs")
            verify_dense(restore_mass(row["result"]), generated[i]["cipher"], source)
            keys = ("expanded", "generated", "source_calls", "pruned_states", "maximum_active_states", "stop_reason", "reading_bound_separated")
            if (any(summary[k] != replay[k] for k in keys) or summary["terminals"] != len(replay["terminals"])
                    or summary["readings"] != len(replay["readings"])):
                raise ValueError("Compact dense accounting differs")
        if array_identity(source) != result["array_identity"]:
            raise ValueError("Source arrays mutated")
        save_new(OUT/"audit.json", {"status": "PASS_artificial_search_and_scalar_replay",
            "workloads": len(rows), "result": artifact(OUT/"result.json"), "trace": result["trace"],
            "resources": resource_report(wall, cpu), "independent_agent_review": False,
            "independent_large_search_algorithm": False, "neural_or_gpu_inference": False})
        print("296 fixed artificial workloads audited", flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
