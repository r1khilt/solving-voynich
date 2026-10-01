"""Registered fresh synthetic comparison of future guidance and matched control."""
from __future__ import annotations

import argparse
import itertools
import math
import signal
import time

import numpy as np

from scripts.run_source_particle_systems001 import (
    ARRAY_NAMES, PATHS as PARENT_PATHS, array_identity, arrays_identity, artifact,
    convert_finite, load_source, reading_bank, recovery_diagnostics, require_frozen,
    resource_report, save_new,
)
from scripts.run_latin_source_model001 import ROOT, limit_resources
from scripts.source_particle_gap_diag import source_log_mass
from voynich.guided_source_particles import run_guided_particles
from voynich.joint_key_proposal import unit_pool

EXP = "SOURCE-GUIDE-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
DESIGNS, SEEDS, PARTICLES = (64, 64, 224, 224), (75511, 75513), 512
PATHS = [*PARENT_PATHS, "scripts/source_particle_gap_diag.py",
    "src/voynich/guided_source_particles.py", "scripts/run_source_guide001.py",
    "scripts/audit_source_guide001.py", "tests/test_guided_source_particles.py",
    "docs/experiments/SOURCE-GUIDE-001.md",
    "docs/research/source-guidance-2026-10-01.md"]


def fixtures(source):
    pool, found = unit_pool(6), []
    for case, length in enumerate(DESIGNS):
        rng = np.random.default_rng(75501+case)
        key = tuple(map(int, rng.integers(42, size=23)))
        texts = []
        for _ in range(2):
            state, text = 0, []
            for _ in range(length):
                letter = int(rng.choice(23, p=source.row(state)))
                text.append(letter)
                state = source.step(state, letter)
            texts.append(tuple(text))
        texts = tuple(texts)
        cipher = tuple(tuple(g for row in text for g in pool[key[row]]) for text in texts)
        found.append({"case": case, "kind": "markov", "generation_length": length,
            "seed": 75501+case, "generation_source": texts, "generation_key": key, "cipher": cipher})
    return found


def diagnose(bank, decision, fixture, source, result):
    # Called only after prediction/population are serialized. No answer enters guide.
    diagnostic = recovery_diagnostics(decision, fixture)
    truth = fixture["generation_source"]
    used = {r for text in truth for r in text}
    key = tuple(fixture["generation_key"][r] if r in used else -1 for r in range(23))
    gold = source_log_mass(truth, source, 1/225)-len(used)*math.log(42)
    present = any(tuple(map(tuple, b["source_records"])) == truth and tuple(b["used_key"]) == key for b in bank)
    best = max((b["literal_log_mass"] for b in bank), default=-math.inf)
    estimate = result["summary"]["log_evidence_estimate"]
    diagnostic.update({"gold_used_leaf_log_mass": gold, "gold_used_leaf_in_bank": present,
        "gold_reading_in_bank": any(tuple(map(tuple, b["source_records"])) == truth for b in bank),
        "best_visited_leaf_log_mass": convert_finite(best),
        "gold_minus_best_visited_leaf_nats": convert_finite(gold-best),
        "gold_leaf_minus_log_evidence_estimate": None if estimate is None else gold-estimate,
        "gold_global_optimality_proven": False, "complete_reading_mass_not_computed": True})
    return diagnostic


def comparisons(rows):
    paired = []
    for case, seed in itertools.product(range(4), SEEDS):
        a, b = [next(r for r in rows if (r["case"], r["seed"], r["guidance"]) == (case, seed, g))
                for g in ("none", "iid")]
        da, db = a["diagnostic"], b["diagnostic"]
        valid = da["best_visited_leaf_log_mass"] is not None and db["best_visited_leaf_log_mass"] is not None
        paired.append({"case": case, "seed": seed,
            "edit_reduction": da["edit_errors"]-db["edit_errors"],
            "guided_best_leaf_gain_nats": (db["best_visited_leaf_log_mass"]-da["best_visited_leaf_log_mass"]) if valid else None})
    edits = {g: sum(r["diagnostic"]["edit_errors"] for r in rows if r["guidance"] == g) for g in ("none", "iid")}
    clauses = {"all_16_complete": all(r["summary"]["status"] == "complete_particles" for r in rows),
        "aggregate_edit_reduction_at_least_10_percent": edits["none"] > 0 and edits["iid"] <= .9*edits["none"],
        "better_best_leaf_in_at_least_6_of_8_pairs": sum(p["guided_best_leaf_gain_nats"] is not None
            and p["guided_best_leaf_gain_nats"] > 0 for p in paired) >= 6}
    return {"pairs": paired, "edit_totals": edits, "exploratory_signal_clauses": clauses,
        "exploratory_signal": "SUPPORTED" if all(clauses.values()) else "NOT_SUPPORTED",
        "fresh_recovery_qualification": False, "pairs_not_independent_corpora": True}


def run(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(1800, 1600)
    wall, cpu, rows, bulk_bytes = time.monotonic(), time.process_time(), [], 0
    BULK.mkdir(parents=True, exist_ok=True)
    try:
        source, selected = load_source()
        identity, cases = array_identity(source), fixtures(source)
        inputs = save_new(BULK/"fixtures.json", {"fixtures": cases, "source": selected["counts"]})
        bulk_bytes += inputs["bytes"]
        for fixture, seed, guidance in itertools.product(cases, SEEDS, ("none", "iid")):
            name = f"case{fixture['case']}-{seed}-{guidance}"
            cell_wall, cell_cpu = time.monotonic(), time.process_time()
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered 2GiB sampled host guard")
            result = run_guided_particles(fixture["cipher"], source.probabilities, source.transitions,
                particles=PARTICLES, seed=seed, schedule="balanced", guidance=guidance, observe=observe)
            bank, decision = reading_bank(result, fixture["cipher"], source)
            with (BULK/f"{name}.npz").open("xb") as handle:
                np.savez_compressed(handle, **{k: result[k] for k in ARRAY_NAMES})
            population = artifact(BULK/f"{name}.npz")
            prediction = save_new(BULK/f"{name}.json.gz", convert_finite({"summary": result["summary"],
                "trace": result["trace"], "bank": bank, "decision": decision}), compressed=True)
            bulk_bytes += population["bytes"]+prediction["bytes"]
            if bulk_bytes > 512*1024**2:
                raise MemoryError("Registered 512MiB bulk guard")
            observe({})
            row = {"name": name, "case": fixture["case"], "seed": seed, "guidance": guidance,
                "particles": PARTICLES, "population": population, "prediction": prediction,
                "arrays": arrays_identity(result), "summary": result["summary"],
                "diversity": {k: v for k, v in decision.items() if not k.startswith("decision_")},
                "diagnostic": diagnose(bank, decision, fixture, source, result),
                "cell_wall_seconds": time.monotonic()-cell_wall, "cell_cpu_seconds": time.process_time()-cell_cpu}
            save_new(OUT/f"{name}.json", convert_finite(row))
            rows.append(row)
            print(f"{name}: {row['diagnostic']['edit_errors']} edits; {row['summary']['guide_tables_built']} guide tables; {row['cell_wall_seconds']:.2f}s", flush=True)
            del result, bank
        if array_identity(source) != identity or len(rows) != 16:
            raise ValueError("Immutable source/fixed16 grid changed")
        save_new(OUT/"result.json", convert_finite({"experiment": EXP, "freeze": freeze,
            "source": selected["counts"], "source_arrays": identity, "fixtures": inputs,
            "numpy_version": np.__version__, "bulk_output_bytes": bulk_bytes,
            "workloads": [artifact(OUT/f"{r['name']}.json") for r in rows],
            "comparison": comparisons(rows), "resources": resource_report(wall, cpu),
            "paid_spend_usd": 0, "scope": "Fresh forced-length synthetic exploration; no historical or mechanism claim."}))
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"freeze": freeze, "error": repr(exc),
            "completed_workloads": len(rows), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
