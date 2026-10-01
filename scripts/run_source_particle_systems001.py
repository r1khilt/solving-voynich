"""Frozen CPU particle systems work and exploratory synthetic recovery diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import signal
import time
from collections import defaultdict

import numpy as np

from scripts.benchmark_source_prefix_systems001 import (
    PATHS as PREFIX_PATHS, array_identity, convert_finite, load_source,
)
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.joint_key_proposal import unit_pool
from voynich.source_key_particles import reconstruct, run_particles, terminal_groups
from voynich.source_prefix_inverse import logsum
from voynich.unit_channel_decision import edit_distance

EXP = "SOURCE-PARTICLE-SYSTEMS-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
COUNTS, SEEDS = (512, 4096), (74411, 74413)
DESIGNS = (("repetitive", 64), ("repetitive", 224), ("markov", 64), ("markov", 224))
ARRAY_NAMES = ("parents", "records", "labels", "keys", "scores", "offsets", "contexts", "closed")
PATHS = [*PREFIX_PATHS, "src/voynich/source_key_particles.py",
    "src/voynich/unit_channel_decision.py", "scripts/run_source_particle_systems001.py",
    "scripts/audit_source_particle_systems001.py", "tests/test_source_key_particles.py",
    "docs/experiments/SOURCE-PARTICLE-SYSTEMS-001.md"]


def fixture_grid(source):
    result, pool = [], unit_pool(6)
    for i, (kind, length) in enumerate(DESIGNS):
        rng = np.random.default_rng(74401+i)
        key = tuple(int(v) for v in rng.integers(42, size=23))
        if kind == "repetitive":
            common = int(np.argmax(source.row(0)))
            texts = ((common,)*length,)*2
        else:
            texts = []
            for _ in range(2):
                state, letters = 0, []
                for _ in range(length):
                    letter = int(rng.choice(23, p=source.row(state)))
                    letters.append(letter)
                    state = source.step(state, letter)
                texts.append(tuple(letters))
            texts = tuple(texts)
        cipher = tuple(tuple(g for r in text for g in pool[key[r]]) for text in texts)
        result.append({"case": i, "kind": kind, "generation_length": length,
            "seed": 74401+i, "generation_key": key, "generation_source": texts, "cipher": cipher})
    return result


def arrays_identity(value):
    return {name: {"shape": list(value[name].shape), "dtype": str(value[name].dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value[name])).hexdigest()} for name in ARRAY_NAMES}


def reading_bank(result, cipher, source):
    """Ancestry, literal emission and alternate scalar-state/prior replay."""
    groups, initial_ancestors = terminal_groups(result)
    leaves = {}
    pool = unit_pool(6)
    for representative, count in groups:
        texts = reconstruct(result, representative, len(cipher))
        key = tuple(map(int, result["keys"][representative]))
        used = {r for text in texts for r in text}
        if (any((k >= 0) != (r in used) for r, k in enumerate(key))
                or tuple(tuple(g for r in text for g in pool[key[r]]) for text in texts) != tuple(map(tuple, cipher))):
            raise ValueError("Reconstructed source/shared key differs from ciphertext or used rows")
        terms = [-len(used)*math.log(42)]
        for text in texts:
            terms += [math.log(1/225), len(text)*math.log1p(-1/225)]
            state = 0
            for row in text:
                terms.append(math.log(float(source.row(state)[row])))
                state = source.step(state, row)
        score = math.fsum(terms)
        if abs(score-float(result["scores"][representative])) > 1e-9:
            raise ValueError("Alternate literal leaf prior/source score differs")
        leaf = leaves.setdefault((texts, key), {"count": 0, "literal_log_mass": score})
        if abs(score-leaf["literal_log_mass"]) > 1e-9:
            raise ValueError("One literal leaf has inconsistent prior mass")
        leaf["count"] += count
    readings = defaultdict(int)
    for (texts, _), leaf in leaves.items():
        readings[texts] += leaf["count"]
    best = min(readings, key=lambda text: (-readings[text], text)) if readings else None
    candidates = [(key, v) for (text, key), v in leaves.items() if text == best]
    key = min(candidates, key=lambda kv: (-kv[1]["count"], kv[0]))[0] if candidates else None
    if groups and sum(readings.values()) != result["summary"]["particles"]:
        raise ValueError("All final particle multiplicities required")
    bank = [{"source_records": texts, "used_key": key, **value}
            for (texts, key), value in sorted(leaves.items())]
    return bank, {"terminal_ancestors": len(groups), "initial_ancestors": initial_ancestors,
        "distinct_used_keys": len({key for _, key in leaves}), "distinct_readings": len(readings),
        "visited_literal_bank_log_mass": convert_finite(logsum(v["literal_log_mass"] for v in leaves.values())),
        "decision_records": best, "decision_key": key,
        "decision_particle_count": readings[best] if best is not None else 0}


def recovery_diagnostics(decision, fixture):
    """Gold is used only after an immutable prediction archive is written."""
    truth = fixture["generation_source"]
    predicted = decision["decision_records"] or ((),)*len(truth)
    gold_used = {r for text in truth for r in text}
    key = decision["decision_key"]
    return {"synthetic_diagnostic_only": True,
        "exact_records": sum(tuple(p) == tuple(t) for p, t in zip(predicted, truth, strict=True)),
        "edit_errors": sum(edit_distance("".join(chr(65+r) for r in p),
            "".join(chr(65+r) for r in t)) for p, t in zip(predicted, truth, strict=True)),
        "true_source_letters": sum(map(len, truth)),
        "gold_used_rows": len(gold_used),
        "matched_gold_used_rows": sum(key is not None and key[r] == fixture["generation_key"][r] for r in gold_used),
        "no_completed_decision": decision["decision_records"] is None}


def run(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(600, 500)
    wall, cpu, summaries, bulk_bytes = time.monotonic(), time.process_time(), [], 0
    BULK.mkdir(parents=True, exist_ok=True)
    try:
        source, selected = load_source()
        identity = array_identity(source)
        fixtures = fixture_grid(source)
        inputs = save_new(BULK/"fixtures.json", {"fixtures": fixtures, "source": selected["counts"]})
        bulk_bytes += inputs["bytes"]
        for fixture, schedule, count, seed in itertools.product(fixtures, ("sequential", "balanced"), COUNTS, SEEDS):
            name = f"case{fixture['case']}-{schedule}-{count}-{seed}"
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered 2GiB sampled particle host guard")
            result = run_particles(fixture["cipher"], source.probabilities, source.transitions,
                particles=count, seed=seed, schedule=schedule, observe=observe)
            bank, decision = reading_bank(result, fixture["cipher"], source)
            with (BULK/f"{name}.npz").open("xb") as handle:
                np.savez_compressed(handle, **{k: result[k] for k in ARRAY_NAMES})
            prediction = save_new(BULK/f"{name}.json.gz", convert_finite({"summary": result["summary"],
                "trace": result["trace"], "bank": bank, "decision": decision}), compressed=True)
            population = artifact(BULK/f"{name}.npz")
            bulk_bytes += prediction["bytes"]+population["bytes"]
            if bulk_bytes > 1024**3:
                raise MemoryError("Registered 1GiB sampled bulk output bound")
            observe({})
            # Deliberately after bank/population serialization; never feed gold to solver.
            diagnostic = recovery_diagnostics(decision, fixture)
            row = {"name": name, "case": fixture["case"], "schedule": schedule,
                "particles": count, "seed": seed, "prediction": prediction,
                "population": population, "arrays": arrays_identity(result),
                "summary": result["summary"],
                "diversity": {k: v for k, v in decision.items() if not k.startswith("decision_")},
                "diagnostic": diagnostic, "resources": resource_report(wall, cpu)}
            save_new(OUT/f"{name}.json", convert_finite(row))
            summaries.append(row)
            print(json.dumps({"name": name, "status": result["summary"]["status"], **diagnostic}), flush=True)
            del result, bank
        if array_identity(source) != identity or len(summaries) != 32:
            raise ValueError("Fixed32-cell grid or immutable source differs")
        save_new(OUT/"result.json", convert_finite({"freeze": freeze,
            "engineering_gate": "PASS_fixed_particle_transport_and_scalar_checks",
            "source": selected["counts"], "source_arrays": identity, "fixtures": inputs,
            "numpy_version": np.__version__, "bulk_output_bytes": bulk_bytes,
            "workloads": [artifact(OUT/f"{row['name']}.json") for row in summaries],
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "scope": "Artificial forced-length diagnostics, not fresh historical/empirical recovery qualification."}))
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"freeze": freeze, "error": repr(exc),
            "completed_workloads": len(summaries), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
