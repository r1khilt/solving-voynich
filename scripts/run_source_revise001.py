"""Bounded whole-key revision from exposed cipher-only particle candidates."""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import signal
import time
from dataclasses import asdict

import numpy as np

from scripts.benchmark_global_search_systems001 import PATHS as NATIVE_PATHS, admission
from scripts.run_source_guide001 import OUT as PARENT, PATHS as GUIDE_PATHS, SEEDS
from scripts.run_source_particle_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.kbest_suffix import record_kbest
from voynich.native_suffix_marginal import NativeMarginal
from voynich.tempered_unit_search import TemperedConfig, search_tempered
from voynich.unit_channel_decision import edit_distance

EXP = "SOURCE-REVISE-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
MAX_NODES, MAX_EDGES = 2_000_000, 8_000_000
DECODE_LIMITS = {"max_nodes": 500_000, "max_edges": 2_000_000, "max_expanded": 50_000}
PATHS = sorted(set([*NATIVE_PATHS, *GUIDE_PATHS,
    "src/voynich/kbest_suffix.py", "src/voynich/shared_key_mixture.py",
    "scripts/run_source_revise001.py", "scripts/audit_source_revise001.py",
    "tests/test_source_revise001.py", "docs/experiments/SOURCE-REVISE-001.md",
    "docs/research/source-key-revision-2026-10-01.md",
    "results/SOURCE-GUIDE-001/result.json", "results/SOURCE-GUIDE-001/audit.json",
    *[f"results/SOURCE-GUIDE-001/case{case}-{seed}-{guidance}.json"
      for case, seed, guidance in itertools.product(range(4), SEEDS, ("none", "iid"))]]))
POOL = tuple("".join(chars) for n in (1, 2) for chars in itertools.product("ABCDEF", repeat=n))


def config_for(case, seed):
    return TemperedConfig(seed=75811+101*case+(seed-75511), iterations=2048,
        max_scored_keys=8192, max_seconds=120.)


def warm_key(bank, case, seed):
    """Highest original literal leaf, lexical tie; unseen rows iid-filled once."""
    if not bank:
        raise ValueError("A completed particle leaf is required")
    best = min(bank, key=lambda b: (-b["literal_log_mass"], tuple(b["used_key"]), tuple(map(tuple, b["source_records"]))))
    partial = tuple(best["used_key"])
    if len(partial) != 23 or any(type(k) is not int or not -1 <= k < 42 for k in partial):
        raise ValueError("Declared partial23/42 dictionary required")
    fill = np.random.default_rng(75701+101*case+(seed-75511)).integers(42, size=23)
    full = tuple(POOL[int(fill[i]) if k < 0 else k] for i, k in enumerate(partial))
    return full, {"partial_indices": partial, "filler_indices": tuple(map(int, fill)),
        "fill_seed": 75701+101*case+(seed-75511), "seed_leaf_log_mass": best["literal_log_mass"]}


def score_key(native, records, key):
    values = [native.score(key, c, 1/225, max_nodes=MAX_NODES, max_edges=MAX_EDGES) for c in records]
    logs = [None if v.log_likelihood == -math.inf else v.log_likelihood for v in values]
    total = None if None in logs else math.fsum(logs)
    return {"units": tuple(key), "record_log_likelihoods": logs,
        "record_nodes": [v.reachable_nodes for v in values], "record_edges": [v.edges for v in values],
        "key_log_prior": -23*math.log(42), "log_key_mass": None if total is None else total-23*math.log(42)}


def read_key(source, native, key, records, expected):
    """Same exact fixed-key Viterbi rule for warm/final/gold diagnostic keys."""
    outputs, scores, graphs = [], [], []
    for i, c in enumerate(records):
        found, total, nodes, edges, expanded = record_kbest(source, key, c, 1/225, 1, **DECODE_LIMITS)
        expected_total = expected["record_log_likelihoods"][i]
        if (not found or expected_total is None or abs(total-expected_total) > 1e-7):
            raise ValueError("Exact native/k-best support or marginal mismatch")
        text, score = found[0]
        emitted = "".join(key[source.alphabet.index(letter)] for letter in text)
        if emitted != c:
            raise ValueError("Decoded reading does not emit literal ciphertext")
        # Alternate source-state scalar score, without reusing graph scores.
        state, terms = source.state(""), [math.log(1/225), len(text)*math.log1p(-1/225)]
        for letter in text:
            row = source.alphabet.index(letter)
            terms.append(math.log(float(source.row(state)[row])))
            state = source.step(state, row)
        if abs(math.fsum(terms)-score) > 1e-7:
            raise ValueError("Decoded scalar source score differs")
        outputs.append(text)
        scores.append(score)
        graphs.append({"nodes": nodes, "edges": edges, "expanded": expanded})
    return {"source_records": outputs, "record_joint_log_probability": scores, "graphs": graphs,
        "decision": "best_key_then_exact_fixed_key_viterbi", "plaintext_marginal_MAP_claimed": False}


def fit_one(records, warm, source, native, config, observe):
    """No gold key/source/length arguments or closure. Full dictionary revisable."""
    work = []
    def score(key):
        row = score_key(native, records, key)
        work.append(row)
        if len(work) % 64 == 0:
            observe({"scored_keys": len(work)})
        return row["log_key_mass"]
    search = search_tempered(score, [warm], POOL, config=config)
    if len(work) != search["scored_keys"] or len(work) > config.max_scored_keys:
        raise ValueError("All unique full-key scores must be retained")
    before = read_key(source, native, warm, records, work[0])
    after = read_key(source, native, search["best_units"], records, work[search["best_index"]])
    if search["best_score"] < work[0]["log_key_mass"]:
        raise ValueError("Search lost its scored warm start")
    return {"search": search, "scores": work, "warm_prediction": before, "final_prediction": after}


def diagnostics(fitted, fixture, source, native):
    """After immutable fit/prediction archives; no evidence returned to fitting."""
    truth = tuple("".join(source.alphabet[r] for r in text) for text in fixture["generation_source"])
    gold_key = tuple(POOL[i] for i in fixture["generation_key"])
    used = {r for text in fixture["generation_source"] for r in text}
    search = fitted["search"]
    rows = {}
    for name, key, prediction in (("warm", fitted["scores"][0]["units"], fitted["warm_prediction"]),
            ("revised", search["best_units"], fitted["final_prediction"])):
        texts = prediction["source_records"]
        rows[name] = {"exact_records": sum(a == b for a, b in zip(texts, truth, strict=True)),
            "edit_errors": sum(edit_distance(a, b) for a, b in zip(texts, truth, strict=True)),
            "true_source_letters": sum(map(len, truth)), "gold_used_rows": len(used),
            "matched_gold_used_rows": sum(key[r] == gold_key[r] for r in used)}
    gold = score_key(native, tuple("".join("ABCDEF"[g] for g in c) for c in fixture["cipher"]), gold_key)
    if gold["log_key_mass"] is None:
        raise ValueError("Generating full key must have support")
    gold_prediction = read_key(source, native, gold_key,
        tuple("".join("ABCDEF"[g] for g in c) for c in fixture["cipher"]), gold)
    rows["gold_key_reader"] = {"exact_records": sum(a == b for a, b in zip(gold_prediction["source_records"], truth, strict=True)),
        "edit_errors": sum(edit_distance(a, b) for a, b in zip(gold_prediction["source_records"], truth, strict=True)),
        "true_source_letters": sum(map(len, truth)), "gold_used_rows": len(used), "matched_gold_used_rows": len(used)}
    rows.update({"score_gain_nats": search["best_score"]-fitted["scores"][0]["log_key_mass"],
        "gold_full_key_log_mass": gold["log_key_mass"],
        "gold_minus_selected_key_nats": gold["log_key_mass"]-search["best_score"],
        "literal_gold_full_key_in_scored_bank": any(tuple(b["units"]) == gold_key for b in search["bank"]),
        "gold_used_mapping_in_scored_bank": any(all(b["units"][r] == gold_key[r] for r in used) for b in search["bank"]),
        "full_key_search_is_not_reading_marginal_search": True})
    return rows


def inputs(freeze):
    require_frozen(freeze, PATHS)
    benchmark = admission(freeze)
    parent = json.loads((PARENT/"result.json").read_text())
    audit = json.loads((PARENT/"audit.json").read_text())
    if audit["result"] != artifact(PARENT/"result.json") or len(parent["workloads"]) != 16:
        raise ValueError("Audited16-cell parent required")
    if artifact(ROOT/parent["fixtures"]["path"]) != parent["fixtures"]:
        raise ValueError("Parent fixture archive changed")
    cases = json.loads((ROOT/parent["fixtures"]["path"]).read_text())["fixtures"]
    return benchmark, parent, cases


def run(freeze):
    benchmark, parent, cases = inputs(freeze)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(2400, 2200)
    wall, cpu, rows, bulk_bytes = time.monotonic(), time.process_time(), [], 0
    BULK.mkdir(parents=True, exist_ok=True)
    try:
        source, selected = load_source()
        native, identity = NativeMarginal(source, benchmark["build"]), array_identity(source)
        for spec in parent["workloads"]:
            metadata = json.loads((ROOT/spec["path"]).read_text())
            name, case, seed = metadata["name"], metadata["case"], metadata["seed"]
            if artifact(ROOT/spec["path"]) != spec or artifact(ROOT/metadata["prediction"]["path"]) != metadata["prediction"]:
                raise ValueError("Parent closed predictions changed")
            bank = json.loads(gzip.decompress((ROOT/metadata["prediction"]["path"]).read_bytes()))["bank"]
            warm, completion = warm_key(bank, case, seed)
            cipher = tuple("".join("ABCDEF"[g] for g in c) for c in cases[case]["cipher"])
            config = config_for(case, seed)
            cell_wall, cell_cpu = time.monotonic(), time.process_time()
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 4*1024**3:
                    raise MemoryError("Registered4GiB sampled host cap")
            # All arrays passed to fit_one are cipher-only; the gold fixture stays outside it.
            fitted = fit_one(cipher, warm, source, native, config, observe)
            packed = save_new(BULK/f"{name}.json.gz", fitted, compressed=True)
            prediction = save_new(BULK/f"{name}-prediction.json", {"completion": completion,
                "warm": fitted["warm_prediction"], "revised": fitted["final_prediction"],
                "warm_key": warm, "revised_key": fitted["search"]["best_units"]})
            bulk_bytes += packed["bytes"]+prediction["bytes"]
            if bulk_bytes > 512*1024**2:
                raise MemoryError("Registered512MiB sampled bulk cap")
            observe({})
            diagnostic = diagnostics(fitted, cases[case], source, native)
            row = {"name": name, "case": case, "seed": seed, "guidance": metadata["guidance"],
                "parent": spec, "config": asdict(config), "fitted": packed, "prediction": prediction,
                "search_summary": {k: v for k, v in fitted["search"].items() if k not in ("bank", "events")},
                "diagnostic": diagnostic, "cell_wall_seconds": time.monotonic()-cell_wall,
                "cell_cpu_seconds": time.process_time()-cell_cpu}
            save_new(OUT/f"{name}.json", row)
            rows.append(row)
            print(f"{name}: {diagnostic['warm']['edit_errors']} -> {diagnostic['revised']['edit_errors']} edits; {fitted['search']['scored_keys']} keys; {fitted['search']['stop_reason']}", flush=True)
            del fitted, bank
        if len(rows) != 16 or array_identity(source) != identity:
            raise ValueError("Fixed16/source identity differs")
        totals = {stage: {metric: sum(r["diagnostic"][stage][metric] for r in rows) for metric in
            ("edit_errors", "exact_records", "true_source_letters", "matched_gold_used_rows", "gold_used_rows")}
            for stage in ("warm", "revised", "gold_key_reader")}
        clauses = {"all16_complete": len(rows) == 16,
            "edits_reduced_at_least_25_percent": totals["warm"]["edit_errors"] > 0 and totals["revised"]["edit_errors"] <= .75*totals["warm"]["edit_errors"],
            "better_edits_in_at_least12_cells": sum(r["diagnostic"]["revised"]["edit_errors"] < r["diagnostic"]["warm"]["edit_errors"] for r in rows) >= 12}
        save_new(OUT/"result.json", {"freeze": freeze, "parent": artifact(PARENT/"result.json"),
            "source": selected["counts"], "source_arrays": identity, "native_build": benchmark["build"],
            "workloads": [artifact(OUT/f"{r['name']}.json") for r in rows], "totals": totals,
            "exploratory_signal_clauses": clauses, "exploratory_signal": "SUPPORTED" if all(clauses.values()) else "NOT_SUPPORTED",
            "resources": resource_report(wall, cpu), "bulk_output_bytes": bulk_bytes, "paid_spend_usd": 0,
            "scope": "Exposed synthetic staged key-search diagnosis; no fresh recovery qualification or posterior coverage claim."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_workloads": len(rows),
            "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
