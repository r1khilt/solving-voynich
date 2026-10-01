"""One registered exposed IID-surrogate calibration of whole-key resample--move."""

from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np

from scripts.run_shared_key_guide001 import PATHS as OLD_PATHS, SEEDS, inputs as parent_inputs, queries
from scripts.run_source_state_systems001 import (
    ROOT, array_identity, artifact, finite, limit_resources, load_source,
    require_frozen, resource_report, save_new,
)
from voynich.dictionary_smc import FixedKeyBridge, bridge_schedule, run_dictionary_smc


EXP = "DICTIONARY-SMC-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
ARMS = ("no_move","row","joint")
PATHS = sorted(set([*OLD_PATHS,
    "src/voynich/dictionary_smc.py","tests/test_dictionary_smc.py",
    "scripts/check_dictionary_smc_theory001.py","results/DICTIONARY-SMC-THEORY-001/result.json",
    "scripts/run_dictionary_smc001.py","scripts/audit_dictionary_smc001.py",
    "tests/test_dictionary_smc001.py","docs/experiments/DICTIONARY-SMC-001.md",
    "docs/research/dictionary-smc-2026-10-01.md",
    "results/SHARED-KEY-GUIDE-001/result.json","results/SHARED-KEY-GUIDE-001/audit.json",
    "scripts/check_prefix_bridge001.py"]))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent,fixtures,cells,archives = parent_inputs(freeze)
    closed = json.loads((ROOT/"results/SHARED-KEY-GUIDE-001/result.json").read_text())
    audit = json.loads((ROOT/"results/SHARED-KEY-GUIDE-001/audit.json").read_text())
    if (audit["status"]!="PASS" or audit["result"]!=artifact(ROOT/"results/SHARED-KEY-GUIDE-001/result.json")
            or closed["queries"]!=queries(fixtures,cells,archives)):
        raise ValueError("Unchanged closed 12-query shared-key calibration required")
    return parent,closed


def evaluate(query,row,seed,arm,bridge_type=FixedKeyBridge):
    if arm not in ARMS:
        raise ValueError("Unregistered whole-key SMC arm")
    q = ((row+1e-8)/(1+23e-8))[None,:].copy()
    remaining = tuple(len(c)-o for c,o in zip(query["cipher"],query["offsets"],strict=True))
    stages = len(bridge_schedule(remaining,4))
    tables = 128*stages*(1 if arm=="no_move" else 3)
    bridge = bridge_type(query["cipher"],q,np.zeros((1,23),dtype=np.uint32),
        offsets=query["offsets"],contexts=(0,0),max_tables=tables,
        max_edges=128*3*stages*23*sum(remaining),max_work_bytes=128*1024**2)
    result = run_dictionary_smc(bridge,query["partial_key"],particles=128,seed=seed,
        stride=4,mutations=0 if arm=="no_move" else 2,kernel="row" if arm=="row" else "joint")
    result["keys"] = result["keys"].tolist()
    result["key_log_likelihoods"] = result["key_log_likelihoods"].tolist()
    return finite(result)


def compact(value):
    trace = value["trace"]
    return {"status":value["status"],"log_evidence_estimate":value["log_evidence_estimate"],
        "stages_completed":sum(not t["extinct"] for t in trace),"tables":value["tables"],"edges":value["edges"],
        "minimum_incremental_ess":min((t["incremental_ess"] for t in trace if not t["extinct"]),default=0.),
        "maximum_incremental_weight":max((t["maximum_incremental_weight"] for t in trace if not t["extinct"]),default=0.),
        "final_distinct_full_keys":int(len(np.unique(value["keys"],axis=0))),
        "accepted_proposals":sum(t.get("accepted_proposals",0) for t in trace),
        "changed_proposals":sum(t.get("changed_proposals",0) for t in trace),
        "proposals":sum(t.get("proposals",0) for t in trace)}


def summarize(rows):
    arms = {}
    for arm in ARMS:
        values = [r for r in rows if r["arm"]==arm]
        spreads = []
        for query in sorted({r["query_id"] for r in values}):
            chosen = [r["summary"]["log_evidence_estimate"] for r in values if r["query_id"]==query]
            spreads.append({"query":query,"all_positive":all(v is not None for v in chosen),
                "seed_log_spread":max(chosen)-min(chosen) if all(v is not None for v in chosen) else None})
        preference = []
        for case in range(4):
            for seed in SEEDS:
                a = next(r["summary"]["log_evidence_estimate"] for r in values if r["query_id"]==f"case{case}-prefix" and r["seed"]==seed)
                b = next(r["summary"]["log_evidence_estimate"] for r in values if r["query_id"]==f"case{case}-altered_binding" and r["seed"]==seed)
                preference.append(a is not None and (b is None or a>b))
        positive = all(s["all_positive"] for s in spreads)
        stable = positive and all(s["seed_log_spread"]<=2. for s in spreads)
        undominated = all(r["summary"]["maximum_incremental_weight"]<=.5 for r in values)
        arms[arm] = {"calls":len(values),"positive_calls":sum(r["summary"]["status"]=="complete_particles" for r in values),
            "spreads":spreads,"all_calls_positive":positive,"all_seed_spreads_at_most_2":stable,
            "all_incremental_weights_at_most_half":undominated,
            "prefix_preferred_to_altered_binding":sum(preference),"preference_denominator":len(preference),
            "exploratory_calibration_supported":positive and stable and undominated and sum(preference)>=12,
            "tables":sum(r["summary"]["tables"] for r in values),
            "row_operations":sum(r["summary"]["edges"] for r in values),
            "summed_call_wall_seconds":sum(r["wall_seconds"] for r in values)}
    return {"arms":arms,"iid_surrogate_not_full_contextual_future":True,
        "known_answer_conditioned_and_exposed":True,"recovery_or_historical_qualification":False,
        "incremental_weights_are_not_independent_prior_contribution_shares":True,
        "cpu_matched_search_claimed":False}


def guard(wall,cpu,bytes_so_far):
    if resource_report(wall,cpu)["peak_rss_bytes"]>2*1024**3 or bytes_so_far>128*1024**2:
        raise MemoryError("Registered 2GiB host /128MiB bulk cap exhausted")


def run(freeze):
    parent,closed = inputs(freeze)
    save_new(OUT/"started.json",{"freeze":freeze,"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,rows,bulk_bytes = time.monotonic(),time.process_time(),[],0
    try:
        source,selected = load_source()
        identity = array_identity(source)
        if selected["counts"]!=parent["source"] or identity!=parent["source_arrays"]:
            raise ValueError("Original source counts/arrays changed")
        for query in closed["queries"]:
            for seed in SEEDS:
                for arm in ARMS:
                    start = time.monotonic()
                    value = evaluate(query,source.probabilities[0],seed,arm)
                    name = f"{query['id']}-{seed}-{arm}"
                    spec = save_new(BULK/f"{name}.json.gz",value,compressed=True)
                    bulk_bytes += spec["bytes"]
                    row = {"query_id":query["id"],"seed":seed,"arm":arm,"output":spec,
                        "summary":compact(value),"wall_seconds":time.monotonic()-start}
                    save_new(OUT/f"{name}.json",row)
                    rows.append(row)
                    guard(wall,cpu,bulk_bytes)
            print(f"Closed {query['id']}: 12 whole-key SMC calls",flush=True)
        if len(rows)!=144 or array_identity(source)!=identity:
            raise ValueError("Fixed call inventory/source changed")
        save_new(OUT/"result.json",{"experiment":EXP,"freeze":freeze,"numpy_version":np.__version__,
            "parent_result":artifact(ROOT/"results/SHARED-KEY-GUIDE-001/result.json"),
            "parent_audit":artifact(ROOT/"results/SHARED-KEY-GUIDE-001/audit.json"),
            "source":selected["counts"],"source_arrays":identity,"queries":closed["queries"],
            "workloads":[artifact(OUT/f"{r['query_id']}-{r['seed']}-{r['arm']}.json") for r in rows],
            "summary":summarize(rows),"bulk_bytes":bulk_bytes,"resources":resource_report(wall,cpu),
            "paid_spend_usd":0,"no_new_native_search_reader_training_or_holdout":True})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json",{"error":repr(exc),"closed_calls":len(rows),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",required=True)
    run(parser.parse_args().freeze)
