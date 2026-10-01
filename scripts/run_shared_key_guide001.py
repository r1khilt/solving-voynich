"""Registered exposed-query stability/cost calibration, not a decoding search."""

from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np

from scripts.run_source_bellman001 import PATHS as OLD_PATHS, load_bound
from scripts.run_source_state_systems001 import (
    ROOT, array_identity, artifact, limit_resources, load_source, require_frozen,
    resource_report, save_new,
)
from voynich.guided_source_particles import IidSuffixGuide
from voynich.shared_dictionary_guide import SharedDictionaryIid

EXP = "SHARED-KEY-GUIDE-001"
OUT = ROOT/"results"/EXP
PRUNE = ROOT/"results/SOURCE-PRUNE-DIAG-002"
SEEDS = (81401,81402,81403,81404)
ARMS = ("mc16","mc672","rb16")
PATHS = sorted(set([*OLD_PATHS,
    "src/voynich/shared_dictionary_guide.py","tests/test_shared_dictionary_guide.py",
    "scripts/run_shared_key_guide001.py","scripts/audit_shared_key_guide001.py",
    "tests/test_shared_key_guide001.py","docs/experiments/SHARED-KEY-GUIDE-001.md",
    "docs/research/shared-dictionary-guide-2026-10-01.md",
    "results/SOURCE-PRUNE-DIAG-002/result.json","results/SOURCE-PRUNE-DIAG-002/audit.json",
    *[f"results/SOURCE-PRUNE-DIAG-002/case{c}-wide_guided.json" for c in range(4)]]))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent = json.loads((PRUNE/"result.json").read_text())
    audit = json.loads((PRUNE/"audit.json").read_text())
    if audit["status"]!="PASS" or audit["result"]!=artifact(PRUNE/"result.json"):
        raise ValueError("Closed complete pruning audit required")
    fixtures = load_bound(parent["fixtures"])["fixtures"]
    if len(fixtures)!=4 or [f["case"] for f in fixtures]!=list(range(4)):
        raise ValueError("Four exposed fixture identities required")
    cells = [load_bound(parent["workloads"][4*c+3]) for c in range(4)]
    archives = [load_bound(cell["diagnostic"]) for cell in cells]
    return parent,fixtures,cells,archives


def queries(fixtures,cells,archives):
    result = []
    for fixture,cell,archive in zip(fixtures,cells,archives,strict=True):
        case = fixture["case"]
        loss = cell["summary"]["first_loss"]
        target = loss["target_glyph_layer"] if loss else next(t["target_glyph_layer"] for t in archive["trace"] if t["target_glyph_layer"]>=10)
        index = next(i for i,t in enumerate(archive["trace"]) if t["target_glyph_layer"]==target)
        state = archive["gold_states"][index]
        if len(state)!=30 or sum(state[:2])!=target or state[29]!=1:
            raise ValueError("Archived known-answer prefix state layout changed")
        partial = state[4:27]
        row = next(i for i,k in enumerate(partial) if k>=0)
        wrong = partial.copy()
        wrong[row] = (wrong[row]+1)%42
        for name,key,offsets in (("root",[-1]*23,[0,0]),("prefix",partial,state[:2]),("altered_binding",wrong,state[:2])):
            result.append({"id":f"case{case}-{name}","case":case,"kind":name,
                "cipher":fixture["cipher"],"partial_key":key,"offsets":offsets,
                "supplied_contexts_ignored_by_iid":state[2:4] if name!="root" else [0,0],
                "prefix_glyph_layer":sum(offsets),"altered_row":row if name=="altered_binding" else None,
                "known_answer_conditioned_diagnostic":name!="root"})
    return result


def evaluate(query,row,seed,arm):
    if arm not in ARMS:
        raise ValueError("Unregistered shared-dictionary arm")
    bank = np.random.default_rng(seed).integers(42,size=(672,23),dtype=np.int32)
    key = query["partial_key"]
    free = [i for i,k in enumerate(key) if k<0]
    chosen = max(free,key=lambda i:(row[i],-i)) if arm=="rb16" else None
    guide = SharedDictionaryIid(query["cipher"],row,max_tables=672)
    return guide.estimate(key,query["offsets"],bank if arm=="mc672" else bank[:16],
                          integrate_row=chosen if arm=="rb16" else None)


def renewal(query,row):
    guide = IidSuffixGuide(query["cipher"],row,cache_entries=1,max_tables=1)
    offsets = np.array([query["offsets"]],dtype=np.int32)
    closed = offsets==np.array(list(map(len,query["cipher"])))
    return float(guide(np.array([query["partial_key"]],dtype=np.int32),offsets,closed)[0])


def summarize(rows):
    arms = {}
    for arm in ARMS:
        values = [r for r in rows if r["arm"]==arm]
        spreads = []
        for name in sorted({r["query"]["id"] for r in values}):
            chosen = [r["estimate"]["log_mean"] for r in values if r["query"]["id"]==name]
            spreads.append({"query":name,"all_positive":all(v is not None for v in chosen),
                "seed_log_spread":max(chosen)-min(chosen) if all(v is not None for v in chosen) else None})
        preference = []
        for case in range(4):
            for seed in SEEDS:
                a = next(r["estimate"]["log_mean"] for r in values if r["query"]["id"]==f"case{case}-prefix" and r["seed"]==seed)
                b = next(r["estimate"]["log_mean"] for r in values if r["query"]["id"]==f"case{case}-altered_binding" and r["seed"]==seed)
                preference.append(a is not None and (b is None or a>b))
        stable = all(s["all_positive"] and s["seed_log_spread"]<=2. for s in spreads)
        undominated = all(r["estimate"]["maximum_base_contribution_share"]<=.5 for r in values)
        arms[arm] = {"calls":len(values),"positive_calls":sum(r["estimate"]["log_mean"] is not None for r in values),
            "fixed_dictionary_tables":sum(r["estimate"]["fixed_dictionary_tables"] for r in values),
            "maximum_contribution_share":max(r["estimate"]["maximum_base_contribution_share"] for r in values),
            "minimum_contribution_ess":min(r["estimate"]["likelihood_contribution_ess"] for r in values),
            "spreads":spreads,"stable_all_queries":stable,"undominated_all_calls":undominated,
            "prefix_preferred_to_altered_binding":sum(preference),"preference_denominator":len(preference),
            "exploratory_calibration_supported":stable and undominated and sum(preference)>=12,
            "summed_call_wall_seconds":sum(r["wall_seconds"] for r in values)}
    return {"arms":arms,"recovery_or_historical_qualification":False,
        "known_answer_conditioned_and_exposed":True,"cpu_matched_search_claimed":False}


def guard(wall,cpu):
    if resource_report(wall,cpu)["peak_rss_bytes"]>2*1024**3:
        raise MemoryError("Sampled host RSS exceeds registered 2GiB")


def run(freeze):
    parent,fixtures,cells,archives = inputs(freeze)
    save_new(OUT/"started.json",{"freeze":freeze,"start_unix":time.time()})
    limit_resources(600,500)
    wall,cpu,rows = time.monotonic(),time.process_time(),[]
    try:
        source,selected = load_source()
        identity = array_identity(source)
        if selected["counts"]!=parent["source"] or identity!=parent["source_arrays"]:
            raise ValueError("Frozen source counts/arrays changed")
        declared = queries(fixtures,cells,archives)
        for query in declared:
            baseline = renewal(query,source.probabilities[0])
            for seed in SEEDS:
                for arm in ARMS:
                    before = time.monotonic()
                    estimate = evaluate(query,source.probabilities[0],seed,arm)
                    row = {"query":query,"seed":seed,"arm":arm,"estimate":estimate,
                        "renewal_log_guide":baseline,"wall_seconds":time.monotonic()-before}
                    save_new(OUT/f"{query['id']}-{seed}-{arm}.json",row)
                    rows.append(row)
                    guard(wall,cpu)
            print(f"Closed {query['id']}: 12 finite estimates",flush=True)
        if len(rows)!=144 or array_identity(source)!=identity:
            raise ValueError("Fixed 144-call inventory/source changed")
        specs = [artifact(OUT/f"{r['query']['id']}-{r['seed']}-{r['arm']}.json") for r in rows]
        if sum(s["bytes"] for s in specs)>16*1024**2:
            raise MemoryError("Compact output byte cap exceeded")
        save_new(OUT/"result.json",{"experiment":EXP,"freeze":freeze,"numpy_version":np.__version__,
            "parent_result":artifact(PRUNE/"result.json"),"parent_audit":artifact(PRUNE/"audit.json"),
            "source":selected["counts"],"source_arrays":identity,"fixtures":parent["fixtures"],
            "queries":declared,"workloads":specs,"summary":summarize(rows),
            "resources":resource_report(wall,cpu),"paid_spend_usd":0,
            "no_new_native_search_fixed_key_source_score_reader_training_or_holdout":True})
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
