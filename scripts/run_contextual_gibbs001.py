"""One registered contextual Gibbs versus matched-table uniform-row MH panel."""

from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np

from scripts.benchmark_censored_context001 import PATHS as OLD_PATHS, inputs as parent_inputs
from scripts.run_dictionary_smc001 import compact
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import (
    ROOT, array_identity, artifact, finite, limit_resources, load_source,
    require_frozen, resource_report, save_new,
)
from voynich.dictionary_gibbs import run_dictionary_gibbs
from voynich.dictionary_smc import bridge_schedule,run_dictionary_smc
from voynich.native_censored_bridge import CensoredNativeBridge
from voynich.native_suffix_marginal import NativeMarginal


EXP = "CONTEXTUAL-GIBBS-001"
OUT,BULK = ROOT/"results"/EXP,ROOT/"outputs"/EXP
SEEDS,ARMS,PARTICLES,STRIDE = (81401,81402),("uniform42","gibbs"),32,16
PATHS = sorted(set([*OLD_PATHS,"src/voynich/dictionary_gibbs.py","tests/test_dictionary_gibbs.py",
    "scripts/run_contextual_gibbs001.py","scripts/audit_contextual_gibbs001.py",
    "tests/test_contextual_gibbs001.py","docs/experiments/CONTEXTUAL-GIBBS-001.md",
    "docs/research/contextual-row-gibbs-2026-10-01.md",
    "results/CENSORED-CONTEXT-001/result.json","results/CENSORED-CONTEXT-001/audit.json"]))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent,native,selected = parent_inputs(freeze)
    result = json.loads((ROOT/"results/CENSORED-CONTEXT-001/result.json").read_text())
    audit = json.loads((ROOT/"results/CENSORED-CONTEXT-001/audit.json").read_text())
    if (audit["status"]!="PASS_complete_contextual_benchmark_replay"
            or audit["result"]!=artifact(ROOT/"results/CENSORED-CONTEXT-001/result.json")
            or result["summary"]["contextual_engineering_gate"]!="PASS"):
        raise ValueError("Complete original contextual bridge qualification required")
    queries = [s["query"] for s in selected if s["query"]["kind"]=="root"]
    assert [q["case"] for q in queries]==list(range(4))
    assert all(q["partial_key"]==[-1]*23 and q["offsets"]==[0,0]
               and q["supplied_contexts_ignored_by_iid"]==[0,0] for q in queries)
    old = json.loads((ROOT/"results/SOURCE-PRUNE-DIAG-002/result.json").read_text())
    fixtures = load_bound(old["fixtures"])["fixtures"]
    assert all(q["cipher"]==f["cipher"] for q,f in zip(queries,fixtures,strict=True))
    return parent,native,queries,fixtures


def evaluate(query,native,seed,arm,edge_limit=2_000_000_000):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unregistered contextual panel seed/arm")
    lengths = tuple(len(c)-o for c,o in zip(query["cipher"],query["offsets"],strict=True))
    stages = len(bridge_schedule(lengths,STRIDE))
    bridge = CensoredNativeBridge(query["cipher"],native,offsets=query["offsets"],
        contexts=query["supplied_contexts_ignored_by_iid"],
        max_tables=PARTICLES*43*stages,max_edges=edge_limit)
    trace = []
    started = time.monotonic()
    try:
        if arm=="gibbs":
            value = run_dictionary_gibbs(bridge,query["partial_key"],particles=PARTICLES,
                seed=seed,stride=STRIDE,observe=trace.append)
        else:
            value = run_dictionary_smc(bridge,query["partial_key"],particles=PARTICLES,
                seed=seed,stride=STRIDE,mutations=len(bridge.units),kernel="row",observe=trace.append)
        value["keys"] = value["keys"].tolist()
        value["key_log_likelihoods"] = value["key_log_likelihoods"].tolist()
        value = finite(value)
    except (RuntimeError,MemoryError) as exc:
        if not any(s in str(exc) for s in ("Exact lattice node cap","Exact lattice edge cap")):
            raise
        value = {"status":"graph_cap","error":str(exc),"trace":trace,
            "tables":bridge.tables,"edges":bridge.edges,"no_partial_likelihood":True}
    return {"value":finite(value),"wall_seconds":time.monotonic()-started,
        "native_calls":bridge.native_calls,"native_nodes":bridge.nodes,
        "maximum_native_nodes":bridge.maximum_native_nodes,
        "maximum_owned_work_envelope":bridge.maximum_owned_work_envelope,
        "nominal_complete_table_budget":PARTICLES*(1+len(bridge.units))*stages,
        "source_context_preserved":True}


def mapping_metric(value,fixture):
    used = sorted({r for text in fixture["generation_source"] for r in text})
    if value["status"]!="complete_particles":
        return {"used_rows":len(used),"selected_matches":None,"selected_complete_used_key":False,
            "bank_contains_complete_used_key":False,"selected_index":None}
    values = value["key_log_likelihoods"]
    assert all(v is not None for v in values)
    selected = max(range(len(values)),key=lambda i:(values[i],-i))
    matches = [sum(k[r]==fixture["generation_key"][r] for r in used) for k in value["keys"]]
    return {"used_rows":len(used),"selected_matches":matches[selected],
        "selected_complete_used_key":matches[selected]==len(used),
        "bank_contains_complete_used_key":max(matches)==len(used),"selected_index":selected}


def metrics(value):
    if value["status"]=="graph_cap":
        return {"status":"graph_cap","log_evidence_estimate":None,"tables":value["tables"],
            "edges":value["edges"],"stages_completed":len(value["trace"])}
    return compact(value)


def summarize(cells):
    arms = {}
    for arm in ARMS:
        rows = [c for c in cells if c["arm"]==arm]
        complete = all(c["summary"]["status"]=="complete_particles" for c in rows)
        spreads = []
        for case in range(4):
            estimates = [c["summary"]["log_evidence_estimate"] for c in rows if c["case"]==case]
            spreads.append({"case":case,"log_spread":max(estimates)-min(estimates) if len(estimates)==2 and all(v is not None for v in estimates) else None})
        matched = sum(c["mapping"]["selected_matches"] or 0 for c in rows)
        denominator = sum(c["mapping"]["used_rows"] for c in rows)
        arms[arm] = {"calls":len(rows),"complete_positive":sum(c["summary"]["status"]=="complete_particles" for c in rows),
            "extinct":sum(c["summary"]["status"]=="extinct" for c in rows),
            "graph_cap":sum(c["summary"]["status"]=="graph_cap" for c in rows),
            "all_complete":complete and len(rows)==8,"seed_spreads":spreads,
            "all_seed_spreads_at_most_2":complete and all(s["log_spread"] is not None and s["log_spread"]<=2 for s in spreads),
            "all_incremental_weights_at_most_half":complete and all(c["summary"]["maximum_incremental_weight"]<=.5 for c in rows),
            "selected_used_matches":matched,"used_row_denominator":denominator,
            "selected_match_fraction_with_failures_counted_zero":matched/denominator,
            "selected_complete_used_keys":sum(c["mapping"]["selected_complete_used_key"] for c in rows),
            "banks_containing_complete_used_key":sum(c["mapping"]["bank_contains_complete_used_key"] for c in rows),
            "tables":sum(c["summary"]["tables"] for c in rows),
            "summed_call_wall_seconds":sum(c["wall_seconds"] for c in rows)}
    improvement = (arms["gibbs"]["all_complete"] and arms["uniform42"]["all_complete"]
        and arms["gibbs"]["selected_match_fraction_with_failures_counted_zero"]>=
            arms["uniform42"]["selected_match_fraction_with_failures_counted_zero"]+.10
        and arms["gibbs"]["selected_complete_used_keys"]>=1)
    return {"arms":arms,"exploratory_mapping_improvement_supported":improvement,
        "stable_calibration_supported":all(a["all_complete"] and a["all_seed_spreads_at_most_2"]
            and a["all_incremental_weights_at_most_half"] for a in arms.values()),
        "table_work_matched_only_when_complete":True,"cpu_matched_claimed":False,
        "all_inputs_exposed":True,"reading_or_historical_qualification":False}


def guard(wall,cpu,bulk_bytes,cells):
    if (resource_report(wall,cpu)["peak_rss_bytes"]>2*1024**3 or bulk_bytes>128*1024**2
            or sum(c["summary"]["edges"] for c in cells)>8_000_000_000):
        raise MemoryError("Registered2GiB host /128MiB bulk /8B edge campaign cap")


def run(freeze):
    parent,benchmark,queries,fixtures = inputs(freeze)
    save_new(OUT/"started.json",{"freeze":freeze,"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells,bulk_bytes = time.monotonic(),time.process_time(),[],0
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent["source"]==chosen["counts"] and parent["source_arrays"]==identity
        native = NativeMarginal(source,benchmark["build"])
        for query,fixture in zip(queries,fixtures,strict=True):
            for seed in SEEDS:
                for arm in ARMS:
                    remaining = 8_000_000_000-sum(c["summary"]["edges"] for c in cells)
                    if remaining<=0:
                        raise RuntimeError("Campaign edge work exhausted")
                    evaluated = evaluate(query,native,seed,arm,min(2_000_000_000,remaining))
                    value = evaluated.pop("value")
                    name = f"case{query['case']}-{seed}-{arm}"
                    spec = save_new(BULK/f"{name}.json.gz",value,compressed=True)
                    bulk_bytes += spec["bytes"]
                    cell = {"case":query["case"],"seed":seed,"arm":arm,"output":spec,
                        "summary":metrics(value),"mapping":mapping_metric(value,fixture),**evaluated}
                    save_new(OUT/f"{name}.json",cell)
                    cells.append(cell)
                    guard(wall,cpu,bulk_bytes,cells)
                    print(f"Closed {name}: {value['status']}, {cell['summary']['tables']} tables",flush=True)
        assert len(cells)==16 and array_identity(source)==identity
        save_new(OUT/"result.json",{"experiment":EXP,"freeze":freeze,"numpy_version":np.__version__,
            "qualification":artifact(ROOT/"results/CENSORED-CONTEXT-001/result.json"),
            "qualification_audit":artifact(ROOT/"results/CENSORED-CONTEXT-001/audit.json"),
            "source":chosen["counts"],"source_arrays":identity,"native_build":benchmark["build"],
            "queries":queries,"workloads":[artifact(OUT/f"case{c['case']}-{c['seed']}-{c['arm']}.json") for c in cells],
            "summary":summarize(cells),"bulk_bytes":bulk_bytes,"resources":resource_report(wall,cpu),"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json",{"error":repr(exc),"closed_calls":len(cells),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",required=True)
    run(parser.parse_args().freeze)
