"""One frozen contextual prefix/closure/cost benchmark on previously closed keys."""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import signal
import time

import numpy as np

from scripts.benchmark_global_search_systems001 import admission
from scripts.run_dictionary_smc001 import PATHS as OLD_PATHS, inputs as parent_inputs
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import (
    ROOT, array_identity, artifact, limit_resources, load_source, require_frozen, resource_report, save_new,
)
from voynich.dictionary_smc import FixedKeyBridge
from voynich.native_censored_bridge import CensoredNativeBridge
from voynich.native_suffix_marginal import NativeMarginal


EXP = "CENSORED-CONTEXT-001"
OUT = ROOT/"results"/EXP
KINDS = ("root","prefix","altered_binding")
PATHS = sorted(set([*OLD_PATHS,
    "src/voynich/native_censored_bridge.py","tests/test_native_censored_bridge.py",
    "scripts/benchmark_censored_context001.py","scripts/audit_censored_context001.py",
    "tests/test_censored_context001.py","docs/experiments/CENSORED-CONTEXT-001.md",
    "docs/research/native-censored-context-2026-10-01.md",
    "results/DICTIONARY-SMC-001/result.json","results/DICTIONARY-SMC-001/audit.json",
    *[f"results/DICTIONARY-SMC-001/case{case}-{kind}-81401-row.json" for case in range(4) for kind in KINDS]]))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    _,closed = parent_inputs(freeze)
    result = json.loads((ROOT/"results/DICTIONARY-SMC-001/result.json").read_text())
    audit = json.loads((ROOT/"results/DICTIONARY-SMC-001/audit.json").read_text())
    if (audit["status"]!="PASS" or audit["result"]!=artifact(ROOT/"results/DICTIONARY-SMC-001/result.json")
            or result["queries"]!=closed["queries"] or len(result["workloads"])!=144):
        raise ValueError("Unchanged complete dictionary-SMC query/trajectory provenance required")
    selected = []
    for query in result["queries"]:
        path = ROOT/f"results/DICTIONARY-SMC-001/{query['id']}-81401-row.json"
        spec = next(s for s in result["workloads"] if s["path"]==str(path.relative_to(ROOT)))
        cell = load_bound(spec)
        output = load_bound(cell["output"])
        if (cell["query_id"]!=query["id"] or cell["seed"]!=81401 or cell["arm"]!="row"
                or output["status"]!="complete_particles" or len(output["keys"])!=128
                or any(len(k)!=23 or any(type(u) is not int or not 0<=u<42 for u in k) for k in output["keys"])
                or any(any(p>=0 and k[r]!=p for r,p in enumerate(query["partial_key"])) for k in output["keys"])
                or len(output["key_log_likelihoods"])!=128 or any(v is None for v in output["key_log_likelihoods"])):
            raise ValueError("Closed supported128-key conditional bank required")
        selected.append({"query":query,"parent_cell":spec,"parent_output":cell["output"],
            "keys":output["keys"][:16],"selection":"first16_final_particles_no_contextual_score_selection"})
    return result,admission(freeze),selected


def stages(lengths):
    return tuple((tuple(min(n,cut) for n in lengths),False) for cut in (1,8,32))+(
        (tuple(lengths),False),(tuple(lengths),True))


def timed(call):
    wall,cpu = time.monotonic(),time.process_time()
    try:
        result = {"status":"complete","log_value":float(call())}
        if result["log_value"]==-math.inf:
            result["log_value"] = None
    except (MemoryError,RuntimeError) as exc:
        # Only per-record/source-state graph caps are registered outcomes.
        # Global work/memory envelopes or other exceptions abort the campaign.
        if not any(s in str(exc) for s in ("Exact lattice node cap","Exact lattice edge cap",
                                           "Bridge source-state allocation cap")):
            raise
        result = {"status":"graph_cap","error":str(exc)}
    return {**result,"wall_seconds":time.monotonic()-wall,"cpu_seconds":time.process_time()-cpu}


def closure(native,query,key):
    units = tuple("".join(unit) for n in (1,2) for unit in itertools.product("ABCDEF",repeat=n))
    code_units = tuple(units[k] for k in key)
    value,nodes,edges,calls = 0.,0,0,0
    for text,offset,context in zip(query["cipher"],query["offsets"],query["supplied_contexts_ignored_by_iid"],strict=True):
        if offset==len(text):
            continue
        conditional = copy.copy(native)
        conditional.root = context
        record = "".join("ABCDEF"[g] for g in text[offset:])
        result = conditional.score(code_units,record,1/225,max_nodes=300_000,max_edges=2_000_000)
        value += result.log_likelihood
        nodes,edges,calls = nodes+result.reachable_nodes,edges+result.edges,calls+1
    return value,nodes,edges,calls


def compare(a,b):
    if a["status"]!="complete" or b["status"]!="complete":
        return None
    if a["log_value"] is None or b["log_value"] is None:
        assert a["log_value"]==b["log_value"]
        return 0.
    delta = abs(a["log_value"]-b["log_value"])
    assert delta<=1e-7
    return delta


def evaluate(spec,source,native,edge_limit=400_000_000):
    query,keys = spec["query"],np.array(spec["keys"],dtype=np.int32)
    started = time.monotonic()
    bridge = CensoredNativeBridge(query["cipher"],native,offsets=query["offsets"],
        contexts=query["supplied_contexts_ignored_by_iid"],max_tables=80,max_edges=edge_limit)
    native_init = time.monotonic()-started
    started = time.monotonic()
    reference = FixedKeyBridge(query["cipher"],source.probabilities,source.transitions,
        offsets=query["offsets"],contexts=query["supplied_contexts_ignored_by_iid"],
        max_tables=5,max_edges=100_000_000,max_states_per_table=100_000,max_work_bytes=768*1024**2)
    python_init = time.monotonic()-started
    rows = []
    for stage,(cuts,closed) in enumerate(stages(bridge.lengths)):
        for index,key in enumerate(keys):
            before = (bridge.nodes,bridge.edges,bridge.native_calls)
            value = timed(lambda:bridge.log_values(key[None,:],cuts,closed=closed)[0])
            value.update({"nodes":bridge.nodes-before[0],"edges":bridge.edges-before[1],
                          "native_calls":bridge.native_calls-before[2]})
            row = {"stage":stage,"cuts":cuts,"closed":closed,"key_index":index,"native":value}
            if index==0:
                row["python"] = timed(lambda:reference.log_values(key[None,:],cuts,closed=closed)[0])
                row["python_log_delta"] = compare(value,row["python"])
            if closed:
                details = []
                def check_closure():
                    found = closure(native,query,key)
                    details.append(found)
                    return found[0]
                row["closure_reference"] = timed(check_closure)
                row["closure_log_delta"] = compare(value,row["closure_reference"])
                if details:
                    _,nodes,edges,calls = details[0]
                    row["closure_reference"].update({"nodes":nodes,"edges":edges,"native_calls":calls})
                    if value["status"]=="complete":
                        assert (value["nodes"],value["edges"],value["native_calls"])==(nodes,edges,calls)
            rows.append(row)
    result = {"query_id":query["id"],"selection":spec,"workloads":rows,
        "native_construction_seconds":native_init,"python_construction_seconds":python_init,
        "tables":bridge.tables,"native_calls":bridge.native_calls,"native_edges":bridge.edges,
        "native_nodes":bridge.nodes,"maximum_native_nodes":bridge.maximum_native_nodes,
        "native_owned_work_envelope":bridge.maximum_owned_work_envelope,"pinned_source_bytes":bridge.source_pinned_bytes,
        "python_tables":reference.tables,"python_row_operations":reference.edges}
    reference = bridge = None
    gc.collect()
    return result


def summarize(cells):
    rows = [r for cell in cells for r in cell["workloads"]]
    native = [r["native"] for r in rows]
    python = [r["python"] for r in rows if "python" in r]
    closure = [r["closure_reference"] for r in rows if "closure_reference" in r]
    deltas = [r[k] for r in rows for k in ("python_log_delta","closure_log_delta") if k in r and r[k] is not None]
    positive = all(r["status"]=="complete" and r["log_value"] is not None for r in native)
    checked = sum(r["status"]=="complete" for r in python)
    closed_checked = all(r["status"]=="complete" for r in closure)
    return {"native_attempts":len(native),"native_complete":sum(r["status"]=="complete" for r in native),
        "native_all_positive":positive,"python_attempts":len(python),"python_complete":checked,
        "closure_attempts":len(closure),"closure_complete":sum(r["status"]=="complete" for r in closure),
        "maximum_comparison_log_delta":max(deltas,default=None),
        "native_summed_wall_seconds":sum(r["wall_seconds"] for r in native),
        "python_summed_wall_seconds":sum(r["wall_seconds"] for r in python),
        "closure_summed_wall_seconds":sum(r["wall_seconds"] for r in closure),
        "contextual_engineering_gate":"PASS" if len(native)==960 and positive and checked>=36 and closed_checked else "FAIL",
        "source_context_preserved":True,"new_smc_recovery_or_historical_qualification":False}


def guard(wall,cpu,cells):
    if (resource_report(wall,cpu)["peak_rss_bytes"]>2*1024**3
            or sum(c["native_edges"] for c in cells)>1_000_000_000):
        raise MemoryError("Registered2GiB RSS /1B completed-query native edges exceeded")


def run(freeze):
    parent,benchmark,selected = inputs(freeze)
    save_new(OUT/"started.json",{"freeze":freeze,"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells = time.monotonic(),time.process_time(),[]
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        if parent["source"]!=chosen["counts"] or parent["source_arrays"]!=identity:
            raise ValueError("Original contextual source counts/arrays changed")
        native = NativeMarginal(source,benchmark["build"])
        for spec in selected:
            remaining = 1_000_000_000-sum(c["native_edges"] for c in cells)
            if remaining<=0:
                raise RuntimeError("Campaign native edge budget exhausted")
            cell = evaluate(spec,source,native,min(400_000_000,remaining))
            save_new(OUT/f"{cell['query_id']}.json",cell)
            cells.append(cell)
            guard(wall,cpu,cells)
            print(f"Closed {cell['query_id']}: 80 native,5Python,16closure comparisons",flush=True)
        assert len(cells)==12 and array_identity(source)==identity
        manifests = [artifact(OUT/f"{c['query_id']}.json") for c in cells]
        if sum(s["bytes"] for s in manifests)>16*1024**2:
            raise MemoryError("Registered compact output cap exceeded")
        save_new(OUT/"result.json",{"experiment":EXP,"freeze":freeze,"numpy_version":np.__version__,
            "parent_result":artifact(ROOT/"results/DICTIONARY-SMC-001/result.json"),
            "parent_audit":artifact(ROOT/"results/DICTIONARY-SMC-001/audit.json"),
            "source":chosen["counts"],"source_arrays":identity,"native_build":benchmark["build"],
            "workloads":manifests,
            "summary":summarize(cells),"resources":resource_report(wall,cpu),"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json",{"error":repr(exc),"closed_queries":len(cells),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",required=True)
    run(parser.parse_args().freeze)
