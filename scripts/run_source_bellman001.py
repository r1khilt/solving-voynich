"""Registered finite early Bellman-backed guide versus unchanged iid guide."""
from __future__ import annotations

import argparse
import gzip
import json
import signal
import time
from pathlib import Path

import numpy as np

from scripts.run_source_state_systems001 import (
    ROOT, NativeMarginal, array_identity, artifact, config as old_config, diagnose,
    finite, inputs as parent_inputs, limit_resources, load_source, predict,
    require_frozen, resource_report, save_new,
)
from scripts.run_source_prune_diag002 import PATHS as PARENT_PATHS
from voynich.source_bellman_guide import NativeBellmanGuide, build_bellman

EXP="SOURCE-BELLMAN-001"
OUT,BULK=ROOT/"results"/EXP,ROOT/"outputs"/EXP
PARENT=ROOT/"results/SOURCE-STATE-SYSTEMS-001"
ARMS=("iid","bellman_one")
PATHS=sorted(set([*PARENT_PATHS,"src/voynich/source_bellman_guide.py",
    "tests/test_source_bellman_guide.py","tests/test_source_bellman001.py",
    "scripts/run_source_bellman001.py","scripts/audit_source_bellman001.py",
    "scripts/check_bellman_theory001.py","docs/experiments/SOURCE-BELLMAN-001.md",
    "docs/research/source-bellman-2026-10-01.md","results/SOURCE-BELLMAN-001/native-build.json",
    "results/SOURCE-BELLMAN-001/theory.json"]))


def config(arm):
    if arm not in ARMS:
        raise ValueError("Unregistered guide arm")
    return {**old_config("wide_guided"),"max_seconds":600.,
        "lookahead_depth":int(arm=="bellman_one"),"lookahead_layers":16}


def load_bound(spec):
    if artifact(ROOT/spec["path"])!=spec:
        raise ValueError("Closed artifact changed")
    raw=(ROOT/spec["path"]).read_bytes()
    return json.loads(gzip.decompress(raw) if spec["path"].endswith(".gz") else raw)


def prepare():
    from scripts.check_bellman_theory001 import check
    build=build_bellman(BULK/"build")
    save_new(OUT/"native-build.json",{"build":build,"library":artifact(Path(build["library_path"])),
        "generated_cpp":artifact(Path(build["generated_cpp_path"])),"no_empirical_search_or_training":True,"paid_spend_usd":0})
    save_new(OUT/"theory.json",check())


def inputs(freeze):
    require_frozen(freeze,PATHS)
    benchmark,parent,cases,_=parent_inputs(freeze)
    closed=json.loads((PARENT/"result.json").read_text())
    audit=json.loads((PARENT/"audit.json").read_text())
    if audit["result"]!=artifact(PARENT/"result.json") or audit["status"]!="PASS_full_native_state_and_prediction_replay":
        raise ValueError("Original complete source-state audit required")
    build=json.loads((OUT/"native-build.json").read_text())
    for spec in (build["library"],build["generated_cpp"]):
        if artifact(ROOT/spec["path"])!=spec:
            raise ValueError("Bellman compiled build changed")
    return benchmark,parent,cases,closed,build


def base_output(result):
    extra=("lookahead_depth","lookahead_layers","lookahead_calls","lookahead_actions","tail_is_heuristic_with_floor")
    return {k:v for k,v in result.items() if k not in extra}


def summarize(rows):
    result={}
    for arm in ARMS:
        selected=[r for r in rows if r["arm"]==arm]
        result[arm]={"cells":len(selected),"read_cells":sum(r["diagnostic"]["edit_errors"] is not None for r in selected),
            "edit_errors_read_cells_only":sum(r["diagnostic"]["edit_errors"] or 0 for r in selected),
            "source_letters":sum(r["diagnostic"]["true_source_letters"] for r in selected),
            "exact_records":sum(r["diagnostic"]["exact_records"] for r in selected),
            "matched_used_rows":sum(r["diagnostic"]["matched_gold_used_rows"] or 0 for r in selected),
            "gold_used_rows":sum(r["diagnostic"]["gold_used_rows"] for r in selected),
            "cells_with_returned_gold_used_key":sum(r["diagnostic"]["gold_used_mapping_in_returned_terminals"] for r in selected),
            "guide_tables":sum(r["summary"]["guide_tables_built"] for r in selected),
            "lookahead_calls":sum(r["summary"]["lookahead_calls"] for r in selected),
            "lookahead_actions":sum(r["summary"]["lookahead_actions"] for r in selected),
            "summed_cell_wall_seconds":sum(r["cell_wall_seconds"] for r in selected)}
    a,b=result["iid"],result["bellman_one"]
    paired=sum(r["diagnostic"]["edit_errors"] is not None for r in rows)==8
    improves=sum(next(r for r in rows if r["case"]==c and r["arm"]=="bellman_one")["diagnostic"]["edit_errors"] is not None
        and next(r for r in rows if r["case"]==c and r["arm"]=="iid")["diagnostic"]["edit_errors"] is not None
        and next(r for r in rows if r["case"]==c and r["arm"]=="bellman_one")["diagnostic"]["edit_errors"]<next(r for r in rows if r["case"]==c and r["arm"]=="iid")["diagnostic"]["edit_errors"] for c in range(4))
    return {"arms":result,"paired_all_readings":paired,"improved_cases":improves,
        "exploratory_signal_supported":paired and b["edit_errors_read_cells_only"]<=.9*a["edit_errors_read_cells_only"] and improves>=2,
        "fresh_recovery_qualification":False,"equal_cpu_claimed":False}


def guard(wall,cpu,size):
    if resource_report(wall,cpu)["peak_rss_bytes"]>2*1024**3 or size>256*1024**2:
        raise MemoryError("Registered sampled2GiB/256MiB output guard")


def run(freeze):
    benchmark,parent,cases,closed,build=inputs(freeze)
    save_new(OUT/"started.json",{"freeze":freeze,"start_unix":time.time(),"paid_spend_usd":0})
    limit_resources(2400,2200)
    wall,cpu,rows,size=time.monotonic(),time.process_time(),[],0
    try:
        source,selected=load_source()
        identity=array_identity(source)
        assert selected["counts"]==closed["source"] and identity==closed["source_arrays"] and parent["fixtures"]==closed["fixtures"]
        engine=NativeBellmanGuide(source.probabilities,source.transitions,build["build"])
        native=NativeMarginal(source,benchmark["build"])
        for fixture in cases:
            for arm in ARMS:
                name=f"case{fixture['case']}-{arm}"
                cw,cc=time.monotonic(),time.process_time()
                fitted=engine.search(fixture["cipher"],**config(arm))
                if arm=="iid":
                    oldrow=load_bound(closed["workloads"][4*fixture["case"]+3])
                    if finite(base_output(fitted))!=load_bound(oldrow["fitted"]):
                        raise ValueError("Depth-zero control differs from original wide search")
                    assert fitted["lookahead_calls"]==fitted["lookahead_actions"]==0
                archive=save_new(BULK/f"{name}.json.gz",finite(fitted),compressed=True)
                prediction=predict(fitted,fixture["cipher"],fixture["case"],source,native)
                pred_spec=save_new(BULK/f"{name}-prediction.json",prediction)
                diagnostic=diagnose(fitted,prediction,fixture,source)
                size+=archive["bytes"]+pred_spec["bytes"]
                guard(wall,cpu,size)
                row={"name":name,"case":fixture["case"],"arm":arm,"config":config(arm),"fitted":archive,
                    "prediction":pred_spec,"diagnostic":diagnostic,
                    "summary":finite({k:v for k,v in fitted.items() if k not in ("terminals","trace")}),
                    "cell_wall_seconds":time.monotonic()-cw,"cell_cpu_seconds":time.process_time()-cc}
                save_new(OUT/f"{name}.json",row)
                rows.append(row)
                print(f"{name}: edits={diagnostic['edit_errors']}, guide={fitted['guide_tables_built']}, backupactions={fitted['lookahead_actions']}, {fitted['stop_reason']}, {row['cell_wall_seconds']:.2f}s",flush=True)
        assert len(rows)==8 and array_identity(source)==identity
        save_new(OUT/"result.json",{"experiment":EXP,"freeze":freeze,"source":selected["counts"],"source_arrays":identity,
            "fixtures":parent["fixtures"],"native_build":benchmark["build"],"bellman_build":artifact(OUT/"native-build.json"),
            "parent_result":artifact(PARENT/"result.json"),"parent_audit":artifact(PARENT/"audit.json"),
            "workloads":[artifact(OUT/f"{r['name']}.json") for r in rows],"summary":summarize(rows),
            "resources":resource_report(wall,cpu),"bulk_output_bytes":size,"numpy_version":np.__version__,
            "paid_spend_usd":0,"scope":"Exposed synthetic guide ablation, matched state budgets but extra guide work; no fresh qualification or historical decipherment."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json",{"error":repr(exc),"completed_workloads":len(rows),"resources":resource_report(wall,cpu),"no_retry":True})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--freeze")
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
    elif args.freeze:
        run(args.freeze)
    else:
        parser.error("Frozen registration required")
