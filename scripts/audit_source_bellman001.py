"""One full guide-search/prediction replay plus alternate full-key scores."""
from __future__ import annotations

import argparse
import json
import math
import signal
import time

import numpy as np

from scripts.run_source_bellman001 import (
    ARMS, OUT, PARENT, NativeBellmanGuide, NativeMarginal, array_identity, artifact,
    base_output, config, diagnose, finite, guard, inputs, limit_resources,
    load_bound, load_source, predict, resource_report, save_new, summarize,
)
from voynich.native_suffix_marginal import marginal_python


def replay_config(stored,arm):
    cfg=config(arm)
    if stored["stop_reason"]=="time_cap":
        if stored["expanded"]<1:
            raise ValueError("Zero-expanded timed call cannot be fully replayed")
        cfg.update(max_expanded=stored["expanded"],max_seconds=1200.)
    return cfg


def compare_replay(stored,actual):
    if stored["stop_reason"]=="time_cap":
        if actual["stop_reason"]!="expansion_cap":
            raise ValueError("Timed prefix replay must stop at frozen expansion count")
        actual={**actual,"stop_reason":"time_cap"}
    if stored!=finite(actual):
        raise ValueError("Full fitted search/guide/terminal/trace replay differs")


def audit():
    result=json.loads((OUT/"result.json").read_text())
    benchmark,parent,cases,closed,build=inputs(result["freeze"])
    save_new(OUT/"audit-started.json",{"freeze":result["freeze"],"start_unix":time.time()})
    limit_resources(2400,2200)
    wall,cpu,rows,size,checked,max_delta=time.monotonic(),time.process_time(),[],0,0,0.
    try:
        source,selected=load_source()
        assert selected["counts"]==result["source"] and array_identity(source)==result["source_arrays"]
        assert parent["fixtures"]==result["fixtures"] and benchmark["build"]==result["native_build"]
        assert result["bellman_build"]==artifact(OUT/"native-build.json")
        assert result["parent_result"]==artifact(PARENT/"result.json") and result["parent_audit"]==artifact(PARENT/"audit.json")
        assert result["numpy_version"]==np.__version__
        assert result["paid_spend_usd"]==result["resources"]["paid_spend_usd"]==0
        assert 0<=result["resources"]["wall_seconds"]<=2400 and 0<=result["resources"]["cpu_seconds"]<=2200
        assert 0<result["resources"]["peak_rss_bytes"]<=2*1024**3
        engine=NativeBellmanGuide(source.probabilities,source.transitions,build["build"])
        native=NativeMarginal(source,benchmark["build"])
        for fixture in cases:
            for arm in ARMS:
                name=f"case{fixture['case']}-{arm}"
                spec=result["workloads"][len(rows)]
                assert spec==artifact(OUT/f"{name}.json")
                row=load_bound(spec)
                assert (row["name"],row["case"],row["arm"],row["config"])==(name,fixture["case"],arm,config(arm))
                stored=load_bound(row["fitted"])
                actual=engine.search(fixture["cipher"],**replay_config(stored,arm))
                compare_replay(stored,actual)
                assert row["summary"]=={k:v for k,v in stored.items() if k not in ("terminals","trace")}
                assert stored["lookahead_actions"]<=64000000 and stored["lookahead_layers"]==16
                if arm=="iid":
                    original=load_bound(closed["workloads"][4*fixture["case"]+3])
                    assert base_output(stored)==load_bound(original["fitted"])
                    assert stored["lookahead_calls"]==stored["lookahead_actions"]==0
                prediction=load_bound(row["prediction"])
                assert finite(predict(actual,fixture["cipher"],fixture["case"],source,native))==prediction
                assert diagnose(stored,prediction,fixture,source)==row["diagnostic"]
                if prediction["status"]=="read":
                    for r,cipher in enumerate(fixture["cipher"]):
                        alternate=marginal_python(source,tuple(prediction["key"]),"".join("ABCDEF"[g] for g in cipher),1/225,max_nodes=2000000,max_edges=8000000)
                        delta=abs(alternate.log_likelihood-prediction["score"]["record_log_likelihoods"][r])
                        assert math.isfinite(delta) and delta<=1e-7
                        checked,max_delta=checked+1,max(max_delta,delta)
                assert 0<=row["cell_wall_seconds"]<=result["resources"]["wall_seconds"]
                assert 0<=row["cell_cpu_seconds"]<=result["resources"]["cpu_seconds"]
                size+=row["fitted"]["bytes"]+row["prediction"]["bytes"]
                guard(wall,cpu,size)
                rows.append(row)
                print(f"Audited {name}: {stored['lookahead_actions']} backup actions",flush=True)
        assert len(rows)==len(result["workloads"])==8 and summarize(rows)==result["summary"]
        assert size==result["bulk_output_bytes"] and array_identity(source)==result["source_arrays"]
        save_new(OUT/"audit.json",{"status":"PASS","result":artifact(OUT/"result.json"),"workloads":8,
            "alternate_full_key_record_scores":checked,"maximum_alternate_delta":max_delta,
            "resources":resource_report(wall,cpu),"paid_spend_usd":0,
            "same_author_and_native_search_implementation":True,"independent_agent_review":False,
            "independent_full_search":False,"depth_zero_matches_all_original_wide_cells":True})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json",{"error":repr(exc),"completed_workloads":len(rows),"resources":resource_report(wall,cpu),"no_retry":True})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    argparse.ArgumentParser().parse_args()
    audit()
