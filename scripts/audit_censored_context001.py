"""One complete contextual benchmark replay, with independent Python prefixes."""

from __future__ import annotations

import json
import signal
import time

import numpy as np

from scripts.benchmark_censored_context001 import (
    OUT, ROOT, NativeMarginal, array_identity, artifact, evaluate, guard, inputs,
    limit_resources, load_bound, load_source, resource_report, save_new, summarize,
)


def logical(value):
    if isinstance(value,dict):
        return {k:logical(v) for k,v in value.items() if not k.endswith("seconds")}
    if isinstance(value,(list,tuple)):
        return [logical(v) for v in value]
    return value


def audit():
    result = json.loads((OUT/"result.json").read_text())
    parent,benchmark,selected = inputs(result["freeze"])
    assert result["parent_result"]==artifact(ROOT/"results/DICTIONARY-SMC-001/result.json")
    assert result["parent_audit"]==artifact(ROOT/"results/DICTIONARY-SMC-001/audit.json")
    assert result["numpy_version"]==np.__version__ and result["native_build"]==benchmark["build"]
    save_new(OUT/"audit-started.json",{"freeze":result["freeze"],"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells = time.monotonic(),time.process_time(),[]
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent["source"]==result["source"]==chosen["counts"]
        assert parent["source_arrays"]==result["source_arrays"]==identity
        native = NativeMarginal(source,benchmark["build"])
        for spec in selected:
            manifest = result["workloads"][len(cells)]
            assert manifest==artifact(OUT/f"{spec['query']['id']}.json")
            cell = load_bound(manifest)
            remaining = 1_000_000_000-sum(c["native_edges"] for c in cells)
            assert remaining>0
            replay = evaluate(spec,source,native,min(400_000_000,remaining))
            assert logical(cell)==logical(replay)
            assert cell["tables"]==80 and cell["python_tables"]==5
            assert len(cell["workloads"])==80
            for row in cell["workloads"]:
                for key in ("native","python","closure_reference"):
                    if key in row:
                        assert 0<=row[key]["wall_seconds"]<=result["resources"]["wall_seconds"]
                        assert 0<=row[key]["cpu_seconds"]<=result["resources"]["cpu_seconds"]
            cells.append(cell)
            guard(wall,cpu,cells)
            print(f"Audited {spec['query']['id']}: every key/prefix/closure and caps",flush=True)
        assert len(cells)==len(result["workloads"])==12
        assert summarize(cells)==result["summary"] and array_identity(source)==identity
        assert sum(s["bytes"] for s in result["workloads"])<=16*1024**2
        for r in (result["resources"],resource_report(wall,cpu)):
            assert 0<=r["wall_seconds"]<=1800 and 0<=r["cpu_seconds"]<=1600
            assert 0<r["peak_rss_bytes"]<=2*1024**3 and r["paid_spend_usd"]==0
        save_new(OUT/"audit.json",{"status":"PASS_complete_contextual_benchmark_replay",
            "result":artifact(OUT/"result.json"),"native_attempts":result["summary"]["native_attempts"],
            "python_attempts":result["summary"]["python_attempts"],
            "legacy_closure_attempts":result["summary"]["closure_attempts"],
            "maximum_comparison_log_delta":result["summary"]["maximum_comparison_log_delta"],
            "resources":resource_report(wall,cpu),"same_author":True,"independent_agent_review":False,
            "no_new_smc_search_reader_training_or_holdout":True,"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json",{"error":repr(exc),"audited_queries":len(cells),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    audit()
