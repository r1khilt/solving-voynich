"""Single complete replay plus alternate forward fixed-dictionary likelihoods."""

from __future__ import annotations

import json
import math
import signal
import time

import numpy as np

from scripts.run_shared_key_guide001 import (
    ARMS, OUT, ROOT, SEEDS, array_identity, artifact, evaluate, guard, inputs,
    limit_resources, load_bound, load_source, queries, renewal, resource_report,
    save_new, summarize,
)


def forward_logs(cipher,offsets,row,keys):
    """Alternate direction; explicit per-letter outgoing edges, no suffix table."""
    n = len(keys)
    weights = np.zeros((n,42))
    for letter,p in enumerate((row+1e-8)/(1+len(row)*1e-8)):
        for code in range(42):
            weights[:,code] += p*(keys[:,letter]==code)
    with np.errstate(divide="ignore"):
        edges = np.log(weights)+math.log1p(-1/225)
    result = np.zeros(n)
    for text,start in zip(cipher,offsets,strict=True):
        if start==len(text):
            continue
        mass = np.full((n,len(text)+1),-math.inf)
        mass[:,start] = 0.
        for pos in range(start,len(text)):
            mass[:,pos+1] = np.logaddexp(mass[:,pos+1],mass[:,pos]+edges[:,text[pos]])
            if pos+1<len(text):
                code = 6+6*text[pos]+text[pos+1]
                mass[:,pos+2] = np.logaddexp(mass[:,pos+2],mass[:,pos]+edges[:,code])
        result += mass[:,len(text)]+math.log(1/225)
    return result


def alternate(query,row,seed,arm,stored):
    bank = np.random.default_rng(seed).integers(42,size=(672,23),dtype=np.int32)
    if arm!="mc672":
        bank = bank[:16]
    key = np.array(query["partial_key"])
    span = 42 if arm=="rb16" else 1
    free = np.flatnonzero(key<0)
    chosen = int(free[np.argmax(row[free])]) if span>1 else None
    logs = []
    for start in range(0,len(bank)*span,64):
        indices = np.arange(start,min(start+64,len(bank)*span))
        keys = bank[indices//span].copy()
        for letter,code in enumerate(key):
            if code>=0:
                keys[:,letter] = code
        if span>1:
            keys[:,chosen] = indices%42
        logs.extend(forward_logs(query["cipher"],query["offsets"],row,keys))
    logs = np.array(logs).reshape(len(bank),span)
    grouped = np.logaddexp.reduce(logs,axis=1)-math.log(span)
    expected = stored["base_log_likelihoods"]
    maximum = 0.
    for a,b in zip(grouped,expected,strict=True):
        if b is None:
            assert a==-math.inf
        else:
            assert math.isfinite(a)
            delta = abs(float(a)-b)
            assert delta<=1e-7
            maximum = max(maximum,delta)
    assert stored["positive_fixed_dictionaries"]==int(np.isfinite(logs).sum())
    return logs.size,maximum


def audit():
    result = json.loads((OUT/"result.json").read_text())
    parent,fixtures,cells,archives = inputs(result["freeze"])
    assert result["parent_result"]==artifact(ROOT/"results/SOURCE-PRUNE-DIAG-002/result.json")
    assert result["parent_audit"]==artifact(ROOT/"results/SOURCE-PRUNE-DIAG-002/audit.json")
    declared = queries(fixtures,cells,archives)
    assert declared==result["queries"] and result["fixtures"]==parent["fixtures"]
    assert result["numpy_version"]==np.__version__ and result["paid_spend_usd"]==0
    save_new(OUT/"audit-started.json",{"freeze":result["freeze"],"start_unix":time.time()})
    limit_resources(600,500)
    wall,cpu,rows,count,maximum = time.monotonic(),time.process_time(),[],0,0.
    try:
        source,selected = load_source()
        identity = array_identity(source)
        assert selected["counts"]==result["source"] and identity==result["source_arrays"]
        for query in declared:
            baseline = renewal(query,source.probabilities[0])
            for seed in SEEDS:
                for arm in ARMS:
                    spec = result["workloads"][len(rows)]
                    assert spec==artifact(OUT/f"{query['id']}-{seed}-{arm}.json")
                    row = load_bound(spec)
                    assert (row["query"],row["seed"],row["arm"],row["renewal_log_guide"])==(query,seed,arm,baseline)
                    assert row["estimate"]==evaluate(query,source.probabilities[0],seed,arm)
                    checked,delta = alternate(query,source.probabilities[0],seed,arm,row["estimate"])
                    count,maximum = count+checked,max(maximum,delta)
                    assert 0<=row["wall_seconds"]<=result["resources"]["wall_seconds"]
                    rows.append(row)
                    guard(wall,cpu)
            print(f"Audited {query['id']}: full bank and forward likelihoods",flush=True)
        assert len(rows)==len(result["workloads"])==144 and summarize(rows)==result["summary"]
        assert sum(s["bytes"] for s in result["workloads"])<=16*1024**2
        assert array_identity(source)==identity
        for r in (result["resources"],resource_report(wall,cpu)):
            assert 0<=r["wall_seconds"]<=600 and 0<=r["cpu_seconds"]<=500
            assert 0<r["peak_rss_bytes"]<=2*1024**3 and r["paid_spend_usd"]==0
        save_new(OUT/"audit.json",{"status":"PASS","result":artifact(OUT/"result.json"),"calls":len(rows),
            "alternate_forward_fixed_dictionary_checks":int(count),"maximum_alternate_log_delta":maximum,
            "resources":resource_report(wall,cpu),"same_author_replay":True,"independent_agent_review":False,
            "no_new_native_search_source_scoring_reader_training_or_holdout":True,"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json",{"error":repr(exc),"audited_calls":len(rows),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    audit()
