"""Complete RNG/core replay with all fixed-key values checked by outgoing DP."""

from __future__ import annotations

import json
import math
import signal
import time

import numpy as np

from scripts.run_dictionary_smc001 import (
    ARMS, BULK, OUT, ROOT, SEEDS, array_identity, artifact, compact, evaluate,
    guard, inputs, limit_resources, load_source, resource_report, save_new, summarize,
)
from scripts.run_shared_key_guide001 import load_bound
from voynich.dictionary_smc import FixedKeyBridge


def forward_values(bridge,keys,cuts,closed):
    """Explicit per-row outgoing edges and minimal crossings; no suffix DP."""
    weights = np.zeros((len(keys),len(bridge.units)))
    for row,p in enumerate(bridge.p[0]):
        for code in range(len(bridge.units)):
            weights[:,code] += p*(keys[:,row]==code)
    with np.errstate(divide="ignore"):
        edges = np.log(weights)+math.log1p(-bridge.rho)
    value = np.zeros(len(keys))
    for text,offset,cut,remaining in zip(bridge.cipher,bridge.offsets,cuts,bridge.lengths,strict=True):
        if not remaining or not cut:
            continue
        y = text[offset:offset+cut]
        dp = np.full((len(keys),cut),-math.inf)
        dp[:,0] = 0.
        crossed = np.full(len(keys),-math.inf)
        for pos in range(cut):
            for length in (1,2):
                end = pos+length
                if end>cut and closed:
                    continue
                if length==1:
                    codes = (y[pos],)
                elif end<=cut:
                    codes = (bridge.glyphs+bridge.glyphs*y[pos]+y[pos+1],)
                else:
                    codes = tuple(bridge.glyphs+bridge.glyphs*y[pos]+g for g in range(bridge.glyphs))
                for code in codes:
                    arriving = dp[:,pos]+edges[:,code]
                    if end>=cut:
                        crossed = np.logaddexp(crossed,arriving)
                    else:
                        dp[:,end] = np.logaddexp(dp[:,end],arriving)
        value += crossed+(math.log(bridge.rho) if closed else 0.)
    return value


class CheckedBridge(FixedKeyBridge):
    checked = 0
    maximum = 0.

    def log_values(self,keys,cuts,*,closed=False):
        actual = super().log_values(keys,cuts,closed=closed)
        expected = forward_values(self,keys,cuts,closed)
        assert np.array_equal(np.isneginf(actual),np.isneginf(expected))
        supported = np.isfinite(actual)
        assert np.all(np.abs(actual[supported]-expected[supported])<=1e-7)
        type(self).checked += len(actual)
        type(self).maximum = max(type(self).maximum,float(np.max(np.abs(actual[supported]-expected[supported]),initial=0.)))
        return actual  # Replay original RNG/acceptance exactly; alternate values are verification only.


def audit():
    result = json.loads((OUT/"result.json").read_text())
    _,closed = inputs(result["freeze"])
    assert result["queries"]==closed["queries"]
    assert result["parent_result"]==artifact(ROOT/"results/SHARED-KEY-GUIDE-001/result.json")
    assert result["parent_audit"]==artifact(ROOT/"results/SHARED-KEY-GUIDE-001/audit.json")
    assert result["numpy_version"]==np.__version__ and result["paid_spend_usd"]==0
    save_new(OUT/"audit-started.json",{"freeze":result["freeze"],"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,rows,bulk_bytes = time.monotonic(),time.process_time(),[],0
    CheckedBridge.checked,CheckedBridge.maximum = 0,0.
    try:
        source,selected = load_source()
        identity = array_identity(source)
        assert selected["counts"]==result["source"] and identity==result["source_arrays"]
        for query in result["queries"]:
            for seed in SEEDS:
                for arm in ARMS:
                    name = f"{query['id']}-{seed}-{arm}"
                    spec = result["workloads"][len(rows)]
                    assert spec==artifact(OUT/f"{name}.json")
                    row = load_bound(spec)
                    assert (row["query_id"],row["seed"],row["arm"])==(query["id"],seed,arm)
                    assert row["output"]==artifact(BULK/f"{name}.json.gz")
                    value = load_bound(row["output"])
                    assert value==evaluate(query,source.probabilities[0],seed,arm,CheckedBridge)
                    assert row["summary"]==compact(value)
                    assert 0<=row["wall_seconds"]<=result["resources"]["wall_seconds"]
                    bulk_bytes += row["output"]["bytes"]
                    rows.append(row)
                    guard(wall,cpu,bulk_bytes)
            print(f"Audited {query['id']}: full trajectories and all outgoing likelihoods",flush=True)
        assert len(rows)==len(result["workloads"])==144 and summarize(rows)==result["summary"]
        assert result["bulk_bytes"]==bulk_bytes<=128*1024**2
        assert CheckedBridge.checked==sum(r["summary"]["tables"] for r in rows)
        assert array_identity(source)==identity
        for r in (result["resources"],resource_report(wall,cpu)):
            assert 0<=r["wall_seconds"]<=1800 and 0<=r["cpu_seconds"]<=1600
            assert 0<r["peak_rss_bytes"]<=2*1024**3 and r["paid_spend_usd"]==0
        save_new(OUT/"audit.json",{"status":"PASS","result":artifact(OUT/"result.json"),"calls":len(rows),
            "alternate_outgoing_fixed_dictionary_checks":CheckedBridge.checked,
            "maximum_alternate_log_delta":CheckedBridge.maximum,"resources":resource_report(wall,cpu),
            "same_author_rng_core_replay":True,"independent_agent_review":False,"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json",{"error":repr(exc),"audited_calls":len(rows),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    audit()
