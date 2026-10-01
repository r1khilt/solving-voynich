"""One full native RNG replay plus independent final-key contextual likelihoods."""

from __future__ import annotations

import gc
import json
import signal
import time

import numpy as np

from scripts.audit_censored_context001 import logical
from scripts.run_contextual_gibbs001 import (
    OUT, ROOT, NativeMarginal, array_identity, artifact, evaluate, guard, inputs,
    limit_resources, load_bound, load_source, mapping_metric, metrics,
    resource_report, save_new, summarize,
)
from voynich.dictionary_smc import FixedKeyBridge


def audit():
    result = json.loads((OUT/"result.json").read_text())
    parent,benchmark,queries,fixtures = inputs(result["freeze"])
    assert result["queries"]==queries and result["native_build"]==benchmark["build"]
    assert result["numpy_version"]==np.__version__
    assert result["qualification"]==artifact(ROOT/"results/CENSORED-CONTEXT-001/result.json")
    assert result["qualification_audit"]==artifact(ROOT/"results/CENSORED-CONTEXT-001/audit.json")
    save_new(OUT/"audit-started.json",{"freeze":result["freeze"],"start_unix":time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells,bulk_bytes,checked,max_delta = time.monotonic(),time.process_time(),[],0,0,0.
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent["source"]==result["source"]==chosen["counts"]
        assert parent["source_arrays"]==result["source_arrays"]==identity
        native = NativeMarginal(source,benchmark["build"])
        for manifest in result["workloads"]:
            cell = load_bound(manifest)
            query,fixture = queries[cell["case"]],fixtures[cell["case"]]
            value = load_bound(cell["output"])
            remaining = 8_000_000_000-sum(c["summary"]["edges"] for c in cells)
            assert remaining>0
            replay = evaluate(query,native,cell["seed"],cell["arm"],min(2_000_000_000,remaining))
            assert replay.pop("value")==value
            assert logical(replay)==logical({k:cell[k] for k in replay})
            assert cell["summary"]==metrics(value) and cell["mapping"]==mapping_metric(value,fixture)
            if value["status"]=="complete_particles":
                assert cell["summary"]["tables"]==cell["nominal_complete_table_budget"]
                reference = FixedKeyBridge(query["cipher"],source.probabilities,source.transitions,
                    contexts=(0,0),max_tables=32,max_edges=1_000_000_000,
                    max_states_per_table=300_000,max_work_bytes=768*1024**2)
                expected = np.array(value["key_log_likelihoods"])
                observed = reference.log_values(np.array(value["keys"],dtype=np.int32),reference.lengths,closed=True)
                assert np.all(np.isfinite(observed))
                delta = float(np.max(np.abs(observed-expected)))
                assert delta<=1e-7
                max_delta,checked = max(max_delta,delta),checked+len(observed)
                reference = None
                gc.collect()
            bulk_bytes += cell["output"]["bytes"]
            cells.append(cell)
            guard(wall,cpu,bulk_bytes,cells)
            print(f"Audited case{cell['case']}-{cell['seed']}-{cell['arm']}: {value['status']}",flush=True)
        assert len(cells)==16 and result["summary"]==summarize(cells)
        assert bulk_bytes==result["bulk_bytes"] and array_identity(source)==identity
        assert checked==32*sum(c["summary"]["status"]=="complete_particles" for c in cells)
        for r in (result["resources"],resource_report(wall,cpu)):
            assert r["wall_seconds"]<=1800 and r["cpu_seconds"]<=1600 and 0<r["peak_rss_bytes"]<=2*1024**3
        save_new(OUT/"audit.json",{"status":"PASS_complete_contextual_sampling_replay",
            "result":artifact(OUT/"result.json"),"calls":len(cells),"independent_final_key_scores":checked,
            "maximum_final_score_log_delta":max_delta,"resources":resource_report(wall,cpu),
            "same_author":True,"independent_agent_review":False,
            "intermediate_score_replay_shares_native_backend":True,"paid_spend_usd":0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json",{"error":repr(exc),"audited_calls":len(cells),"no_retry":True,
            "resources":resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    audit()
