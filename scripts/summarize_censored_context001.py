"""Read-only post-outcome diagnostics; never calls a source/scoring model."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from scripts.benchmark_censored_context001 import OUT, PATHS, ROOT
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import artifact, require_frozen, save_new


def describe(values):
    a = np.array(values,dtype=float)
    assert len(a) and np.all(np.isfinite(a))
    return {"count":len(a),"minimum":float(a.min()),"median":float(np.median(a)),
        "p95":float(np.quantile(a,.95)),"maximum":float(a.max()),"sum":float(a.sum())}


def weight_diagnostic(logs):
    a = np.array(logs,dtype=float)
    weights = np.exp(a-np.logaddexp.reduce(a))
    # Independent scalar normalization checks the NumPy calculation.
    scalar = [math.exp(float(v-a.max())) for v in a]
    total = math.fsum(scalar)
    scalar = [v/total for v in scalar]
    assert np.max(np.abs(weights-np.array(scalar)))<1e-12
    assert abs(float(1/np.square(weights).sum())-1/math.fsum(v*v for v in scalar))<1e-10
    return {"maximum_normalized_weight":max(scalar),
        "inverse_squared_weight_sum":1/math.fsum(v*v for v in scalar),
        "log_range":float(a.max()-a.min())}


def summarize_saved():
    result = json.loads((OUT/"result.json").read_text())
    audit = json.loads((OUT/"audit.json").read_text())
    require_frozen(result["freeze"],PATHS)
    assert audit["status"]=="PASS_complete_contextual_benchmark_replay"
    assert audit["result"]==artifact(OUT/"result.json")
    assert result["summary"]["contextual_engineering_gate"]=="PASS"
    rows,cells,diagnostics = [],[],[]
    for manifest in result["workloads"]:
        cell = load_bound(manifest)
        spec = cell["selection"]
        assert spec["parent_cell"]==artifact(ROOT/spec["parent_cell"]["path"])
        parent = load_bound(spec["parent_cell"])
        assert parent["output"]==spec["parent_output"]
        bank = load_bound(spec["parent_output"])
        assert spec["keys"]==bank["keys"][:16]
        observed = cell["workloads"]
        assert len(observed)==80 and all(r["native"]["status"]=="complete" for r in observed)
        values = [[r["native"]["log_value"] for r in observed if r["stage"]==s] for s in range(5)]
        assert all(len(v)==16 and all(x is not None for x in v) for v in values)
        for index in range(16):
            assert all(values[s+1][index]<=values[s][index]+1e-7 for s in range(4))
        contextual,iid = np.array(values[4]),np.array(bank["key_log_likelihoods"][:16])
        correction = contextual-iid
        lengths = tuple(len(c)-o for c,o in zip(spec["query"]["cipher"],spec["query"]["offsets"],strict=True))
        active_records = sum(n>0 for n in lengths)
        closure_delta = np.array(values[4])-np.array(values[3])
        assert np.max(closure_delta)<=active_records*math.log(1/225)+1e-7
        inversions,comparable = 0,0
        for i in range(16):
            for j in range(i):
                if abs(float(contextual[i]-contextual[j]))>1e-10 and abs(float(iid[i]-iid[j]))>1e-10:
                    comparable += 1
                    inversions += int((contextual[i]>contextual[j])!=(iid[i]>iid[j]))
        diagnostics.append({"query_id":cell["query_id"],"particles":16,
            "distinct_full_keys":len({tuple(k) for k in spec["keys"]}),
            "contextual_minus_iid_log_values":describe(correction),
            "fixed_bank_correction_weights":weight_diagnostic(correction),
            "ranking_inversions":inversions,"comparable_pairs":comparable,
            "full_prefix_to_closed_log_increment":describe(closure_delta),
            "fixed_bank_closure_weights":weight_diagnostic(closure_delta),
            "constant_exact_endpoint_log_increment":active_records*math.log(1/225)})
        rows.extend(observed)
        cells.append(cell)
    assert len(cells)==12 and len(rows)==960
    return {"kind":"post_outcome_read_only_closed_scores_no_new_model_calls",
        "result":artifact(OUT/"result.json"),"audit":artifact(OUT/"audit.json"),
        "queries":diagnostics,
        "native_wall_seconds_by_stage":{str(s):describe([r["native"]["wall_seconds"] for r in rows if r["stage"]==s]) for s in range(5)},
        "native_nodes_per_attempt":describe([r["native"]["nodes"] for r in rows]),
        "native_edges_per_attempt":describe([r["native"]["edges"] for r in rows]),
        "native_total_edges":sum(c["native_edges"] for c in cells),
        "maximum_owned_work_envelope":max(c["native_owned_work_envelope"] for c in cells),
        "python_construction_seconds":describe([c["python_construction_seconds"] for c in cells]),
        "native_construction_seconds":describe([c["native_construction_seconds"] for c in cells]),
        "restrictions":["First16 correlated IID-bank particles; no additional selection or scoring",
            "Fixed-bank normalized-weight statistics are not effective independent posterior sample counts",
            "No evidence estimate, whole-space posterior coverage, or recovered reading is computed",
            "Timings are bank-specific; changing keys or sampler can change graph size/cost"]}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    value = summarize_saved()
    if args.save:
        save_new(OUT/"post-outcome.json",value)
    print(json.dumps({"native_total_edges":value["native_total_edges"],
        "queries":value["queries"],"native_wall_seconds_by_stage":value["native_wall_seconds_by_stage"]},indent=2))
