"""Read-only outcome checks and lossless local archival; no guide/model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction as F

import numpy as np

from scripts.run_shared_key_guide001 import (
    ARMS, OUT, PATHS, ROOT, artifact, load_bound, require_frozen, save_new, summarize,
)


def future_rows(query,fixture):
    future = set()
    for offset,text in zip(query["offsets"],fixture["generation_source"],strict=True):
        consumed = 0
        for row in text:
            if consumed>=offset:
                future.add(row)
            consumed += 1 if fixture["generation_key"][row]<6 else 2
            if consumed>offset and consumed-(1 if fixture["generation_key"][row]<6 else 2)<offset:
                raise ValueError("Prefix offset is inside a true emission unit")
    return future


def check():
    result = json.loads((OUT/"result.json").read_text())
    audit = json.loads((OUT/"audit.json").read_text())
    require_frozen(result["freeze"],PATHS)
    assert audit["status"]=="PASS" and audit["result"]==artifact(OUT/"result.json")
    assert audit["alternate_forward_fixed_dictionary_checks"]==65280
    assert audit["maximum_alternate_log_delta"]<=1e-7
    fixtures = load_bound(result["fixtures"])["fixtures"]
    rows,archive,compact,path_checks = [],[],[],[]
    for spec in result["workloads"]:
        row = load_bound(spec)
        raw = (ROOT/spec["path"]).read_text()
        archive.append({"original":spec,"utf8_text":raw})
        e = row["estimate"]
        values = [x for x in e["base_log_likelihoods"] if x is not None]
        assert len(e["base_log_likelihoods"])==e["base_samples"]
        assert e["positive_base_groups"]==len(values)
        assert e["fixed_dictionary_tables"]==(16 if row["arm"]=="mc16" else 672)
        if values:
            peak = max(values)
            total = peak+math.log(sum(math.exp(v-peak) for v in values))
            shares = [math.exp(v-total) for v in values]
            assert abs(e["log_mean"]-(total-math.log(e["base_samples"])))<=1e-9
            assert abs(e["maximum_base_contribution_share"]-max(shares))<=1e-9
            assert abs(e["likelihood_contribution_ess"]-1/sum(x*x for x in shares))<=1e-9
        else:
            assert e["log_mean"] is None and e["likelihood_contribution_ess"]==e["maximum_base_contribution_share"]==0
        compact.append({"query_id":row["query"]["id"],"seed":row["seed"],"arm":row["arm"],
            "estimate":{k:v for k,v in e.items() if k!="base_log_likelihoods"},
            "renewal_log_guide":row["renewal_log_guide"],"wall_seconds":row["wall_seconds"],"original":spec})
        if row["query"]["kind"] in ("root","prefix"):
            q = row["query"]
            fixture = fixtures[q["case"]]
            remaining = future_rows(q,fixture)
            assert all(q["partial_key"][r] in (-1,fixture["generation_key"][r]) for r in remaining)
            free = sorted(r for r in remaining if q["partial_key"][r]<0)
            integrated = e["integrated_row"]
            constrained = [r for r in free if r!=integrated]
            n = e["base_samples"]
            bank = np.random.default_rng(row["seed"]).integers(42,size=(672,23),dtype=np.int32)[:n]
            compatible = sum(all(sample[r]==fixture["generation_key"][r] for r in constrained) for sample in bank)
            p = F(1,42**len(constrained))
            path_checks.append({"query":q["id"],"arm":row["arm"],"seed":row["seed"],
                "future_unknown_used_rows":len(free),"base_random_constraints":len(constrained),
                "compatible_base_samples":compatible,"base_success_probability":str(p),
                "union_bound_any_compatible_base":str(min(F(1),n*p)),
                "relative_variance_of_one_prescribed_path_component":str((1/p-1)/n),
                "not_a_lower_bound_on_total_future_marginal_variance":True})
        rows.append(row)
    assert len(rows)==144 and summarize(rows)==result["summary"]
    assert sum(s["bytes"] for s in result["workloads"])<=16*1024**2
    for resource in (result["resources"],audit["resources"]):
        assert 0<=resource["wall_seconds"]<=600 and 0<=resource["cpu_seconds"]<=500
        assert 0<resource["peak_rss_bytes"]<=2*1024**3 and resource["paid_spend_usd"]==0
    stats = {}
    for arm in ARMS:
        es = [r["estimate"] for r in rows if r["arm"]==arm]
        positives = [e for e in es if e["log_mean"] is not None]
        stats[arm] = {"positive_calls":len(positives),"positive_calls_with_share_at_least_point99":sum(e["maximum_base_contribution_share"]>=.99 for e in positives),
            "minimum_positive_contribution_ess":min(e["likelihood_contribution_ess"] for e in positives),
            "maximum_positive_contribution_ess":max(e["likelihood_contribution_ess"] for e in positives),
            "positive_fixed_dictionaries":sum(e["positive_fixed_dictionaries"] for e in es)}
    return result,audit,archive,compact,{"status":"PASS_read_only_outcome_and_archive_checks",
        "result":artifact(OUT/"result.json"),"audit":artifact(OUT/"audit.json"),"frozen_paths":len(PATHS),
        "calls":len(rows),"concentration":stats,"prescribed_future_path_diagnostics":path_checks,
        "no_new_guide_source_scoring_search_reader_training_or_neural_calls":True}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result,audit,archive,compact,publication = check()
    if args.save:
        bulk = save_new(ROOT/"outputs/SHARED-KEY-GUIDE-001/closed-calls.json.gz",archive,compressed=True)
        restored = load_bound(bulk)
        assert len(restored)==144
        for entry in restored:
            data = entry["utf8_text"].encode()
            assert len(data)==entry["original"]["bytes"] and hashlib.sha256(data).hexdigest()==entry["original"]["sha256"]
        publication["lossless_original_calls_archive"] = bulk
        save_new(OUT/"compact-calls.json",compact)
        publication["compact_calls"] = artifact(OUT/"compact-calls.json")
        save_new(OUT/"publication-audit.json",publication)
    print(json.dumps({k:v for k,v in publication.items() if k!="prescribed_future_path_diagnostics"},indent=2,allow_nan=False))
