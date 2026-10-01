"""Post-outcome closed-artifact checks with alternate edit/encoding arithmetic."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from scripts.run_source_bellman001 import (
    ARMS, OUT, PATHS, artifact, base_output, config, load_bound, require_frozen, save_new,
)
from scripts.run_source_revise001 import POOL


ALPHABET="abcdefghiklmnopqrstuxyz"


def levenshtein(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        previous,row=row,[i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
    return row[-1]


def check():
    result=json.loads((OUT/"result.json").read_text())
    audit=json.loads((OUT/"audit.json").read_text())
    require_frozen(result["freeze"],PATHS)
    assert audit["status"]=="PASS" and audit["result"]==artifact(OUT/"result.json")
    assert audit["depth_zero_matches_all_original_wide_cells"] is True
    assert audit["maximum_alternate_delta"]<=1e-7
    parent=load_bound(result["parent_result"])
    parent_audit=load_bound(result["parent_audit"])
    assert parent_audit["result"]==result["parent_result"]
    fixtures=load_bound(result["fixtures"])["fixtures"]
    assert len(fixtures)==4
    counts,total,rows=0,0,[]
    for fixture in fixtures:
        for arm in ARMS:
            name=f"case{fixture['case']}-{arm}"
            spec=result["workloads"][len(rows)]
            assert spec==artifact(OUT/f"{name}.json")
            r=load_bound(spec)
            assert (r["name"],r["case"],r["arm"],r["config"])==(name,fixture["case"],arm,config(arm))
            fit,pred=load_bound(r["fitted"]),load_bound(r["prediction"])
            assert {k:v for k,v in fit.items() if k not in ("terminals","trace")}==r["summary"]
            assert fit["lookahead_calls"]<=16*16384 and fit["lookahead_actions"]<=16*16384*46
            if arm=="iid":
                old=load_bound(parent["workloads"][4*fixture["case"]+3])
                assert base_output(fit)==load_bound(old["fitted"])
            assert fit["interval_arithmetic_certificate"] is False and fit["full_key_posterior_claimed"] is False
            d=r["diagnostic"]
            used=sorted({x for text in fixture["generation_source"] for x in text})
            assert d["gold_used_rows"]==len(used)
            assert d["true_source_letters"]==sum(map(len,fixture["generation_source"]))
            assert d["gold_used_mapping_in_returned_terminals"]==any(all(t["used_key"][x]==fixture["generation_key"][x] for x in used) for t in fit["terminals"])
            if pred["status"]=="read":
                terminal=fit["terminals"][0]
                assert pred["partial_key"]==terminal["used_key"]
                fill=list(map(int,np.random.default_rng(79101+fixture["case"]).integers(42,size=23)))
                assert pred["fill_seed"]==79101+fixture["case"] and pred["filler_indices"]==fill
                assert pred["key"]==[POOL[k if k>=0 else fill[i]] for i,k in enumerate(terminal["used_key"])]
                texts=pred["prediction"]["source_records"]
                target=["".join(ALPHABET[x] for x in record) for record in fixture["generation_source"]]
                edits=[levenshtein(a,b) for a,b in zip(texts,target,strict=True)]
                assert d["record_edits"]==edits and d["edit_errors"]==sum(edits)
                assert d["exact_records"]==sum(a==b for a,b in zip(texts,target,strict=True))
                assert d["matched_gold_used_rows"]==sum(pred["key"][x]==POOL[fixture["generation_key"][x]] for x in used)
                for text,cipher in zip(texts,fixture["cipher"],strict=True):
                    assert "".join(pred["key"][ALPHABET.index(letter)] for letter in text)=="".join("ABCDEF"[g] for g in cipher)
                    counts+=1
                # Completing unused priors preserves this entire selected
                # partial-key history group in the full-key marginal.
                expected_lower=terminal["log_mass"]-sum(k<0 for k in terminal["used_key"])*math.log(42)
                assert pred["score"]["log_key_mass"]>=expected_lower-1e-7
                assert pred["score"]["key_log_prior"]==-23*math.log(42)
            else:
                assert not fit["terminals"] and d["edit_errors"] is None
            total+=r["fitted"]["bytes"]+r["prediction"]["bytes"]
            rows.append(r)
    assert len(rows)==len(result["workloads"])==audit["workloads"]==8
    assert total==result["bulk_output_bytes"]
    direct={arm:{"edits":sum(r["diagnostic"]["edit_errors"] or 0 for r in rows if r["arm"]==arm),
        "exact_records":sum(r["diagnostic"]["exact_records"] for r in rows if r["arm"]==arm),
        "matched_used_rows":sum(r["diagnostic"]["matched_gold_used_rows"] or 0 for r in rows if r["arm"]==arm)} for arm in ARMS}
    for arm in ARMS:
        stored=result["summary"]["arms"][arm]
        assert (stored["edit_errors_read_cells_only"],stored["exact_records"],stored["matched_used_rows"])==(direct[arm]["edits"],direct[arm]["exact_records"],direct[arm]["matched_used_rows"])
    all_read=all(r["diagnostic"]["edit_errors"] is not None for r in rows)
    improved=sum(rows[2*c+1]["diagnostic"]["edit_errors"] is not None and rows[2*c]["diagnostic"]["edit_errors"] is not None and rows[2*c+1]["diagnostic"]["edit_errors"]<rows[2*c]["diagnostic"]["edit_errors"] for c in range(4))
    assert result["summary"]["improved_cases"]==improved
    assert result["summary"]["exploratory_signal_supported"]==(all_read and direct["bellman_one"]["edits"]<=.9*direct["iid"]["edits"] and improved>=2)
    for resource in (result["resources"],audit["resources"]):
        assert 0<=resource["wall_seconds"]<=2400 and 0<=resource["cpu_seconds"]<=2200
        assert 0<resource["peak_rss_bytes"]<=2*1024**3 and resource["paid_spend_usd"]==0
    return {"status":"PASS_read_only_publication_checks","result":artifact(OUT/"result.json"),
        "audit":artifact(OUT/"audit.json"),"frozen_paths":len(PATHS),"workloads":len(rows),
        "alternate_record_encoding_edit_checks":counts,"direct_totals":direct,"bulk_bytes":total,
        "no_new_source_scoring_search_training_or_neural_calls":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args=parser.parse_args()
    result=check()
    if args.save:
        save_new(OUT/"publication-audit.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
