"""Read-only inventory/label ceilings and extinction-stage diagnostics."""

from __future__ import annotations

import argparse
import collections
import itertools
import json

from scripts.run_contextual_gibbs001 import OUT,PATHS,ROOT
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import artifact,require_frozen,save_new


def inventory_ceiling(key,gold,used):
    available = collections.Counter(key)
    needed = collections.Counter(gold[r] for r in used)
    deficit = sum(max(0,count-available[code]) for code,count in needed.items())
    return len(used)-deficit,deficit


def exact_tiny_ceiling_checks():
    checks = 0
    for gold in ((0,0,1),(0,1,2),(2,2,2)):
        for key in itertools.product(range(6),repeat=3):
            for mask in range(1,8):
                used = [r for r in range(3) if mask&(1<<r)]
                expected = max(sum(p[r]==gold[r] for r in used) for p in itertools.permutations(key))
                ceiling,deficit = inventory_ceiling(key,gold,used)
                assert ceiling==expected and deficit==len(used)-expected
                checks += 1
    return checks


def summarize_saved():
    result = json.loads((OUT/"result.json").read_text())
    audit = json.loads((OUT/"audit.json").read_text())
    require_frozen(result["freeze"],PATHS)
    assert audit["status"]=="PASS_complete_contextual_sampling_replay"
    assert audit["result"]==artifact(OUT/"result.json")
    parent = json.loads((ROOT/"results/SOURCE-PRUNE-DIAG-002/result.json").read_text())
    fixtures = load_bound(parent["fixtures"])["fixtures"]
    diagnostics = []
    for manifest in result["workloads"]:
        cell = load_bound(manifest)
        value = load_bound(cell["output"])
        fixture = fixtures[cell["case"]]
        used = sorted({r for text in fixture["generation_source"] for r in text})
        trace = value["trace"]
        item = {"case":cell["case"],"seed":cell["seed"],"arm":cell["arm"],"status":value["status"],
            "completed_stages":sum(not t["extinct"] for t in trace),"used_rows":len(used)}
        if value["status"]=="extinct":
            assert trace[-1]["extinct"]
            item["extinction_stage"] = trace[-1]["stage"]
            item["extinction_cuts"] = trace[-1]["cuts"]
            item["extinction_at_closed_eos"] = trace[-1]["closed"]
        if value["status"]=="graph_cap":
            item["error"] = value["error"]
        if value["status"]=="complete_particles":
            chosen = cell["mapping"]["selected_index"]
            assert chosen==max(range(len(value["keys"])),key=lambda i:(value["key_log_likelihoods"][i],-i))
            ceilings = [inventory_ceiling(k,fixture["generation_key"],used) for k in value["keys"]]
            ceiling,deficit = ceilings[chosen]
            key = value["keys"][chosen]
            direct = sum(key[r]==fixture["generation_key"][r] for r in used)
            assert direct==cell["mapping"]["selected_matches"] and direct<=ceiling<=len(used)
            item.update({"selected_matches":direct,"selected_maximum_matches_after_any_row_permutation":ceiling,
                "minimum_code_replacements_before_permutation_can_complete_used_key":deficit,
                "bank_has_permutation_compatible_inventory":any(d==0 for _,d in ceilings),
                "best_bank_permutation_ceiling":max(c for c,_ in ceilings),
                "selected_unique_codes":len(set(key)),
                "selected_code_counts":dict(sorted(collections.Counter(key).items()))})
        conditional = [t for t in trace if not t["extinct"] and "minimum_positive_codes" in t]
        if conditional:
            item["minimum_positive_conditional_codes"] = min(t["minimum_positive_codes"] for t in conditional)
            item["maximum_conditional_weight"] = max(t["maximum_conditional_weight"] for t in conditional)
        diagnostics.append(item)
    assert len(diagnostics)==16
    return {"kind":"post_outcome_closed_bank_inventory_analysis_no_scoring",
        "result":artifact(OUT/"result.json"),"audit":artifact(OUT/"audit.json"),
        "tiny_exhaustive_permutation_ceiling_checks":exact_tiny_ceiling_checks(),"calls":diagnostics,
        "interpretation_limits":["Oracle inventory ceilings use exposed synthetic answers and are evaluation only",
            "A feasible permutation is not found, scored, or guaranteed to be preferred by the source",
            "Count-deficit replacements ignore intermediate support and likelihood barriers",
            "Inventory-compatible bank does not mean the correct labeling or reading was recovered"],
        "empirical_model_calls":0,"paid_spend_usd":0}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = summarize_saved()
    if args.save:
        save_new(OUT/"post-outcome.json",result)
    print(json.dumps(result,indent=2))
