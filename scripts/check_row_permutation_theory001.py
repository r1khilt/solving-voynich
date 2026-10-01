"""Finite rational support/balance checks for dictionary-row permutations."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from fractions import Fraction as F

from scripts.check_prefix_bridge001 import CONTEXTS,POOL,enumerated
from scripts.run_source_state_systems001 import ROOT,artifact,save_new


def image_support(observed,key,closed):
    """Independent Boolean word break on code inventory, no source probabilities."""
    if not observed:
        return True
    reachable = {0}
    for pos in range(len(observed)):
        if pos not in reachable:
            continue
        for code in set(key):
            unit = POOL[code]
            end = pos+len(unit)
            if closed:
                if end<=len(observed) and unit==observed[pos:end]:
                    reachable.add(end)
            else:
                visible = min(len(unit),len(observed)-pos)
                if unit[:visible]==observed[pos:pos+visible]:
                    reachable.add(min(end,len(observed)))
    return len(observed) in reachable


def check():
    support_checks = balance_checks = 0
    for key in itertools.product(range(6),repeat=2):
        swapped = key[::-1]
        for ctx in range(3):
            for n in range(1,4):
                for text in itertools.product(range(2),repeat=n):
                    for cut,closed in [(c,False) for c in range(n+1)]+[(n,True)]:
                        observed = text[:cut]
                        a = enumerated(observed,key,ctx,CONTEXTS,closed=closed)
                        b = enumerated(observed,swapped,ctx,CONTEXTS,closed=closed)
                        assert (a>0)==(b>0)==image_support(observed,key,closed)
                        forward = min(F(1),b/a) if a else F(0)
                        reverse = min(F(1),a/b) if b else F(0)
                        assert a*forward==b*reverse
                        support_checks += 1
                        balance_checks += 1
    # Counting prior mass by emission histogram versus explicit dictionaries.
    histogram_counts = collections.Counter(tuple(sorted(k)) for k in itertools.product(range(6),repeat=3))
    for histogram,count in histogram_counts.items():
        multiplicities = collections.Counter(histogram)
        expected = math.factorial(3)//math.prod(math.factorial(v) for v in multiplicities.values())
        assert count==expected
    assert len(histogram_counts)==math.comb(8,3) and sum(histogram_counts.values())==6**3
    # Two-record Gibbs barrier is traversed by a deterministic transposition MH.
    a = enumerated((0,),(0,1),1,CONTEXTS,closed=True)*enumerated((1,),(0,1),2,CONTEXTS,closed=True)
    b = enumerated((0,),(1,0),1,CONTEXTS,closed=True)*enumerated((1,),(1,0),2,CONTEXTS,closed=True)
    assert a>0 and b>0 and a*min(F(1),b/a)==b*min(F(1),a/b)
    # Positivity is essential: a source that cannot emit row1 breaks the claim.
    zero_rows = ((F(1),F(0)),)*3
    positive = enumerated((0,),(0,1),0,zero_rows,closed=True)
    unsupported = enumerated((0,),(1,0),0,zero_rows,closed=True)
    assert positive>0 and unsupported==0
    return {"status":"PASS_finite_row_permutation_support_and_balance",
        "support_and_boolean_inventory_checks":support_checks,"exact_balance_checks":balance_checks,
        "three_row_histograms":len(histogram_counts),"three_row_dictionaries":sum(histogram_counts.values()),
        "two_record_barrier":{"original_likelihood":str(a),"swapped_likelihood":str(b),
            "forward_swap_acceptance":str(min(F(1),b/a)),"reverse_swap_acceptance":str(min(F(1),a/b))},
        "strict_positivity_counterexample":{"original":str(positive),"swapped":str(unsupported)},
        "actual_23_row_42_code_count_vectors":math.comb(64,23),
        "actual_whole_dictionary_count":42**23,
        "inputs":[artifact(ROOT/"scripts/check_prefix_bridge001.py"),artifact(ROOT/"scripts/check_row_permutation_theory001.py")],
        "scope":"Finite arithmetic only; no corpus/native/model/sampler calls, no mixing or recovery result",
        "empirical_panel_calls":0,"paid_spend_usd":0}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/"results/ROW-PERMUTATION-THEORY-001/result.json",result)
    print(json.dumps(result,indent=2))
