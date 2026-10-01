"""Exact post-outcome bridge-support mathematics, without empirical model calls."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.run_shared_key_guide001 import OUT, save_new

POOL = ((0,),(1,),(0,0),(0,1),(1,0),(1,1))
CONTEXTS = ((F(1,3),F(2,3)),(F(3,4),F(1,4)),(F(1,5),F(4,5)))
CONT, STOP = F(3,4), F(1,4)


def prefix_dp(observed,key,context,rows):
    """Probability of reaching observed glyphs, allowing a cut inside a unit."""
    if not observed:
        return F(1)
    states = {(0,context):F(1)}
    total = F(0)
    for offset in range(len(observed)):
        for ctx in range(len(rows)):
            mass = states.get((offset,ctx),F(0))
            for letter in range(2):
                unit = POOL[key[letter]]
                seen = min(len(unit),len(observed)-offset)
                if unit[:seen]!=observed[offset:offset+seen]:
                    continue
                weight = mass*CONT*rows[ctx][letter]
                end = offset+len(unit)
                if end>=len(observed):
                    total += weight
                else:
                    target = (end,letter+1)
                    states[target] = states.get(target,F(0))+weight
    return total


def enumerated(observed,key,context,rows,closed=False):
    if not observed:
        return STOP if closed else F(1)
    total = F(0)
    for n in range(1,len(observed)+1):
        for text in itertools.product(range(2),repeat=n):
            encoded = tuple(g for r in text for g in POOL[key[r]])
            before_last = len(encoded)-len(POOL[key[text[-1]]])
            if closed:
                match = encoded==observed
            else:
                match = encoded[:len(observed)]==observed and before_last<len(observed)<=len(encoded)
            if not match:
                continue
            weight,ctx = CONT**n,context
            for letter in text:
                weight *= rows[ctx][letter]
                ctx = letter+1
            total += weight*STOP if closed else weight
    return total


def check():
    cases = 0
    supported = 0
    for rows in (CONTEXTS,((F(1),F(0)),)*3):
        for key in itertools.product(range(6),repeat=2):
            for ctx in range(3):
                for length in range(1,4):
                    for observed in itertools.product(range(2),repeat=length):
                        final = enumerated(observed,key,ctx,rows,closed=True)
                        last = F(1)
                        for cut in range(length+1):
                            prefix = observed[:cut]
                            dp = prefix_dp(prefix,key,ctx,rows)
                            assert dp==enumerated(prefix,key,ctx,rows)
                            assert final<=dp<=last
                            assert final==0 or dp>0
                            last = dp
                            cases += 1
                        supported += final>0
    key = (2,2)  # Each source row emits 00; a one-glyph cut is mid-emission.
    valid = prefix_dp((0,),key,0,CONTEXTS)
    bad = enumerated((0,),key,0,CONTEXTS,closed=True)
    final = enumerated((0,0),key,0,CONTEXTS,closed=True)
    assert valid==F(3,4) and bad==0 and final==F(3,16)
    return {"status":"PASS_exact_prefix_bridge_support","prefix_state_checks":cases,
        "positive_full_observation_cases":supported,
        "mid_unit_witness":{"key_codes":list(key),"observed_final":[0,0],
            "valid_first_glyph_prefix":str(valid),"incorrect_forced_closed_prefix":str(bad),"positive_final_likelihood":str(final)},
        "scope":"All36two-row/six-unit dictionaries, three supplied contexts, every binary observation length1..3, each prefix cut, positive and zero-transition source fixtures. No SMC or decoding performance claim.",
        "post_outcome_analysis":True,"empirical_panel_calls":0,"paid_spend_usd":0}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(OUT/"prefix-bridge-post-outcome.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
