"""Count dictionaries using support-mask cardinalities, not independent bits."""

import argparse
import itertools
import json
import math

from scripts.check_observation_initialization001 import support_counts
from scripts.run_source_state_systems001 import ROOT,artifact,save_new


def exact_occupancy_count(n,k,base):
    """n draws: every one of k named units occurs; base other codes allowed."""
    return sum((-1)**j*math.comb(k,j)*(base+k-j)**n for j in range(k+1))


def cardinality_coefficients(predicate,known_mask,variables):
    unknown = [j for j in range(variables) if not known_mask&(1<<j)]
    coefficients = [0]*(len(unknown)+1)
    for subset in range(1<<len(unknown)):
        mask = known_mask
        for i,j in enumerate(unknown):
            if subset&(1<<i):
                mask |= 1<<j
        coefficients[subset.bit_count()] += predicate[mask]
    return coefficients


def check():
    checks = 0
    pool = ((0,),(1,),(0,0),(0,1),(1,0),(1,1))
    for length in range(1,4):
        for observed in itertools.product(range(2),repeat=length):
            for known in ((),(0,),(3,),(0,1)):
                counts,bits,start = support_counts(observed,pool,3,known)
                coefficients = cardinality_coefficients(counts[0],start,len(bits))
                missing_variables = len(coefficients)-1
                base = len(pool)-missing_variables
                for n in range(4):
                    count = sum(b*exact_occupancy_count(n,k,base) for k,b in enumerate(coefficients))
                    assert count==counts[n][start]
                    checks += 1
    # An independence shortcut is false even for a single free row.
    both_present_exact = exact_occupancy_count(1,2,4)
    assert both_present_exact==0 and (1-(5/6))**2>0
    large_pool = tuple(u for length in (1,2) for u in itertools.product(range(6),repeat=length))
    counts,bits,start = support_counts((0,1,2,3),large_pool,23)
    coefficients = cardinality_coefficients(counts[0],start,len(bits))
    count = sum(b*exact_occupancy_count(23,k,42-len(bits)) for k,b in enumerate(coefficients))
    assert count==counts[23][start]
    return {'status':'PASS_exact_occupancy_cardinality_counts','small_profile_layer_checks':checks,
        'generic_four_glyph_relevant_codes':len(bits),'generic_cardinality_coefficients':coefficients,
        'generic_supported_dictionary_count':str(count),'independent_presence_null_rejected':True,
        'empirical_model_or_sampler_calls':0,'bdd_compiler_implemented':False,
        'scope':'Own exact finite occupancy/counting identity, not a compiled full-record sampler or recovery result',
        'inputs':[artifact(ROOT/p) for p in ('scripts/check_occupancy_cardinality001.py',
            'scripts/check_observation_initialization001.py')],'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/OCCUPANCY-CARDINALITY-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
