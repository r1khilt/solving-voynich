"""Finite exact qualification; not a corpus search or recovery result."""

import argparse
import itertools
import json

from scripts.run_source_state_systems001 import ROOT,artifact,save_new
from tests.test_coordinated_dictionary import checked_laws,pool
from voynich.coordinated_dictionary import PAIR_OPERATIONS,completion_count,pair_involution


def check():
    checks = 0
    for g in (1,2,6):
        units = pool(g)
        for a,b in itertools.product(range(len(units)),repeat=2):
            for op in PAIR_OPERATIONS:
                aa,bb = pair_involution(a,b,units,op)
                assert pair_involution(aa,bb,units,op)==(a,b)
                checks += 1
    return {'status':'PASS_exact_coordinated_laws',**checked_laws(),
        'pair_involution_checks':checks,'large_coverage_count':str(completion_count(23,6,42)),
        'large_coverage_denominator':str(42**23),'paid_spend_usd':0,
        'scope':'Exact own finite mathematics; no posterior, decoding, or historical qualification',
        'inputs':[artifact(ROOT/p) for p in ('src/voynich/coordinated_dictionary.py',
            'tests/test_coordinated_dictionary.py','scripts/check_coordinated_dictionary001.py')]}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/COORDINATED-DICTIONARY-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
