"""Exact finite compiler/occupancy/factorization/SMC qualification only."""

import argparse
import json

from scripts.run_source_state_systems001 import ROOT,artifact,save_new
from tests.test_inventory_bdd import finite_checks,full_support_smc_law,generic_checks


def check():
    return {'status':'PASS_exact_inventory_bdd_laws',**finite_checks(),
        **full_support_smc_law(),**generic_checks(),'actual_corpus_or_model_calls':0,
        'scope':'Own finite Boolean/occupancy/ranking/factorization mathematics; no empirical recovery claim',
        'inputs':[artifact(ROOT/p) for p in ('src/voynich/inventory_bdd.py',
            'tests/test_inventory_bdd.py','scripts/check_inventory_bdd001.py',
            'scripts/check_observation_initialization001.py','scripts/check_prefix_bridge001.py',
            'tests/test_coordinated_dictionary.py','src/voynich/coordinated_dictionary.py')],
        'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/INVENTORY-BDD-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
