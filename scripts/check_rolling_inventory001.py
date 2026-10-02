"""Own exact finite protected-root/occupancy/sampling qualification, no corpus."""

import argparse
import json

from scripts.run_source_state_systems001 import ROOT,artifact,save_new
from tests.test_rolling_inventory_bdd import finite_checks,six_glyph_checks


def check():
    return {'status':'PASS_exact_rolling_inventory_laws',
        'intervals':{str(i):finite_checks(i) for i in (1,16)},**six_glyph_checks(),
        'actual_corpus_or_source_model_calls':0,'scope':'Own finite mathematical qualification, not recovery',
        'inputs':[artifact(ROOT/p) for p in ('src/voynich/rolling_inventory_bdd.py',
            'tests/test_rolling_inventory_bdd.py','scripts/check_rolling_inventory001.py',
            'src/voynich/inventory_bdd.py','tests/test_inventory_bdd.py',
            'results/INVENTORY-BDD-THEORY-001/result.json')],'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/ROLLING-INVENTORY-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
