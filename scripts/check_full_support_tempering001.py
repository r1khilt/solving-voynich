"""Own finite exact tempering law; no corpus, training or empirical search."""

import argparse
import json

from scripts.run_source_state_systems001 import ROOT,artifact,save_new
from tests.test_tempered_inventory import finite_tempering_checks


def check():
    paths = ('src/voynich/tempered_inventory.py','tests/test_tempered_inventory.py',
        'scripts/check_full_support_tempering001.py','src/voynich/coordinated_dictionary.py',
        'src/voynich/inventory_bdd.py','tests/test_coordinated_dictionary.py',
        'tests/test_inventory_bdd.py','scripts/check_prefix_bridge001.py',
        'src/voynich/source_key_particles.py','src/voynich/dictionary_smc.py')
    return {'status':'PASS_exact_full_support_tempering_laws',**finite_tempering_checks(),
        'scope':'Own finite rational balance, N1 unnormalized law and N2 first-increment resampling; not a posterior/mixing/recovery certificate',
        'actual_corpus_or_model_calls':0,'paid_spend_usd':0,
        'inputs':[artifact(ROOT/p) for p in paths]}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/FULL-SUPPORT-TEMPERING-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
