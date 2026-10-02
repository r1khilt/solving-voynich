"""Own exact-count/float-entropy decomposition of closed exposed profiles."""

import argparse
import json
import math
from fractions import Fraction

from scripts import run_compiled_inventory003 as run
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import ROOT,artifact,require_frozen,save_new
from voynich.inventory_bdd import InventoryProfile


def entropy_decomposition():
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    require_frozen(result['freeze'],run.PATHS)
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==artifact(run.OUT/'result.json')
    rows,inputs = [],[artifact(run.OUT/'result.json'),artifact(run.OUT/'audit.json'),
        artifact(ROOT/'scripts/summarize_inventory_entropy001.py')]
    prior_count = 42**23
    for spec in result['workloads']:
        cell = load_bound(spec)
        if cell['order']!='frequency':
            continue
        value = load_bound(cell['output'])
        assert value['status']=='compiled_sampled'
        h = int(value['dictionary_count'])
        coefficients = value['coefficients']
        weights = list(map(int,value['cardinality_weights']))
        onto = [InventoryProfile.onto_count(23,k,k) for k in range(len(weights))]
        assert weights==[b*t for b,t in zip(coefficients,onto,strict=True)]
        assert sum(weights)==h and Fraction(value['prior_support_probability'])==Fraction(h,prior_count)
        parts = [(k,float(Fraction(w,h)),coefficients[k],onto[k]) for k,w in enumerate(weights) if w]
        cardinality_entropy = -math.fsum(p*math.log2(p) for _,p,_,_ in parts)
        subset_entropy = math.fsum(p*math.log2(b) for _,p,b,_ in parts)
        assignment_entropy = math.fsum(p*math.log2(t) for _,p,_,t in parts)
        conditional_entropy = math.log2(h)
        assert abs(cardinality_entropy+subset_entropy+assignment_entropy-conditional_entropy)<=1e-10
        row = {'case':cell['case'],'dictionary_count':str(h),
            'prior_key_entropy_bits':math.log2(prior_count),
            'conditional_supported_key_entropy_bits':conditional_entropy,
            'support_information_bits':math.log2(prior_count)-conditional_entropy,
            'cardinality_entropy_bits':cardinality_entropy,'conditional_subset_entropy_bits':subset_entropy,
            'conditional_label_assignment_entropy_bits':assignment_entropy,
            'mean_distinct_codes_given_support':math.fsum(k*p for k,p,_,_ in parts),
            'cardinality_probabilities':[{'k':k,'probability':p} for k,p,_,_ in parts],
            'exact_count_law_verified':True,'float_chain_rule_tolerance':1e-10}
        inputs.extend((spec,cell['output']))
        rows.append(row)
    assert len(rows)==4
    return {'status':'PASS_closed_support_entropy_decomposition','rows':rows,'inputs':inputs,
        'scope':'Exposed conditional prior, not posterior entropy or used-key/plaintext uncertainty; no Gold',
        'actual_corpus_compiler_sampler_or_source_calls':0,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = entropy_decomposition()
    if args.save:
        save_new(ROOT/'results/INVENTORY-ENTROPY-001/result.json',result)
    print(json.dumps(result,indent=2))
