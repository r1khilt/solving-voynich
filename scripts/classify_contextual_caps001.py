"""Post-outcome correction of cap labels, without editing frozen code/results."""

from __future__ import annotations

import argparse
import json

from scripts.run_contextual_gibbs001 import OUT,PATHS,ROOT
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import artifact,require_frozen,save_new


def classify():
    result = json.loads((OUT/'result.json').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    require_frozen(result['freeze'],PATHS)
    assert audit['status']=='PASS_complete_contextual_sampling_replay'
    assert audit['result']==artifact(OUT/'result.json')
    spent,caps = 0,[]
    for index,manifest in enumerate(result['workloads']):
        cell = load_bound(manifest)
        value = load_bound(cell['output'])
        budget = min(2_000_000_000,8_000_000_000-spent)
        assert budget>0 and cell['summary']['edges']==value['edges']
        if value['status']=='graph_cap':
            exhausted = value['edges']>budget
            assert exhausted and value['edges']==budget+1 and 'edge cap' in value['error']
            caps.append({'call_index':index,'case':cell['case'],'seed':cell['seed'],'arm':cell['arm'],
                'original_status':value['status'],'corrected_cause':'cumulative_per_call_edge_work_budget',
                'registered_cumulative_edge_budget':budget,'counter_including_first_rejected_edge':value['edges'],
                'remaining_predefined_calls_executed_after_this_cap':len(result['workloads'])-index-1,
                'no_per_record_lattice_size_claim':True})
        spent += value['edges']
    assert len(caps)==2 and spent<=8_000_000_000
    return {'kind':'post_outcome_cap_classification_correction','result':artifact(OUT/'result.json'),
        'audit':artifact(OUT/'audit.json'),'caps':caps,'total_edges_including_rejected_operations':spent,
        'protocol_adherence':'DEVIATION_cumulative_call_budget_error_was_allowed_to_continue_as_graph_cap',
        'source_of_deviation':'Original wrapper lowers native per-record edge cap to remaining cumulative budget; runner classified native message only',
        'replay_and_positive_final_likelihood_checks_remain_valid':True,
        'original_recovery_and_calibration_gates_remain_failed':True,
        'original_frozen_files_and_receipts_preserved':True,
        'strict_future_adapter':artifact(ROOT/'src/voynich/strict_censored_bridge.py'),
        'empirical_model_calls':0,'no_retries_or_extensions':True,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = classify()
    if args.save:
        save_new(OUT/'cap-classification.json',result)
    print(json.dumps(result,indent=2))
