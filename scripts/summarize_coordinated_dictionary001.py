"""Read-only closed-bank inventory/termination accounting; no scorer calls."""

import argparse
from collections import Counter
import json

from scripts.run_coordinated_dictionary001 import ARMS,OUT,artifact,load_bound,save_new


def summarize():
    result = json.loads((OUT/'result.json').read_text())
    fixtures = load_bound(result['gold'])['fixtures']
    rows = []
    for spec in result['workloads']:
        cell = load_bound(spec)
        closed = load_bound(cell['output'])
        value,fixture = closed['value'],fixtures[cell['case']]
        used = sorted({r for text in fixture['generation_source'] for r in text})
        true = Counter(fixture['generation_key'][r] for r in used)
        row = {'case':cell['case'],'seed':cell['seed'],'arm':cell['arm'],'status':value['status'],
            'edge_work':value['edges'],'used_rows':len(used),'inventory_ceiling_counts':[],
            'selected_minimum_inventory_replacements':None,'termination_stage':None,
            'termination_cuts':None,'termination_closed':None}
        if value['status']=='complete_particles':
            ceilings = []
            for key in value['keys']:
                counts = Counter(key)
                ceilings.append(sum(min(n,counts[code]) for code,n in true.items()))
            selected = closed['prediction']['selected_index']
            row['inventory_ceiling_counts'] = ceilings
            row['selected_minimum_inventory_replacements'] = len(used)-ceilings[selected]
        elif value['status']=='extinct':
            end = value['trace'][-1]
            assert end['extinct']
            row.update(termination_stage=end['stage'],termination_cuts=end['cuts'],termination_closed=end['closed'])
        elif value['status']=='work_cap':
            assert value['edges']==1_000_000_001
        else:
            assert value['status']=='graph_cap'
        rows.append(row)
    arms = {}
    for arm in ARMS:
        chosen = [r for r in rows if r['arm']==arm]
        ceilings = [(v,r['used_rows']) for r in chosen for v in r['inventory_ceiling_counts']]
        complete = [r for r in chosen if r['status']=='complete_particles']
        arms[arm] = {'completed_final_particles':len(ceilings),
            'inventories_allowing_full_used_truth_by_any_permutation':sum(v==n for v,n in ceilings),
            'selected_minimum_inventory_replacements':[r['selected_minimum_inventory_replacements'] for r in complete],
            'work_caps':sum(r['status']=='work_cap' for r in chosen),
            'pre_eos_extinctions':sum(r['status']=='extinct' and not r['termination_closed'] for r in chosen),
            'fitting_edge_work':sum(r['edge_work'] for r in chosen)}
    assert len(rows)==24
    return {'status':'PASS_closed_bank_accounting','result':artifact(OUT/'result.json'),
        'arms':arms,'rows':rows,'source_or_sampler_calls':0,'paid_spend_usd':0,
        'truth_used_only_for_post_outcome_diagnostic':True,
        'scope':'Oracle inventory ceilings ignore intermediate support and do not supply a solver proposal or language claim'}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = summarize()
    if args.save:
        save_new(OUT/'post-outcome.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
