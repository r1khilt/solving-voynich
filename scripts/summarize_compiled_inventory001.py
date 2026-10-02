"""Read-only archive/freeze validation and final-root reachability diagnostic."""

import argparse
import json
import subprocess
from fractions import Fraction

from scripts.run_compiled_inventory001 import (
    OUT,PATHS,artifact,load_bound,require_frozen,save_new,summarize,
)


def reachable(nodes,roots):
    seen,todo = set(),list(roots)
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        if node>1:
            todo.extend(nodes[node][1:])
    return len(seen)


def summarize_closed():
    result = json.loads((OUT/'result.json').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    require_frozen(result['freeze'],PATHS)
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==artifact(OUT/'result.json')
    cells,diagnostics,bulk = [],[],0
    for spec in result['workloads']:
        cell = load_bound(spec)
        value = load_bound(cell['output'])
        subprocess.run(['git','check-ignore','--quiet',cell['output']['path']],check=True)
        assert value['status']==cell['summary']['status']
        bulk += cell['output']['bytes']
        cells.append(cell)
        if value['status']=='compiled_sampled':
            count = len(value['nodes'])
            root_count = reachable(value['nodes'],[value['root']])
            protected = reachable(value['nodes'],[value['root'],*value['record_roots'],*range(2,44)])
            diagnostics.append({'case':cell['case'],'order':cell['order'],'retained_nodes':count,
                'final_root_reachable_nodes':root_count,'final_record_and_variable_union_nodes':protected,
                'fraction_unused_by_final_root':1-root_count/count,
                'dictionary_count':value['dictionary_count'],
                'prior_support_probability':value['prior_support_probability'],
                'prior_support_probability_float':float(Fraction(value['prior_support_probability']))})
        else:
            assert value['no_partial_count_or_sampler'] and not any(k in value for k in ('nodes','keys'))
    assert bulk==result['bulk_bytes'] and summarize(cells)==result['summary']
    assert audit['independent_supported_keys']==result['summary']['whole_record_supported_sampled_keys']
    assert audit['independent_first_key_source_checks']==2
    for r in (result['resources'],audit['resources']):
        assert r['wall_seconds']<=600 and r['cpu_seconds']<=500 and r['peak_rss_bytes']<=2*1024**3
        assert r['thread_environment']['OPENBLAS_NUM_THREADS']==r['thread_environment']['OMP_NUM_THREADS']=='1'
    assert bulk<=128*1024**2
    return {'status':'PASS_closed_compiler_publication','result':artifact(OUT/'result.json'),
        'audit':artifact(OUT/'audit.json'),'frozen_paths':len(PATHS),'archives':len(cells),
        'bulk_bytes':bulk,'all_bulk_hash_bound_and_git_ignored':True,'summary':result['summary'],
        'final_graph_diagnostics':diagnostics,'actual_corpus_compiler_or_source_calls':0,
        'interpretation':'Many retained nodes are unused by the completed final root. This does not bound the largest graph or live roots during compilation, nor prove a collector will pass capped cases.',
        'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    result = summarize_closed()
    if parser.parse_args().save:
        save_new(OUT/'post-outcome.json',result)
    print(json.dumps(result,indent=2))
