"""Read-only publication checks of closed full-source tempered searches."""

import argparse
import json
import math
import subprocess

from scripts import run_tempered_inventory001 as run
from scripts.run_shared_key_guide001 import load_bound


def summarize_closed():
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    run.require_frozen(result['freeze'],run.PATHS)
    assert audit['status']=='PASS_full_tempered_engineering_replay'
    assert audit['result']==run.artifact(run.OUT/'result.json')
    cells,details,bulk,hashes = [],[],0,{}
    for spec in result['workloads']:
        cell = load_bound(spec)
        value = load_bound(cell['output'])
        subprocess.run(['git','check-ignore','--quiet',cell['output']['path']],check=True)
        assert cell['summary']==run.metrics(value)
        assert value.get('tables',0)<=run.MAX_TABLES and value.get('edges',0)<=run.MAX_EDGES+1
        assert value.get('maximum_native_nodes',0)<=300_001
        assert value.get('maximum_owned_work_envelope',0)<=128*1024**2
        if value['status']=='complete_tempered_particles':
            assert value['seed']==83211+cell['case'] and value['kernel']==cell['kernel']
            assert len(value['keys'])==run.PARTICLES
            assert len(value['trace'])==len(run.BETAS)-1
            init = value['initialization']
            assert init['prior_full_support_probability']==cell['compiler']['prior_support_probability']
            assert abs(init['initial_log_normalizer']+sum(t['increment_log_mean'] for t in value['trace'])
                       -value['log_evidence_estimate'])<=1e-9
            hashes.setdefault(cell['case'],[]).append(init['initial_key_bank_sha256'])
            assert math.isfinite(value['log_evidence_estimate'])
        else:
            assert value['status'] in ('compiler_cap','native_graph_cap','call_work_cap')
            assert 'keys' not in value and 'key_log_likelihoods' not in value
        details.append({'case':cell['case'],'kernel':cell['kernel'],**cell['summary']})
        cells.append(cell)
        bulk += cell['output']['bytes']
    assert all(len(set(h))==1 for h in hashes.values())
    assert run.summarize(cells)==result['summary'] and bulk==result['bulk_bytes']<=128*1024**2
    assert result['summary']['native_edge_work']<=run.GLOBAL_EDGES
    complete = result['summary']['complete_calls']
    assert audit['independent_final_key_source_checks']==4*complete
    assert audit['literal_final_key_checks']==run.PARTICLES*complete
    assert audit['maximum_source_log_delta']<=1e-7
    for r in (result['resources'],audit['resources']):
        assert r['wall_seconds']<=1200 and r['cpu_seconds']<=1000 and r['peak_rss_bytes']<=2*1024**3
    return {'status':'PASS_closed_tempered_engineering_publication',
        'result':run.artifact(run.OUT/'result.json'),'audit':run.artifact(run.OUT/'audit.json'),
        'frozen_paths':len(run.PATHS),'bulk_bytes':bulk,'all_archives_hash_bound_and_ignored':True,
        'shared_initial_banks_agree':True,'summary':result['summary'],'cells':details,
        'actual_corpus_compiler_sampler_or_source_calls':0,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = summarize_closed()
    if args.save:
        run.save_new(run.OUT/'post-outcome.json',result)
    print(json.dumps(result,indent=2))
