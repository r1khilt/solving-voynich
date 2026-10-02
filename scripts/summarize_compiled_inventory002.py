"""Read-only closed collected-compiler publication and paired001 comparison."""

import argparse
import json
import subprocess
from fractions import Fraction

from scripts import run_compiled_inventory002 as run


def summarize_closed():
    base = run.original
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    old = json.loads((base.ROOT/'results/COMPILED-INVENTORY-001/result.json').read_text())
    base.require_frozen(result['freeze'],run.PATHS)
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==base.artifact(run.OUT/'result.json')
    assert result['queries']==old['queries'] and result['source_arrays']==old['source_arrays']
    assert result['native_build']==old['native_build'] and result['experiment']==run.EXP
    cells,details,bulk = [],[],0
    for spec,old_spec in zip(result['workloads'],old['workloads'],strict=True):
        cell,old_cell = base.load_bound(spec),base.load_bound(old_spec)
        value,old_value = base.load_bound(cell['output']),base.load_bound(old_cell['output'])
        subprocess.run(['git','check-ignore','--quiet',cell['output']['path']],check=True)
        assert (cell['case'],cell['order'])==(old_cell['case'],old_cell['order'])
        assert cell['summary']==base.metrics(value)
        stats = value['stats']
        assert stats['created_nodes']+2-stats['collected_nodes']==stats['nodes']
        assert stats['maximum_live_nodes']<=150_000 and stats['maximum_apply_entries']<=300_000
        assert stats['collection_work']<=20_000_000 and stats['maximum_owned_envelope']<=512*1024**2
        row = {'case':cell['case'],'order':cell['order'],'status':value['status'],'stats':stats,
            'original_status':old_value['status']}
        if value['status']=='compiled_sampled':
            assert stats['apply_steps']<=3_000_000 and value['source_check']['status']=='positive_scores_agree'
            row.update(dictionary_count=value['dictionary_count'],
                prior_support_probability=float(Fraction(value['prior_support_probability'])),
                source_check=value['source_check'])
            if old_value['status']=='compiled_sampled':
                for field in ('coefficients','cardinality_weights','dictionary_count','prior_support_probability',
                              'keys','control_keys','control_supported','source_check'):
                    assert value[field]==old_value[field],field
                row['original_keys_counts_scores_exactly_unchanged'] = True
                row['peak_live_node_reduction_fraction'] = 1-stats['maximum_live_nodes']/old_value['stats']['nodes']
        else:
            assert value['error']=='Inventory Boolean apply-work cap exhausted'
            assert stats['apply_steps']==3_000_001 and value['no_partial_count_or_sampler']
            assert not any(k in value for k in ('nodes','keys','dictionary_count'))
        details.append(row)
        cells.append(cell)
        bulk += cell['output']['bytes']
    assert result['summary']==base.summarize(cells) and result['bulk_bytes']==bulk<=128*1024**2
    assert audit['independent_supported_keys']==160 and audit['independent_first_key_source_checks']==5
    for resource in (result['resources'],audit['resources']):
        assert resource['wall_seconds']<=600 and resource['cpu_seconds']<=500
        assert resource['peak_rss_bytes']<=2*1024**3
    return {'status':'PASS_closed_collected_compiler_publication','result':base.artifact(run.OUT/'result.json'),
        'audit':base.artifact(run.OUT/'audit.json'),'original_result':base.artifact(base.ROOT/'results/COMPILED-INVENTORY-001/result.json'),
        'frozen_paths':len(run.PATHS),'archives':len(cells),'bulk_bytes':bulk,'all_bulk_hash_bound_and_ignored':True,
        'summary':result['summary'],'cells':details,'original_compiled_calls':old['summary']['compiled_calls'],
        'additional_completed_cells':result['summary']['compiled_calls']-old['summary']['compiled_calls'],
        'actual_corpus_compiler_or_source_calls':0,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = summarize_closed()
    if args.save:
        run.original.save_new(run.OUT/'post-outcome.json',result)
    print(json.dumps(result,indent=2))
