"""Read only the closed higher-work panel; never compile or score again."""

import argparse
import json
import subprocess
from fractions import Fraction

from scripts import run_compiled_inventory003 as run


def summarize_closed():
    base = run.previous.original
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    old_path = run.ROOT/'results/COMPILED-INVENTORY-002/result.json'
    old = json.loads(old_path.read_text())
    base.require_frozen(result['freeze'],run.PATHS)
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==base.artifact(run.OUT/'result.json')
    assert result['experiment']==run.EXP
    for field in ('queries','source','source_arrays','native_build'):
        assert result[field]==old[field],field
    cells,details,bulk,unchanged = [],[],0,0
    for spec,old_spec in zip(result['workloads'],old['workloads'],strict=True):
        cell,old_cell = base.load_bound(spec),base.load_bound(old_spec)
        value,old_value = base.load_bound(cell['output']),base.load_bound(old_cell['output'])
        subprocess.run(['git','check-ignore','--quiet',cell['output']['path']],check=True)
        assert (cell['case'],cell['order'])==(old_cell['case'],old_cell['order'])
        assert cell['summary']==base.metrics(value)
        stats = value['stats']
        assert stats['created_nodes']+2-stats['collected_nodes']==stats['nodes']
        assert stats['maximum_live_nodes']<=150_000 and stats['maximum_apply_entries']<=300_000
        assert stats['maximum_owned_envelope']<=512*1024**2
        row = {'case':cell['case'],'order':cell['order'],'status':value['status'],
            'stats':stats,'previous_status':old_value['status']}
        if value['status']=='compiled_sampled':
            assert stats['apply_steps']<=12_000_000
            assert stats['collection_work']<=20_000_000
            assert value['source_check']['status']=='positive_scores_agree'
            row.update(dictionary_count=value['dictionary_count'],
                prior_support_probability=float(Fraction(value['prior_support_probability'])),
                source_check=value['source_check'])
            if old_value['status']=='compiled_sampled':
                for field in ('coefficients','cardinality_weights','dictionary_count','prior_support_probability',
                              'keys','control_keys','control_supported','source_check','stats'):
                    assert value[field]==old_value[field],field
                row['previous_keys_counts_scores_counters_exactly_unchanged'] = True
                unchanged += 1
        else:
            assert value['no_partial_count_or_sampler']
            if value['error']=='Inventory collection-work cap exhausted':
                # Attempted batched charge is recorded before work, then raises.
                assert 20_000_000<stats['collection_work']<=20_000_000+stats['nodes']+stats['apply_entries']
                assert stats['apply_steps']<=12_000_000
            else:
                assert value['error']=='Inventory Boolean apply-work cap exhausted'
                assert stats['apply_steps']==12_000_001 and stats['collection_work']<=20_000_000
            assert not any(k in value for k in ('nodes','keys','dictionary_count'))
        details.append(row)
        cells.append(cell)
        bulk += cell['output']['bytes']
    assert unchanged==old['summary']['compiled_calls']==5
    assert result['summary']==base.summarize(cells)
    assert result['bulk_bytes']==bulk<=128*1024**2
    complete = result['summary']['compiled_calls']
    assert audit['independent_supported_keys']==32*complete
    assert audit['independent_first_key_source_checks']==complete
    for resource in (result['resources'],audit['resources']):
        assert resource['wall_seconds']<=600 and resource['cpu_seconds']<=500
        assert resource['peak_rss_bytes']<=2*1024**3
    return {'status':'PASS_closed_higher_work_compiler_publication',
        'result':base.artifact(run.OUT/'result.json'),'audit':base.artifact(run.OUT/'audit.json'),
        'previous_result':base.artifact(old_path),'frozen_paths':len(run.PATHS),
        'archives':len(cells),'bulk_bytes':bulk,'all_bulk_hash_bound_and_ignored':True,
        'summary':result['summary'],'cells':details,'previous_compiled_calls':5,
        'additional_completed_cells':complete-5,'previous_completed_cells_exactly_unchanged':unchanged,
        'actual_corpus_compiler_or_source_calls':0,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = summarize_closed()
    if args.save:
        run.previous.original.save_new(run.OUT/'post-outcome.json',result)
    print(json.dumps(result,indent=2))
