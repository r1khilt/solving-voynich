"""Read-only, separately computed publication accounting for the closed panel."""

import argparse
import json
import math
import subprocess

from scripts import run_tempered_recovery001 as run
from scripts.run_shared_key_guide001 import load_bound


COMPLETE = 'complete_tempered_particles'
FAILURES = ('compiler_cap','native_graph_cap','call_work_cap')


def independent_summary(cells):
    expected = {(case,seed,arm) for case in range(8) for seed in run.SEEDS for arm in run.ARMS}
    indexed = {(c['case'],c['seed'],c['arm']):c for c in cells}
    if len(cells)!=64 or len(indexed)!=64 or set(indexed)!=expected:
        raise ValueError('Exactly the registered 64 distinct calls are required')
    arms = {}
    for arm in run.ARMS:
        rows = [indexed[case,seed,arm] for case in range(8) for seed in run.SEEDS]
        positive = [c for c in rows if c['case']%2==0]
        for c in rows:
            assert c['summary']['status'] in (COMPLETE,*FAILURES)
            assert c['recovery']['kind']==('positive' if c['case']%2==0 else 'shuffle')
            if c['summary']['status']==COMPLETE:
                assert math.isfinite(c['summary']['log_evidence_estimate'])
            elif c['case']%2==0:
                r = c['recovery']
                assert r['selected_matches'] is None and not r['selected_complete_used_key']
                assert not r['bank_contains_complete_used_key'] and r['exact_records']==0
                assert r['edit_errors_with_failed_readings_counted_full_length']==r['true_letters']
        spreads, margins = [],[]
        for case in range(8):
            pair = [indexed[case,seed,arm]['summary'] for seed in run.SEEDS]
            spreads.append(abs(pair[0]['log_evidence_estimate']-pair[1]['log_evidence_estimate'])
                           if all(p['status']==COMPLETE for p in pair) else None)
        for case in range(0,8,2):
            for seed in run.SEEDS:
                a,b = (indexed[i,seed,arm]['summary'] for i in (case,case+1))
                margins.append(a['log_evidence_estimate']-b['log_evidence_estimate']
                               if a['status']==b['status']==COMPLETE else None)
        matched = sum(c['recovery']['selected_matches'] or 0 for c in positive)
        used = sum(c['recovery']['used_rows'] for c in positive)
        complete = sum(c['summary']['status']==COMPLETE for c in rows)
        arms[arm] = {
            'calls':16,'complete_calls':complete,'positive_calls':8,
            'positive_complete':sum(c['summary']['status']==COMPLETE for c in positive),
            'all_positive_complete':all(c['summary']['status']==COMPLETE for c in positive),
            'selected_used_row_matches':matched,'used_row_denominator':used,
            'used_row_match_fraction_with_failed_calls_zero':matched/used if used else 0.,
            'selected_complete_used_keys':sum(c['recovery']['selected_complete_used_key'] for c in positive),
            'banks_containing_complete_used_key':sum(c['recovery']['bank_contains_complete_used_key'] for c in positive),
            'edit_errors':sum(c['recovery']['edit_errors_with_failed_readings_counted_full_length'] for c in positive),
            'true_letters':sum(c['recovery']['true_letters'] for c in positive),
            'exact_records':sum(c['recovery']['exact_records'] for c in positive),
            'seed_log_evidence_spreads':spreads,'positive_minus_shuffle_log_evidence_margins':margins,
            'calibration_gate':complete==16 and all(x<=2 for x in spreads)
                and all(c['summary']['maximum_incremental_weight']<=.5 for c in rows),
            'positive_shuffle_gate':all(x is not None and x>0 for x in margins),
            'native_edge_work':sum(c['summary']['edges'] for c in rows),
            **{f'{name}s':sum(c['summary']['status']==name for c in rows) for name in FAILURES}}
    new,control = arms['supported_refresh'],arms['coordinated']
    recovery = (new['all_positive_complete'] and control['all_positive_complete']
        and new['selected_complete_used_keys']>=1 and new['exact_records']>=1
        and new['used_row_match_fraction_with_failed_calls_zero']>=control['used_row_match_fraction_with_failed_calls_zero']+.10
        and new['edit_errors']<=.90*control['edit_errors'])
    return {'arms':arms,'exploratory_recovery_gate':recovery,
        'combined_qualification_gate':recovery and new['calibration_gate'] and new['positive_shuffle_gate'],
        'fresh_development_keys':True,'mechanism_language_or_historical_holdout':False,
        'historical_or_neural_qualification':False,'equal_actual_compute_claimed':False}


def summarize_closed():
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    run.require_frozen(result['freeze'],run.PATHS)
    assert audit['status']=='PASS_full_fresh_tempered_recovery_replay'
    assert audit['result']==run.artifact(run.OUT/'result.json') and audit['calls']==64
    assert audit['fresh_fixture_generation_replayed'] and audit['all_predictions_sealed_before_metrics']
    assert result['particles']==run.PARTICLES and result['mutations']==run.MUTATIONS
    assert result['betas']==list(run.BETAS) and result['arms']==list(run.ARMS) and result['seeds']==list(run.SEEDS)
    queries = load_bound(result['queries'])['queries']
    assert len(queries)==8
    for case,query in enumerate(queries):
        run.validate_query(query)
        assert query['case']==case
    seal = load_bound(result['prediction_seal'])
    assert seal['queries']==result['queries'] and seal['gold_not_used_by_fit_or_prediction']
    assert run.artifact(run.ROOT/result['gold']['path'])==result['gold']
    cells,initial_banks,compilers,bulk = [],{}, {},result['gold']['bytes']
    for spec in result['workloads']:
        cell = load_bound(spec)
        name = f"case{cell['case']}-{cell['seed']}-{cell['arm']}"
        assert spec==run.artifact(run.OUT/f'{name}.json')
        assert load_bound(run.artifact(run.OUT/f'{name}-sealed.json'))=={k:v for k,v in cell.items() if k!='recovery'}
        value = load_bound(cell['output'])['value']
        subprocess.run(['git','check-ignore','--quiet',cell['output']['path']],check=True)
        assert cell['summary']==run.engineering.metrics(value)
        maximum = run.PARTICLES*(1+(len(run.BETAS)-1)*(0 if cell['arm']=='supported_prior' else run.MUTATIONS))
        assert value.get('tables',0)<=maximum and value.get('edges',0)<=run.CALL_EDGES+1
        assert value.get('maximum_native_nodes',0)<=300_001
        assert value.get('maximum_owned_work_envelope',0)<=128*1024**2
        compilers.setdefault(cell['case'],[]).append(cell['compiler'])
        if value['status']==COMPLETE:
            assert value['seed']==cell['seed'] and value['arm']==cell['arm']
            assert len(value['keys'])==len(value['key_log_likelihoods'])==run.PARTICLES
            assert len(value['trace'])==len(run.BETAS)-1
            init = value['initialization']
            assert init['prior_full_support_probability']==cell['compiler']['prior_support_probability']
            assert abs(init['initial_log_normalizer']+sum(t['increment_log_mean'] for t in value['trace'])
                       -value['log_evidence_estimate'])<=1e-9
            initial_banks.setdefault((cell['case'],cell['seed']),[]).append(init['initial_key_bank_sha256'])
        else:
            assert value['status'] in FAILURES and 'keys' not in value and 'key_log_likelihoods' not in value
        cells.append(cell)
        bulk += cell['output']['bytes']
    assert seal['outputs']==[c['output'] for c in cells]
    assert all(len(set(h))==1 for h in initial_banks.values())
    assert all(all(c==group[0] for c in group) for group in compilers.values())
    summary = independent_summary(cells)
    assert summary==result['summary']
    assert bulk==result['bulk_bytes']<=128*1024**2
    assert sum(a['native_edge_work'] for a in summary['arms'].values())<=run.GLOBAL_EDGES
    complete = sum(a['complete_calls'] for a in summary['arms'].values())
    assert audit['literal_final_key_checks']==run.PARTICLES*complete
    assert audit['independent_fixed_final_key_source_checks']==min(8,run.PARTICLES)*complete
    assert audit['maximum_source_log_delta']<=1e-7
    for r,wall,cpu in ((result['resources'],run.FIT_WALL,run.FIT_CPU),(audit['resources'],run.AUDIT_WALL,run.AUDIT_CPU)):
        assert r['wall_seconds']<=wall and r['cpu_seconds']<=cpu and r['peak_rss_bytes']<=2*1024**3
    return {'status':'PASS_closed_tempered_recovery_publication',
        'result':run.artifact(run.OUT/'result.json'),'audit':run.artifact(run.OUT/'audit.json'),
        'frozen_paths':len(run.PATHS),'bulk_bytes':bulk,'all_archives_hash_bound_and_ignored':True,
        'shared_initial_banks_and_compilers_agree':True,'independent_summary':summary,
        'actual_corpus_compiler_sampler_or_source_calls':0,'gold_contents_opened':False,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = summarize_closed()
    if args.save:
        run.save_new(run.OUT/'post-outcome.json',result)
    print(json.dumps(result,indent=2))
