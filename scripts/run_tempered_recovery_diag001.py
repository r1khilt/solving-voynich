"""Separately bounded known-answer diagnosis; no fitting or generation."""

import argparse
import itertools
import json
import math
import signal
import time

import numpy as np

from scripts import run_tempered_recovery001 as parent
from scripts.run_shared_key_guide001 import load_bound
from voynich.dictionary_smc import FixedKeyBridge
from voynich.key_bank_diagnostics import full_key_bank_bounds,inventory_ceiling
from voynich.native_suffix_marginal import NativeMarginal


EXP = 'TEMPERED-RECOVERY-DIAG-001'
ROOT,OUT,BULK = parent.ROOT,parent.ROOT/'results'/EXP,parent.ROOT/'outputs'/EXP
WALL,CPU = 600,500
POOL = tuple(''.join(x) for n in (1,2) for x in itertools.product('ABCDEF',repeat=n))
PATHS = sorted(set([*parent.PATHS,'src/voynich/key_bank_diagnostics.py',
    'tests/test_key_bank_diagnostics.py','scripts/run_tempered_recovery_diag001.py',
    'scripts/audit_tempered_recovery_diag001.py','tests/test_tempered_recovery_diag001.py',
    'docs/experiments/TEMPERED-RECOVERY-DIAG-001.md',
    'docs/research/tempered-recovery-next-diagnosis-2026-10-01.md',
    'results/TEMPERED-RECOVERY-001/result.json','results/TEMPERED-RECOVERY-001/predictions-sealed.json',
    'results/TEMPERED-RECOVERY-001/queries.json',
    *[f'results/TEMPERED-RECOVERY-001/case{case}-{seed}-{arm}{suffix}.json'
      for case,seed,arm,suffix in itertools.product(range(8),parent.SEEDS,parent.ARMS,('','-sealed'))]]))


def inputs(freeze):
    parent.require_frozen(freeze,PATHS)
    _,benchmark = parent.inputs(freeze)
    result = json.loads((parent.OUT/'result.json').read_text())
    assert result['source_arrays'] and result['experiment']==parent.EXP
    assert result['native_build']==benchmark['build']
    assert not (parent.OUT/'failure.json').exists() and not (parent.OUT/'audit-failure.json').exists()
    audit = None
    if (parent.OUT/'audit.json').exists():
        audit = json.loads((parent.OUT/'audit.json').read_text())
        assert audit['status']=='PASS_full_fresh_tempered_recovery_replay'
        assert audit['result']==parent.artifact(parent.OUT/'result.json')
    else:
        assert (parent.OUT/'audit-started.json').exists()
    cells = [load_bound(s) for s in result['workloads']]
    seal = load_bound(result['prediction_seal'])
    queries = load_bound(result['queries'])['queries']
    assert len(cells)==64 and len(queries)==8
    assert seal['outputs']==[c['output'] for c in cells] and seal['queries']==result['queries']
    assert seal['gold_not_used_by_fit_or_prediction']
    expected = list(itertools.product(range(8),parent.SEEDS,parent.ARMS))
    assert [(c['case'],c['seed'],c['arm']) for c in cells]==expected
    for c in cells:
        path = parent.OUT/f"case{c['case']}-{c['seed']}-{c['arm']}-sealed.json"
        assert load_bound(parent.artifact(path))=={k:v for k,v in c.items() if k!='recovery'}
        assert parent.artifact(ROOT/c['output']['path'])==c['output']
    gold = load_bound(result['gold'])['fixtures']
    assert len(gold)==8 and queries==list(map(parent.query_for,gold))
    return result,benchmark,cells,gold,audit


def source_point(source,texts,rho=1/225):
    direct,adapter = [],[]
    for text in texts:
        assert text and all(type(a) is int and 0<=a<len(source.alphabet) for a in text)
        direct.extend((math.log(rho),len(text)*math.log1p(-rho)))
        adapter.extend((math.log(rho),len(text)*math.log1p(-rho)))
        state,other = source.state(''),source.state('')
        for row in text:
            direct.append(math.log(float(source.probabilities[state,row])))
            adapter.append(math.log(float(source.row(other)[row])))
            state = int(source.transitions[state,row])
            other = source.step(other,row)
        assert state==other
    a,b = math.fsum(direct),math.fsum(adapter)
    assert math.isfinite(a) and a<=0 and abs(a-b)<=1e-10
    return a


def oracle(source,native,fixture):
    key = fixture['generation_key']
    assert len(key)==len(source.alphabet)==23 and all(type(i) is int and 0<=i<42 for i in key)
    units = tuple(POOL[i] for i in key)
    assert [[ord(c)-ord('A') for row in text for c in units[row]] for text in fixture['generation_source']]==fixture['cipher']
    point = source_point(source,fixture['generation_source'])
    used = sorted({r for text in fixture['generation_source'] for r in text})
    prediction = {'status':'no_complete_key','selected_index':None}
    total,delta,nodes,edges = None,None,[],[]
    status,error = 'complete',None
    try:
        scores = [native.score(units,''.join('ABCDEF'[g] for g in c),1/225,
            max_nodes=100_000,max_edges=400_000) for c in fixture['cipher']]
    except RuntimeError as exc:
        if not any(t in str(exc) for t in ('Exact lattice node cap','Exact lattice edge cap')):
            raise
        status,error = 'oracle_native_graph_cap',str(exc)
    else:
        total = math.fsum(v.log_likelihood for v in scores)
        assert math.isfinite(total) and total>=point-1e-7
        nodes,edges = [v.reachable_nodes for v in scores],[v.edges for v in scores]
        reference = FixedKeyBridge(fixture['cipher'],native.probabilities,native.transitions,
            max_tables=1,max_edges=800_000_000,max_states_per_table=300_000,max_work_bytes=768*1024**2)
        checked = float(reference.log_values(np.array([key]),reference.lengths,closed=True)[0])
        delta = abs(checked-total)
        assert math.isfinite(checked) and delta<=1e-7
        value = {'status':'complete_tempered_particles','keys':[key],'key_log_likelihoods':[total]}
        prediction = parent.predict(value,parent.query_for(fixture),source,native)
        if prediction['status']!='read':
            status = 'oracle_reader_cap'
    value = {'status':'complete_tempered_particles','keys':[key],'key_log_likelihoods':[total]} if total is not None else {'status':'oracle_native_graph_cap'}
    recovery = parent.recovery(value,prediction,fixture,source.alphabet)
    return {'case':fixture['case'],'status':status,'error':error,'used_rows':used,
        'known_source_joint_log_probability':point,'known_key_log_likelihood':total,
        'independent_source_log_delta':delta,'native_record_nodes':nodes,'native_record_edges':edges,
        'conditional_known_source_log_probability':None if total is None else point-total,
        'recovery':recovery,'prediction':prediction,'oracle_supplied_generating_dictionary':True}


def bank_diagnostic(cell,stored,known):
    value = stored['value']
    if value['status']!='complete_tempered_particles':
        return {'case':cell['case'],'seed':cell['seed'],'arm':cell['arm'],'status':'failed_parent_call',
            'parent_status':value['status'],'no_partial_bank_diagnostic':True}
    fixture,oracle_value = known
    keys,scores = value['keys'],value['key_log_likelihoods']
    assert len(keys)==len(scores)==parent.PARTICLES
    ceilings = [inventory_ceiling(k,fixture['generation_key'],oracle_value['used_rows'],pool_size=42) for k in keys]
    selected = max(range(len(keys)),key=lambda i:(scores[i],-i))
    assert selected==stored['prediction']['selected_index']
    matches = ceilings[selected]['current_used_matches']
    assert matches==cell['recovery']['selected_matches']
    assert all(c['current_used_matches']<=c['permutation_used_match_ceiling'] for c in ceilings)
    return {'case':cell['case'],'seed':cell['seed'],'arm':cell['arm'],'status':'complete',
        'selected_inventory_ceiling':ceilings[selected],
        'inventory_compatible_particles':sum(c['inventory_permits_complete_used_mapping'] for c in ceilings),
        'best_permutation_used_match_ceiling':max(c['permutation_used_match_ceiling'] for c in ceilings),
        'minimum_inventory_replacements_across_particles':min(c['minimum_inventory_replacements'] for c in ceilings),
        'mean_inventory_replacements_over_particles':sum(c['minimum_inventory_replacements'] for c in ceilings)/len(keys),
        'known_path_minus_best_key_log_likelihood':oracle_value['known_source_joint_log_probability']-max(scores),
        'known_key_minus_best_key_log_likelihood':None if oracle_value['known_key_log_likelihood'] is None else oracle_value['known_key_log_likelihood']-max(scores),
        'bounds':full_key_bank_bounds(keys,scores,oracle_value['known_source_joint_log_probability'],oracle_value['used_rows'],
            pool_size=42,known_key_log_likelihood=oracle_value['known_key_log_likelihood'])}


def analyze(source,native,cells,gold,observe=lambda:None):
    known,oracles,rows = {},[],[]
    for case in (0,2,4,6):
        assert gold[case]['case']==case and gold[case]['kind']=='positive'
        value = oracle(source,native,gold[case])
        known[case] = gold[case],value
        oracles.append(value)
        observe()
        print(f"Diagnosed generating key case{case}: {value['status']}",flush=True)
    for cell in cells:
        if cell['case']%2==0:
            rows.append(bank_diagnostic(cell,load_bound(cell['output']),known[cell['case']]))
            observe()
    assert len(rows)==32
    complete = [r for r in rows if r['status']=='complete']
    return {'oracles':oracles,'banks':rows,'summary':{
        'positive_parent_calls':32,'complete_parent_banks':len(complete),
        'oracle_calls':4,'oracle_complete_readings':sum(o['prediction']['status']=='read' for o in oracles),
        'oracle_exact_records':sum(o['recovery']['exact_records'] for o in oracles),
        'oracle_edit_errors':sum(o['recovery']['edit_errors_with_failed_readings_counted_full_length'] for o in oracles),
        'oracle_true_letters':sum(o['recovery']['true_letters'] for o in oracles),
        'banks_without_compatible_inventory':sum(r['inventory_compatible_particles']==0 for r in complete),
        'banks_known_path_exceeds_best_key':sum(r['known_path_minus_best_key_log_likelihood']>1e-7 for r in complete),
        'banks_known_key_exceeds_best_key':sum(r['known_key_minus_best_key_log_likelihood'] is not None
            and r['known_key_minus_best_key_log_likelihood']>1e-7 for r in complete),
        'independent_known_key_source_checks':sum(o['independent_source_log_delta'] is not None for o in oracles),
        'maximum_known_key_source_log_delta':max((o['independent_source_log_delta'] for o in oracles if o['independent_source_log_delta'] is not None),default=None),
        'fitting_generation_or_mutation_calls':0,'historical_or_recovery_qualification':False}}


def guard(wall,cpu):
    r = parent.resource_report(wall,cpu)
    if r['wall_seconds']>WALL or r['cpu_seconds']>CPU or r['peak_rss_bytes']>2*1024**3:
        raise RuntimeError('Known-answer diagnostic resource cap; no retry')


def run(freeze):
    previous,benchmark,cells,gold,audit = inputs(freeze)
    parent.save_new(OUT/'started.json',{'freeze':freeze,'start_unix':time.time()})
    parent.limit_resources(WALL,CPU)
    wall,cpu = time.monotonic(),time.process_time()
    try:
        source,chosen = parent.load_source()
        identity = parent.array_identity(source)
        assert chosen['counts']==previous['source'] and identity==previous['source_arrays']
        analysis = analyze(source,NativeMarginal(source,benchmark['build']),cells,gold,lambda:guard(wall,cpu))
        bulk = parent.save_new(BULK/'analysis.json.gz',analysis,compressed=True)
        assert bulk['bytes']<=8*1024**2 and parent.array_identity(source)==identity
        guard(wall,cpu)
        parent.save_new(OUT/'result.json',{'experiment':EXP,'freeze':freeze,'parent_result':parent.artifact(parent.OUT/'result.json'),
            'parent_prediction_seal':previous['prediction_seal'],'parent_full_audit_at_admission':'PASS' if audit else 'ACTIVE',
            'source':previous['source'],'source_arrays':identity,'native_build':benchmark['build'],'analysis':bulk,
            'oracles':[{k:v for k,v in o.items() if k not in ('prediction','used_rows')} for o in analysis['oracles']],
            'banks':analysis['banks'],'summary':analysis['summary'],'resources':parent.resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        parent.save_new(OUT/'failure.json',{'error':repr(exc),'no_retry':True,'resources':parent.resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    run(parser.parse_args().freeze)
