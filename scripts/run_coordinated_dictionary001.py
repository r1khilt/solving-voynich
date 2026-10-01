"""One fresh-key three-arm panel, predictions sealed before gold evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import signal
import time

import numpy as np

from scripts.run_contextual_gibbs001 import PATHS as OLD_PATHS,inputs as parent_inputs
from scripts.run_dictionary_smc001 import compact
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import (
    ROOT,array_identity,artifact,finite,limit_resources,load_source,require_frozen,
    resource_report,save_new,
)
from voynich.coordinated_dictionary import run_coordinated_smc
from voynich.dictionary_smc import bridge_schedule
from voynich.kbest_suffix import record_kbest
from voynich.native_suffix_marginal import NativeMarginal
from voynich.strict_censored_bridge import CensoredWorkBudgetExceeded,StrictCensoredNativeBridge
from voynich.unit_channel_decision import edit_distance


EXP = 'COORDINATED-DICTIONARY-001'
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
ARMS = ('prior_row','covered_row','covered_coordinated')
SEEDS,GENERATION_SEEDS,DESIGNS = (82611,82612),(82601,82602,82603,82604),(64,64,224,224)
PARTICLES,STRIDE,MUTATIONS = 32,4,4
CALL_EDGES,CAMPAIGN_EDGES = 1_000_000_000,16_000_000_016
READER_LIMITS = {'max_nodes':100_000,'max_edges':400_000,'max_expanded':50_000}
PATHS = sorted(set([*OLD_PATHS,
    'src/voynich/coordinated_dictionary.py','src/voynich/strict_censored_bridge.py',
    'tests/test_strict_censored_bridge.py','tests/test_coordinated_dictionary.py',
    'scripts/check_coordinated_dictionary001.py','results/COORDINATED-DICTIONARY-THEORY-001/result.json',
    'scripts/check_inventory_barrier001.py','scripts/check_prefix_bridge001.py',
    'scripts/run_coordinated_dictionary001.py','scripts/audit_coordinated_dictionary001.py',
    'tests/test_coordinated_dictionary001.py','docs/experiments/COORDINATED-DICTIONARY-001.md',
    'docs/research/coordinated-dictionary-2026-10-01.md']))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent,benchmark,_,_ = parent_inputs(freeze)
    theory = json.loads((ROOT/'results/COORDINATED-DICTIONARY-THEORY-001/result.json').read_text())
    if theory['status']!='PASS_exact_coordinated_laws':
        raise ValueError('Exact finite proposal/weight qualification required')
    for spec in theory['inputs']:
        assert artifact(ROOT/spec['path'])==spec
    return parent,benchmark


def fixtures(source):
    """Only generator uses known source lengths; query constructor drops gold."""
    result = []
    units = tuple(u for n in (1,2) for u in itertools.product(range(6),repeat=n))
    for case,(seed,length) in enumerate(zip(GENERATION_SEEDS,DESIGNS,strict=True)):
        rng = np.random.default_rng(seed)
        key = list(map(int,rng.integers(42,size=23)))
        texts,cipher = [],[]
        for _ in range(2):
            text,state = [],source.state('')
            for _ in range(length):
                row = int(rng.choice(23,p=source.row(state)))
                text.append(row)
                state = source.step(state,row)
            texts.append(text)
            cipher.append([int(g) for r in text for g in units[key[r]]])
        result.append({'case':case,'generation_seed':seed,'generation_length':length,
            'generation_key':key,'generation_source':texts,'cipher':cipher})
    return result


def query_for(fixture):
    return {'case':fixture['case'],'cipher':fixture['cipher'],'partial_key':[-1]*23}


def evaluate(query,native,seed,arm):
    if seed not in SEEDS or arm not in ARMS or set(query)!= {'case','cipher','partial_key'}:
        raise ValueError('Registered cipher-only query and arm/seed required')
    stages = len(bridge_schedule(tuple(map(len,query['cipher'])),STRIDE))
    bridge = StrictCensoredNativeBridge(query['cipher'],native,
        max_tables=PARTICLES*(1+MUTATIONS)*stages,max_edges=CALL_EDGES)
    trace = []
    started = time.monotonic()
    try:
        value = run_coordinated_smc(bridge,query['partial_key'],particles=PARTICLES,seed=seed,
            stride=STRIDE,mutations=MUTATIONS,covered=arm!='prior_row',
            kernel='coordinated' if arm=='covered_coordinated' else 'row',observe=trace.append)
        value['keys'] = value['keys'].tolist()
        value['key_log_likelihoods'] = value['key_log_likelihoods'].tolist()
    except CensoredWorkBudgetExceeded as exc:
        value = {'status':'work_cap','error':str(exc),'trace':trace,'tables':bridge.tables,
            'edges':bridge.edges,'no_partial_likelihood':True}
    except RuntimeError as exc:
        if not any(s in str(exc) for s in ('Exact lattice node cap','Exact lattice edge cap')):
            raise
        value = {'status':'graph_cap','error':str(exc),'trace':trace,'tables':bridge.tables,
            'edges':bridge.edges,'no_partial_likelihood':True}
    return {'value':finite(value),'wall_seconds':time.monotonic()-started,
        'native_calls':bridge.native_calls,'native_nodes':bridge.nodes,
        'maximum_native_nodes':bridge.maximum_native_nodes,
        'maximum_owned_work_envelope':bridge.maximum_owned_work_envelope,
        'nominal_complete_table_budget':PARTICLES*(1+MUTATIONS)*stages}


def selected_index(value):
    if value['status']!='complete_particles':
        return None
    return max(range(len(value['keys'])),key=lambda i:(value['key_log_likelihoods'][i],-i))


def predict(value,query,source,native):
    """Gold-free highest-final-key likelihood then conditional fixed-key Viterbi."""
    selected = selected_index(value)
    if selected is None:
        return {'status':'no_complete_key','selected_index':None}
    alphabet = tuple(source.alphabet)
    pool = tuple(''.join(chars) for n in (1,2) for chars in itertools.product('ABCDEF',repeat=n))
    key = tuple(pool[c] for c in value['keys'][selected])
    outputs,logs,marginals,graphs = [],[],[],[]
    try:
        for record in query['cipher']:
            cipher = ''.join('ABCDEF'[g] for g in record)
            native_value = native.score(key,cipher,1/225,max_nodes=100_000,max_edges=400_000)
            found,total,nodes,edges,expanded = record_kbest(source,key,cipher,1/225,1,**READER_LIMITS)
            assert found and abs(total-native_value.log_likelihood)<=1e-7
            text,score = found[0]
            assert ''.join(key[alphabet.index(c)] for c in text)==cipher
            state,terms = source.state(''),[math.log(1/225),len(text)*math.log1p(-1/225)]
            for letter in text:
                r = alphabet.index(letter)
                terms.append(math.log(float(source.row(state)[r])))
                state = source.step(state,r)
            assert abs(math.fsum(terms)-score)<=1e-7
            outputs.append(text)
            logs.append(score)
            marginals.append(total)
            graphs.append({'nodes':nodes,'edges':edges,'expanded':expanded})
        assert abs(math.fsum(marginals)-value['key_log_likelihoods'][selected])<=1e-7
        return {'status':'read','selected_index':selected,'source_records':outputs,
            'record_joint_log_probability':logs,'record_marginals':marginals,'graphs':graphs,
            'selection':'highest_final_key_likelihood_first_index_then_conditional_viterbi',
            'native_reader_edge_work_bound':800002,'python_reader_edge_work_bound':800002}
    except RuntimeError as exc:
        if not any(s in str(exc) for s in ('lattice node cap','lattice edge cap','K-best prefix cap')):
            raise
        return {'status':'reader_cap','selected_index':selected,'error':str(exc),
            'no_partial_reading':True,'native_reader_edge_work_bound':800002,
            'python_reader_edge_work_bound':800002}


def metrics(value):
    if value['status'] in ('work_cap','graph_cap'):
        return {'status':value['status'],'log_evidence_estimate':None,
            'tables':value['tables'],'edges':value['edges'],'stages_completed':len(value['trace'])}
    return compact(value)


def recovery(value,prediction,fixture,alphabet):
    used = sorted({r for t in fixture['generation_source'] for r in t})
    selected = selected_index(value)
    matches = [] if selected is None else [sum(k[r]==fixture['generation_key'][r] for r in used) for k in value['keys']]
    gold = [''.join(alphabet[r] for r in t) for t in fixture['generation_source']]
    total = sum(map(len,gold))
    return {'used_rows':len(used),'selected_matches':None if selected is None else matches[selected],
        'selected_complete_used_key':selected is not None and matches[selected]==len(used),
        'bank_contains_complete_used_key':bool(matches) and max(matches)==len(used),
        'true_letters':total,'exact_records':sum(a==b for a,b in zip(gold,prediction.get('source_records',[]))),
        'edit_errors_with_failed_readings_counted_full_length':sum(edit_distance(a,b) for a,b in zip(
            gold,prediction['source_records'],strict=True)) if prediction['status']=='read' else total}


def summarize(cells):
    arms = {}
    for arm in ARMS:
        rows = [c for c in cells if c['arm']==arm]
        complete = len(rows)==8 and all(c['summary']['status']=='complete_particles' for c in rows)
        spreads = []
        for case in range(4):
            values = [c['summary']['log_evidence_estimate'] for c in rows if c['case']==case]
            spreads.append(max(values)-min(values) if len(values)==2 and all(v is not None for v in values) else None)
        denom = sum(c['recovery']['used_rows'] for c in rows)
        arms[arm] = {'calls':len(rows),'complete_positive':sum(c['summary']['status']=='complete_particles' for c in rows),
            'extinct':sum(c['summary']['status']=='extinct' for c in rows),
            'graph_cap':sum(c['summary']['status']=='graph_cap' for c in rows),
            'work_cap':sum(c['summary']['status']=='work_cap' for c in rows),'all_complete':complete,
            'selected_match_fraction_with_failures_counted_zero':sum(c['recovery']['selected_matches'] or 0 for c in rows)/denom,
            'selected_complete_used_keys':sum(c['recovery']['selected_complete_used_key'] for c in rows),
            'banks_containing_complete_used_key':sum(c['recovery']['bank_contains_complete_used_key'] for c in rows),
            'edit_errors':sum(c['recovery']['edit_errors_with_failed_readings_counted_full_length'] for c in rows),
            'true_letters':sum(c['recovery']['true_letters'] for c in rows),
            'exact_records':sum(c['recovery']['exact_records'] for c in rows),
            'seed_spreads':spreads,'stable_calibration':complete and all(v is not None and v<=2 for v in spreads)
                and all(c['summary']['maximum_incremental_weight']<=.5 for c in rows),
            'tables':sum(c['summary']['tables'] for c in rows),
            'summed_call_wall_seconds':sum(c['wall_seconds'] for c in rows)}
    new,control = arms['covered_coordinated'],arms['covered_row']
    supported = (new['all_complete'] and control['all_complete'] and new['selected_complete_used_keys']>=1
        and new['selected_match_fraction_with_failures_counted_zero']>=control['selected_match_fraction_with_failures_counted_zero']+.10
        and new['edit_errors']<=.90*control['edit_errors'])
    return {'arms':arms,'exploratory_coordinated_recovery_supported':supported,
        'all_fresh_development_synthetics':True,'mechanism_or_language_holdout':False,
        'table_work_matched_only_when_complete':True,'equal_cpu_claimed':False,
        'historical_or_neural_qualification':False}


def guard(wall,cpu,bulk_bytes,cells):
    r = resource_report(wall,cpu)
    if (r['peak_rss_bytes']>2*1024**3 or bulk_bytes>128*1024**2
            or sum(c['summary']['edges'] for c in cells)>CAMPAIGN_EDGES
            or r['wall_seconds']>1800 or r['cpu_seconds']>1600):
        raise RuntimeError('Global campaign resource cap; abort without continuing calls')


def run(freeze):
    parent,benchmark = inputs(freeze)
    save_new(OUT/'started.json',{'freeze':freeze,'start_unix':time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells,bulk_bytes = time.monotonic(),time.process_time(),[],0
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==chosen['counts'] and parent['source_arrays']==identity
        gold = fixtures(source)
        queries = list(map(query_for,gold))
        gold_spec = save_new(BULK/'gold.json.gz',{'fixtures':gold},compressed=True)
        bulk_bytes += gold_spec['bytes']
        query_spec = save_new(OUT/'queries.json',{'queries':queries})
        native = NativeMarginal(source,benchmark['build'])
        # No recovery diagnostics until ALL predetermined fitting calls close.
        for query in queries:
            for seed in SEEDS:
                for arm in ARMS:
                    if CAMPAIGN_EDGES-sum(c['summary']['edges'] for c in cells)<CALL_EDGES+1:
                        raise RuntimeError('Insufficient global work for next fixed call; abort')
                    evaluated = evaluate(query,native,seed,arm)
                    value = evaluated.pop('value')
                    prediction = predict(value,query,source,native)
                    name = f"case{query['case']}-{seed}-{arm}"
                    spec = save_new(BULK/f'{name}.json.gz',{'value':value,'prediction':prediction},compressed=True)
                    bulk_bytes += spec['bytes']
                    cells.append({'case':query['case'],'seed':seed,'arm':arm,'output':spec,
                        'summary':metrics(value),'prediction_status':prediction['status'],**evaluated})
                    guard(wall,cpu,bulk_bytes,cells)
                    print(f"Sealed {name}: {value['status']}, reader {prediction['status']}",flush=True)
        assert len(cells)==24
        seal = save_new(OUT/'predictions-sealed.json',{'outputs':[c['output'] for c in cells],
            'queries':query_spec,'gold_not_used_by_fit_or_prediction':True})
        workloads = []
        for cell in cells:
            closed = load_bound(cell['output'])
            cell['recovery'] = recovery(closed['value'],closed['prediction'],gold[cell['case']],source.alphabet)
            name = f"case{cell['case']}-{cell['seed']}-{cell['arm']}"
            workloads.append(save_new(OUT/f'{name}.json',cell))
        assert array_identity(source)==identity
        guard(wall,cpu,bulk_bytes,cells)
        save_new(OUT/'result.json',{'experiment':EXP,'freeze':freeze,'numpy_version':np.__version__,
            'source':chosen['counts'],'source_arrays':identity,'native_build':benchmark['build'],
            'queries':query_spec,'gold':gold_spec,'prediction_seal':seal,'workloads':workloads,
            'summary':summarize(cells),'bulk_bytes':bulk_bytes,'resources':resource_report(wall,cpu),
            'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/'failure.json',{'error':repr(exc),'closed_calls':len(cells),'no_retry':True,
            'resources':resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    run(parser.parse_args().freeze)
