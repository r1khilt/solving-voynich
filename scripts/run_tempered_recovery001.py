"""Fixed fresh-key positive/shuffle campaign; seal all predictions before Gold."""

import argparse
import gc
import itertools
import json
import signal
import time

import numpy as np

from scripts import run_coordinated_dictionary001 as reader
from scripts import run_tempered_inventory001 as engineering
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import (
    ROOT,array_identity,artifact,limit_resources,load_source,require_frozen,resource_report,save_new,
)
from voynich.native_suffix_marginal import NativeMarginal
from voynich.strict_censored_bridge import CensoredWorkBudgetExceeded,StrictCensoredNativeBridge
from voynich.tempered_inventory import run_tempered_inventory_smc


EXP = 'TEMPERED-RECOVERY-001'
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
ARMS = ('supported_prior','row','coordinated','supported_refresh')
SEEDS,GENERATION_SEEDS,SHUFFLE_SEEDS = (89511,89512),(89501,89502,89503,89504),(89531,89532,89533,89534)
DESIGNS = (64,64,224,224)
PARTICLES,MUTATIONS = 128,4
BETAS = tuple((i/64)**3 for i in range(65))
CALL_EDGES,GLOBAL_EDGES = 4_000_000_000,256_000_000_128
FIT_WALL,FIT_CPU,AUDIT_WALL,AUDIT_CPU = 7200,6500,10800,10000
PATHS = sorted(set([*engineering.PATHS,'scripts/run_tempered_recovery001.py',
    'scripts/audit_tempered_recovery001.py','tests/test_tempered_recovery001.py',
    'docs/experiments/TEMPERED-RECOVERY-001.md',
    'docs/research/tempered-recovery-2026-10-01.md',
    'results/TEMPERED-INVENTORY-001/result.json','results/TEMPERED-INVENTORY-001/audit.json',
    'results/TEMPERED-INVENTORY-001/post-outcome.json','docs/experiments/TEMPERED-INVENTORY-001-results.md']))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent,benchmark,_ = engineering.inputs(freeze)
    result = json.loads((ROOT/'results/TEMPERED-INVENTORY-001/result.json').read_text())
    audit = json.loads((ROOT/'results/TEMPERED-INVENTORY-001/audit.json').read_text())
    assert audit['status']=='PASS_full_tempered_engineering_replay'
    assert audit['result']==artifact(ROOT/'results/TEMPERED-INVENTORY-001/result.json')
    assert result['summary']['engineering_gate']=='PASS'
    assert result['source']==parent['source'] and result['source_arrays']==parent['source_arrays']
    assert result['native_build']==benchmark['build']
    return parent,benchmark


def fixtures(source):
    """Generation only; fixed lengths and keys never reach inference."""
    units = tuple(u for n in (1,2) for u in itertools.product(range(6),repeat=n))
    result = []
    for pair,(seed,shuffle_seed,length) in enumerate(zip(GENERATION_SEEDS,SHUFFLE_SEEDS,DESIGNS,strict=True)):
        rng = np.random.default_rng(seed)
        key = rng.integers(42,size=23).tolist()
        texts,cipher = [],[]
        for _ in range(2):
            text,state = [],source.state('')
            for _ in range(length):
                row = int(rng.choice(23,p=source.row(state)))
                text.append(row)
                state = source.step(state,row)
            texts.append(text)
            cipher.append([g for row in text for g in units[key[row]]])
        result.append({'case':2*pair,'pair':pair,'kind':'positive','generation_seed':seed,
            'generation_length':length,'generation_key':key,'generation_source':texts,'cipher':cipher})
        shuffle = np.random.default_rng(shuffle_seed)
        result.append({'case':2*pair+1,'pair':pair,'kind':'shuffle','shuffle_seed':shuffle_seed,
            'cipher':[shuffle.permutation(record).tolist() for record in cipher],
            'accuracy_not_defined':True})
    return result


def query_for(fixture):
    return {'case':fixture['case'],'cipher':fixture['cipher'],'partial_key':[-1]*23}


def validate_query(query):
    if (set(query)!={'case','cipher','partial_key'} or type(query['case']) is not int
            or not 0<=query['case']<8 or query['partial_key']!=[-1]*23):
        raise ValueError('Registered fresh ciphertext-only all-unknown query required')


def compile_profile(query):
    validate_query(query)
    # The old compiler validates four accounting IDs; its algorithm uses no ID.
    # Preserve exact cipher/profile, mapping only the public case identifier.
    return engineering.compile_profile({**query,'case':query['case']//2})


def evaluate(query,native,profile,seed,arm,monitor=None):
    validate_query(query)
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError('Registered recovery seed and arm required')
    mutations = 0 if arm=='supported_prior' else MUTATIONS
    maximum_tables = PARTICLES*(1+(len(BETAS)-1)*mutations)
    bridge = StrictCensoredNativeBridge(query['cipher'],native,max_tables=maximum_tables,max_edges=CALL_EDGES)
    started = time.monotonic()
    trace = []
    def observe(row):
        trace.append(row)
        if monitor is not None:
            monitor(bridge)
        if (row['stage']+1)%16==0:
            print(f"Progress case{query['case']}/{seed}/{arm}: temperature{row['stage']+1}/{len(BETAS)-1}",flush=True)
    try:
        value = run_tempered_inventory_smc(bridge,profile,betas=BETAS,particles=PARTICLES,
            mutations=mutations,seed=seed,kernel='row' if arm=='supported_prior' else arm,observe=observe)
    except CensoredWorkBudgetExceeded as exc:
        value = {'status':'call_work_cap','error':str(exc),'completed_trace':trace,
            'no_partial_particle_result_or_score_substitution':True}
    except RuntimeError as exc:
        if not any(s in str(exc) for s in ('Exact lattice node cap','Exact lattice edge cap')):
            raise
        value = {'status':'native_graph_cap','error':str(exc),'completed_trace':trace,
            'no_partial_particle_result_or_score_substitution':True}
    else:
        value['keys'] = value['keys'].tolist()
        value['key_log_likelihoods'] = value['key_log_likelihoods'].tolist()
    value.update(tables=bridge.tables,edges=bridge.edges,native_calls=bridge.native_calls,
        native_nodes=bridge.nodes,maximum_native_nodes=bridge.maximum_native_nodes,
        maximum_owned_work_envelope=bridge.maximum_owned_work_envelope,
        seed=seed,arm=arm,nominal_complete_table_budget=maximum_tables,
        wall_seconds=time.monotonic()-started)
    return value


def reading_value(value):
    # Reuse the immutable qualified reader, without modifying its old status law.
    return {**value,'status':'complete_particles'} if value['status']=='complete_tempered_particles' else value


def predict(value,query,source,native):
    validate_query(query)
    return reader.predict(reading_value(value),query,source,native)


def recovery(value,prediction,fixture,alphabet):
    if fixture['kind']=='shuffle':
        return {'kind':'shuffle','accuracy_not_defined':True}
    return {'kind':'positive',**reader.recovery(reading_value(value),prediction,fixture,alphabet)}


def summarize(cells):
    arms = {}
    for arm in ARMS:
        all_rows = [c for c in cells if c['arm']==arm]
        positive = [c for c in all_rows if c['recovery']['kind']=='positive']
        complete = len(positive)==8 and all(c['summary']['status']=='complete_tempered_particles' for c in positive)
        spreads = []
        for case in range(8):
            values = [c['summary'].get('log_evidence_estimate') for c in all_rows if c['case']==case]
            spreads.append(max(values)-min(values) if len(values)==2 and all(v is not None for v in values) else None)
        margins = []
        for pair in range(4):
            for seed in SEEDS:
                pos = [c for c in all_rows if c['case']==2*pair and c['seed']==seed]
                null = [c for c in all_rows if c['case']==2*pair+1 and c['seed']==seed]
                a = pos[0]['summary'].get('log_evidence_estimate') if pos else None
                b = null[0]['summary'].get('log_evidence_estimate') if null else None
                margins.append(a-b if a is not None and b is not None else None)
        matched = sum(c['recovery']['selected_matches'] or 0 for c in positive)
        used = sum(c['recovery']['used_rows'] for c in positive)
        weights = [c['summary'].get('maximum_incremental_weight') for c in all_rows]
        arms[arm] = {'calls':len(all_rows),'complete_calls':sum(c['summary']['status']=='complete_tempered_particles' for c in all_rows),
            'positive_calls':len(positive),'positive_complete':sum(c['summary']['status']=='complete_tempered_particles' for c in positive),
            'all_positive_complete':complete,'selected_used_row_matches':matched,'used_row_denominator':used,
            'used_row_match_fraction_with_failed_calls_zero':matched/used if used else 0.,
            'selected_complete_used_keys':sum(c['recovery']['selected_complete_used_key'] for c in positive),
            'banks_containing_complete_used_key':sum(c['recovery']['bank_contains_complete_used_key'] for c in positive),
            'edit_errors':sum(c['recovery']['edit_errors_with_failed_readings_counted_full_length'] for c in positive),
            'true_letters':sum(c['recovery']['true_letters'] for c in positive),
            'exact_records':sum(c['recovery']['exact_records'] for c in positive),
            'seed_log_evidence_spreads':spreads,'positive_minus_shuffle_log_evidence_margins':margins,
            'calibration_gate':len(all_rows)==16 and all(v is not None and v<=2 for v in spreads)
                and all(w is not None and w<=.5 for w in weights),
            'positive_shuffle_gate':len(margins)==8 and all(m is not None and m>0 for m in margins),
            'native_edge_work':sum(c['summary']['edges'] for c in all_rows),
            'native_graph_caps':sum(c['summary']['status']=='native_graph_cap' for c in all_rows),
            'call_work_caps':sum(c['summary']['status']=='call_work_cap' for c in all_rows),
            'compiler_caps':sum(c['summary']['status']=='compiler_cap' for c in all_rows)}
    new,control = arms['supported_refresh'],arms['coordinated']
    recovery_gate = (new['all_positive_complete'] and control['all_positive_complete']
        and new['selected_complete_used_keys']>=1 and new['exact_records']>=1
        and new['used_row_match_fraction_with_failed_calls_zero']>=control['used_row_match_fraction_with_failed_calls_zero']+.10
        and new['edit_errors']<=.90*control['edit_errors'])
    return {'arms':arms,'exploratory_recovery_gate':recovery_gate,
        'combined_qualification_gate':recovery_gate and new['calibration_gate'] and new['positive_shuffle_gate'],
        'fresh_development_keys':True,'mechanism_language_or_historical_holdout':False,
        'historical_or_neural_qualification':False,'equal_actual_compute_claimed':False}


def guard(wall,cpu,bulk,cells,extra_edges=0,*,auditing=False):
    r = resource_report(wall,cpu)
    if (r['wall_seconds']>(AUDIT_WALL if auditing else FIT_WALL)
            or r['cpu_seconds']>(AUDIT_CPU if auditing else FIT_CPU)
            or r['peak_rss_bytes']>2*1024**3 or bulk>128*1024**2
            or sum(c['summary']['edges'] for c in cells)+extra_edges>GLOBAL_EDGES):
        raise RuntimeError('Global fresh tempered campaign resource cap; abort without continuing calls')


def run(freeze):
    parent,benchmark = inputs(freeze)
    save_new(OUT/'started.json',{'freeze':freeze,'start_unix':time.time()})
    limit_resources(FIT_WALL,FIT_CPU)
    wall,cpu,cells,bulk = time.monotonic(),time.process_time(),[],0
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==chosen['counts'] and parent['source_arrays']==identity
        gold = fixtures(source)
        queries = list(map(query_for,gold))
        gold_spec = save_new(BULK/'gold.json.gz',{'fixtures':gold},compressed=True)
        bulk += gold_spec['bytes']
        query_spec = save_new(OUT/'queries.json',{'queries':queries})
        native = NativeMarginal(source,benchmark['build'])
        for query in queries:
            profile,compiled = compile_profile(query)
            for seed in SEEDS:
                for arm in ARMS:
                    if GLOBAL_EDGES-sum(c['summary']['edges'] for c in cells)<CALL_EDGES+1:
                        raise RuntimeError('Insufficient global work for next fixed call; abort')
                    value = evaluate(query,native,profile,seed,arm,
                        monitor=lambda b:guard(wall,cpu,bulk,cells,b.edges)) if profile is not None else dict(compiled)
                    prediction = predict(value,query,source,native)
                    name = f"case{query['case']}-{seed}-{arm}"
                    spec = save_new(BULK/f'{name}.json.gz',{'value':value,'prediction':prediction},compressed=True)
                    bulk += spec['bytes']
                    cell = {'case':query['case'],'seed':seed,'arm':arm,'compiler':compiled,'output':spec,
                        'summary':engineering.metrics(value),'prediction_status':prediction['status']}
                    cells.append(cell)
                    save_new(OUT/f'{name}-sealed.json',cell)
                    guard(wall,cpu,bulk,cells)
                    print(f"Sealed {name}: {value['status']}, reader{prediction['status']}",flush=True)
                    gc.collect()
            profile = None
            gc.collect()
        assert len(cells)==64
        seal = save_new(OUT/'predictions-sealed.json',{'outputs':[c['output'] for c in cells],
            'queries':query_spec,'gold_not_used_by_fit_or_prediction':True})
        workloads = []
        for cell in cells:
            closed = load_bound(cell['output'])
            cell['recovery'] = recovery(closed['value'],closed['prediction'],gold[cell['case']],source.alphabet)
            workloads.append(save_new(OUT/f"case{cell['case']}-{cell['seed']}-{cell['arm']}.json",cell))
        assert array_identity(source)==identity
        guard(wall,cpu,bulk,cells)
        save_new(OUT/'result.json',{'experiment':EXP,'freeze':freeze,'numpy_version':np.__version__,
            'source':chosen['counts'],'source_arrays':identity,'native_build':benchmark['build'],
            'queries':query_spec,'gold':gold_spec,'prediction_seal':seal,'workloads':workloads,
            'betas':BETAS,'particles':PARTICLES,'mutations':MUTATIONS,'arms':ARMS,'seeds':SEEDS,
            'summary':summarize(cells),'bulk_bytes':bulk,'resources':resource_report(wall,cpu),'paid_spend_usd':0})
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
