"""One full-record compiler/count/sampling engineering panel on exposed ciphers."""

import argparse
import gc
import json
import math
import signal
import time

import numpy as np

from scripts.benchmark_global_search_systems001 import admission as native_admission
from scripts.run_coordinated_dictionary001 import PATHS as OLD_PATHS
from scripts.run_shared_key_guide001 import load_bound
from scripts.run_source_state_systems001 import (
    ROOT,array_identity,artifact,limit_resources,load_source,require_frozen,
    resource_report,save_new,
)
from tests.test_inventory_bdd import literal_support
from voynich.dictionary_smc import FixedKeyBridge
from voynich.inventory_bdd import InventoryBDD,InventoryBudgetExceeded,InventoryProfile
from voynich.native_suffix_marginal import NativeMarginal
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


EXP = 'COMPILED-INVENTORY-001'
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
ORDERS = ('natural','frequency')
LIMITS = {'max_nodes':150_000,'max_apply_steps':3_000_000,'max_apply_entries':300_000,
    'max_polynomial_cells':3_000_000,'max_polynomial_work':50_000_000,'max_work_bytes':512*1024**2}
PATHS = sorted(set([*OLD_PATHS,'src/voynich/inventory_bdd.py','tests/test_inventory_bdd.py',
    'scripts/check_inventory_bdd001.py','results/INVENTORY-BDD-THEORY-001/result.json',
    'scripts/run_compiled_inventory001.py','scripts/audit_compiled_inventory001.py',
    'tests/test_compiled_inventory001.py','docs/experiments/COMPILED-INVENTORY-001.md',
    'docs/research/compiled-inventory-sampling-2026-10-01.md',
    'scripts/check_observation_initialization001.py',
    'results/COORDINATED-DICTIONARY-001/result.json','results/COORDINATED-DICTIONARY-001/audit.json',
    'results/COORDINATED-DICTIONARY-001/queries.json']))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    benchmark = native_admission(freeze)
    result = json.loads((ROOT/'results/COORDINATED-DICTIONARY-001/result.json').read_text())
    audit = json.loads((ROOT/'results/COORDINATED-DICTIONARY-001/audit.json').read_text())
    assert audit['status']=='PASS_full_coordinated_panel_replay'
    assert audit['result']==artifact(ROOT/'results/COORDINATED-DICTIONARY-001/result.json')
    contextual = json.loads((ROOT/'results/CENSORED-CONTEXT-001/result.json').read_text())
    contextual_audit = json.loads((ROOT/'results/CENSORED-CONTEXT-001/audit.json').read_text())
    assert contextual_audit['status']=='PASS_complete_contextual_benchmark_replay'
    assert contextual_audit['result']==artifact(ROOT/'results/CENSORED-CONTEXT-001/result.json')
    assert contextual['summary']['contextual_engineering_gate']=='PASS'
    assert result['source']==contextual['source']==benchmark['source']
    assert result['source_arrays']==contextual['source_arrays']
    assert result['native_build']==contextual['native_build']==benchmark['build']
    queries = load_bound(result['queries'])['queries']
    assert [q['case'] for q in queries]==list(range(4))
    assert all(set(q)=={'case','cipher','partial_key'} and q['partial_key']==[-1]*23 for q in queries)
    proof = json.loads((ROOT/'results/INVENTORY-BDD-THEORY-001/result.json').read_text())
    assert proof['status']=='PASS_exact_inventory_bdd_laws'
    assert all(artifact(ROOT/s['path'])==s for s in proof['inputs'])
    return result,benchmark,queries


def variable_order(cipher,kind):
    if kind not in ORDERS:
        raise ValueError('Registered variable order required')
    if kind=='natural':
        return tuple(range(42))
    counts = [0]*42
    for record in cipher:
        for i,glyph in enumerate(record):
            counts[glyph] += 1
            if i+1<len(record):
                counts[6+6*glyph+record[i+1]] += 1
    return tuple(sorted(range(42),key=lambda c:(-counts[c],c)))


def evaluate(query,kind,native):
    if (set(query)!={'case','cipher','partial_key'} or type(query['case']) is not int
            or not 0<=query['case']<4 or query['partial_key']!=[-1]*23):
        raise ValueError('Registered cipher-only all-unknown query required')
    start = time.monotonic()
    bdd = InventoryBDD.__new__(InventoryBDD)
    try:
        # Keep the object for counter-only reporting when construction hits a cap.
        InventoryBDD.__init__(bdd,query['cipher'],order=variable_order(query['cipher'],kind),**LIMITS)
        profile = InventoryProfile(bdd,np.array(query['partial_key']))
        assert profile.total>0 and 0<profile.prior_event_probability<=1
        rng = np.random.default_rng(82711+query['case'])
        keys = [list(map(int,profile.sample(rng))) for _ in range(32)]
        controls = np.random.default_rng(82701+query['case']).integers(42,size=(64,23)).tolist()
        assert all(bdd.accepts(k)==literal_support(query['cipher'],bdd.units,set(k)) for k in controls)
        assert all(literal_support(query['cipher'],bdd.units,set(k)) for k in keys)
        value = {'status':'compiled_sampled','order':list(bdd.order),'nodes':[list(n) for n in bdd.nodes],
            'root':bdd.root,'record_roots':list(bdd.record_roots),'coefficients':list(profile.coefficients),
            'cardinality_weights':[str(w) for w in profile.cardinality_weights],
            'dictionary_count':str(profile.total),'prior_support_probability':str(profile.prior_event_probability),
            'keys':keys,'control_keys':controls,'control_supported':[bdd.accepts(k) for k in controls],
            'sampler_seed':82711+query['case'],'control_seed':82701+query['case'],'stats':bdd.stats()}
    except InventoryBudgetExceeded as exc:
        return {'value':{'status':'compiler_cap','error':str(exc),'stats':bdd.stats(),
            'no_partial_count_or_sampler':True},'wall_seconds':time.monotonic()-start}
    # One fixed first sampled key, not selected by gold or source likelihood.
    bridge = StrictCensoredNativeBridge(query['cipher'],native,max_tables=1,max_edges=5_000_000)
    try:
        native_log = float(bridge.log_values(np.array([keys[0]],dtype=np.int32),bridge.lengths,closed=True)[0])
    except RuntimeError as exc:
        if not any(s in str(exc) for s in ('Exact lattice node cap','Exact lattice edge cap')):
            raise
        value['source_check'] = {'status':'native_graph_cap','error':str(exc),'native_calls':bridge.native_calls,
            'native_nodes':bridge.nodes,'native_edges':bridge.edges,'no_partial_likelihood':True}
    else:
        assert math.isfinite(native_log)
        reference = FixedKeyBridge(query['cipher'],native.probabilities,native.transitions,
            max_tables=1,max_edges=100_000_000,max_states_per_table=300_000,max_work_bytes=768*1024**2)
        python_log = float(reference.log_values(np.array([keys[0]],dtype=np.int32),reference.lengths,closed=True)[0])
        assert math.isfinite(python_log) and abs(python_log-native_log)<=1e-7
        value['source_check'] = {'status':'positive_scores_agree','native_log':native_log,'python_log':python_log,
            'delta':abs(python_log-native_log),'native_calls':bridge.native_calls,
            'native_nodes':bridge.nodes,'native_edges':bridge.edges,'python_edge_work':reference.edges}
        reference = None
        gc.collect()
    return {'value':value,'wall_seconds':time.monotonic()-start}


def metrics(value):
    return {'status':value['status'],'stats':value['stats'],
        'dictionary_count':value.get('dictionary_count'),
        'sampled_keys':len(value.get('keys',[])),
        'source_check_status':value.get('source_check',{}).get('status'),
        'source_score_delta':value.get('source_check',{}).get('delta')}


def summarize(cells):
    equal = []
    for case in range(4):
        rows = [c for c in cells if c['case']==case]
        counts = [c['summary']['dictionary_count'] for c in rows]
        equal.append(len(rows)==2 and None not in counts and counts[0]==counts[1])
    passed = (len(cells)==8 and all(c['summary']['status']=='compiled_sampled'
        and c['summary']['sampled_keys']==32 and c['summary']['source_check_status']=='positive_scores_agree'
        for c in cells) and all(equal))
    return {'engineering_gate':'PASS' if passed else 'FAIL','calls':len(cells),
        'compiled_calls':sum(c['summary']['status']=='compiled_sampled' for c in cells),
        'compiler_caps':sum(c['summary']['status']=='compiler_cap' for c in cells),
        'whole_record_supported_sampled_keys':sum(c['summary']['sampled_keys'] for c in cells),
        'variable_order_count_invariance':equal,'recovery_or_historical_claim':False,
        'all_queries_exposed_development':True}


def guard(wall,cpu,bulk_bytes):
    r = resource_report(wall,cpu)
    if (r['wall_seconds']>600 or r['cpu_seconds']>500 or r['peak_rss_bytes']>2*1024**3
            or bulk_bytes>128*1024**2):
        raise RuntimeError('Global compiler campaign resource cap; abort')


def run(freeze):
    parent,benchmark,queries = inputs(freeze)
    save_new(OUT/'started.json',{'freeze':freeze,'start_unix':time.time()})
    limit_resources(600,500)
    wall,cpu,cells,bulk = time.monotonic(),time.process_time(),[],0
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==chosen['counts'] and parent['source_arrays']==identity
        native = NativeMarginal(source,benchmark['build'])
        for query in queries:
            for kind in ORDERS:
                evaluated = evaluate(query,kind,native)
                value = evaluated.pop('value')
                name = f"case{query['case']}-{kind}"
                spec = save_new(BULK/f'{name}.json.gz',value,compressed=True)
                bulk += spec['bytes']
                cells.append({'case':query['case'],'order':kind,'output':spec,'summary':metrics(value),**evaluated})
                save_new(OUT/f'{name}.json',cells[-1])
                guard(wall,cpu,bulk)
                print(f"Closed {name}: {value['status']}",flush=True)
                gc.collect()
        assert array_identity(source)==identity
        save_new(OUT/'result.json',{'experiment':EXP,'freeze':freeze,'queries':queries,
            'source':chosen['counts'],'source_arrays':identity,'native_build':benchmark['build'],
            'numpy_version':np.__version__,'workloads':[artifact(OUT/f"case{c['case']}-{c['order']}.json") for c in cells],
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
