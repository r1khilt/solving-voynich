"""One fixed original-source engineering panel; ciphertext inputs only."""

import argparse
import gc
import json
import signal
import time

import numpy as np

from scripts import run_compiled_inventory003 as compiler
from scripts.run_source_state_systems001 import (
    ROOT,array_identity,artifact,limit_resources,load_source,require_frozen,resource_report,save_new,
)
from voynich.inventory_bdd import InventoryBudgetExceeded,InventoryProfile
from voynich.native_suffix_marginal import NativeMarginal
from voynich.rolling_inventory_bdd import RollingInventoryBDD
from voynich.strict_censored_bridge import CensoredWorkBudgetExceeded,StrictCensoredNativeBridge
from voynich.tempered_inventory import run_tempered_inventory_smc


EXP = 'TEMPERED-INVENTORY-001'
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
KERNELS = ('row','supported_refresh')
BETAS = tuple((i/16)**3 for i in range(17))
PARTICLES,MUTATIONS = 32,4
MAX_TABLES,MAX_EDGES = 2080,500_000_000
GLOBAL_EDGES = 4_000_000_016
PATHS = sorted(set([*compiler.PATHS,'src/voynich/tempered_inventory.py',
    'tests/test_tempered_inventory.py','scripts/check_full_support_tempering001.py',
    'results/FULL-SUPPORT-TEMPERING-THEORY-001/result.json',
    'docs/research/full-support-tempering-2026-10-01.md',
    'results/COMPILED-INVENTORY-003/result.json','results/COMPILED-INVENTORY-003/audit.json',
    'results/COMPILED-INVENTORY-003/post-outcome.json',
    'scripts/run_tempered_inventory001.py','scripts/audit_tempered_inventory001.py',
    'tests/test_tempered_inventory001.py','docs/experiments/TEMPERED-INVENTORY-001.md']))


def inputs(freeze):
    require_frozen(freeze,PATHS)
    parent,benchmark,queries = compiler.inputs(freeze)
    result = json.loads((ROOT/'results/COMPILED-INVENTORY-003/result.json').read_text())
    audit = json.loads((ROOT/'results/COMPILED-INVENTORY-003/audit.json').read_text())
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==artifact(ROOT/'results/COMPILED-INVENTORY-003/result.json')
    assert result['queries']==queries and result['source_arrays']==parent['source_arrays']
    assert result['source']==parent['source'] and result['native_build']==benchmark['build']
    proof = json.loads((ROOT/'results/FULL-SUPPORT-TEMPERING-THEORY-001/result.json').read_text())
    assert proof['status']=='PASS_exact_full_support_tempering_laws'
    assert all(artifact(ROOT/a['path'])==a for a in proof['inputs'])
    return parent,benchmark,queries


def compile_profile(query):
    if (set(query)!={'case','cipher','partial_key'} or type(query['case']) is not int
            or not 0<=query['case']<4 or query['partial_key']!=[-1]*23):
        raise ValueError('Registered ciphertext-only all-unknown query required')
    bdd = RollingInventoryBDD.__new__(RollingInventoryBDD)
    try:
        RollingInventoryBDD.__init__(bdd,query['cipher'],
            order=compiler.previous.original.variable_order(query['cipher'],'frequency'),**compiler.LIMITS)
        profile = InventoryProfile(bdd,np.array(query['partial_key']))
        assert profile.total>0
        return profile,{'status':'compiled','stats':bdd.stats(),
            'dictionary_count':str(profile.total),'prior_support_probability':str(profile.prior_event_probability)}
    except InventoryBudgetExceeded as exc:
        return None,{'status':'compiler_cap','error':str(exc),'stats':bdd.stats(),
            'no_partial_graph_count_or_sampler':True}


def evaluate(query,kernel,profile,native):
    if kernel not in KERNELS:
        raise ValueError('Fixed engineering kernel required')
    bridge = StrictCensoredNativeBridge(query['cipher'],native,max_tables=MAX_TABLES,max_edges=MAX_EDGES)
    start = time.monotonic()
    trace = []
    try:
        value = run_tempered_inventory_smc(bridge,profile,betas=BETAS,particles=PARTICLES,
            mutations=MUTATIONS,seed=83211+query['case'],kernel=kernel,observe=trace.append)
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
        seed=83211+query['case'],kernel=kernel,wall_seconds=time.monotonic()-start)
    return value


def metrics(value):
    trace = value.get('trace',value.get('completed_trace',[]))
    row = {'status':value['status'],'tables':value.get('tables',0),'edges':value.get('edges',0),
        'completed_stages':len(trace),'minimum_incremental_ess':min((t['incremental_ess'] for t in trace),default=None),
        'maximum_incremental_weight':max((t['maximum_incremental_weight'] for t in trace),default=None),
        'accepted_changes':sum(t['changed_proposals'] for t in trace),
        'supported_refresh_draws':sum(t['supported_refresh_draws'] for t in trace),
        'support_rejected_without_scoring':sum(t['support_rejected_without_scoring'] for t in trace)}
    if value['status']=='complete_tempered_particles':
        row.update(final_distinct_keys=len(set(map(tuple,value['keys']))),
            log_evidence_estimate=value['log_evidence_estimate'],
            final_minimum_log_likelihood=min(value['key_log_likelihoods']),
            final_maximum_log_likelihood=max(value['key_log_likelihoods']))
    return row


def summarize(cells):
    complete = sum(c['summary']['status']=='complete_tempered_particles' for c in cells)
    return {'engineering_gate':'PASS' if len(cells)==8 and complete==8 else 'FAIL',
        'calls':len(cells),'complete_calls':complete,'failed_calls':len(cells)-complete,
        'native_edge_work':sum(c['summary']['edges'] for c in cells),
        'all_queries_exposed_development':True,'recovery_or_calibration_or_historical_claim':False}


def guard(wall,cpu,bulk,edges):
    r = resource_report(wall,cpu)
    if (r['wall_seconds']>1200 or r['cpu_seconds']>1000 or r['peak_rss_bytes']>2*1024**3
            or bulk>128*1024**2 or edges>GLOBAL_EDGES):
        raise RuntimeError('Global tempering engineering resource cap; abort')


def run(freeze):
    parent,benchmark,queries = inputs(freeze)
    save_new(OUT/'started.json',{'freeze':freeze,'start_unix':time.time()})
    limit_resources(1200,1000)
    wall,cpu,cells,bulk = time.monotonic(),time.process_time(),[],0
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==chosen['counts'] and parent['source_arrays']==identity
        native = NativeMarginal(source,benchmark['build'])
        for query in queries:
            profile,compiled = compile_profile(query)
            for kernel in KERNELS:
                value = evaluate(query,kernel,profile,native) if profile is not None else dict(compiled)
                name = f"case{query['case']}-{kernel}"
                spec = save_new(BULK/f'{name}.json.gz',value,compressed=True)
                bulk += spec['bytes']
                cell = {'case':query['case'],'kernel':kernel,'compiler':compiled,'output':spec,'summary':metrics(value)}
                cells.append(cell)
                save_new(OUT/f'{name}.json',cell)
                guard(wall,cpu,bulk,sum(c['summary']['edges'] for c in cells))
                print(f"Closed {name}: {value['status']}",flush=True)
                gc.collect()
            profile = None
            gc.collect()
        assert array_identity(source)==identity
        save_new(OUT/'result.json',{'experiment':EXP,'freeze':freeze,'queries':queries,
            'source':chosen['counts'],'source_arrays':identity,'native_build':benchmark['build'],
            'numpy_version':np.__version__,'betas':BETAS,'particles':PARTICLES,'mutations':MUTATIONS,
            'kernels':KERNELS,'max_tables':MAX_TABLES,'max_edges_per_call':MAX_EDGES,
            'workloads':[artifact(OUT/f"case{c['case']}-{c['kernel']}.json") for c in cells],
            'summary':summarize(cells),'bulk_bytes':bulk,'resources':resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/'failure.json',{'error':repr(exc),'closed_calls':len(cells),
            'no_retry':True,'resources':resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    run(parser.parse_args().freeze)
