"""One full fresh-fixture/search/reader/Gold replay, including failures."""

import gc
import json
import signal
import time

import numpy as np

from scripts import run_tempered_recovery001 as run
from scripts.audit_censored_context001 import logical
from scripts.run_shared_key_guide001 import load_bound
from tests.test_inventory_bdd import literal_support
from voynich.dictionary_smc import FixedKeyBridge
from voynich.native_suffix_marginal import NativeMarginal


def audit():
    result = json.loads((run.OUT/'result.json').read_text())
    parent,benchmark = run.inputs(result['freeze'])
    assert result['experiment']==run.EXP
    assert result['numpy_version']==np.__version__ and result['native_build']==benchmark['build']
    assert result['betas']==list(run.BETAS) and result['particles']==run.PARTICLES
    assert result['mutations']==run.MUTATIONS and result['arms']==list(run.ARMS) and result['seeds']==list(run.SEEDS)
    run.save_new(run.OUT/'audit-started.json',{'freeze':result['freeze'],'start_unix':time.time()})
    run.limit_resources(run.AUDIT_WALL,run.AUDIT_CPU)
    wall,cpu,cells,checked,literal,max_delta = time.monotonic(),time.process_time(),[],0,0,0.
    bulk = result['gold']['bytes']
    try:
        source,chosen = run.load_source()
        identity = run.array_identity(source)
        assert parent['source']==result['source']==chosen['counts']
        assert parent['source_arrays']==result['source_arrays']==identity
        gold = load_bound(result['gold'])['fixtures']
        assert gold==run.fixtures(source)
        queries = load_bound(result['queries'])['queries']
        assert queries==list(map(run.query_for,gold))
        seal = load_bound(result['prediction_seal'])
        assert seal['queries']==result['queries'] and seal['gold_not_used_by_fit_or_prediction']
        native = NativeMarginal(source,benchmark['build'])
        for query in queries:
            profile,compiled = run.compile_profile(query)
            for seed in run.SEEDS:
                for arm in run.ARMS:
                    spec = result['workloads'][len(cells)]
                    name = f"case{query['case']}-{seed}-{arm}"
                    assert spec==run.artifact(run.OUT/f'{name}.json')
                    cell = load_bound(spec)
                    assert (cell['case'],cell['seed'],cell['arm'])==(query['case'],seed,arm)
                    assert cell['compiler']==compiled
                    sealed_cell = load_bound(run.artifact(run.OUT/f'{name}-sealed.json'))
                    assert sealed_cell=={k:v for k,v in cell.items() if k!='recovery'}
                    stored = load_bound(cell['output'])
                    value = run.evaluate(query,native,profile,seed,arm,
                        monitor=lambda b:run.guard(wall,cpu,bulk,cells,b.edges,auditing=True)) if profile is not None else dict(compiled)
                    assert logical(value)==logical(stored['value'])
                    prediction = run.predict(value,query,source,native)
                    assert prediction==stored['prediction'] and cell['prediction_status']==prediction['status']
                    assert cell['summary']==run.engineering.metrics(value)
                    assert cell['recovery']==run.recovery(value,prediction,gold[cell['case']],source.alphabet)
                    if value['status']=='complete_tempered_particles':
                        assert len(value['keys'])==run.PARTICLES and len(value['trace'])==len(run.BETAS)-1
                        assert all(literal_support(query['cipher'],profile.bdd.units,set(k)) for k in value['keys'])
                        literal += len(value['keys'])
                        # Fixed FIRSTeight, without deduplication; no Gold-based selection.
                        size = min(8,run.PARTICLES)
                        reference = FixedKeyBridge(query['cipher'],native.probabilities,native.transitions,
                            max_tables=size,max_edges=800_000_000,max_states_per_table=300_000,max_work_bytes=768*1024**2)
                        scores = reference.log_values(np.array(value['keys'][:size]),reference.lengths,closed=True)
                        assert np.isfinite(scores).all()
                        delta = float(np.max(np.abs(scores-np.array(value['key_log_likelihoods'][:size]))))
                        assert delta<=1e-7
                        max_delta,checked = max(max_delta,delta),checked+len(scores)
                        reference = None
                        gc.collect()
                    else:
                        assert value['status'] in ('compiler_cap','native_graph_cap','call_work_cap')
                        assert 'keys' not in value and 'key_log_likelihoods' not in value
                    bulk += cell['output']['bytes']
                    cells.append(cell)
                    run.guard(wall,cpu,bulk,cells,auditing=True)
                    print(f"Audited {name}: {value['status']}, reader{prediction['status']}",flush=True)
            profile = None
            gc.collect()
        assert len(cells)==len(result['workloads'])==64
        assert seal['outputs']==[c['output'] for c in cells]
        assert result['summary']==run.summarize(cells) and result['bulk_bytes']==bulk
        assert run.array_identity(source)==identity
        assert result['resources']['wall_seconds']<=run.FIT_WALL and result['resources']['cpu_seconds']<=run.FIT_CPU
        assert result['resources']['peak_rss_bytes']<=2*1024**3
        run.save_new(run.OUT/'audit.json',{'status':'PASS_full_fresh_tempered_recovery_replay',
            'result':run.artifact(run.OUT/'result.json'),'calls':len(cells),'literal_final_key_checks':literal,
            'independent_fixed_final_key_source_checks':checked,'maximum_source_log_delta':max_delta,
            'fresh_fixture_generation_replayed':True,'all_predictions_sealed_before_metrics':True,
            'same_author':True,'replay_shares_compiler_sampler_native_backend':True,'independent_agent_review':False,
            'resources':run.resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        run.save_new(run.OUT/'audit-failure.json',{'error':repr(exc),'audited_calls':len(cells),'no_retry':True,
            'resources':run.resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    audit()
