"""Replay every fixed compiler/search call, including typed failures."""

import gc
import json
import signal
import time

import numpy as np

from scripts import run_tempered_inventory001 as run
from scripts.audit_censored_context001 import logical
from scripts.run_shared_key_guide001 import load_bound
from voynich.dictionary_smc import FixedKeyBridge
from voynich.native_suffix_marginal import NativeMarginal
from tests.test_inventory_bdd import literal_support


def audit():
    result = json.loads((run.OUT/'result.json').read_text())
    parent,benchmark,queries = run.inputs(result['freeze'])
    assert result['experiment']==run.EXP and result['queries']==queries
    assert result['numpy_version']==np.__version__ and result['native_build']==benchmark['build']
    assert result['betas']==list(run.BETAS) and result['kernels']==list(run.KERNELS)
    assert result['particles']==run.PARTICLES and result['mutations']==run.MUTATIONS
    assert result['max_tables']==run.MAX_TABLES and result['max_edges_per_call']==run.MAX_EDGES
    run.save_new(run.OUT/'audit-started.json',{'freeze':result['freeze'],'start_unix':time.time()})
    run.limit_resources(1200,1000)
    wall,cpu,cells,bulk,checked,delta = time.monotonic(),time.process_time(),[],0,0,0.
    try:
        source,chosen = run.load_source()
        identity = run.array_identity(source)
        assert result['source']==parent['source']==chosen['counts']
        assert result['source_arrays']==parent['source_arrays']==identity
        native = NativeMarginal(source,benchmark['build'])
        for query in queries:
            profile,compiled = run.compile_profile(query)
            for kernel in run.KERNELS:
                index = len(cells)
                spec = result['workloads'][index]
                assert spec==run.artifact(run.OUT/f"case{query['case']}-{kernel}.json")
                cell = load_bound(spec)
                assert (cell['case'],cell['kernel'])==(query['case'],kernel)
                assert cell['compiler']==compiled
                value = load_bound(cell['output'])
                replay = run.evaluate(query,kernel,profile,native) if profile is not None else dict(compiled)
                assert logical(value)==logical(replay)
                assert cell['summary']==run.metrics(value)
                if value['status']=='complete_tempered_particles':
                    assert len(value['keys'])==run.PARTICLES and len(value['trace'])==len(run.BETAS)-1
                    assert all(literal_support(query['cipher'],profile.bdd.units,set(k)) for k in value['keys'])
                    # FIRST four final particles, fixed before any panel outcome.
                    reference = FixedKeyBridge(query['cipher'],native.probabilities,native.transitions,
                        max_tables=4,max_edges=400_000_000,max_states_per_table=300_000,max_work_bytes=768*1024**2)
                    scores = reference.log_values(np.array(value['keys'][:4]),reference.lengths,closed=True)
                    assert np.isfinite(scores).all()
                    difference = float(np.max(np.abs(scores-np.array(value['key_log_likelihoods'][:4]))))
                    assert difference<=1e-7
                    delta = max(delta,difference)
                    checked += len(scores)
                    reference = None
                    gc.collect()
                else:
                    assert value['status'] in ('compiler_cap','call_work_cap','native_graph_cap')
                    assert 'keys' not in value and 'key_log_likelihoods' not in value
                bulk += cell['output']['bytes']
                cells.append(cell)
                run.guard(wall,cpu,bulk,sum(c['summary']['edges'] for c in cells))
                print(f"Audited case{query['case']}-{kernel}: {value['status']}",flush=True)
            profile = None
            gc.collect()
        assert len(cells)==len(result['workloads'])==8
        assert result['summary']==run.summarize(cells) and result['bulk_bytes']==bulk
        assert run.array_identity(source)==identity
        r = result['resources']
        assert r['wall_seconds']<=1200 and r['cpu_seconds']<=1000 and r['peak_rss_bytes']<=2*1024**3
        run.save_new(run.OUT/'audit.json',{'status':'PASS_full_tempered_engineering_replay',
            'result':run.artifact(run.OUT/'result.json'),'calls':len(cells),
            'independent_final_key_source_checks':checked,'maximum_source_log_delta':delta,
            'literal_final_key_checks':run.PARTICLES*result['summary']['complete_calls'],
            'same_author':True,'replay_shares_compiler_sampler_native_backend':True,
            'independent_agent_review':False,'resources':run.resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        run.save_new(run.OUT/'audit-failure.json',{'error':repr(exc),'audited_calls':len(cells),'no_retry':True,
            'resources':run.resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    audit()
