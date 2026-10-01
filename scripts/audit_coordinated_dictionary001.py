"""One full RNG/native/reader replay and independent final-bank Python DP."""

import gc
import json
import signal
import time

import numpy as np

from scripts.audit_censored_context001 import logical
from scripts.run_coordinated_dictionary001 import (
    OUT,NativeMarginal,array_identity,artifact,evaluate,fixtures,guard,inputs,
    limit_resources,load_bound,load_source,metrics,predict,query_for,recovery,
    resource_report,save_new,summarize,
)
from voynich.dictionary_smc import FixedKeyBridge


def audit():
    result = json.loads((OUT/'result.json').read_text())
    parent,benchmark = inputs(result['freeze'])
    assert result['numpy_version']==np.__version__ and result['native_build']==benchmark['build']
    save_new(OUT/'audit-started.json',{'freeze':result['freeze'],'start_unix':time.time()})
    limit_resources(1800,1600)
    wall,cpu,cells,checked,max_delta = time.monotonic(),time.process_time(),[],0,0.
    bulk_bytes = result['gold']['bytes']
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==result['source']==chosen['counts']
        assert parent['source_arrays']==result['source_arrays']==identity
        gold = load_bound(result['gold'])['fixtures']
        assert gold==fixtures(source)
        queries = load_bound(result['queries'])['queries']
        assert queries==list(map(query_for,gold))
        seal = load_bound(result['prediction_seal'])
        assert seal['queries']==result['queries'] and seal['gold_not_used_by_fit_or_prediction']
        native = NativeMarginal(source,benchmark['build'])
        for manifest in result['workloads']:
            cell = load_bound(manifest)
            stored = load_bound(cell['output'])
            query = queries[cell['case']]
            replay = evaluate(query,native,cell['seed'],cell['arm'])
            value = replay.pop('value')
            assert value==stored['value']
            assert logical(replay)==logical({k:cell[k] for k in replay})
            prediction = predict(value,query,source,native)
            assert prediction==stored['prediction'] and cell['prediction_status']==prediction['status']
            assert cell['summary']==metrics(value)
            assert cell['recovery']==recovery(value,prediction,gold[cell['case']],source.alphabet)
            if value['status']=='complete_particles':
                assert cell['summary']['tables']==cell['nominal_complete_table_budget']
                reference = FixedKeyBridge(query['cipher'],source.probabilities,source.transitions,
                    max_tables=32,max_edges=1_000_000_000,max_states_per_table=300_000,
                    max_work_bytes=768*1024**2)
                observed = reference.log_values(np.array(value['keys'],dtype=np.int32),reference.lengths,closed=True)
                expected = np.array(value['key_log_likelihoods'])
                assert np.all(np.isfinite(observed))
                delta = float(np.max(np.abs(observed-expected)))
                assert delta<=1e-7
                max_delta,checked = max(max_delta,delta),checked+len(observed)
                reference = None
                gc.collect()
            bulk_bytes += cell['output']['bytes']
            cells.append(cell)
            guard(wall,cpu,bulk_bytes,cells)
            print(f"Audited case{cell['case']}-{cell['seed']}-{cell['arm']}: {value['status']}",flush=True)
        assert len(cells)==24 and result['summary']==summarize(cells)
        assert seal['outputs']==[c['output'] for c in cells]
        assert bulk_bytes==result['bulk_bytes'] and array_identity(source)==identity
        assert checked==32*sum(c['summary']['status']=='complete_particles' for c in cells)
        assert result['resources']['wall_seconds']<=1800 and result['resources']['cpu_seconds']<=1600
        assert result['resources']['peak_rss_bytes']<=2*1024**3
        save_new(OUT/'audit.json',{'status':'PASS_full_coordinated_panel_replay',
            'result':artifact(OUT/'result.json'),'calls':len(cells),'independent_final_key_scores':checked,
            'maximum_final_score_log_delta':max_delta,'resources':resource_report(wall,cpu),
            'same_author':True,'independent_agent_review':False,'intermediate_replay_shares_native_backend':True,
            'fresh_fixture_generation_replayed':True,'all_predictions_sealed_before_metrics':True,
            'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/'audit-failure.json',{'error':repr(exc),'audited_calls':len(cells),
            'no_retry':True,'resources':resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    audit()
