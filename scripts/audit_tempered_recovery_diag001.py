"""One bounded all-case known-answer replay; parent audit stays separate."""

import json
import signal
import time

from scripts import run_tempered_recovery_diag001 as run
from scripts.audit_censored_context001 import logical
from scripts.run_shared_key_guide001 import load_bound
from voynich.native_suffix_marginal import NativeMarginal


def audit():
    result = json.loads((run.OUT/'result.json').read_text())
    previous,benchmark,cells,gold,_ = run.inputs(result['freeze'])
    assert result['experiment']==run.EXP
    assert result['parent_result']==run.parent.artifact(run.parent.OUT/'result.json')
    assert result['parent_prediction_seal']==previous['prediction_seal']
    assert result['native_build']==benchmark['build']
    run.parent.save_new(run.OUT/'audit-started.json',{'freeze':result['freeze'],'start_unix':time.time()})
    run.parent.limit_resources(run.WALL,run.CPU)
    wall,cpu = time.monotonic(),time.process_time()
    try:
        source,chosen = run.parent.load_source()
        identity = run.parent.array_identity(source)
        assert chosen['counts']==result['source']==previous['source']
        assert identity==result['source_arrays']==previous['source_arrays']
        stored = load_bound(result['analysis'])
        actual = run.analyze(source,NativeMarginal(source,benchmark['build']),cells,gold,lambda:run.guard(wall,cpu))
        assert logical(actual)==logical(stored)
        assert result['summary']==actual['summary'] and result['banks']==actual['banks']
        assert result['oracles']==[{k:v for k,v in o.items() if k not in ('prediction','used_rows')} for o in actual['oracles']]
        assert run.parent.array_identity(source)==identity and result['analysis']['bytes']<=8*1024**2
        for r in (result['resources'],run.parent.resource_report(wall,cpu)):
            assert r['wall_seconds']<=run.WALL and r['cpu_seconds']<=run.CPU and r['peak_rss_bytes']<=2*1024**3
        run.parent.save_new(run.OUT/'audit.json',{'status':'PASS_full_known_answer_diagnostic_replay',
            'result':run.parent.artifact(run.OUT/'result.json'),'generating_key_cases':4,'positive_bank_cases':32,
            'independent_known_key_source_checks':actual['summary']['independent_known_key_source_checks'],
            'maximum_known_key_source_log_delta':actual['summary']['maximum_known_key_source_log_delta'],
            'same_author':True,'replay_shares_native_backend':True,'independent_agent_review':False,
            'parent_full_replay_separately_required_for_final_publication':True,
            'resources':run.parent.resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        run.parent.save_new(run.OUT/'audit-failure.json',{'error':repr(exc),'no_retry':True,
            'resources':run.parent.resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    audit()
