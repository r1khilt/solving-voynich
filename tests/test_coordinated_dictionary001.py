"""Full fresh-panel transport, fit/gold boundary, and distinct stop semantics."""

import json

import pytest

from scripts import audit_coordinated_dictionary001 as audit
from scripts import run_coordinated_dictionary001 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('cap',[None,'graph','work'])
def test_full24_calls_one_full_audit_and_no_retries(environment,tmp_path,monkeypatch,cap):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    source_id = {'fixture':True}
    parent = {'source':source_id,'source_arrays':array_identity(model)}
    out,bulk = tmp_path/'results/COORDINATED-DICTIONARY-001',tmp_path/'outputs/COORDINATED-DICTIONARY-001'
    if cap:
        class CapBridge(StrictCensoredNativeBridge):
            def __init__(self,*args,**kwargs):
                setting = {'max_nodes_per_record':1} if cap=='graph' else {'max_edges':1}
                super().__init__(*args,**{**kwargs,**setting})
        monkeypatch.setattr(run,'StrictCensoredNativeBridge',CapBridge)
    for module in (run,audit):
        monkeypatch.setattr(module,'OUT',out)
        monkeypatch.setattr(module,'inputs',lambda _:(parent,{'build':build}))
        monkeypatch.setattr(module,'load_source',lambda:(model,{'counts':source_id}))
        monkeypatch.setattr(module,'limit_resources',lambda *args:None)
        monkeypatch.setattr(module,'guard',lambda *args:None)
    monkeypatch.setattr(run,'DESIGNS',(2,2,2,2))
    monkeypatch.setattr(run,'BULK',bulk)
    # This trips if gold evaluation happens before every output is sealed.
    original = run.recovery
    def only_after_seal(*args):
        assert (out/'predictions-sealed.json').exists()
        assert len(json.loads((out/'predictions-sealed.json').read_text())['outputs'])==24
        return original(*args)
    monkeypatch.setattr(run,'recovery',only_after_seal)
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert len(result['workloads'])==verified['calls']==24
    assert verified['status']=='PASS_full_coordinated_panel_replay'
    if cap:
        assert all(a[cap+'_cap']==8 for a in result['summary']['arms'].values())
        assert verified['independent_final_key_scores']==0
        assert not result['summary']['exploratory_coordinated_recovery_supported']
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')


def test_cipher_only_query_drops_every_generator_answer():
    fixture = {'case':3,'cipher':[[0,1],[2,3]],'generation_key':[1]*23,
        'generation_source':[[7,8],[3,4]],'generation_length':2,'generation_seed':99}
    assert run.query_for(fixture)=={'case':3,'cipher':fixture['cipher'],'partial_key':[-1]*23}
    with pytest.raises(ValueError,match='cipher-only'):
        run.evaluate(fixture,None,run.SEEDS[0],run.ARMS[0])


def test_global_caps_abort_instead_of_allowed_per_call_work_outcome(monkeypatch):
    monkeypatch.setattr(run,'resource_report',lambda *_:{'peak_rss_bytes':1,'wall_seconds':1,'cpu_seconds':1})
    with pytest.raises(RuntimeError,match='Global campaign'):
        run.guard(0,0,0,[{'summary':{'edges':run.CAMPAIGN_EDGES+1}}])
