"""Whole fixed-panel transport, separate caps, no-gold admission, and tampering."""

import json

import pytest

from scripts import audit_compiled_inventory001 as audit
from scripts import run_compiled_inventory001 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('cap',[None,'compiler','native'])
def test_eight_calls_full_replay_and_duplicate_refusal(environment,tmp_path,monkeypatch,cap):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    source_id = {'fixture':True}
    parent = {'source':source_id,'source_arrays':array_identity(model)}
    queries = [{'case':i,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23} for i in range(4)]
    out = tmp_path/'results/COMPILED-INVENTORY-001'
    for module in (run,audit):
        monkeypatch.setattr(module,'OUT',out)
        monkeypatch.setattr(module,'inputs',lambda _:(parent,{'build':build},queries))
        monkeypatch.setattr(module,'load_source',lambda:(model,{'counts':source_id}))
        monkeypatch.setattr(module,'limit_resources',lambda *args:None)
        monkeypatch.setattr(module,'guard',lambda *args:None)
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/COMPILED-INVENTORY-001')
    if cap=='compiler':
        monkeypatch.setattr(run,'LIMITS',{**run.LIMITS,'max_nodes':2})
    if cap=='native':
        class CapBridge(StrictCensoredNativeBridge):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**{**kwargs,'max_nodes_per_record':1})
        monkeypatch.setattr(run,'StrictCensoredNativeBridge',CapBridge)
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert verified['status']=='PASS_full_compiled_inventory_replay' and verified['calls']==8
    assert result['summary']['engineering_gate']==('FAIL' if cap else 'PASS')
    assert verified['independent_supported_keys']==(0 if cap=='compiler' else 256)
    assert verified['independent_first_key_source_checks']==(0 if cap else 8)
    assert result['summary']['compiler_caps']==(8 if cap=='compiler' else 0)
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_cipher_only_admission_precedes_compilation():
    query = {'case':0,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23,'gold_key':[0]*23}
    with pytest.raises(ValueError,match='cipher-only'):
        run.evaluate(query,'natural',None)


def test_real_admission_opens_only_the_cipher_query_archive(monkeypatch):
    # Closed metadata admission only; no source loading, compilation or scoring.
    monkeypatch.setattr(run,'require_frozen',lambda *args:None)
    original = run.load_bound
    opened = []
    def only_query(spec):
        opened.append(spec['path'])
        assert spec['path']=='results/COORDINATED-DICTIONARY-001/queries.json'
        return original(spec)
    monkeypatch.setattr(run,'load_bound',only_query)
    parent,benchmark,queries = run.inputs('9f6f6a26feeaf39c1cce0af92a1b196b16cbccb6')
    assert parent['source']==benchmark['source'] and [q['case'] for q in queries]==list(range(4))
    assert opened==['results/COORDINATED-DICTIONARY-001/queries.json']


def test_global_cap_is_an_abort(monkeypatch):
    monkeypatch.setattr(run,'resource_report',lambda *_:{'peak_rss_bytes':2*1024**3+1,
        'wall_seconds':1,'cpu_seconds':1})
    with pytest.raises(RuntimeError,match='Global compiler'):
        run.guard(0,0,0)


def test_stored_count_tamper_rejected(environment):
    from voynich.native_suffix_marginal import NativeMarginal
    model,build = environment
    query = {'case':0,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23}
    value = run.evaluate(query,'natural',NativeMarginal(model,build))['value']
    assert audit.verify_graph(value,query)==32
    value['dictionary_count'] = str(int(value['dictionary_count'])+1)
    with pytest.raises(AssertionError):
        audit.verify_graph(value,query)
