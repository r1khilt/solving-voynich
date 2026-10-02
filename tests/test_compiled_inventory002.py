"""Full collected-panel transport, failure counters and isolated runtime wiring."""

import json

import pytest

from scripts import audit_compiled_inventory002 as audit
from scripts import run_compiled_inventory002 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('cap',[None,'compiler','native','collection'])
def test_full_eight_cells_audit_and_duplicate_refusal(environment,tmp_path,monkeypatch,cap):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    parent = {'source':{'fixture':True},'source_arrays':array_identity(model)}
    queries = [{'case':i,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23} for i in range(4)]
    out = tmp_path/'results/COMPILED-INVENTORY-002'
    monkeypatch.setattr(run,'OUT',out)
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/COMPILED-INVENTORY-002')
    monkeypatch.setattr(run,'inputs',lambda _:(parent,{'build':build},queries))
    for module in (run.original,run.auditor):
        monkeypatch.setattr(module,'load_source',lambda:(model,{'counts':parent['source']}))
        monkeypatch.setattr(module,'limit_resources',lambda *args:None)
        monkeypatch.setattr(module,'guard',lambda *args:None)
    if cap in ('compiler','collection'):
        setting = {'max_nodes':2} if cap=='compiler' else {'max_collection_work':1}
        monkeypatch.setattr(run,'LIMITS',{**run.LIMITS,**setting})
    if cap=='native':
        class CapBridge(StrictCensoredNativeBridge):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**{**kwargs,'max_nodes_per_record':1})
        monkeypatch.setattr(run.original,'StrictCensoredNativeBridge',CapBridge)
    original_class,original_out = run.original.InventoryBDD,run.original.OUT
    run.run('tiny-frozen')
    audit.audit()
    assert run.original.InventoryBDD is original_class and run.original.OUT==original_out
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert result['experiment']==run.EXP and verified['calls']==8
    assert result['summary']['engineering_gate']==('FAIL' if cap else 'PASS')
    assert verified['independent_supported_keys']==(0 if cap in ('compiler','collection') else 256)
    assert verified['independent_first_key_source_checks']==(0 if cap else 8)
    for spec in result['workloads']:
        stats = run.original.load_bound(spec)['summary']['stats']
        assert 'collection_work' in stats and stats['maximum_live_nodes']<=150_000
        if cap=='compiler':
            assert stats['maximum_live_nodes']==stats['nodes']==2
        if not cap:
            assert stats['collections']>=3
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_namespace_restores_bindings_after_exception():
    before = run.original.OUT,run.original.InventoryBDD,run.original.inputs,run.auditor.OUT
    with pytest.raises(RuntimeError,match='fixture'):
        with run.namespace():
            assert run.original.EXP==run.EXP
            raise RuntimeError('fixture')
    assert before==(run.original.OUT,run.original.InventoryBDD,run.original.inputs,run.auditor.OUT)
