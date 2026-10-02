"""Actual tiny eight-call transport/replay, allowed caps and cipher boundary."""

import json

import pytest

from scripts import audit_tempered_inventory001 as audit
from scripts import run_tempered_inventory001 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('forced',[None,'compiler','native','work'])
def test_actual_full_panel_and_audit_with_preserved_failures(environment,tmp_path,monkeypatch,forced):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    parent = {'source':{'fixture':True},'source_arrays':array_identity(model)}
    queries = [{'case':i,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23} for i in range(4)]
    out = tmp_path/'results/TEMPERED-INVENTORY-001'
    monkeypatch.setattr(run,'OUT',out)
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/TEMPERED-INVENTORY-001')
    monkeypatch.setattr(run,'inputs',lambda _:(parent,{'build':build},queries))
    monkeypatch.setattr(run,'load_source',lambda:(model,{'counts':parent['source']}))
    monkeypatch.setattr(run,'limit_resources',lambda *args:None)
    monkeypatch.setattr(run,'guard',lambda *args:None)
    monkeypatch.setattr(run,'PARTICLES',4)
    monkeypatch.setattr(run,'BETAS',(0,.5,1))
    monkeypatch.setattr(run,'MUTATIONS',1)
    if forced=='compiler':
        monkeypatch.setattr(run.compiler,'LIMITS',{**run.compiler.LIMITS,'max_apply_steps':1})
    if forced in ('native','work'):
        def bridge(*args,**kwargs):
            kwargs.update({'max_nodes_per_record':1} if forced=='native' else {'max_edges':1})
            return StrictCensoredNativeBridge(*args,**kwargs)
        monkeypatch.setattr(run,'StrictCensoredNativeBridge',bridge)
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert result['summary']['engineering_gate']==('PASS' if forced is None else 'FAIL')
    assert verified['calls']==8 and verified['independent_final_key_source_checks']==(32 if forced is None else 0)
    for spec in result['workloads']:
        cell = audit.load_bound(spec)
        value = audit.load_bound(cell['output'])
        assert value['status']=={None:'complete_tempered_particles','compiler':'compiler_cap',
            'native':'native_graph_cap','work':'call_work_cap'}[forced]
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_gold_containing_query_rejected_before_compilation():
    with pytest.raises(ValueError,match='ciphertext-only'):
        run.compile_profile({'case':0,'cipher':[[0],[1]],'partial_key':[-1]*23,'gold':'no'})


def test_fixed_table_ceiling_and_cube_schedule():
    assert run.MAX_TABLES==run.PARTICLES*(1+(len(run.BETAS)-1)*run.MUTATIONS)==2080
    assert run.BETAS==tuple((i/16)**3 for i in range(17))
    assert run.GLOBAL_EDGES>=8*(run.MAX_EDGES+1)


def test_unexpected_scorer_failure_aborts_entire_campaign(environment,tmp_path,monkeypatch):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    parent = {'source':{'fixture':True},'source_arrays':array_identity(model)}
    queries = [{'case':0,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23}]
    out = tmp_path/'results/TEMPERED-INVENTORY-001'
    monkeypatch.setattr(run,'OUT',out)
    monkeypatch.setattr(run,'inputs',lambda _:(parent,{'build':build},queries))
    monkeypatch.setattr(run,'load_source',lambda:(model,{'counts':parent['source']}))
    monkeypatch.setattr(run,'limit_resources',lambda *args:None)
    def fail(*args,**kwargs):
        raise RuntimeError('Exact integer sampler rejection budget exhausted')
    monkeypatch.setattr(run,'run_tempered_inventory_smc',fail)
    with pytest.raises(RuntimeError,match='integer sampler'):
        run.run('tiny-frozen')
    failure = json.loads((out/'failure.json').read_text())
    assert failure['closed_calls']==0 and failure['no_retry']
    assert not (out/'result.json').exists()
