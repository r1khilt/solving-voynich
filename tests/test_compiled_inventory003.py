"""Nested namespace, actual tiny full-panel transport, and work-limit identity."""

import json

import pytest

from scripts import audit_compiled_inventory003 as audit
from scripts import run_compiled_inventory003 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support


environment = fixture_support.environment


def test_full_eight_calls_and_audit_under_new_namespace(environment,tmp_path,monkeypatch):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    parent = {'source':{'fixture':True},'source_arrays':array_identity(model)}
    queries = [{'case':i,'cipher':[[0,1],[1,0]],'partial_key':[-1]*23} for i in range(4)]
    out = tmp_path/'results/COMPILED-INVENTORY-003'
    monkeypatch.setattr(run,'OUT',out)
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/COMPILED-INVENTORY-003')
    monkeypatch.setattr(run,'inputs',lambda _:(parent,{'build':build},queries))
    for module in (run.previous.original,run.previous.auditor):
        monkeypatch.setattr(module,'load_source',lambda:(model,{'counts':parent['source']}))
        monkeypatch.setattr(module,'limit_resources',lambda *args:None)
        monkeypatch.setattr(module,'guard',lambda *args:None)
    base_limits = run.previous.original.LIMITS.copy()
    run.run('tiny-frozen')
    audit.audit()
    assert run.previous.original.LIMITS==base_limits
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert result['experiment']==run.EXP and result['summary']['engineering_gate']=='PASS'
    assert verified['calls']==8 and verified['independent_supported_keys']==256
    assert verified['independent_first_key_source_checks']==8
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_only_apply_budget_changes_and_nested_restoration():
    previous = run.previous
    before = previous.EXP,previous.OUT,previous.LIMITS,previous.original.LIMITS,previous.inputs
    assert set(run.LIMITS)==set(previous.LIMITS)
    assert {k for k in run.LIMITS if run.LIMITS[k]!=previous.LIMITS[k]}=={'max_apply_steps'}
    with pytest.raises(RuntimeError,match='fixture'):
        with run.namespace():
            with previous.namespace():
                assert previous.original.EXP==run.EXP
                assert previous.original.LIMITS['max_apply_steps']==12_000_000
                raise RuntimeError('fixture')
    assert before==(previous.EXP,previous.OUT,previous.LIMITS,previous.original.LIMITS,previous.inputs)
