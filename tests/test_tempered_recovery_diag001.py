"""Native tiny oracle transport, independent reference and complete replay."""

import json

import numpy as np
import pytest

from scripts import audit_tempered_recovery_diag001 as audit
from scripts import run_tempered_recovery_diag001 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support


environment = fixture_support.environment


def test_actual_tiny_four_oracles_all32_banks_and_complete_audit(environment,tmp_path,monkeypatch):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001','scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    monkeypatch.setattr(run,'OUT',tmp_path/'results/TEMPERED-RECOVERY-DIAG-001')
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/TEMPERED-RECOVERY-DIAG-001')
    monkeypatch.setattr(run.parent,'OUT',tmp_path/'results/TEMPERED-RECOVERY-001')
    monkeypatch.setattr(run.parent,'GENERATION_SEEDS',(11,12,13,14))
    monkeypatch.setattr(run.parent,'SHUFFLE_SEEDS',(21,22,23,24))
    monkeypatch.setattr(run.parent,'DESIGNS',(2,2,2,2))
    monkeypatch.setattr(run.parent,'PARTICLES',4)
    gold = run.parent.fixtures(model)
    native = run.NativeMarginal(model,build)
    cells = []
    for fixture in gold:
        if fixture['case']%2:
            continue
        key = fixture['generation_key']
        reference = run.FixedKeyBridge(fixture['cipher'],native.probabilities,native.transitions,
            max_tables=1,max_edges=800_000_000,max_states_per_table=300_000,max_work_bytes=768*1024**2)
        likelihood = float(reference.log_values(np.array([key]),reference.lengths,closed=True)[0])
        for seed in run.parent.SEEDS:
            for arm in run.parent.ARMS:
                output = run.parent.save_new(tmp_path/f"bulk-{fixture['case']}-{seed}-{arm}.json.gz",
                    {'value':{'status':'complete_tempered_particles','keys':[key]*4,'key_log_likelihoods':[likelihood]*4},
                     'prediction':{'selected_index':0}},compressed=True)
                cells.append({'case':fixture['case'],'seed':seed,'arm':arm,'output':output,
                    'recovery':{'selected_matches':len({r for t in fixture['generation_source'] for r in t})}})
    previous = {'source':{'fixture':True},'source_arrays':array_identity(model),'prediction_seal':{'fixture':True}}
    run.parent.save_new(run.parent.OUT/'result.json',previous)
    monkeypatch.setattr(run,'inputs',lambda _:(previous,{'build':build},cells,gold,None))
    monkeypatch.setattr(run.parent,'load_source',lambda:(model,{'counts':previous['source']}))
    monkeypatch.setattr(run.parent,'limit_resources',lambda *args:None)
    monkeypatch.setattr(run,'guard',lambda *args:None)
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((run.OUT/'result.json').read_text())
    verified = json.loads((run.OUT/'audit.json').read_text())
    assert verified['status']=='PASS_full_known_answer_diagnostic_replay'
    assert result['summary']['complete_parent_banks']==32
    assert result['summary']['banks_without_compatible_inventory']==0
    assert result['summary']['independent_known_key_source_checks']==4
    assert all(b['bounds']['distinct_full_keys']==1 for b in result['banks'])
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_failed_parent_bank_never_becomes_a_partial_coverage_result():
    result = run.bank_diagnostic({'case':0,'seed':1,'arm':'row'},
        {'value':{'status':'call_work_cap'}},None)
    assert result['status']=='failed_parent_call' and 'bounds' not in result


def test_oracle_reencoding_mismatch_refused_before_source_score(environment):
    model,build = environment
    with pytest.raises(AssertionError):
        run.oracle(model,run.NativeMarginal(model,build),
            {'generation_key':[0]*23,'generation_source':[[0]],'cipher':[[1]]})


def test_oracle_native_cap_preserves_point_bound_but_no_partial_marginal(environment):
    model,build = environment
    original = run.NativeMarginal(model,build)
    class Capped:
        def score(self,*args,**kwargs):
            return original.score(*args,**{**kwargs,'max_nodes':1})
    fixture = {'case':0,'kind':'positive','generation_key':[0]*23,
        'generation_source':[[0],[1]],'cipher':[[0],[0]]}
    value = run.oracle(model,Capped(),fixture)
    assert value['status']=='oracle_native_graph_cap'
    assert value['known_key_log_likelihood'] is None and value['independent_source_log_delta'] is None
    assert value['known_source_joint_log_probability']<0
    assert value['recovery']['edit_errors_with_failed_readings_counted_full_length']==2


def test_independent_reference_failure_is_not_a_partial_diagnostic(environment,monkeypatch):
    model,build = environment
    def fail(*args,**kwargs):
        raise MemoryError('Reference work envelope cap')
    monkeypatch.setattr(run,'FixedKeyBridge',fail)
    fixture = {'case':0,'kind':'positive','generation_key':[0]*23,
        'generation_source':[[0],[1]],'cipher':[[0],[0]]}
    with pytest.raises(MemoryError,match='Reference'):
        run.oracle(model,run.NativeMarginal(model,build),fixture)
