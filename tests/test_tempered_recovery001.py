"""Tiny actual64-call sealed recovery/null panel and entire same-author audit."""

import json

import pytest

from scripts import audit_tempered_recovery001 as audit
from scripts import run_tempered_recovery001 as run
from scripts.run_source_state_systems001 import array_identity
from tests import test_censored_context001 as fixture_support
from voynich.strict_censored_bridge import StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('cap',[None,'compiler','graph','work'])
def test_actual64calls_sealed_before_gold_and_full_audit(environment,tmp_path,monkeypatch,cap):
    model,build = environment
    registered_generation = set(run.GENERATION_SEEDS)
    original_fixtures = run.fixtures
    def tiny_fixtures(source):
        assert registered_generation.isdisjoint(run.GENERATION_SEEDS)
        return original_fixtures(source)
    monkeypatch.setattr(run,'fixtures',tiny_fixtures)
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001',
                   'scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    parent = {'source':{'fixture':True},'source_arrays':array_identity(model)}
    out = tmp_path/'results/TEMPERED-RECOVERY-001'
    monkeypatch.setattr(run,'OUT',out)
    monkeypatch.setattr(run,'BULK',tmp_path/'outputs/TEMPERED-RECOVERY-001')
    monkeypatch.setattr(run,'inputs',lambda _:(parent,{'build':build}))
    monkeypatch.setattr(run,'load_source',lambda:(model,{'counts':parent['source']}))
    monkeypatch.setattr(run,'limit_resources',lambda *args:None)
    monkeypatch.setattr(run,'guard',lambda *args,**kwargs:None)
    monkeypatch.setattr(run,'DESIGNS',(2,2,2,2))
    monkeypatch.setattr(run,'GENERATION_SEEDS',(11,12,13,14))
    monkeypatch.setattr(run,'SHUFFLE_SEEDS',(21,22,23,24))
    monkeypatch.setattr(run,'PARTICLES',4)
    monkeypatch.setattr(run,'BETAS',(0,.5,1))
    monkeypatch.setattr(run,'MUTATIONS',1)
    if cap=='compiler':
        monkeypatch.setattr(run.engineering.compiler,'LIMITS',{**run.engineering.compiler.LIMITS,'max_apply_steps':1})
    if cap in ('graph','work'):
        def bridge(*args,**kwargs):
            kwargs.update({'max_nodes_per_record':1} if cap=='graph' else {'max_edges':1})
            return StrictCensoredNativeBridge(*args,**kwargs)
        monkeypatch.setattr(run,'StrictCensoredNativeBridge',bridge)
    original = run.recovery
    def only_after_seal(*args):
        seal = json.loads((out/'predictions-sealed.json').read_text())
        assert len(seal['outputs'])==64
        return original(*args)
    monkeypatch.setattr(run,'recovery',only_after_seal)
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert verified['calls']==len(result['workloads'])==64
    assert verified['independent_fixed_final_key_source_checks']==(256 if cap is None else 0)
    for spec in result['workloads']:
        cell = audit.load_bound(spec)
        assert cell['recovery']['kind']==('positive' if cell['case']%2==0 else 'shuffle')
        if cap:
            assert cell['summary']['status']=={'compiler':'compiler_cap','graph':'native_graph_cap','work':'call_work_cap'}[cap]
    if cap:
        assert not result['summary']['combined_qualification_gate']
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_all_generator_answers_and_null_labels_drop_at_query_boundary():
    fixture = {'case':7,'cipher':[[0],[1]],'kind':'shuffle','generation_key':[0]*23,
        'generation_source':[[1],[2]],'generation_length':1,'generation_seed':99}
    assert run.query_for(fixture)=={'case':7,'cipher':fixture['cipher'],'partial_key':[-1]*23}
    with pytest.raises(ValueError,match='ciphertext-only'):
        run.compile_profile(fixture)
    with pytest.raises(ValueError,match='ciphertext-only'):
        run.predict(fixture,fixture,None,None)


def test_shuffles_preserve_every_record_length_and_unigram(environment,monkeypatch):
    model,_ = environment
    monkeypatch.setattr(run,'DESIGNS',(2,2,2,2))
    monkeypatch.setattr(run,'GENERATION_SEEDS',(11,12,13,14))
    monkeypatch.setattr(run,'SHUFFLE_SEEDS',(21,22,23,24))
    gold = run.fixtures(model)
    assert [f['case'] for f in gold]==list(range(8))
    for i in range(4):
        assert gold[2*i]['kind']=='positive' and gold[2*i+1]['kind']=='shuffle'
        for a,b in zip(gold[2*i]['cipher'],gold[2*i+1]['cipher'],strict=True):
            assert len(a)==len(b) and sorted(a)==sorted(b)


def test_normalizer_spread_and_failed_call_denominators_are_not_hidden():
    cells = []
    for case in range(8):
        for seed in run.SEEDS:
            for arm in run.ARMS:
                recovered = {'kind':'shuffle','accuracy_not_defined':True} if case%2 else {
                    'kind':'positive','selected_matches':None,'used_rows':10,
                    'selected_complete_used_key':False,'bank_contains_complete_used_key':False,
                    'edit_errors_with_failed_readings_counted_full_length':128,'true_letters':128,'exact_records':0}
                cells.append({'case':case,'seed':seed,'arm':arm,'recovery':recovered,
                    'summary':{'status':'compiler_cap','edges':0}})
    summary = run.summarize(cells)
    assert not summary['combined_qualification_gate']
    for arm in summary['arms'].values():
        assert arm['used_row_denominator']==80 and arm['true_letters']==arm['edit_errors']==1024
        assert arm['selected_used_row_matches']==0 and not arm['calibration_gate'] and not arm['positive_shuffle_gate']
        assert arm['seed_log_evidence_spreads']==[None]*8


def test_global_resource_stop_is_distinct_from_allowed_call_work_cap(monkeypatch):
    monkeypatch.setattr(run,'resource_report',lambda *_:{'wall_seconds':1,'cpu_seconds':1,'peak_rss_bytes':1})
    with pytest.raises(RuntimeError,match='Global fresh'):
        run.guard(0,0,0,[],run.GLOBAL_EDGES+1)
    assert run.PARTICLES*(1+(len(run.BETAS)-1)*run.MUTATIONS)==32896
    assert len(run.ARMS)*len(run.SEEDS)*8==64
