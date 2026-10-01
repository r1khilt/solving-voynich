"""Full16-call transport/audit, deliberate cap outcomes and real admission only."""

import json

import pytest

from scripts import audit_contextual_gibbs001 as audit
from scripts import run_contextual_gibbs001 as run
from scripts.run_source_state_systems001 import array_identity,save_new
from tests import test_censored_context001 as fixture_support
from voynich.native_censored_bridge import CensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('force_caps',[False,True])
def test_full16_calls_and_one_audit_on_tiny_source(environment,tmp_path,monkeypatch,force_caps):
    model,build = environment
    for module in ('scripts.run_blind_channel_dev004','scripts.run_latin_source_model001','scripts.run_source_bellman001'):
        monkeypatch.setattr(module+'.ROOT',tmp_path)
    source_id = {'fixture':True}
    parent = {'source':source_id,'source_arrays':array_identity(model)}
    queries = [{'id':f'case{case}-root','case':case,'cipher':[[0,1],[0,1]],
        'offsets':[0,0],'supplied_contexts_ignored_by_iid':[0,0],'partial_key':[-1]*23} for case in range(4)]
    fixtures = [{'generation_source':[[0,1],[0,1]],'generation_key':[0,1]+[2]*21} for _ in range(4)]
    if force_caps:
        class CapBridge(CensoredNativeBridge):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**{**kwargs,'max_nodes_per_record':1})
        monkeypatch.setattr(run,'CensoredNativeBridge',CapBridge)
    out,bulk = tmp_path/'results/CONTEXTUAL-GIBBS-001',tmp_path/'outputs/CONTEXTUAL-GIBBS-001'
    for module in (run,audit):
        monkeypatch.setattr(module,'ROOT',tmp_path)
        monkeypatch.setattr(module,'OUT',out)
        monkeypatch.setattr(module,'inputs',lambda _:(parent,{'build':build},queries,fixtures))
        monkeypatch.setattr(module,'load_source',lambda:(model,{'counts':source_id}))
        monkeypatch.setattr(module,'limit_resources',lambda *args:None)
        monkeypatch.setattr(module,'guard',lambda *args:None)
    monkeypatch.setattr(run,'BULK',bulk)
    save_new(tmp_path/'results/CENSORED-CONTEXT-001/result.json',{'fixture':True})
    save_new(tmp_path/'results/CENSORED-CONTEXT-001/audit.json',{'fixture':True})
    run.run('tiny-frozen')
    audit.audit()
    result = json.loads((out/'result.json').read_text())
    verified = json.loads((out/'audit.json').read_text())
    assert len(result['workloads'])==verified['calls']==16
    assert verified['status']=='PASS_complete_contextual_sampling_replay'
    if force_caps:
        assert all(a['graph_cap']==8 and not a['all_complete'] for a in result['summary']['arms'].values())
        assert verified['independent_final_key_scores']==0
        assert not result['summary']['exploratory_mapping_improvement_supported']
    else:
        assert all(a['all_complete'] for a in result['summary']['arms'].values())
        assert verified['independent_final_key_scores']==512
        assert result['summary']['arms']['gibbs']['tables']==result['summary']['arms']['uniform42']['tables']
    with pytest.raises(FileExistsError):
        run.run('tiny-frozen')


def test_original_qualification_and_root_only_inputs_admitted_without_scoring(monkeypatch):
    for name in ('scripts.run_contextual_gibbs001','scripts.benchmark_censored_context001',
                 'scripts.run_dictionary_smc001','scripts.run_shared_key_guide001',
                 'scripts.benchmark_global_search_systems001'):
        monkeypatch.setattr(name+'.require_frozen',lambda *args:None)
    _,_,queries,fixtures = run.inputs('read-only-admission')
    assert len(queries)==len(fixtures)==4
    assert all(q['kind']=='root' and q['partial_key']==[-1]*23 for q in queries)
    original = run.artifact
    def corrupt(path):
        spec = original(path)
        return {**spec,'sha256':'0'*64} if path.name=='result.json' and path.parent.name=='CENSORED-CONTEXT-001' else spec
    monkeypatch.setattr(run,'artifact',corrupt)
    with pytest.raises(ValueError,match='qualification'):
        run.inputs('read-only-admission')
