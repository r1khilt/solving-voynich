"""Fixed ablation, real admission, closed prediction boundary and full tiny audit."""
import inspect
import json

import numpy as np
import pytest

from scripts.audit_source_bellman001 import compare_replay,replay_config
from scripts.run_source_bellman001 import ARMS,base_output,config,summarize
from scripts.run_source_state_systems001 import config as old_config,predict
from voynich.source_bellman_guide import NativeBellmanGuide


def test_fixed_one_step_early_only_ablation_and_no_answer_inputs():
    a,b=config("iid"),config("bellman_one")
    assert {**a,"lookahead_depth":1}==b
    assert a=={**old_config("wide_guided"),"max_seconds":600.,"lookahead_depth":0,"lookahead_layers":16}
    assert "generation_key" not in inspect.signature(NativeBellmanGuide.search).parameters
    assert "fixture" not in inspect.signature(predict).parameters
    with pytest.raises(ValueError):
        config("retune")


def test_timed_prefix_replay_work_and_output_metadata():
    stored={"stop_reason":"time_cap","expanded":123,"lookahead_actions":456}
    cfg=replay_config(stored,"bellman_one")
    assert cfg["max_expanded"]==123 and cfg["max_seconds"]==1200 and cfg["lookahead_depth"]==1
    compare_replay(stored,{**stored,"stop_reason":"expansion_cap"})
    with pytest.raises(ValueError):
        compare_replay(stored,{**stored,"lookahead_actions":457,"stop_reason":"expansion_cap"})
    with pytest.raises(ValueError):
        replay_config({**stored,"expanded":0},"iid")
    assert base_output({"lookahead_calls":7,"lookahead_depth":1,"expanded":23})=={"expanded":23}


def test_missing_readings_cannot_pass_edit_gate():
    rows=[]
    for case in range(4):
        for arm in ARMS:
            rows.append({"case":case,"arm":arm,"cell_wall_seconds":1.,"diagnostic":{
                "edit_errors":100 if arm=="iid" else 50,"true_source_letters":200,"exact_records":0,
                "matched_gold_used_rows":1,"gold_used_rows":2,"gold_used_mapping_in_returned_terminals":False},
                "summary":{"guide_tables_built":3,"lookahead_calls":0,"lookahead_actions":0}})
    assert summarize(rows)["exploratory_signal_supported"]
    rows[-1]["diagnostic"]["edit_errors"]=None
    assert not summarize(rows)["exploratory_signal_supported"] and not summarize(rows)["paired_all_readings"]


def test_real_admission_join_and_tamper(tmp_path,monkeypatch):
    import scripts.run_source_bellman001 as runner
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_blind_channel_dev004 as storage
    for module in (runner,artifacts,storage):
        monkeypatch.setattr(module,"ROOT",tmp_path)
    out,parent=tmp_path/"results/new",tmp_path/"results/parent"
    monkeypatch.setattr(runner,"OUT",out)
    monkeypatch.setattr(runner,"PARENT",parent)
    monkeypatch.setattr(runner,"require_frozen",lambda *_:None)
    monkeypatch.setattr(runner,"parent_inputs",lambda _:({},{},[],{}))
    result=runner.save_new(parent/"result.json",{})
    runner.save_new(parent/"audit.json",{"result":result,"status":"PASS_full_native_state_and_prediction_replay"})
    library,cpp=tmp_path/"library",tmp_path/"cpp"
    library.write_bytes(b"library")
    cpp.write_bytes(b"cpp")
    build={"library":runner.artifact(library),"generated_cpp":runner.artifact(cpp)}
    runner.save_new(out/"native-build.json",build)
    assert runner.inputs("tiny")[-1]==build
    cpp.write_bytes(b"changed")
    with pytest.raises(ValueError,match="compiled build changed"):
        runner.inputs("tiny")


def test_complete_actual8_tiny_calls_prediction_closure_and_full_audit(tmp_path,monkeypatch):
    import scripts.run_source_bellman001 as runner
    import scripts.audit_source_bellman001 as auditor
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_blind_channel_dev004 as storage
    from voynich.native_source_state import NativeStateLattice,build_source_state_native
    from voynich.native_suffix_marginal import marginal_python

    class Source:
        alphabet=tuple("abcdefghiklmnopqrstuxyz")
        probabilities=np.full((1,23),1/23,dtype=np.float64)
        transitions=np.zeros((1,23),dtype=np.uint32)
        def state(self,_history):
            return 0
        def row(self,_state):
            return self.probabilities[0]
        def step(self,_state,_row):
            return 0

    class Native:
        def __init__(self,source,_build):
            self.source=source
        def score(self,key,record,rho,**limits):
            return marginal_python(self.source,key,record,rho,**limits)

    for module in (runner,artifacts,storage):
        monkeypatch.setattr(module,"ROOT",tmp_path)
    out,bulk,parent=tmp_path/"results/new",tmp_path/"outputs/new",tmp_path/"results/parent"
    monkeypatch.setattr(runner,"BULK",bulk)
    for module in (runner,auditor):
        monkeypatch.setattr(module,"OUT",out)
        monkeypatch.setattr(module,"PARENT",parent)
        monkeypatch.setattr(module,"limit_resources",lambda *_:None)
        monkeypatch.setattr(module,"load_source",lambda:(Source(),{"counts":{"tiny":True}}))
        monkeypatch.setattr(module,"NativeMarginal",Native)
    runner.prepare()
    build=json.loads((out/"native-build.json").read_text())
    original=NativeStateLattice(Source.probabilities,Source.transitions,build_source_state_native(tmp_path/"oldbuild"))
    cases=[{"case":i,"cipher":[[0],[0]],"generation_source":[[0],[1]],"generation_key":[0]*23} for i in range(4)]
    fixtures=runner.save_new(bulk/"fixtures.json",{"fixtures":cases})
    cells=[]
    for case in range(4):
        for arm in ("unmerged","merged","guided","wide_guided"):
            fitted=runner.save_new(tmp_path/f"outputs/old/{case}-{arm}.json.gz",runner.finite(original.search(cases[case]["cipher"],**old_config(arm))),compressed=True)
            cells.append(runner.save_new(parent/f"{case}-{arm}.json",{"fitted":fitted}))
    closed={"source":{"tiny":True},"source_arrays":runner.array_identity(Source()),"fixtures":fixtures,"workloads":cells}
    spec=runner.save_new(parent/"result.json",closed)
    runner.save_new(parent/"audit.json",{"result":spec})
    for module in (runner,auditor):
        monkeypatch.setattr(module,"inputs",lambda _:({"build":{}},{"fixtures":fixtures},cases,closed,build))
    original_diagnose=runner.diagnose
    def closed_before_gold(fitted,prediction,fixture,source):
        assert len(list(bulk.glob("case*.json.gz")))==len(list(bulk.glob("case*-prediction.json")))>0
        return original_diagnose(fitted,prediction,fixture,source)
    monkeypatch.setattr(runner,"diagnose",closed_before_gold)
    runner.run("tiny")
    auditor.audit()
    result=json.loads((out/"result.json").read_text())
    audit=json.loads((out/"audit.json").read_text())
    assert len(result["workloads"])==audit["workloads"]==8 and audit["status"]=="PASS"
    assert audit["alternate_full_key_record_scores"]==16
    assert result["summary"]["arms"]["bellman_one"]["lookahead_actions"]>0
