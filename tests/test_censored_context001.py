"""Complete 960-cell transport/audit plus cap/provenance controls on toy inputs."""

import itertools
import json
import shutil

import numpy as np
import pytest

from scripts import audit_censored_context001 as audit
from scripts import benchmark_censored_context001 as run
from scripts.run_source_state_systems001 import array_identity,save_new
from tests.test_native_suffix_marginal import source
from voynich.native_censored_bridge import CensoredNativeBridge
from voynich.native_suffix_marginal import build_native


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    if not (shutil.which("clang++") or shutil.which("c++")):
        pytest.skip("Optional local C++ compiler unavailable")
    model = source(order=1,alphabet="abcdefghijklmnopqrstuvw")
    build = build_native(tmp_path_factory.mktemp("censored-transport")/"build")
    return model,build


@pytest.mark.parametrize("force_caps",[False,True])
def test_complete960_cells_60_python_192_closure_and_full_audit(environment,tmp_path,monkeypatch,force_caps):
    model,build = environment
    for module in ("scripts.run_blind_channel_dev004","scripts.run_latin_source_model001",
                   "scripts.run_source_bellman001"):
        monkeypatch.setattr(module+".ROOT",tmp_path)
    source_id = {"fixture":True}
    parent = {"source":source_id,"source_arrays":array_identity(model)}
    selected = []
    for case,kind in itertools.product(range(4),run.KINDS):
        partial = [-1]*23
        if kind!="root":
            partial[0] = 0 if kind=="prefix" else 2
        query = {"id":f"case{case}-{kind}","cipher":[[0,1],[0,1]],
            "offsets":[0,0] if kind=="root" else [1,1],"partial_key":partial,
            "supplied_contexts_ignored_by_iid":[0,0] if kind=="root" else [1,1]}
        keys = np.random.default_rng(81921+case).integers(42,size=(16,23),dtype=np.int32)
        keys[:,0] = 0 if kind!="altered_binding" else 2
        keys[:,1] = 1
        selected.append({"query":query,"keys":keys.tolist(),"selection":"fixture"})
    if force_caps:
        class CapBridge(CensoredNativeBridge):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**{**kwargs,"max_nodes_per_record":1})
        monkeypatch.setattr(run,"CensoredNativeBridge",CapBridge)
    out = tmp_path/"results/CENSORED-CONTEXT-001"
    for module in (run,audit):
        monkeypatch.setattr(module,"ROOT",tmp_path)
        monkeypatch.setattr(module,"OUT",out)
        monkeypatch.setattr(module,"inputs",lambda _: (parent,{"build":build},selected))
        monkeypatch.setattr(module,"load_source",lambda:(model,{"counts":source_id}))
        monkeypatch.setattr(module,"limit_resources",lambda *args:None)
        monkeypatch.setattr(module,"guard",lambda *args:None)
    save_new(tmp_path/"results/DICTIONARY-SMC-001/result.json",{"fixture":True})
    save_new(tmp_path/"results/DICTIONARY-SMC-001/audit.json",{"fixture":True})
    run.run("tiny-frozen")
    audit.audit()
    result = json.loads((out/"result.json").read_text())
    verified = json.loads((out/"audit.json").read_text())
    summary = result["summary"]
    assert summary["native_attempts"]==verified["native_attempts"]==960
    assert summary["python_attempts"]==verified["python_attempts"]==60
    assert summary["closure_attempts"]==verified["legacy_closure_attempts"]==192
    assert summary["contextual_engineering_gate"]==("FAIL" if force_caps else "PASS")
    assert summary["native_complete"]==(0 if force_caps else 960)
    assert verified["status"]=="PASS_complete_contextual_benchmark_replay"
    with pytest.raises(FileExistsError):
        run.run("tiny-frozen")


def test_original_closed_banks_admission_is_fixed_and_parent_tamper_rejected(monkeypatch):
    # Artifact admission only. No source loading, bridge scoring or new sampler.
    for name in ("scripts.benchmark_censored_context001","scripts.run_dictionary_smc001",
                 "scripts.run_shared_key_guide001","scripts.benchmark_global_search_systems001"):
        monkeypatch.setattr(name+".require_frozen",lambda *args:None)
    parent,_,selected = run.inputs("read-only-fixture-admission")
    assert len(selected)==12 and len(parent["workloads"])==144
    for spec in selected:
        assert len(spec["keys"])==16
        assert spec["selection"]=="first16_final_particles_no_contextual_score_selection"
    original = run.artifact
    parent_path = run.ROOT/"results/DICTIONARY-SMC-001/result.json"
    def tamper(path):
        found = original(path)
        return {**found,"sha256":"0"*64} if path==parent_path else found
    monkeypatch.setattr(run,"artifact",tamper)
    with pytest.raises(ValueError,match="provenance"):
        run.inputs("read-only-fixture-admission")


def test_prefix_stage_definitions_and_graph_caps_not_zero_mass():
    assert run.stages((4,1))==(((1,1),False),((4,1),False),((4,1),False),((4,1),False),((4,1),True))
    def cap():
        raise RuntimeError("Exact lattice node cap; no pruning")
    value = run.timed(cap)
    assert value["status"]=="graph_cap" and "log_value" not in value
    def other():
        raise RuntimeError("Campaign native edge budget exhausted")
    with pytest.raises(RuntimeError):
        run.timed(other)
