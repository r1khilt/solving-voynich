"""Full registered transport on tiny stand-ins; alternate DP checks every call."""

import json
import math

import numpy as np
import pytest

from scripts import audit_dictionary_smc001 as audit
from scripts import run_dictionary_smc001 as run
from scripts.run_source_state_systems001 import array_identity, artifact, save_new
from tests.test_dictionary_smc import KEYS, bridge
from voynich.dictionary_smc import bridge_schedule


@pytest.mark.parametrize("cipher",[((0,0),(1,0)),((0,1,0),(0,0))])
def test_alternate_outgoing_direction_matches_every_tiny_key_stage(cipher):
    b = bridge(cipher,iid=True)
    for cuts,closed in bridge_schedule(b.lengths):
        expected = audit.forward_values(b,KEYS,cuts,closed)
        actual = b.log_values(KEYS,cuts,closed=closed)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)


def test_complete144_tiny_run_and_all_trajectory_audit(tmp_path,monkeypatch):
    out,bulk = tmp_path/"results",tmp_path/"bulk"
    for module in ("scripts.run_blind_channel_dev004","scripts.run_latin_source_model001",
                   "scripts.run_source_bellman001"):
        monkeypatch.setattr(module+".ROOT",tmp_path)
    for module in (run,audit):
        monkeypatch.setattr(module,"OUT",out)
        monkeypatch.setattr(module,"BULK",bulk)
        monkeypatch.setattr(module,"ROOT",tmp_path)
        monkeypatch.setattr(module,"limit_resources",lambda *args:None)
        monkeypatch.setattr(module,"guard",lambda *args:None)
    p = np.array([[1/23]*23],dtype=np.float64)
    class Source:
        probabilities = p
        transitions = np.zeros((1,23),dtype=np.uint32)
    source = Source()
    parent = {"source":{"stand_in":True},"source_arrays":array_identity(source)}
    declared = []
    for case in range(4):
        for kind in ("root","prefix","altered_binding"):
            key = [-1]*23
            if kind!="root":
                key[0] = 0 if kind=="prefix" else 1
            declared.append({"id":f"case{case}-{kind}","cipher":[[0,1],[1,0]],
                "offsets":[0,0],"partial_key":key})
    closed = {"queries":declared}
    for module in (run,audit):
        monkeypatch.setattr(module,"inputs",lambda _: (parent,closed))
        monkeypatch.setattr(module,"load_source",lambda:(source,{"counts":parent["source"]}))
    save_new(tmp_path/"results/SHARED-KEY-GUIDE-001/result.json",closed)
    save_new(tmp_path/"results/SHARED-KEY-GUIDE-001/audit.json",{"fixture":True})
    run.run("tiny-frozen")
    result = json.loads((out/"result.json").read_text())
    assert len(result["workloads"])==144
    audit.audit()
    checked = json.loads((out/"audit.json").read_text())
    assert checked["calls"]==144 and checked["status"]=="PASS"
    assert checked["alternate_outgoing_fixed_dictionary_checks"]==sum(v["tables"] for v in result["summary"]["arms"].values())
    assert checked["maximum_alternate_log_delta"]<1e-12
    for spec in result["workloads"]:
        row = json.loads((tmp_path/spec["path"]).read_text())
        assert row["summary"]["tables"]>0
    assert result["parent_result"]==artifact(tmp_path/"results/SHARED-KEY-GUIDE-001/result.json")


def test_registered_arm_admission_and_conditional_prior_telescoping():
    query = {"cipher":[[0],[0]],"offsets":[0,0],"partial_key":[0]*23}
    p = np.array([1/23]*23)
    expected = 2*math.log((1-1/225)/225)
    for arm in run.ARMS:
        value = run.evaluate(query,p,81401,arm)
        assert value["status"]=="complete_particles"
        assert value["log_evidence_estimate"]==pytest.approx(expected,abs=1e-13)
        assert run.compact(value)["proposals"]==0
    with pytest.raises(ValueError):
        run.evaluate(query,p,81401,"unknown")


def test_actual_closed_query_admission_rejects_parent_result_tamper(monkeypatch):
    # Read-only admission: no real source loading, scoring, sampler, or model call.
    parent_path = run.ROOT/"results/SHARED-KEY-GUIDE-001/result.json"
    original_artifact = run.artifact
    monkeypatch.setattr(run,"require_frozen",lambda *args:None)
    monkeypatch.setattr("scripts.run_shared_key_guide001.require_frozen",lambda *args:None)
    _,closed = run.inputs("admission-fixture-only")
    assert len(closed["queries"])==12
    def tampered(path):
        value = original_artifact(path)
        return {**value,"sha256":"0"*64} if path==parent_path else value
    monkeypatch.setattr(run,"artifact",tampered)
    with pytest.raises(ValueError,match="Unchanged closed"):
        run.inputs("admission-fixture-only")
