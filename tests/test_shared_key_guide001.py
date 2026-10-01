"""Real tiny transport/replay and unambiguous zero/stability decision checks."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.audit_shared_key_guide001 as audit_module
import scripts.run_shared_key_guide001 as runner


def fixture_bundle():
    fixtures,cells,archives = [],[],[]
    for case in range(4):
        fixtures.append({"case":case,"cipher":[[0,1],[0,1]]})
        state = [1,1,0,0,0]+[-1]*22+[0,1,1]
        trace = [{"target_glyph_layer":2}]
        cells.append({"summary":{"first_loss":{"target_glyph_layer":2}}})
        archives.append({"trace":trace,"gold_states":[state]})
    return fixtures,cells,archives


def test_complete144_call_tiny_transport_and_all_forward_replays(tmp_path,monkeypatch):
    fixtures,cells,archives = fixture_bundle()
    source = SimpleNamespace(probabilities=np.full((1,23),1/23),transitions=np.zeros((1,23),dtype=np.uint32))
    counts = {"path":"source","bytes":1,"sha256":"synthetic"}
    identity = {"probabilities":"synthetic-p","transitions":"synthetic-goto"}
    parent = {"source":counts,"source_arrays":identity,"fixtures":{"synthetic":True}}
    out = tmp_path/"results/SHARED-KEY-GUIDE-001"
    def artifact(path):
        path = path.resolve()
        if path.exists():
            raw = path.read_bytes()
            return {"path":str(path.relative_to(tmp_path)),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        return {"path":str(path.relative_to(tmp_path)),"bytes":0,"sha256":"stub-parent"}
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x") as f:
            json.dump(value,f,allow_nan=False)
    for mod in (runner,audit_module):
        monkeypatch.setattr(mod,"ROOT",tmp_path)
        monkeypatch.setattr(mod,"OUT",out)
        monkeypatch.setattr(mod,"inputs",lambda freeze:(parent,fixtures,cells,archives))
        monkeypatch.setattr(mod,"load_source",lambda:(source,{"counts":counts}))
        monkeypatch.setattr(mod,"array_identity",lambda source:identity)
        monkeypatch.setattr(mod,"artifact",artifact)
        monkeypatch.setattr(mod,"save_new",save)
        monkeypatch.setattr(mod,"limit_resources",lambda *args:None)
        monkeypatch.setattr(mod,"load_bound",lambda spec:json.loads((tmp_path/spec["path"]).read_text()))
    monkeypatch.setattr(runner,"PRUNE",tmp_path/"results/SOURCE-PRUNE-DIAG-002")
    runner.run("tiny-frozen")
    audit_module.audit()
    result = json.loads((out/"result.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    assert len(result["workloads"])==audit["calls"]==144
    assert audit["alternate_forward_fixed_dictionary_checks"]==65280
    assert audit["maximum_alternate_log_delta"]<=1e-7
    assert audit["result"]==artifact(out/"result.json")
    with pytest.raises(FileExistsError):
        runner.run("tiny-frozen")


def test_queries_are_closed_prefix_plus_stated_hypothetical_binding():
    q = runner.queries(*fixture_bundle())
    assert len(q)==12
    for i in range(4):
        root,prefix,wrong = q[3*i:3*i+3]
        assert root["partial_key"]==[-1]*23
        assert prefix["offsets"]==wrong["offsets"]==[1,1]
        assert sum(a!=b for a,b in zip(prefix["partial_key"],wrong["partial_key"],strict=True))==1


def test_summary_never_counts_zero_estimate_as_stability_or_retunes_threshold():
    rows = []
    for query in runner.queries(*fixture_bundle()):
        for seed in runner.SEEDS:
            for arm in runner.ARMS:
                rows.append({"query":query,"seed":seed,"arm":arm,"wall_seconds":.001,
                    "estimate":{"log_mean":-3. if query["kind"]=="altered_binding" else -2.,
                        "fixed_dictionary_tables":16 if arm=="mc16" else 672,
                        "maximum_base_contribution_share":.25,"likelihood_contribution_ess":4.}})
    assert all(v["exploratory_calibration_supported"] for v in runner.summarize(rows)["arms"].values())
    broken = copy.deepcopy(rows)
    broken[0]["estimate"]["log_mean"] = None
    broken[1]["estimate"]["maximum_base_contribution_share"] = .51
    broken[2]["estimate"]["log_mean"] = -5.
    assert not any(v["exploratory_calibration_supported"] for v in runner.summarize(broken)["arms"].values())


def test_alternate_checks_literal_zero_support():
    query = {"cipher":[[0,1,0]],"offsets":[0],"partial_key":[0]*23}
    row = np.full(23,1/23)
    result = runner.evaluate(query,row,81401,"mc16")
    assert result["log_mean"] is None


def test_actual_input_admission_binds_archived_queries_and_rejects_tampering(tmp_path,monkeypatch):
    fixtures,cells,archives = fixture_bundle()
    monkeypatch.setattr(runner,"PRUNE",tmp_path)
    monkeypatch.setattr(runner,"require_frozen",lambda *args:None)
    def save(name,obj):
        path = tmp_path/name
        path.write_text(json.dumps(obj))
        return {"path":name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    def artifact(path):
        return {"path":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    def load_bound(spec):
        path = tmp_path/spec["path"]
        if artifact(path)!=spec:
            raise ValueError("Closed artifact changed")
        return json.loads(path.read_text())
    monkeypatch.setattr(runner,"artifact",artifact)
    monkeypatch.setattr(runner,"load_bound",load_bound)
    workloads = []
    for case in range(4):
        cell = {**cells[case],"diagnostic":save(f"archive{case}.json",archives[case])}
        spec = save(f"cell{case}.json",cell)
        workloads.extend([spec]*4)
    parent = {"fixtures":save("fixtures.json",{"fixtures":fixtures}),"workloads":workloads}
    spec = save("result.json",parent)
    save("audit.json",{"status":"PASS","result":spec})
    actual = runner.inputs("tiny")
    assert actual[1]==fixtures and actual[3]==archives
    (tmp_path/"archive2.json").write_text("{}")
    with pytest.raises(ValueError,match="Closed artifact"):
        runner.inputs("tiny")
