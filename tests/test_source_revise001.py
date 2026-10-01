"""Staged revision laws, immutable prediction-before-gold and actual transport."""
import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.audit_source_revise001 import replay_rng_trace
from scripts.run_source_revise001 import POOL, config_for, fit_one, score_key, warm_key
from voynich.native_suffix_marginal import marginal_python
from voynich.tempered_unit_search import search_tempered


def test_warm_completion_preserves_bound_units_and_ignores_gold():
    bank = [{"used_key": [0, 7]+[-1]*21, "source_records": [[0, 1]], "literal_log_mass": -3},
            {"used_key": [1, 8]+[-1]*21, "source_records": [[1, 0]], "literal_log_mass": -4}]
    a, meta = warm_key(bank, 2, 75511)
    assert a[:2] == (POOL[0], POOL[7]) and len(a) == 23 and set(a) <= set(POOL)
    assert warm_key(bank[::-1], 2, 75511) == (a, meta)
    assert warm_key(bank, 2, 75511) == (a, meta)
    assert set(inspect.signature(fit_one).parameters) == {"records", "warm", "source", "native", "config", "observe"}


@pytest.mark.parametrize("change", ["empty", "short", "invalid"])
def test_invalid_warm_bank_refused(change):
    bank = [] if change == "empty" else [{"used_key": [-1]*(22 if change == "short" else 23),
        "literal_log_mass": 0, "source_records": [[]]}]
    if change == "invalid":
        bank[0]["used_key"][0] = 42
    with pytest.raises(ValueError):
        warm_key(bank, 0, 75511)


def test_rng_replay_includes_one_start_replicated_and_all_mixture_components():
    cfg = replace(config_for(0, 75511), iterations=8, max_scored_keys=1000)
    def score(key):
        return -sum(map(len, key))
    search = search_tempered(score, [POOL[:23]], POOL, config=cfg)
    result = replay_rng_trace(json.loads(json.dumps(search)), cfg)
    assert result["proposals"] == 64
    assert all(v["proposed"] > 0 for v in search["proposal_counts"].values())
    assert not search["stationary_distribution_claimed"]


def test_resource_failure_never_becomes_zero_likelihood():
    class Capped:
        def score(self, *_args, **_kwargs):
            raise RuntimeError("Exact lattice edge cap; no pruning")
    with pytest.raises(RuntimeError, match="cap"):
        score_key(Capped(), ("ABC",), POOL[:23])


def test_actual_16_cell_run_and_full_audit_on_mock23_row_source(tmp_path, monkeypatch):
    import scripts.audit_source_revise001 as auditor
    import scripts.run_blind_channel_dev004 as storage
    import scripts.run_latin_source_model001 as artifacts
    import scripts.run_source_guide001 as guide_runner
    import scripts.run_source_revise001 as runner

    class Source:
        alphabet = tuple("abcdefghiklmnopqrstuxyz")
        probabilities = np.full((1, 23), 1/23, dtype=np.float64)
        transitions = np.zeros((1, 23), dtype=np.uint32)

        def state(self, history):
            return 0

        def row(self, state):
            return self.probabilities[state]

        def step(self, state, row):
            return 0

    class Native:
        def __init__(self, source, _build):
            self.source = source

        def score(self, key, record, rho, **limits):
            return marginal_python(self.source, key, record, rho, **limits)

    for module in (runner, auditor, storage, artifacts, guide_runner):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    parent_out = tmp_path/"results/SOURCE-GUIDE-001"
    parent_bulk = tmp_path/"outputs/SOURCE-GUIDE-001"
    monkeypatch.setattr(guide_runner, "OUT", parent_out)
    monkeypatch.setattr(guide_runner, "BULK", parent_bulk)
    monkeypatch.setattr(guide_runner, "DESIGNS", (3, 3, 5, 5))
    monkeypatch.setattr(guide_runner, "PARTICLES", 4)
    monkeypatch.setattr(guide_runner, "require_frozen", lambda *_: None)
    monkeypatch.setattr(guide_runner, "limit_resources", lambda *_: None)
    monkeypatch.setattr(guide_runner, "load_source", lambda: (Source(), {"counts": {"mock": True}}))
    guide_runner.run("mock")
    parent = json.loads((parent_out/"result.json").read_text())
    fixture = json.loads((tmp_path/parent["fixtures"]["path"]).read_text())["fixtures"]
    out, bulk = tmp_path/"results/SOURCE-REVISE-001", tmp_path/"outputs/SOURCE-REVISE-001"
    monkeypatch.setattr(runner, "PARENT", parent_out)
    monkeypatch.setattr(runner, "BULK", bulk)
    for module in (runner, auditor):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "inputs", lambda *_: ({"build": {}}, parent, fixture))
        monkeypatch.setattr(module, "load_source", lambda: (Source(), {"counts": {"mock": True}}))
        monkeypatch.setattr(module, "NativeMarginal", Native)
        monkeypatch.setattr(module, "config_for", lambda case, seed: replace(config_for(case, seed), iterations=2, max_scored_keys=16))
    original = runner.diagnostics

    def guarded_diagnostic(fitted, case, source, native):
        matching = list(bulk.glob("*-prediction.json"))
        assert matching and len(matching) == len(list(bulk.glob("case*.json.gz")))
        return original(fitted, case, source, native)

    monkeypatch.setattr(runner, "diagnostics", guarded_diagnostic)
    runner.run("mock")
    auditor.audit()
    assert len(json.loads((out/"result.json").read_text())["workloads"]) == 16
    assert json.loads((out/"audit.json").read_text())["workloads"] == 16
