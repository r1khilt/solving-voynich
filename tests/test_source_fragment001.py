"""Whole blocks, immutable round bases, cache and no-gold fit boundary."""
import inspect

import pytest

from scripts import run_source_fragment001 as run
from scripts.audit_source_fragment001 import event_check


def toy_fit(monkeypatch, max_keys=8):
    def score(_, records, key):
        return {"units": key, "log_key_mass": sum(u == "AA" for u in key)}
    monkeypatch.setattr(run, "score_key", score)
    monkeypatch.setattr(run, "read_key", lambda source, native, key, records, expected: {"source_records": list(key)})
    windows = [{"scan": {"matches": [{"partial_key": [6, -1]}, {"partial_key": [-1, 6]}, {"partial_key": [6, -1]}]}}]
    return run.fit_one(("AAAA",), ("A", "A"), windows, None, None, max_keys=max_keys, rounds=3), windows


def test_multi_round_composition_and_duplicate_cache(monkeypatch):
    fitted, windows = toy_fit(monkeypatch)
    assert fitted["scores"][fitted["best_index"]]["units"] == ("AA", "AA")
    assert len(fitted["scores"]) == 4
    assert len(fitted["events"]) == 9
    assert event_check(fitted, windows) == 9
    assert not fitted["posterior_sampling_claimed"]


def test_budget_preserves_supported_baseline_and_prefix_events(monkeypatch):
    fitted, windows = toy_fit(monkeypatch, max_keys=1)
    assert len(fitted["scores"]) == 1 and fitted["best_index"] == 0
    assert fitted["events"] == [] and fitted["stop_reason"] == "key_cap"
    assert event_check(fitted, windows) == 0
    with pytest.raises(ValueError):
        run.fit_one(("AA",), ("A", "A"), windows, None, None, max_keys=0)


def test_random_blocks_preserve_mask_and_lengths_and_fit_has_no_gold():
    windows = [{"record": 0, "offset": 1, "scan": {"matches": [{"gram": 0, "consumed": 3, "partial_key": [0, -1, 6]}]}}]
    random = run.random_blocks(windows, 0)
    assert random == run.random_blocks(windows, 0) and windows[0]["scan"]["matches"][0]["partial_key"] == [0, -1, 6]
    partial = random[0]["scan"]["matches"][0]["partial_key"]
    assert partial[1] == -1 and len(run.POOL[partial[0]]) == 1 and len(run.POOL[partial[2]]) == 2
    assert "fixture" not in inspect.signature(run.fit_one).parameters
    assert "gold" not in inspect.getsource(run.fit_one).split('"""')[-1]
