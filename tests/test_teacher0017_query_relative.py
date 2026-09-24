"""Visible query-relative coding and independent metadata checks."""

import json
from pathlib import Path

import pytest

from scripts.teacher0015_finite_audit import audit_control_permutations
from scripts.teacher0017_query_relative import (
    _counts, _decode, _encode, _rank_prediction, evaluate_split,
)
from scripts.teacher0017_query_relative_audit import _expected
from scripts.teacher0016_suite_audit import _episode
from voynich.workspace.teacher14_tasks import SYMBOL_COUNT, SYMBOL_START


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("split", ["discovery", "confirmation"])
def test_query_relative_visible_state_and_metadata_audit(split: str) -> None:
    groups = json.loads((ROOT / "outputs/TEACH-0017" /
                         f"{split}.json").read_text())["groups"]
    deranged, _ = audit_control_permutations(groups)
    rows = evaluate_split({"groups": groups})
    assert len(rows) == 1024
    expected = _expected(groups[0], groups[deranged[0]], 0,
                         True, 0, 0, 0)
    assert rows[0][0] == expected
    selected = _counts(rows)
    assert selected["same_query_attempts"] == 9216
    assert selected["shifted_rank_offdiag_attempts"] > 4096
    assert selected["shifted_rank_offdiag_hits"]["transfer"] == selected[
        "shifted_rank_offdiag_attempts"]
    assert selected["shifted_rank_offdiag_hits"]["rank"] == 0
    assert selected["shifted_rank_offdiag_hits"]["wrong_key"] == 0


def test_relative_code_changes_with_query_and_detects_missing_row() -> None:
    groups = json.loads((ROOT / "outputs/TEACH-0017/discovery.json").read_text())[
        "groups"]
    source = _episode(groups[0], 1, 0, 0, True, 0)
    base = _episode(groups[0], 0, 1, 1, True, 1)
    assert _decode(base, _encode(source)) == groups[0]["recipient_outputs"][1][1]
    prediction, shifted = _rank_prediction(source, base)
    if shifted:
        assert prediction != groups[0]["recipient_outputs"][1][1]
    missing = next(symbol for symbol in range(
        SYMBOL_START, SYMBOL_START + SYMBOL_COUNT)
        if symbol not in {left for left, _ in base["serialized_rows"]})
    query = base["query"]
    assert _decode(base, (missing - query) % SYMBOL_COUNT) == -1
