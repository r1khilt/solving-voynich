"""Cross-check the independent TEACH-0013 discovery rescoring implementation."""

import importlib.util
import json
from pathlib import Path

from voynich.workspace.teacher13_score import (
    rank_secondary_residual_sites,
    select_residual_site,
)


_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_discovery_audit",
    Path(__file__).parents[1] / "scripts/teacher0013_discovery_audit.py")
assert _SPEC is not None and _SPEC.loader is not None
auditor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(auditor)


def _rows(*, family, replicate, cut, label, correct=True):
    rows = []
    for render in ("marked", "marker_free"):
        for half in ("first_half", "second_half"):
            for group in range(2):
                fixed = 100 + group * 3
                for recipient in range(3):
                    target = fixed + recipient
                    rows.append({
                        "family": family, "replicate": replicate, "cut_index": cut,
                        "semantic_label": label,
                        "group_id": f"{render}:{half}:{group}", "recipient": recipient,
                        "target": target, "prediction": target if correct else 999,
                        "fixed_donor_answer": fixed, "target_probability": .6,
                        "base_target_probability": .1, "render_stratum": render,
                        "position_stratum": half,
                    })
    return rows


def normalized(value):
    return json.loads(json.dumps(value))


def test_independent_primary_selection_matches_registered_implementation():
    rows = []
    for replicate in (0, 1):
        rows += _rows(family="f_content", replicate=replicate, cut=1, label="query")
        rows += _rows(family="f_content", replicate=replicate, cut=0,
                      label="answer", correct=False)
    labels = ["answer", "query"]
    expected = select_residual_site(rows, tuple(labels), expected_groups_per_render=4)
    assert auditor.primary_selection(rows, labels, 4) == normalized(expected)


def test_independent_secondary_ranking_matches_registered_implementation():
    rows = []
    for replicate in (0, 1):
        rows += _rows(family="g_content", replicate=replicate, cut=1, label="query")
        rows += _rows(family="g_content", replicate=replicate, cut=0,
                      label="answer", correct=False)
    labels = ["answer", "query"]
    expected = rank_secondary_residual_sites(rows, tuple(labels), expected_groups=8)
    assert auditor.secondary_selection(rows, labels, 8, "g_content") == normalized(expected)
