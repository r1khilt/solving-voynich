"""Independent-auditor checks added before TEACH-0012 report access."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from voynich.workspace.teacher12_tasks import evaluation_suite


_SPEC = importlib.util.spec_from_file_location(
    "teacher0012_analyze", Path(__file__).parents[1] / "scripts/teacher0012_analyze.py")
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def test_independent_suite_matches_native_generator_on_development_size():
    native = evaluation_suite(72311, 4)
    independent = analyzer.expected_suite(72311, 4)
    assert native.keys() == independent.keys()
    for name in native:
        left = [analyzer.normalized_episode(asdict(episode)) for episode in native[name]]
        right = [analyzer.normalized_episode(episode) for episode in independent[name]]
        assert left == right


def test_registered_diagnostics_reconstruct_positions_and_wrong_destinations():
    episodes = analyzer.expected_suite(72311, 1)["first_hop_confirm"]
    rows = []
    for index, episode in enumerate(episodes):
        row = analyzer.normalized_episode(episode)
        row.update({"index": index, "length": len(episode["tokens"]),
                    "prediction": episode["answer"], "target_probability": .9,
                    "candidate_member": True})
        rows.append(row)
    rows[0]["prediction"] = rows[0]["g_rows"][0][1]
    result = analyzer.registered_diagnostics(rows)
    assert result["overall"]["total"] == 1
    assert result["overall"]["correct"] == 0
    assert result["wrong_answer_destinations"]["g_value"] == 1
    assert sum(bucket["total"] for bucket in result["by_answer_row_index"].values()) == 1
    assert sum(bucket["total"] for bucket in
               result["by_query_to_answer_source_distance"].values()) == 1


def test_loss_history_recomputes_every_archived_summary():
    losses = [index / 10_000 for index in range(8000)]
    history = analyzer.loss_history(losses)
    assert len(history) == 32
    assert history[0] == {
        "step": 250,
        "last_100_loss": pytest.approx(sum(losses[150:250]) / 100),
        "loss": losses[249],
    }
    assert history[-1]["step"] == 8000
    assert history[-1]["loss"] == losses[-1]
