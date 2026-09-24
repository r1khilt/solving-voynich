"""Campaign input, scoring and source-independent audit helpers."""

import gzip
import json
from pathlib import Path

from scripts.teacher0022_audit import _competent, _visible_path
from scripts.teacher0022_suite_audit import identities
from scripts.teacher0022_train import canonical
from voynich.workspace.teacher14_tasks import training_batch


ROOT = Path(__file__).resolve().parents[1]


def test_training_stream_matches_frozen_oracle_seed_and_step():
    for rep, seed in enumerate((84221, 84231)):
        baseline = json.loads(gzip.decompress((ROOT /
            f"results/TEACH-0014-v3/losses-rep{rep}-oracle_rows_workspace.json.gz"
        ).read_bytes()))
        for step in (0, 137, 5999):
            episodes, answers = training_batch(seed + step, 32, step=step)
            assert canonical([item.render_id for item in episodes]) == baseline[
                step]["answer_input_sha256"]
            assert canonical(answers) == baseline[
                step]["answer_label_sha256"]


def test_visible_path_tracks_first_two_rows_without_metadata():
    episodes, _ = training_batch(84221 + 37, 32, step=37)
    for episode in episodes:
        row = {"tokens": list(episode.tokens), "hops": episode.hops,
               "answer": episode.answer}
        path = _visible_path(row)
        assert len(path) == min(episode.hops, 2)
        assert all(0 <= index < len(episode.serialized_rows)
                   for index in path)


def test_competence_requires_all_registered_absolute_cells():
    panels = {
        "first_hop_confirm": {"items": {"accuracy": .91}},
        "direct_confirm": {"items": {"accuracy": .91}},
        "copy_confirm": {"items": {"accuracy": .96}},
        "factorial": {"groups": {"accuracy": .71}},
    }
    assert _competent(panels)
    panels["factorial"]["groups"]["accuracy"] = .69
    assert not _competent(panels)


def test_suite_identities_are_set_based():
    manifest = {"panels": {"a": [{"graph_id": "g", "logical_id": "l",
                                   "render_id": "r"}],
                           "b": [{"graph_id": "g", "logical_id": "l",
                                   "render_id": "r2"}]}}
    assert identities(manifest) == {
        "graph_id": {"g"}, "logical_id": {"l"},
        "render_id": {"r", "r2"}}
