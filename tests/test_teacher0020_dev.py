"""TEACH-0020 independent archive audit must reject identity/logit drift."""

import copy
from dataclasses import asdict
import json

import pytest

from scripts.teacher0020_dev_audit import _score
from scripts.teacher0020_dev_replay import _compare
from scripts.teacher0020_dev_run import episode_from_json
from voynich.workspace.teacher14_objectives import padded_tokens
from voynich.workspace.teacher14_tasks import Episode


PANEL_NAMES = (
    "composed_train_train", "composed_confirm_train",
    "composed_train_confirm", "composed_confirm_confirm",
    "first_hop_confirm", "direct_confirm", "copy_confirm",
    "first_hop_query_groups", "direct_query_groups", "factorial",
    "order_groups", "boundary_groups", "distractor_groups",
    "long_ood", "alias_terminal", "alias_inner", "hop_3", "hop_4",
    "hop_4_long_ood",
)


def _fixture() -> tuple[dict, dict]:
    manifest = {"panels": {
        name: [{"render_id": f"{name}-{i}", "answer": 16}
               for i in range(4)] for name in PANEL_NAMES}}
    logits = [0.0] * 2064
    logits[16] = 1.0
    archive = {"panels": {}, "samples": {}}
    for name, source in manifest["panels"].items():
        archive["panels"][name] = [
            {"render_id": row["render_id"], "prediction": 16}
            for row in source]
        archive["samples"][name] = [
            {"index": i, "render_id": source[i]["render_id"],
             "logits": logits.copy()} for i in (0, 2, 3)]
    return manifest, archive


def test_all_panel_integrity_and_mutations() -> None:
    manifest, archive = _fixture()
    checked = _score(manifest, archive)
    assert checked["development_twohop_thresholds_met"] is True
    assert checked["development_extrapolation_thresholds_met"] is True

    reordered = copy.deepcopy(archive)
    rows = reordered["panels"]["factorial"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="prediction identity"):
        _score(manifest, reordered)

    changed_logit = copy.deepcopy(archive)
    sample = changed_logit["samples"]["factorial"][0]
    sample["logits"][16] = 0.0
    sample["logits"][17] = 2.0
    with pytest.raises(ValueError, match="argmax"):
        _score(manifest, changed_logit)


def test_checkpoint_replay_tolerance_and_answer() -> None:
    logits = [0.0] * 2064
    logits[16] = 3.0
    changed = logits.copy()
    changed[100] = 0.001
    assert _compare(changed, logits, 16) == pytest.approx(0.001)
    changed[100] = 0.01
    with pytest.raises(ValueError, match="tolerance"):
        _compare(changed, logits, 16)
    with pytest.raises(ValueError, match="argmax"):
        _compare(logits, logits, 17)


def test_archived_episode_restores_frozen_model_input_type() -> None:
    original = Episode(
        tokens=(1, 16, 17, 4, 16, 8), signal_paths=((16, 17),),
        distractor_paths=(), serialized_rows=((16, 17),),
        row_positions=((1, 2),), query=16, answer=17, task="direct",
        hops=1, stage_partitions=("confirm",), graph_partition="confirm",
        graph_id="g", logical_id="l", render_id="r", marker_dropout=0.0,
        alias_mode="none")
    source = json.loads(json.dumps(asdict(original)))
    episode = episode_from_json(source)
    assert isinstance(episode.tokens, tuple)
    assert isinstance(episode.signal_paths, tuple)
    assert padded_tokens([episode], "cpu").shape[0] == 1
