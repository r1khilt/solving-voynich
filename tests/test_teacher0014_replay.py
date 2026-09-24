"""Sampled checkpoint-logit replay validation without a scientific campaign."""

from copy import deepcopy
from dataclasses import asdict
import gzip
import json
import random

import pytest
import torch

from scripts.teacher0014_artifact_audit import ARMS, _audit_replay_archive
from scripts.teacher0014_replay import _verify_vector
from scripts import teacher0014_replay as replay_module
from voynich.workspace.teacher14_tasks import RenderSpec, sample_episode
from voynich.workspace.teacher14_train import (
    Config, _predict_panels, new_model,
)


def test_replay_vector_requires_full_logits_and_same_answer():
    expected = torch.linspace(-3, 3, 2064)
    predicted = int(expected[16:].argmax().item()) + 16
    assert _verify_vector(expected.clone(), expected.tolist(), predicted) == 0
    with pytest.raises(ValueError, match="answer replay"):
        _verify_vector(expected.clone(), expected.tolist(), 16)
    changed = expected.clone()
    changed[100] += .1
    with pytest.raises(ValueError, match="logit replay differs"):
        _verify_vector(changed, expected.tolist(), predicted)
    with pytest.raises(ValueError, match="shape mismatch"):
        _verify_vector(expected[:-1], expected.tolist(), predicted)


def _small_archive():
    episodes = [{"render_id": f"render-{index}"} for index in range(3)]
    manifest = {"panels": {"small": episodes}}
    logits = [-1.0] * 2064
    logits[30] = 1.0
    rows = [{"index": index, "render_id": episode["render_id"],
             "logits": logits.copy()} for index, episode in enumerate(episodes)]
    archive = {"experiment": "TEACH-0014", "manifest_sha256": "a" * 64,
               "runs": {arm: {rep: {"small": deepcopy(rows)} for rep in ("0", "1")}
                        for arm in ARMS}}
    predictions = {"manifest_sha256": "a" * 64,
                   "runs": {arm: {rep: {"panels": {"small": [
                       {"prediction": 30} for _ in episodes]}}
                       for rep in ("0", "1")} for arm in ARMS}}
    return manifest, archive, predictions


def test_archive_rejects_missing_logits_wrong_ids_and_prediction_drift():
    manifest, archive, predictions = _small_archive()
    assert _audit_replay_archive(archive, manifest, predictions) == len(ARMS) * 2 * 3
    broken = deepcopy(archive)
    broken["runs"]["raw_null"]["0"]["small"][1]["index"] = True
    with pytest.raises(ValueError, match="sample identity"):
        _audit_replay_archive(broken, manifest, predictions)
    broken = deepcopy(archive)
    del broken["runs"]["raw_null"]["0"]["small"][1]
    with pytest.raises(ValueError, match="sample count"):
        _audit_replay_archive(broken, manifest, predictions)
    broken = deepcopy(archive)
    broken["runs"]["raw_null"]["0"]["small"][0]["logits"][30] = float("nan")
    with pytest.raises(ValueError, match="invalid replay logits"):
        _audit_replay_archive(broken, manifest, predictions)
    wrong = deepcopy(predictions)
    wrong["runs"]["raw_null"]["0"]["panels"]["small"][2]["prediction"] = 31
    with pytest.raises(ValueError, match="logit/prediction mismatch"):
        _audit_replay_archive(archive, manifest, wrong)


def test_checkpoint_replay_reconstructs_logits_from_saved_weights(
        tmp_path, monkeypatch):
    rng = random.Random(140199)
    episodes = [sample_episode(
        rng, signal_hops=2, task="composed", distractors=1,
        spec=RenderSpec(.25, 1, ("prefix", "infix", "suffix")),
        stage_partitions=("train", "train")) for _ in range(3)]
    suite = {"development": episodes}
    arm = "oracle_rows_workspace"
    config = Config()
    model, optimizer = new_model(config, arm, 0, "cpu")
    predictions, sampled_logits = _predict_panels(model, arm, suite, "cpu")
    outputs = tmp_path / "outputs"
    results = tmp_path / "results"
    outputs.mkdir()
    results.mkdir()
    for rep in (0, 1):
        torch.save({"model": model.state_dict(), "config": asdict(config),
                    "replicate": rep, "arm": arm, "step": config.steps_per_arm,
                    "parameters": model.parameter_count},
                   outputs / f"rep{rep}-{arm}-step{config.steps_per_arm}.pt")
    del model, optimizer
    report = {"config": json.loads(json.dumps(asdict(config))),
              "source_sha256": {}, "source_git_head": "a" * 40,
              "training": {arm: {rep: {"parameters": 24_766_465}
                                 for rep in ("0", "1")}}}
    (results / "report.json").write_text(json.dumps(report))
    artifact_sha = replay_module._canonical_sha({"dummy": True})
    all_predictions = {"runs": {arm: {rep: predictions
                                      for rep in ("0", "1")}}}
    all_logits = {"runs": {arm: {rep: sampled_logits
                                 for rep in ("0", "1")}}}
    (results / "predictions.json.gz").write_bytes(gzip.compress(
        json.dumps(all_predictions).encode()))
    (results / "replay-logits.json.gz").write_bytes(gzip.compress(
        json.dumps(all_logits).encode()))
    monkeypatch.setattr(replay_module, "audit_artifacts", lambda *args: {
        "manifest_sha256": artifact_sha, "replay_logit_samples": 6})
    monkeypatch.setattr(replay_module, "SOURCE_PATHS", ())
    monkeypatch.setattr(replay_module, "ARMS", {arm})
    monkeypatch.setattr(replay_module, "evaluation_suite", lambda *args: suite)
    monkeypatch.setattr(replay_module, "suite_manifest",
                        lambda *args, **kwargs: {"dummy": True})
    result = replay_module.replay(results, outputs, tmp_path)
    assert result["audit"] == "pass"
    assert result["samples"] == 6
    assert result["max_abs_logit_error"] < 1e-5
    checkpoint = outputs / f"rep0-{arm}-step{config.steps_per_arm}.pt"
    tampered = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tampered["model"]["symbol.weight"] = torch.zeros_like(
        tampered["model"]["symbol.weight"])
    tampered["model"]["unembedding.weight"] = torch.zeros_like(
        tampered["model"]["unembedding.weight"])
    torch.save(tampered, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint logit replay differs"):
        replay_module.replay(results, outputs, tmp_path)
