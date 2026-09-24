"""Independent diagnostic archive validation on random-weight observations."""

from dataclasses import asdict
import json
from pathlib import Path
import random

import pytest

from scripts.teacher0014_diagnostic_audit import (
    _benchmark_check, _paired_bootstrap, _validate_row, _visible_path,
)
from scripts.teacher0014_diagnostic_run import _primary_gate, _sha
from voynich.workspace.teacher14_diagnose import evaluate_batch
from voynich.workspace.teacher14_tasks import RenderSpec, sample_episode
from voynich.workspace.teacher14_train import Config, new_model


def _fixture():
    episode = sample_episode(
        random.Random(149911), signal_hops=2, task="composed",
        distractors=4, spec=RenderSpec(1.0, 2,
                                       ("prefix", "infix", "suffix")),
        stage_partitions=("train", "train"))
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    row = evaluate_batch(model, [episode], device="cpu")[0]
    for condition in row["conditions"].values():
        condition["sampled_logits"] = condition.pop("logits")
    return asdict(episode), row


def test_no_model_diagnostic_row_audit_checks_visible_path_and_predictions():
    source, row = _fixture()
    path, count = _visible_path(source)
    assert path == row["path_candidate_indices"]
    assert count == row["candidate_count"]
    hits = _validate_row(
        source, row, index=0,
        primary_prediction=row["conditions"]["native"]["prediction"],
        primary_logits=row["conditions"]["native"]["sampled_logits"],
        sampled=True)
    assert set(hits) == set(row["conditions"])
    row["random_selected_indices"][0] = -1
    with pytest.raises(ValueError, match="random assignment"):
        _validate_row(source, row, index=0,
                      primary_prediction=row["conditions"]["native"][
                          "prediction"], primary_logits=None, sampled=True)


def test_no_model_audit_rejects_missing_sampled_condition_logits():
    source, row = _fixture()
    row["conditions"]["gold"].pop("sampled_logits")
    with pytest.raises(ValueError, match="logit selection"):
        _validate_row(source, row, index=0,
                      primary_prediction=row["conditions"]["native"][
                          "prediction"], primary_logits=None, sampled=True)


def test_no_model_audit_rejects_wrong_gold_gate_assignment():
    source, row = _fixture()
    row["gate_assignment_logits"]["gold"][0] = -20.0
    with pytest.raises(ValueError, match="gold gate assignment"):
        _validate_row(source, row, index=0,
                      primary_prediction=row["conditions"]["native"][
                          "prediction"], primary_logits=None, sampled=True)


def test_diagnostic_benchmark_arithmetic_is_independently_checked(tmp_path):
    names = ("factorial", "boundary_groups", "long_ood",
             "hop_4_long_ood", "copy_confirm", "alias_inner")
    timings = {name: [.01] * 4 for name in names}
    source = {"placeholder": "hash"}
    manifest = {"panels": {"one": [{}] * 33}}
    row = {
        "experiment": "TEACH-0014-gate-diagnostic",
        "diagnostic_source_sha256": source,
        "primary_suite_sha256": "suitehash",
        "timings_seconds": timings,
        "panel_median_seconds": {name: .01 for name in names},
        "slowest_panel_median_seconds": .01,
        "batch_size": 32, "random_rep": 0,
        "warmup_batches": 6, "timed_batches": 24,
        "batches_per_seed": 2,
        "conservative_projected_seconds": 1.75 * .01 * 2 * 2 + 300,
        "admitted": True, "max_seconds": 7200,
        "max_mps_bytes": 12 * 1024**3,
        "max_artifact_bytes": 2 * 1024**3,
        "sampled_mps_allocated_bytes": 100,
        "peak_sampled_mps_allocated_bytes": 100,
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(row))
    _benchmark_check(path, manifest, source, "suitehash")
    row["conservative_projected_seconds"] = 300
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(path, manifest, source, "suitehash")


def test_paired_bootstrap_preserves_pairing_and_fixed_seed():
    result = _paired_bootstrap([True] * 8, [False] * 8,
                               ["fixture", "gold-native"], draws=100)
    assert result["difference"] == 1.0
    assert result["lower_95"] == result["upper_95"] == 1.0
    assert result == _paired_bootstrap([True] * 8, [False] * 8,
                                       ["fixture", "gold-native"], draws=100)


def test_trained_diagnosis_refuses_failed_primary_entry_gate(tmp_path,
                                                             monkeypatch):
    root = Path(__file__).resolve().parents[1]
    audited = {"audit": "pass", "manifest_sha256": "suite"}
    monkeypatch.setattr("scripts.teacher0014_artifact_audit.audit_artifacts",
                        lambda *_: audited)
    (tmp_path / "artifact-audit.json").write_text(json.dumps(audited))
    (tmp_path / "replay-audit.json").write_text(json.dumps({
        "audit": "pass", "source_git_head": "source",
        "manifest_sha256": "suite"}))
    (tmp_path / "report.json").write_text(json.dumps({
        "status": "complete", "source_git_head": "source",
        "source_sha256": {
            "src/voynich/workspace/teacher14_models.py": _sha(
                root / "src/voynich/workspace/teacher14_models.py")}}))
    behavior = {"decisions": {"null_valid": True, "oracle_twohop": True},
                "scores": {"latent_rows_answer": {
                    str(rep): {"twohop_absolute": False} for rep in (0, 1)}}}
    (tmp_path / "behavior-audit.json").write_text(json.dumps(behavior))
    parser = {"runs": {"latent_rows_answer": {
        str(rep): {"parser_qualified": False} for rep in (0, 1)}}}
    (tmp_path / "parser-audit.json").write_text(json.dumps(parser))
    _primary_gate(tmp_path, root)
    behavior["decisions"]["null_valid"] = False
    (tmp_path / "behavior-audit.json").write_text(json.dumps(behavior))
    with pytest.raises(ValueError, match="not entered"):
        _primary_gate(tmp_path, root)
