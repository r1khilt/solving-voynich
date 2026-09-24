"""Fresh clean-screen denominator, benchmark and entry controls."""

import json

import pytest

from scripts.teacher0015_clean_audit import _competent, score_split
from scripts.teacher0015_clean_run import _episodes, _predict
from scripts.teacher0016_clean_audit import (
    EXPECTED_MANIFESTS, MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS,
    _benchmark_check, eligible_arms,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher16_tasks import generate_split, split_manifest


def test_new_suite_random_weight_screen_and_all_correct_oracle():
    manifest = split_manifest(generate_split("discovery", 2), "discovery")
    episodes = _episodes(manifest)
    assert len(episodes) == 132
    model = CandidateEdgeWorkspace()
    rows, samples = _predict(model, episodes, "cpu")
    score = score_split(manifest, rows, samples, ["TEACH-0016", "random"])
    assert score["composed_items"]["total"] == 96
    assert score["recipient_triples"]["total"] == 32
    assert score["marked_free_pairs"]["total"] == 48
    assert [sample["index"] for sample in samples] == [0, 66, 131]
    oracle = [{"render_id": episode.render_id,
               "prediction": episode.answer} for episode in episodes]
    for index in (0, 66, 131):
        sample = next(row for row in samples if row["index"] == index)
        sample["logits"] = [0.0] * 2064
        sample["logits"][oracle[index]["prediction"]] = 1.0
    correct = score_split(manifest, oracle, samples, ["TEACH-0016", "oracle"])
    assert _competent(correct)
    oracle[1]["render_id"] = "tampered"
    with pytest.raises(ValueError, match="render ID"):
        score_split(manifest, oracle, samples, "corrupt")


def test_new_screen_benchmark_arithmetic_and_finite_entry(tmp_path):
    for split in ("discovery", "confirmation"):
        (tmp_path / f"{split}.json").write_text(split)
    from scripts.teacher0015_clean_audit import _sha
    projected = 1.75 * .05 * 528 * 6 + 300
    benchmark = {
        "screen_source_sha256": {"x": "y"},
        "manifest_sha256": EXPECTED_MANIFESTS,
        "suite_file_sha256": {
            split: _sha(tmp_path / f"{split}.json")
            for split in ("discovery", "confirmation")},
        "batch_size": 32, "batches_per_arm_seed": 528,
        "warmup_batches": 6, "timed_batches": 24,
        "timings_seconds": [.05] * 24, "median_batch_seconds": .05,
        "conservative_projected_seconds": projected,
        "max_seconds": MAX_SECONDS, "max_mps_bytes": MAX_MPS_BYTES,
        "max_artifact_bytes": MAX_ARTIFACT_BYTES,
        "peak_sampled_mps_allocated_bytes": 100,
        "elapsed_seconds": 10.0, "artifact_bytes": 0,
        "admitted": True,
    }
    _benchmark_check(benchmark, {"x": "y"}, tmp_path)
    benchmark["conservative_projected_seconds"] = 301
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(benchmark, {"x": "y"}, tmp_path)
    arm = "latent_rows_answer"
    decision = {"audit": "pass",
                "candidate_arms_pending_checkpoint_replay": [arm]}
    path = tmp_path / "decision-audit.json"
    path.write_text(json.dumps(decision))
    report = {"status": "complete", "decision_audit_sha256": _sha(path)}
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "replay-audit.json").write_text(json.dumps({
        "audit": "pass", "portable_state_supported_arms": [arm]}))
    assert eligible_arms(tmp_path) == [arm]
    (tmp_path / "replay-audit.json").write_text(json.dumps({
        "audit": "pass", "portable_state_supported_arms": []}))
    with pytest.raises(ValueError, match="prior portable-state gate"):
        eligible_arms(tmp_path)
