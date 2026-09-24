"""Known-answer controls for the fresh TEACH-0015 clean screen."""

import json

import pytest
import torch

from scripts.teacher0015_clean_audit import (
    _benchmark_check, _competent, eligible_arms,
    score_split,
)
from scripts.teacher0015_clean_run import _episodes, _load_model, _predict, _sha
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher15_tasks import generate_split, split_manifest


def _fixture():
    manifest = split_manifest(generate_split("discovery", 2), "discovery")
    cells = [cell for group in manifest["groups"] for cell in group["cells"]]
    rows = [{"render_id": cell["episode"]["render_id"],
             "prediction": cell["episode"]["answer"]} for cell in cells]
    samples = []
    for index in (0, len(cells) // 2, len(cells) - 1):
        logits = [0.0] * 2064
        logits[rows[index]["prediction"]] = 1.0
        samples.append({"index": index, "render_id": rows[index]["render_id"],
                        "logits": logits})
    return manifest, rows, samples


def test_clean_screen_exact_group_denominators_and_single_error():
    manifest, rows, samples = _fixture()
    rows[1]["prediction"] = (rows[1]["prediction"] - 16 + 1) % 2048 + 16
    scored = score_split(manifest, rows, samples, "fixture")
    assert scored["composed_items"]["total"] == 96
    assert scored["composed_items"]["correct"] == 95
    assert scored["recipient_triples"]["total"] == 32
    assert scored["recipient_triples"]["correct"] == 31
    assert scored["marked_free_pairs"]["total"] == 48
    assert scored["marked_free_pairs"]["correct"] == 47
    for task in ("first_hop", "direct", "copy"):
        assert scored[f"{task}_items"]["correct"] == 12
    assert _competent(scored)
    assert all(value["group_bootstrap"]["groups"] == 2
               for value in scored.values())


def test_clean_screen_refuses_wrong_render_id():
    manifest, rows, samples = _fixture()
    rows[1]["render_id"] = "wrong"
    with pytest.raises(ValueError, match="render ID"):
        score_split(manifest, rows, samples, "fixture")


def test_clean_screen_benchmark_projection_and_arm_entry(tmp_path):
    manifests = {split: {"groups": [{"cells": [None] * 66}] * 128}
                 for split in ("discovery", "confirmation")}
    projected = 1.75 * .01 * 528 * 3 * 2 + 300
    row = {"screen_source_sha256": {"a": "b"},
           "batch_size": 32, "warmup_batches": 6, "timed_batches": 24,
           "timings_seconds": [.01] * 24, "median_batch_seconds": .01,
           "batches_per_arm_seed": 528,
           "conservative_projected_seconds": projected,
           "admitted": True, "max_seconds": 3600,
           "max_mps_bytes": 12 * 1024**3,
           "max_artifact_bytes": 1024**3,
           "peak_sampled_mps_allocated_bytes": 100,
           "manifest_sha256": {
               "discovery": "ff044ec42ceaaea180a4cde43bbb1bd0dd73dc0c4f53c476876855831402261c",
               "confirmation": "941cce744278d677655d01ef71f4dfa4943352e7e477ce66efcbd5233c4db975"}}
    _benchmark_check(row, manifests, {"a": "b"})
    row["conservative_projected_seconds"] = 300
    with pytest.raises(ValueError, match="benchmark resource gate"):
        _benchmark_check(row, manifests, {"a": "b"})

    behavior = {"decisions": {"null_valid": True, "oracle_twohop": True},
                "scores": {arm: {str(rep): {"twohop_absolute": True}
                                 for rep in (0, 1)} for arm in (
                    "latent_rows_answer", "latent_rows_causal",
                    "latent_rows_edge_aux")}}
    parser = {"runs": {arm: {str(rep): {"parser_qualified": True}
                             for rep in (0, 1)} for arm in behavior["scores"]}}
    (tmp_path / "behavior-audit.json").write_text(json.dumps(behavior))
    (tmp_path / "parser-audit.json").write_text(json.dumps(parser))
    assert len(eligible_arms(tmp_path)) == 3
    parser["runs"]["latent_rows_answer"]["1"]["parser_qualified"] = False
    (tmp_path / "parser-audit.json").write_text(json.dumps(parser))
    assert "latent_rows_answer" not in eligible_arms(tmp_path)


def test_clean_screen_loads_frozen_checkpoint_and_parameter_metadata(tmp_path):
    model = CandidateEdgeWorkspace()
    path = (tmp_path / "outputs/TEACH-0014-v3" /
            "rep0-latent_rows_answer-step6000.pt")
    path.parent.mkdir(parents=True)
    torch.save({"arm": "latent_rows_answer", "replicate": 0,
                "step": 6000, "model": model.state_dict()}, path)
    checksum = _sha(path)
    report = {"training": {"latent_rows_answer": {"0": {
        "final_checkpoint": {"sha256": checksum},
        "parameters": model.parameter_count}}}}
    loaded, observed_sha = _load_model(
        tmp_path, report, "latent_rows_answer", 0, "cpu")
    assert observed_sha == checksum
    assert loaded.parameter_count == model.parameter_count


def test_random_weight_runner_and_no_model_scorer_agree_on_complete_mini_grid():
    manifest = split_manifest(generate_split("discovery", 2), "discovery")
    model = CandidateEdgeWorkspace()
    rows, samples = _predict(model, _episodes(manifest), "cpu")
    scored = score_split(manifest, rows, samples, "random-weight-fixture")
    assert len(rows) == 132
    assert [sample["index"] for sample in samples] == [0, 66, 131]
    assert scored["composed_items"]["total"] == 96
    assert scored["recipient_triples"]["total"] == 32
