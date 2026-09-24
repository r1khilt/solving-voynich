"""Denominator, resource and archive controls for the finite key assay."""

from copy import deepcopy
import statistics

import numpy as np
import pytest

from scripts.teacher0015_finite_audit import audit_surface
from scripts.teacher0015_finite_result_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SURFACES,
    TOTAL_WORST_SURFACES, VECTOR_NAMES, _benchmark_check, score_rows,
)
from scripts.teacher0015_finite_run import _pack_surface, _surface
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher15_intervene import evaluate_surface_batched
from voynich.workspace.teacher15_pairs import control_permutations
from voynich.workspace.teacher15_tasks import generate_split, split_manifest


def test_perfect_recipient_transfer_scores_full_denominators():
    rows = []
    for _group in range(128):
        for _, marked, _ in SURFACES:
            for g in range(3):
                rows.append({
                    "g": g, "marked": marked,
                    "base_answer": 16 + g, "target_answer": 24 + g,
                    "fixed_donor_answer": 24,
                    "base_prediction": 16 + g,
                    "target_prediction": 24 + g,
                    "donor_prediction": 24,
                    "transfer_prediction": 24 + g,
                    "reverse_prediction": 16 + g,
                    "wrong_key_prediction": 40,
                    "deranged_prediction": 41,
                    "random_prediction": 42,
                    "same_key_prediction": 24 + g,
                    "final_donor_prediction": 24,
                })
    result = score_rows(rows, ["synthetic", 0])
    assert result["candidate_reusable_key_state_pending_replay"]
    scores = result["scores"]
    assert scores["transfer"]["total"] == 3072
    assert scores["triples"]["total"] == 1024
    assert scores["changed_g_non_injection"]["total"] == 2048
    assert scores["marked"]["total"] == 1536
    assert scores["reverse_eligible"]["total"] == 3072
    assert scores["transfer"]["group_bootstrap"]["groups"] == 128
    bad = deepcopy(rows)
    for row in bad:
        row["transfer_prediction"] = row["fixed_donor_answer"]
    assert not score_rows(bad, ["synthetic", 1])[
        "candidate_reusable_key_state_pending_replay"]
    with pytest.raises(ValueError, match="denominator"):
        score_rows(rows[:-1], ["synthetic", 2])


def test_benchmark_math_and_hard_cap_are_independent(tmp_path):
    paths = {}
    hashes = {}
    for split in ("discovery", "confirmation"):
        path = tmp_path / f"{split}.json"
        path.write_text(split)
        paths[split] = {"path": path}
        from scripts.teacher0015_clean_audit import _sha
        hashes[split] = _sha(path)
    from scripts.teacher0015_clean_audit import EXPECTED_MANIFESTS
    timings = {str(index): [.1, .11, .12] for index in range(8)}
    slowest = max(statistics.median(values) for values in timings.values())
    row = {"timings_seconds": timings, "finite_source_sha256": {"x": "y"},
           "manifest_sha256": EXPECTED_MANIFESTS,
           "suite_file_sha256": hashes,
           "warmup_surfaces": 8, "timed_surfaces": 24,
           "slowest_type_median_seconds": slowest,
           "conservative_projected_seconds": (
               1.75 * slowest * TOTAL_WORST_SURFACES + 300),
           "worst_case_surfaces": TOTAL_WORST_SURFACES,
           "max_seconds": MAX_SECONDS, "max_mps_bytes": MAX_MPS_BYTES,
           "max_artifact_bytes": MAX_ARTIFACT_BYTES,
           "peak_sampled_mps_allocated_bytes": 100,
           "admitted": True}
    _benchmark_check(row, {"x": "y"}, paths)
    bad = {**row, "conservative_projected_seconds": 1.0}
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(bad, {"x": "y"}, paths)


def test_batched_compact_archive_restores_auditable_vectors():
    groups = generate_split("discovery", 3)
    manifest = split_manifest(groups, "discovery")
    model = CandidateEdgeWorkspace()
    def donor(group):
        return next(cell.episode for cell in group.cells if (
            cell.f, cell.g, cell.distractor, cell.marked, cell.order,
            cell.task) == (1, 0, 0, True, 0, "composed"))
    rows = evaluate_surface_batched(
        model, groups[0], distractor=0, marked=True, order=0,
        device="cpu", wrong_donor=donor(groups[1]),
        deranged_donor=donor(groups[2]), full_logits=False)
    compact, vectors = _pack_surface(rows)
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, len(VECTOR_NAMES), 512)
    restored = []
    for recipient, row in enumerate(compact):
        copy = dict(row)
        copy["replacement_vectors"] = {
            name: vectors[recipient, index].tolist()
            for index, name in enumerate(VECTOR_NAMES)}
        restored.append(copy)
    assert audit_surface(
        manifest["groups"][0], restored,
        distractor=0, marked=True, order=0,
        wrong_group=manifest["groups"][1],
        deranged_group=manifest["groups"][2],
        full_logits=False)["audit"] == "pass"


def test_all_eight_frozen_surface_types_pass_no_model_vector_audit():
    groups = generate_split("confirmation")
    records = split_manifest(groups, "confirmation")["groups"]
    wrong, deranged = control_permutations(groups)
    model = CandidateEdgeWorkspace()
    for surface_index, (distractor, marked, order) in enumerate(SURFACES):
        sampled = surface_index == 0
        rows = _surface(model, groups, 0, surface_index,
                        (wrong, deranged), device="cpu", full_logits=sampled)
        compact, vectors = _pack_surface(rows)
        restored = []
        for recipient, row in enumerate(compact):
            copy = dict(row)
            copy["replacement_vectors"] = {
                name: vectors[recipient, index].tolist()
                for index, name in enumerate(VECTOR_NAMES)}
            restored.append(copy)
        assert audit_surface(
            records[0], restored, distractor=distractor, marked=marked,
            order=order, wrong_group=records[wrong[0]],
            deranged_group=records[deranged[0]],
            full_logits=sampled)["audit"] == "pass"
