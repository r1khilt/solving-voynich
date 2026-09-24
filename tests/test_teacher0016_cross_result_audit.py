"""Source/resource gate and complete-surface packing checks."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.teacher0015_clean_audit import _sha
from scripts.teacher0016_clean_audit import EXPECTED_MANIFESTS
from scripts.teacher0016_cross_result_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, TOTAL_WORST_SURFACES,
    VECTOR_NAMES, _benchmark_check,
)
from scripts.teacher0016_cross_run import _pack_surface
from voynich.workspace.teacher14_train import Config, new_model
from voynich.workspace.teacher16_intervene import evaluate_cross_surface
from voynich.workspace.teacher16_tasks import generate_split


def test_causal_benchmark_gate_rejects_timing_and_resource_tampering():
    suite_dir = Path("outputs/TEACH-0016")
    source = {"placeholder": "frozen"}
    times = {str(index): [.12, .13, .14] for index in range(8)}
    projected = 1.75 * .13 * TOTAL_WORST_SURFACES + 300
    benchmark = {
        "cross_source_sha256": source,
        "manifest_sha256": EXPECTED_MANIFESTS,
        "suite_file_sha256": {
            split: _sha(suite_dir / f"{split}.json")
            for split in ("discovery", "confirmation")},
        "vector_names": list(VECTOR_NAMES),
        "timings_seconds": times,
        "warmup_surfaces": 8, "timed_surfaces": 24,
        "worst_case_surfaces": TOTAL_WORST_SURFACES,
        "slowest_type_median_seconds": .13,
        "conservative_projected_seconds": projected,
        "max_seconds": MAX_SECONDS,
        "max_mps_bytes": MAX_MPS_BYTES,
        "max_artifact_bytes": MAX_ARTIFACT_BYTES,
        "peak_sampled_mps_allocated_bytes": 100_000_000,
        "elapsed_seconds": 10.0, "artifact_bytes": 0,
        "admitted": True,
    }
    _benchmark_check(benchmark, source, suite_dir)
    wrong = deepcopy(benchmark)
    wrong["vector_names"][0] = "wrong"
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(wrong, source, suite_dir)
    wrong = deepcopy(benchmark)
    wrong["timings_seconds"]["4"][1] = 5.0
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(wrong, source, suite_dir)
    wrong = deepcopy(benchmark)
    wrong["peak_sampled_mps_allocated_bytes"] = MAX_MPS_BYTES + 1
    with pytest.raises(ValueError, match="resource gate"):
        _benchmark_check(wrong, source, suite_dir)


def test_full_nine_pair_surface_pack_preserves_all_actual_vectors():
    groups = generate_split("discovery", 4)
    group, distinct = next((left, right) for left in groups for right in groups
                           if left.group_id != right.group_id and
                           left.key1 != right.key1)
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    rows = evaluate_cross_surface(model, group, distinct, distractor=1,
                                  marked=False, source_order=1,
                                  device="cpu", full_logits=False)
    compact, vectors = _pack_surface(rows)
    assert vectors.dtype == np.float32
    assert vectors.shape == (9, len(VECTOR_NAMES), 512)
    assert np.isfinite(vectors).all()
    assert all("replacement_vectors" not in row for row in compact)
    for attempt, row in enumerate(rows):
        for index, name in enumerate(VECTOR_NAMES):
            assert vectors[attempt, index].tolist() == row[
                "replacement_vectors"][name]
