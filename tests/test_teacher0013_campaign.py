"""Static resource and archive checks for the TEACH-0013 campaign runner."""

import gzip
import importlib.util
import json
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_campaign", Path(__file__).parents[1] / "scripts/teacher0013_campaign.py")
assert _SPEC is not None and _SPEC.loader is not None
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)


def test_benchmark_projection_applies_group_seed_and_conservative_factors():
    result = campaign.benchmark_projection(
        100, benchmark_groups=32, discovery_groups=128,
        benchmark_seeds=1, campaign_seeds=2, materialized_bytes=1024)
    assert result["scale"] == 36
    assert result["family_item_factor"] == 4.5
    assert result["conservative_stage_b_seconds"] == 5400
    assert result["conservative_complete_seconds"] == 10_800
    assert result["projected_materialized_activation_bytes"] == 55_296
    assert result["stage_b_pass"] and result["complete_pass"] and result["traffic_pass"]


def test_row_archive_is_deterministic_and_roundtrips(tmp_path):
    rows = [{"group_id": "a", "probability": .25},
            {"group_id": "b", "probability": .75}]
    left, right = tmp_path / "left.json.gz", tmp_path / "right.json.gz"
    assert campaign._write_rows(left, rows) == campaign._write_rows(right, rows)
    assert left.read_bytes() == right.read_bytes()
    assert json.loads(gzip.decompress(left.read_bytes())) == rows


def test_benchmark_decision_is_recomputed_before_discovery():
    projection = campaign.benchmark_projection(
        10, benchmark_groups=32, discovery_groups=128, materialized_bytes=1024)
    manifest = {"split_counts": {"discovery": 128},
                "suite_gzip_sha256": "suite",
                "checkpoints": {"0": {"8000": {"sha256": "checkpoint"}}}}
    provenance = {"campaign_source_sha256": {"source": "hash"}}
    report = {"elapsed_seconds": 10, "groups": 32,
              "materialized_activation_bytes": 1024,
              "peak_sampled_current_allocated_bytes": 1,
              "checkpoint_sha256": "checkpoint", "suite_gzip_sha256": "suite",
              "campaign_source_sha256": {"source": "hash"},
              "projection": projection, "status": "pass"}
    campaign.validate_benchmark(report, manifest, provenance)
    report["projection"] = {**projection, "scale": 1}
    try:
        campaign.validate_benchmark(report, manifest, provenance)
    except RuntimeError as exc:
        assert "recompute" in str(exc)
    else:
        raise AssertionError("Tampered benchmark projection was accepted")
