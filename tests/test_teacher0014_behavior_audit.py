"""Prediction artifact identity and decision-gate regression checks."""

from copy import deepcopy

import pytest

from scripts.teacher0014_behavior_audit import ARMS, audit_behavior
from scripts.teacher0014_suite_audit import audit_manifest
from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest


@pytest.fixture(scope="module")
def complete_artifacts():
    manifest = suite_manifest(evaluation_suite(74117, size=2), seed=74117,
                              group_count=2)
    digest = audit_manifest(manifest)["manifest_sha256"]
    runs = {}
    for arm in ARMS:
        runs[arm] = {}
        for replicate in ("0", "1"):
            runs[arm][replicate] = {"panels": {
                name: [{"render_id": row["render_id"],
                        "prediction": 16 if arm == "raw_null" else row["answer"]}
                       for row in rows]
                for name, rows in manifest["panels"].items()}}
    return manifest, {"experiment": "TEACH-0014",
                      "manifest_sha256": digest, "runs": runs}


def test_complete_artifact_scores_every_panel_without_implied_raw_advantage(
        complete_artifacts):
    manifest, artifact = complete_artifacts
    result = audit_behavior(manifest, artifact)
    assert result["audit"] == "pass"
    assert result["scope"] == "answer_behavior_only"
    assert result["decisions"]["oracle_twohop"]
    assert result["decisions"]["null_valid"]
    assert not result["decisions"]["raw_twohop_and_advantage"]
    assert result["decisions"]["primary"] == "RAW_BEHAVIOR_NOT_QUALIFIED"
    assert result["scores"]["latent_rows_answer"]["0"]["panels"][
        "factorial"]["groups"]["total"] == 2


def test_reject_missing_reordered_and_invalid_prediction(complete_artifacts):
    manifest, artifact = complete_artifacts
    missing = deepcopy(artifact)
    del missing["runs"]["latent_rows_answer"]["0"]["panels"]["factorial"][-1]
    with pytest.raises(ValueError, match="prediction count"):
        audit_behavior(manifest, missing)
    reordered = deepcopy(artifact)
    rows = reordered["runs"]["latent_rows_answer"]["0"]["panels"]["factorial"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="invalid identity"):
        audit_behavior(manifest, reordered)
    invalid = deepcopy(artifact)
    invalid["runs"]["latent_rows_answer"]["0"]["panels"]["factorial"][0][
        "prediction"] = True
    with pytest.raises(ValueError, match="invalid identity"):
        audit_behavior(manifest, invalid)


def test_null_leakage_overrides_all_other_behavior(complete_artifacts):
    manifest, artifact = complete_artifacts
    leaked = deepcopy(artifact)
    for replicate in ("0", "1"):
        for name, rows in manifest["panels"].items():
            for source, prediction in zip(
                    rows, leaked["runs"]["raw_null"][replicate]["panels"][name],
                    strict=True):
                prediction["prediction"] = source["answer"]
    result = audit_behavior(manifest, leaked)
    assert result["decisions"]["primary"] == "INVALID_NULL_LEAKAGE"
