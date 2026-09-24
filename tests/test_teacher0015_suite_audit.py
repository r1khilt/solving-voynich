"""Fresh three-recipient generator and independent structural audit fixtures."""

from copy import deepcopy

import pytest

from scripts.teacher0015_suite_audit import audit_manifest, audit_splits
from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest
from voynich.workspace.teacher15_tasks import generate_split, split_manifest


@pytest.fixture(scope="module")
def manifests():
    return (
        split_manifest(generate_split("discovery", 2), "discovery"),
        split_manifest(generate_split("confirmation", 2), "confirmation"),
    )


def test_three_recipient_groups_have_complete_visible_factorial(manifests):
    discovery, confirmation = manifests
    for split, manifest in zip(("discovery", "confirmation"), manifests,
                               strict=True):
        audit = audit_manifest(manifest)
        assert audit["split"] == split
        assert audit["groups"] == 2
        assert audit["episodes"] == 132
        for group in manifest["groups"]:
            assert len(group["cells"]) == 66
            assert len({value for pair in group["recipient_outputs"]
                        for value in pair}) == 6
            assert group["donor_answer"] == group["recipient_outputs"][0][1]
    exposed = suite_manifest(evaluation_suite(74117, 1), seed=74117,
                             group_count=1)
    pair = audit_splits(discovery, confirmation, [exposed])
    assert pair["audit"] == "pass"
    assert len(pair["teacher14_exposure_sha256"]) == 1


def test_counterfactual_tampering_fails_closed(manifests):
    discovery, _ = manifests
    broken = deepcopy(discovery)
    broken["groups"][0]["recipient_outputs"][1][1] += 1
    with pytest.raises(ValueError, match="distinctions invalid"):
        audit_manifest(broken)
    broken = deepcopy(discovery)
    broken["groups"][0]["cells"].pop()
    with pytest.raises(ValueError, match="Incomplete 66-cell"):
        audit_manifest(broken)
    broken = deepcopy(discovery)
    broken["groups"][0]["cells"][0]["episode"]["stage_partitions"] = [
        "train", "confirm"]
    with pytest.raises(ValueError, match="Split or identity hash mismatch"):
        audit_manifest(broken)
    broken = deepcopy(discovery)
    broken["groups"][0]["group_id"] = "0" * 64
    with pytest.raises(ValueError, match="group ID"):
        audit_manifest(broken)


def test_generation_is_repeatable_without_model_state(manifests):
    discovery, confirmation = manifests
    assert split_manifest(generate_split("discovery", 2), "discovery") == discovery
    assert split_manifest(generate_split("confirmation", 2),
                          "confirmation") == confirmation
