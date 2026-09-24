"""Fresh cross-order suite identity and visible-pair safeguards."""

from copy import deepcopy

import pytest

from scripts.teacher0015_suite_audit import audit_group
from scripts.teacher0016_suite_audit import _index, audit_manifest
from voynich.workspace.teacher16_tasks import generate_split, split_manifest


def test_new_seeded_groups_reproduce_and_have_valid_visible_cells():
    first = generate_split("discovery", 2)
    repeated = generate_split("discovery", 2)
    other = generate_split("confirmation", 2)
    assert [group.group_id for group in first] == [
        group.group_id for group in repeated]
    assert {group.group_id for group in first}.isdisjoint(
        {group.group_id for group in other})
    manifest = split_manifest(first, "discovery")
    assert manifest["seed"] == 86111
    assert manifest["namespace"] == "TEACH-0016-cross-order-v1"
    assert all(audit_group(group, "discovery")["group_id"] ==
               group["group_id"] for group in manifest["groups"])
    with pytest.raises(ValueError, match="count"):
        audit_manifest(manifest)
    corrupted = deepcopy(manifest)
    corrupted["seed"] = 85111
    with pytest.raises(ValueError, match="seed/count"):
        audit_manifest(corrupted)


def test_cross_order_slot_is_visible_and_unique():
    group = split_manifest(generate_split("discovery", 1),
                           "discovery")["groups"][0]
    cells = [cell for cell in group["cells"] if (
        cell["f"], cell["g"], cell["distractor"], cell["marked"],
        cell["task"]) == (0, 0, 0, True, "composed")]
    assert len(cells) == 2
    positions = [_index(cell["episode"], group["key1"],
                        group["recipient_outputs"][0][1])
                 for cell in cells]
    assert all(0 <= position < len(cells[0]["episode"]["serialized_rows"])
               for position in positions)
    broken = deepcopy(cells[0]["episode"])
    broken["serialized_rows"][positions[0]] = [group["key1"], -1]
    with pytest.raises(ValueError, match="missing/duplicate"):
        _index(broken, group["key1"], group["recipient_outputs"][0][1])
