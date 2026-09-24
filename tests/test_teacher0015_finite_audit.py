"""Independent vector/identity controls for the future finite key assay."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.teacher0015_finite_audit import (
    audit_control_indices, audit_control_permutations, audit_surface,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher15_intervene import evaluate_surface
from voynich.workspace.teacher15_pairs import control_indices, control_permutations
from voynich.workspace.teacher15_tasks import generate_split, split_manifest


def _donor(group):
    return next(cell.episode for cell in group.cells if (
        cell.f, cell.g, cell.distractor, cell.marked,
        cell.order, cell.task) == (1, 0, 0, True, 0, "composed"))


def test_finite_surface_auditor_checks_actual_replacement_vectors():
    groups = generate_split("discovery", 3)
    manifest = split_manifest(groups, "discovery")
    model = CandidateEdgeWorkspace()
    rows = evaluate_surface(
        model, groups[0], distractor=0, marked=True, order=0,
        device="cpu", wrong_donor=_donor(groups[1]),
        deranged_donor=_donor(groups[2]))
    result = audit_surface(
        manifest["groups"][0], rows, distractor=0,
        marked=True, order=0, wrong_group=manifest["groups"][1],
        deranged_group=manifest["groups"][2])
    assert result["audit"] == "pass"
    assert result["recipient_rows"] == 3
    compact = deepcopy(rows)
    for row in compact:
        for name in tuple(row):
            if name.endswith("_logits"):
                del row[name]
    assert audit_surface(
        manifest["groups"][0], compact, distractor=0,
        marked=True, order=0, wrong_group=manifest["groups"][1],
        deranged_group=manifest["groups"][2],
        full_logits=False)["audit"] == "pass"
    rows[1]["replacement_vectors"]["random"][0] += 1.0
    with pytest.raises(ValueError, match="random norm mismatch|random vector differs"):
        audit_surface(manifest["groups"][0], rows, distractor=0,
                      marked=True, order=0,
                      wrong_group=manifest["groups"][1],
                      deranged_group=manifest["groups"][2])


def test_control_pair_assignment_has_independent_equivalent_audit():
    groups = [SimpleNamespace(group_id=f"{index:064x}", key1=index + 16,
                              split="discovery") for index in range(128)]
    records = [vars(group) for group in groups]
    wrong_perm, deranged_perm = control_permutations(groups)
    assert (wrong_perm, deranged_perm) == audit_control_permutations(records)
    assert len(set(wrong_perm)) == len(set(deranged_perm)) == 128
    for index in range(128):
        wrong, deranged = control_indices(groups, index)
        assert (wrong, deranged) == audit_control_indices(records, index)
        assert len({index, wrong, deranged}) == 3


@pytest.mark.parametrize("split", ["discovery", "confirmation"])
def test_frozen_group_control_permutations_are_bijective(split):
    groups = generate_split(split)
    records = split_manifest(groups, split)["groups"]
    wrong, deranged = control_permutations(groups)
    assert (wrong, deranged) == audit_control_permutations(records)
    assert len(set(wrong)) == len(set(deranged)) == 128
    assert all(groups[index].key1 != groups[wrong[index]].key1 and
               groups[index].key1 != groups[deranged[index]].key1 and
               wrong[index] != deranged[index] for index in range(128))
