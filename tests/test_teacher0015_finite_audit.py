"""Independent vector/identity controls for the future finite key assay."""

import pytest

from scripts.teacher0015_finite_audit import audit_surface
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher15_intervene import evaluate_surface
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
    rows[1]["replacement_vectors"]["random"][0] += 1.0
    with pytest.raises(ValueError, match="random norm mismatch|random vector differs"):
        audit_surface(manifest["groups"][0], rows, distractor=0,
                      marked=True, order=0,
                      wrong_group=manifest["groups"][1],
                      deranged_group=manifest["groups"][2])
