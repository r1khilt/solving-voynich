"""Pure selection and artifact-shape checks for the Stage-D campaign runner."""

import importlib.util
from pathlib import Path

import torch

from voynich.workspace.teacher13_geometry import blocked_contrast_geometry


_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_subspace_campaign",
    Path(__file__).parents[1] / "scripts/teacher0013_subspace_campaign.py")
assert _SPEC is not None and _SPEC.loader is not None
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_subspace_audit",
    Path(__file__).parents[1] / "scripts/teacher0013_subspace_audit.py")
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(audit)


def _summary(rank1, rank2, full=1.0):
    def cell(item, group, gain):
        return {"item_accuracy": item, "group_accuracy": group,
                "mean_probability_gain": gain}
    return {"full": cell(1, 1, full), "1": cell(*rank1), "2": cell(*rank2)}


def test_joint_rank_decision_requires_both_seeds_and_registered_effect_fraction():
    summaries = {
        "0": _summary((.81, .61, .96), (.9, .8, .99)),
        "1": _summary((.7, .5, .96), (.85, .7, .97)),
    }
    result = campaign.joint_rank_decision(summaries, "content")
    assert result["selection"] == 2
    summaries["1"]["full"]["mean_probability_gain"] = 0
    stopped = campaign.joint_rank_decision(summaries, "content")
    assert stopped["selection"] is None
    assert stopped["reason"] == "nonpositive_full_probability_gain"


def test_rank_measurements_excludes_full_reference():
    summary = _summary((.8, .6, .95), (.9, .8, .99))
    rows = campaign.rank_measurements(summary)
    assert [row["rank"] for row in rows] == [1, 2]


def test_independent_auditor_reproduces_joint_rank_selection():
    summaries = {
        "0": _summary((.81, .61, .96), (.9, .8, .99)),
        "1": _summary((.7, .5, .96), (.85, .7, .97)),
    }
    assert audit.joint_selection(summaries, "content") \
        == campaign.joint_rank_decision(summaries, "content")


def test_independent_auditor_reproduces_blocked_geometry_subspace():
    states, factor, nuisance, blocks = [], [], [], []
    for block, direction in (("a", torch.tensor([1., 0., 0.])),
                             ("b", torch.tensor([0., 2., 0.]))):
        for nuisance_cell, offset in (("x", torch.tensor([0., 0., 1.])),
                                      ("y", torch.tensor([0., 0., -1.]))):
            for level, sign in ((0, -1.), (1, 1.)):
                states.append(offset + sign * direction)
                factor.append(level)
                nuisance.append(nuisance_cell)
                blocks.append(block)
    states = torch.stack(states)
    expected = blocked_contrast_geometry(states, factor, nuisance, blocks)
    actual = audit.blocked_geometry({
        "states": states, "factor": factor, "nuisance": nuisance, "blocks": blocks})
    assert torch.allclose(actual["mean"], expected.mean)
    assert torch.allclose(actual["eigenvalues"], expected.eigenvalues)
    assert torch.allclose(actual["basis"] @ actual["basis"].T,
                          expected.basis @ expected.basis.T)
