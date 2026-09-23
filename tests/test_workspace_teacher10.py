import torch

from voynich.workspace.teacher10_answer_readout import (
    cyclic_recipient,
    repeat_g0,
    select_heads,
)


def _cell(groups, items):
    return {"donor_groups": {"accuracy": groups}, "donor_items": {"accuracy": items}}


def test_teacher10_selects_smallest_shared_qualified_head_subset():
    screens = {rep: {str(mask): _cell(0, 0) for mask in range(1, 16)}
               for rep in ("0", "1")}
    for rep in screens:
        screens[rep]["3"] = _cell(.9, .9)
        screens[rep]["1"] = _cell(.7, .8)
        screens[rep]["2"] = _cell(.8, .85)
    selected = select_heads(screens)
    assert selected["mask"] == 2
    assert selected["heads"] == [1]
    assert selected["discovery_qualified"] is True


def test_teacher10_cyclic_recipient_stays_within_group_and_preserves_norms():
    delta = torch.arange(24, dtype=torch.float32).view(6, 4) + 1
    shifted = cyclic_recipient(delta, groups=2)
    assert torch.allclose(shifted[0] / shifted[0].norm(), delta[1] / delta[1].norm())
    assert torch.allclose(shifted[2] / shifted[2].norm(), delta[0] / delta[0].norm())
    assert torch.allclose(shifted.norm(dim=-1), delta.norm(dim=-1))


def test_teacher10_repeat_g0_reuses_first_recipient():
    delta = torch.arange(24, dtype=torch.float32).view(6, 4)
    repeated = repeat_g0(delta, groups=2)
    assert torch.equal(repeated[:3], delta[0].expand(3, -1))
    assert torch.equal(repeated[3:], delta[3].expand(3, -1))
