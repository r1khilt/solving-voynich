import torch

from voynich.workspace.teacher4_models import DenseRows
from voynich.workspace.teacher5_intervene import build_groups
from voynich.workspace.teacher7_dense_mechanism import (
    Config,
    manual_dense,
    same_key_episodes,
    select_site,
)


def test_teacher7_manual_dense_matches_native():
    torch.manual_seed(17)
    net = DenseRows().eval()
    groups = build_groups(67111, 2)
    ids = torch.tensor([tokens for group in groups for tokens in group["base"]])
    with torch.no_grad():
        native = net(ids)[0]
        manual, states = manual_dense(net, ids)
    assert len(states) == 5
    assert torch.equal(native, manual)


def test_teacher7_same_key_distractors_preserve_answer_and_change_surface():
    groups = build_groups(67111, 4)
    episodes = same_key_episodes(groups)
    for group, tokens in zip(groups, episodes.tolist(), strict=True):
        assert tokens[12] == group["query"]
        assert (tokens[2], tokens[4]) != (group["base"][0][2], group["base"][0][4])
        assert tokens[3 if tokens[2] == group["query"] else 5] == group["base_key"]


def _cell(group, item, cross):
    return {"donor_groups": {"accuracy": group}, "donor_items": {"accuracy": item},
            "cross_g_non_injection": {"accuracy": cross}}


def test_teacher7_site_selection_uses_earliest_qualified_cut_and_shared_minimum():
    screens = {rep: {f"{cut}:{slot}": _cell(0.0, 0.0, 1.0)
                     for cut in range(5) for slot in range(6)} for rep in ("0", "1")}
    screens["0"]["2:5"] = _cell(.9, .9, 1.0)
    screens["1"]["2:5"] = _cell(.7, .8, .95)
    screens["0"]["3:4"] = _cell(1.0, 1.0, 1.0)
    screens["1"]["3:4"] = _cell(1.0, 1.0, 1.0)
    selected = select_site(screens)
    assert Config().discovery_groups == 128
    assert selected["discovery_qualified"] is True
    assert (selected["cut"], selected["slot"]) == (2, 5)
