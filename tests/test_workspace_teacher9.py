import torch

from voynich.workspace.teacher4_models import DenseRows
from voynich.workspace.teacher5_intervene import build_groups
from voynich.workspace.teacher7_dense_mechanism import _ids, manual_dense
from voynich.workspace.teacher9_qkv import (
    hybrid_query,
    qkv_parts,
    select_slots,
)


def test_teacher9_qkv_and_source_contributions_reconstruct_heads():
    torch.manual_seed(41)
    net = DenseRows().eval()
    groups = build_groups(69111, 3)
    with torch.no_grad():
        _, states = manual_dense(net, _ids(groups, "base"))
        parts = qkv_parts(net.transformer.layers[1], states[1])
    assert parts["head_reconstruction_error"] < 1e-6
    assert torch.allclose(parts["contributions"].sum(2), parts["head_query"],
                          atol=1e-7, rtol=0)


def test_teacher9_hybrid_endpoints_match_base_and_donor():
    torch.manual_seed(43)
    net = DenseRows().eval()
    groups = build_groups(69111, 3)
    with torch.no_grad():
        _, base_states = manual_dense(net, _ids(groups, "base", first_only=True))
        _, donor_states = manual_dense(net, _ids(groups, "donor", first_only=True))
        base = qkv_parts(net.transformer.layers[1], base_states[1])
        donor = qkv_parts(net.transformer.layers[1], donor_states[1])
        bbb, _, _ = hybrid_query(base, donor, "BBB")
        ddd, _, _ = hybrid_query(base, donor, "DDD")
    assert torch.allclose(bbb, base["head_query"], atol=1e-7, rtol=0)
    assert torch.allclose(ddd, donor["head_query"], atol=1e-7, rtol=0)


def _cell(groups, items, cross):
    return {"donor_groups": {"accuracy": groups}, "donor_items": {"accuracy": items},
            "cross_g_non_injection": {"accuracy": cross}}


def test_teacher9_selects_smallest_shared_source_subset():
    screens = {rep: {str(mask): _cell(0, 0, 1) for mask in range(1, 64)}
               for rep in ("0", "1")}
    for rep in screens:
        screens[rep]["3"] = _cell(.95, .98, 1)
        screens[rep]["1"] = _cell(.7, .8, 1)
        screens[rep]["2"] = _cell(.8, .85, 1)
    selected = select_slots(screens)
    assert selected["mask"] == 2
    assert selected["slots"] == [1]
    assert selected["discovery_qualified"] is True
