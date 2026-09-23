import torch

from voynich.workspace.teacher4_models import DenseRows
from voynich.workspace.teacher5_intervene import build_groups
from voynich.workspace.teacher10_answer_readout import prepare as prepare_answer_run
from voynich.workspace.teacher11_g_readout import (
    Config,
    hybrid_query,
    qkv_parts,
    select_slots,
)


def test_teacher11_query_qkv_reconstructs_native_weight_heads():
    torch.manual_seed(53)
    net = DenseRows().eval()
    groups = build_groups(71111, 3)
    with torch.no_grad():
        prepared = prepare_answer_run(net, groups, Config())
        base = qkv_parts(net.transformer.layers[2], prepared["base_states"][2])
        edited = qkv_parts(net.transformer.layers[2], prepared["edited_cut2"])
    assert base["head_reconstruction_error"] < 1e-6
    assert edited["head_reconstruction_error"] < 1e-6
    assert torch.allclose(base["contributions"].sum(2), base["head_query"],
                          atol=1e-6, rtol=0)


def test_teacher11_hybrid_endpoints_match_base_and_edited():
    torch.manual_seed(59)
    net = DenseRows().eval()
    groups = build_groups(71111, 3)
    with torch.no_grad():
        prepared = prepare_answer_run(net, groups, Config())
        base = qkv_parts(net.transformer.layers[2], prepared["base_states"][2])
        edited = qkv_parts(net.transformer.layers[2], prepared["edited_cut2"])
        bbb, _, _ = hybrid_query(base, edited, "BBB")
        ddd, _, _ = hybrid_query(base, edited, "DDD")
    assert torch.allclose(bbb, base["head_query"], atol=1e-7, rtol=0)
    assert torch.allclose(ddd, edited["head_query"], atol=1e-7, rtol=0)


def _cell(groups, items):
    return {"donor_groups": {"accuracy": groups}, "donor_items": {"accuracy": items}}


def test_teacher11_selects_smallest_shared_source_subset():
    screens = {rep: {str(mask): _cell(0, 0) for mask in range(1, 64)}
               for rep in ("0", "1")}
    for rep in screens:
        screens[rep]["12"] = _cell(.9, .9)
        screens[rep]["4"] = _cell(.7, .8)
        screens[rep]["8"] = _cell(.8, .85)
    selected = select_slots(screens)
    assert selected["mask"] == 8
    assert selected["slots"] == [3]
    assert selected["discovery_qualified"] is True
