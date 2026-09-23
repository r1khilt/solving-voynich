import torch

from voynich.workspace.teacher4_models import DenseRows
from voynich.workspace.teacher5_intervene import build_groups
from voynich.workspace.teacher7_dense_mechanism import _ids, manual_dense
from voynich.workspace.teacher8_components import (
    layer_parts,
    select_heads,
    selected_head_delta,
)


def test_teacher8_explicit_attention_and_layer_match_native():
    torch.manual_seed(23)
    net = DenseRows().eval()
    groups = build_groups(68111, 2)
    ids = _ids(groups, "base")
    with torch.no_grad():
        _, states = manual_dense(net, ids)
        parts = layer_parts(net.transformer.layers[1], states[1])
        native = net.transformer.layers[1](states[1])
    assert parts["errors"]["attention"] < 1e-6
    assert parts["errors"]["layer"] < 1e-6
    assert torch.equal(parts["post_mlp"], native)


def test_teacher8_all_head_deltas_sum_to_attention_difference_at_query():
    torch.manual_seed(29)
    net = DenseRows().eval()
    groups = build_groups(68111, 2)
    with torch.no_grad():
        _, base_states = manual_dense(net, _ids(groups, "base", first_only=True))
        _, donor_states = manual_dense(net, _ids(groups, "donor", first_only=True))
        base = layer_parts(net.transformer.layers[1], base_states[1])
        donor = layer_parts(net.transformer.layers[1], donor_states[1])
        delta = selected_head_delta(net.transformer.layers[1], base["heads"], donor["heads"],
                                   [0, 1, 2, 3])
    assert torch.allclose(base["head_attention"][:, 5] + delta,
                          donor["head_attention"][:, 5], atol=1e-6)


def _cell(groups, items, cross):
    return {"donor_groups": {"accuracy": groups}, "donor_items": {"accuracy": items},
            "cross_g_non_injection": {"accuracy": cross}}


def test_teacher8_selects_smallest_shared_qualified_subset():
    screens = {rep: {str(mask): _cell(0, 0, 1) for mask in range(1, 16)}
               for rep in ("0", "1")}
    for rep in screens:
        screens[rep]["3"] = _cell(.9, .9, 1)
        screens[rep]["1"] = _cell(.7, .8, 1)
        screens[rep]["2"] = _cell(.8, .85, 1)
    selected = select_heads(screens)
    assert selected["mask"] == 2
    assert selected["heads"] == [1]
    assert selected["discovery_qualified"] is True
