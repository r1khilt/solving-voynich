"""Tests for dynamic-position and subspace interventions."""

import torch

from voynich.workspace.teacher12_models import model_for_arm
from voynich.workspace.teacher13_intervene import (
    attention_sites,
    cached_position_patch,
    numerical_reconstruction,
    position_patch,
    raw_forward,
    residual_sites,
)
from voynich.workspace.teacher13_tasks import counterfactual_suite, semantic_layout


def _group():
    return counterfactual_suite(73141, discovery_groups=1, confirmation_groups=1)[
        "discovery"][0]


def test_site_names_cover_raw_and_looped_execution_depth():
    assert len(residual_sites("raw_deep")) == 13
    assert residual_sites("raw_deep")[0] == "embed"
    assert residual_sites("raw_deep")[-1] == "blocks.11.resid_post"
    assert len(residual_sites("raw_looped")) == 13
    assert residual_sites("raw_looped")[-1] == "passes.2.blocks.3.resid_post"
    assert attention_sites("raw_looped", "result")[4] == \
        "passes.1.blocks.0.attn.result"


def test_identity_dynamic_position_patch_reconstructs_instrumented_logits():
    torch.manual_seed(17)
    net = model_for_arm("raw_shallow").eval()
    group = _group()
    episodes = group.base
    site = "blocks.1.resid_post"
    clean = raw_forward(net, episodes, cache_names=(site,))
    patched = cached_position_patch(net, episodes, episodes, site=site, role="answer")
    assert torch.equal(clean.logits, patched.logits)
    assert numerical_reconstruction(net, episodes, site=site) < 1e-5


def test_full_head_and_subspace_patches_change_only_requested_coordinates():
    group = _group()
    bases, donors = group.base, group.donor
    base_layouts = tuple(semantic_layout(episode) for episode in bases)
    donor_layouts = tuple(semantic_layout(episode) for episode in donors)
    batch, time, width = 3, max(len(ep.tokens) for ep in bases), 8
    base = torch.zeros(batch, time, width)
    donor = torch.randn(batch, time, width)
    full = position_patch(donor, base_layouts, donor_layouts, base_role="query")(base)
    for index, layout in enumerate(base_layouts):
        position = layout.positions("query")[0]
        assert torch.equal(full[index, position], donor[index, position])
        assert torch.count_nonzero(full[index, :position]) == 0

    basis = torch.eye(width)[:, :2]
    subspace = position_patch(
        donor, base_layouts, donor_layouts, base_role="query",
        basis=basis, component="subspace")(base)
    complement = position_patch(
        donor, base_layouts, donor_layouts, base_role="query",
        basis=basis, component="complement")(base)
    assert torch.allclose(subspace + complement, full)

    head_base = torch.zeros(batch, time, 4, width)
    head_donor = torch.randn_like(head_base)
    head_patch = position_patch(
        head_donor, base_layouts, donor_layouts, base_role="answer", heads=(1, 3))
    changed = head_patch(head_base)
    for index, layout in enumerate(base_layouts):
        position = layout.positions("answer")[0]
        assert torch.equal(changed[index, position, 1], head_donor[index, position, 1])
        assert torch.equal(changed[index, position, 3], head_donor[index, position, 3])
        assert torch.count_nonzero(changed[index, position, (0, 2)]) == 0
