"""CPU smoke checks for the sealed Stage-D subspace confirmation runner."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import torch

from voynich.workspace.teacher12_models import model_for_arm
from voynich.workspace.teacher13_confirm import MediatorSpec
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_subspace import (
    factor_rank_screen,
    factor_state_panel,
    fit_registered_factor_geometries,
)
from voynich.workspace.teacher13_tasks import counterfactual_suite, semantic_layout


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


confirm = _load("teacher0013_subspace_confirm", "scripts/teacher0013_subspace_confirm.py")
campaign = _load("teacher0013_subspace_campaign_test_dependency",
                 "scripts/teacher0013_subspace_campaign.py")
audit = _load("teacher0013_subspace_audit_test_dependency",
              "scripts/teacher0013_subspace_audit.py")
prepare = _load("teacher0013_prepare_test_dependency", "scripts/teacher0013_prepare.py")


def test_stage_d_confirmation_seed_smoke_covers_primary_controls_and_specificity(monkeypatch):
    suite = counterfactual_suite(74411, discovery_groups=2, confirmation_groups=1)
    discovery_groups = []
    for group in suite["discovery"]:
        record = asdict(group)
        record["semantic_layouts"] = {
            name: [asdict(semantic_layout(episode))
                   for episode in prepare._episodes(group, name)]
            for name in prepare.VARIANTS}
        discovery_groups.append(record)
    confirmation_groups = [asdict(group) for group in suite["confirmation"]]
    torch.manual_seed(71)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    panels = {name: factor_state_panel(
        net, discovery_groups, spec, factor_name=name,
        episode_loader=episode_from_record, batch_size=8)
        for name in ("content", "binding", "order")}
    fitted = fit_registered_factor_geometries(
        panels["content"], panels["binding"], panels["order"])
    geometry = campaign.geometry_payload(panels, fitted)
    audit.verify_panel_binding(
        geometry["panels"]["content"], "content", discovery_groups, asdict(spec))
    rank_rows = factor_rank_screen(
        net, discovery_groups, spec, geometry["orthogonal"]["forward"]["content"][:, :1],
        factor_name="content", episode_loader=episode_from_record, candidate_ranks=(1,))
    audit.verify_rank_grid(rank_rows, "content", discovery_groups)
    discovery = {"rank_selections": {
        "content": {"selection": 1}, "binding": {"selection": None},
        "order": {"selection": None}}}
    monkeypatch.setattr(confirm, "CONTROL_COUNT", 1)
    rows, controls = confirm.run_seed(
        net, confirmation_groups, spec, discovery, geometry, "0",
        lambda *args: None, device="cpu")
    conditions = {row["condition"] for row in rows}
    assert {"content_selected_marked", "content_complement_marked",
            "content_haar_00_marked", "content_deranged_00_marked",
            "content_equal_energy_00_marked", "content_mean_ablation",
            "content_equal_norm_corruption", "content_native_restoration",
            "content_first_hop", "content_direct", "content_copy"} <= conditions
    assert len(controls["content"]["haar"]) == 1
    assert len(controls["content"]["deranged"]) == 1
