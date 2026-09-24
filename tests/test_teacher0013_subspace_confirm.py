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
confirmation_audit = _load(
    "teacher0013_subspace_confirmation_audit_parity_dependency",
    "scripts/teacher0013_subspace_confirmation_audit.py")
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
        "order": {"selection": 1}}}
    monkeypatch.setattr(confirm, "CONTROL_COUNT", 1)
    rows, controls = confirm.run_seed(
        net, confirmation_groups, spec, discovery, geometry, "0",
        lambda *args: None, device="cpu")
    conditions = {row["condition"] for row in rows}
    assert {"content_selected_marked", "content_complement_marked",
            "content_haar_00_marked", "content_deranged_00_marked",
            "content_equal_energy_00_marked", "content_mean_ablation_f0",
            "content_mean_ablation_f1", "content_equal_norm_corruption_f0",
            "content_equal_norm_corruption_f1", "content_native_restoration_f0",
            "content_native_restoration_f1",
            "content_first_hop", "content_direct", "content_copy"} <= conditions
    assert len(controls["content"]["haar"]) == 1
    assert len(controls["content"]["deranged"]) == 1
    assert len(controls["physical_order_oracles"]) == 24
    shortcut = confirm.score_order_shortcut(
        rows, controls["physical_order_oracles"], "order")
    assert shortcut["oracle_kind"] == "f_slot"
    assert shortcut["overall"]["registered_items"] == 24
    assert shortcut["overall"]["eligible_items"] <= 24
    decision = confirm.decide_confirmation(
        {"0": rows, "1": rows}, {"0": controls, "1": controls}, discovery)
    assert decision["status"] == "candidate_decisions_pending_independent_artifact_audit"
    assert set(decision["candidate_labels"]) == {
        "content", "cross_seed_content", "binding", "order"}
    audit_controls = {
        seed: {"binding_specificity_errors": [],
               "physical_order_records": controls["physical_order_oracles"]}
        for seed in ("0", "1")}
    assert confirmation_audit.recompute_decisions(
        {"0": rows, "1": rows}, audit_controls, discovery) == decision


def test_cross_seed_content_transport_runs_both_directions_with_exact_identity():
    suite = counterfactual_suite(74431, discovery_groups=2, confirmation_groups=1)
    discovery_groups = [asdict(group) for group in suite["discovery"]]
    confirmation_groups = [asdict(group) for group in suite["confirmation"]]
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    torch.manual_seed(81)
    source = model_for_arm("raw_shallow").eval()
    torch.manual_seed(82)
    target = model_for_arm("raw_shallow").eval()
    panel = factor_state_panel(
        source, discovery_groups, spec, factor_name="content",
        episode_loader=episode_from_record, batch_size=8)
    basis = fit_registered_factor_geometries(panel, panel, panel).content.basis[:, :1]
    width = basis.shape[0]
    rows = confirm.cross_seed_content_rows(
        source, target, confirmation_groups, spec, basis, torch.eye(width),
        lambda *args: None, source_seed="0", target_seed="1", map_name="shared",
        device="cpu")
    assert len(rows) == 6
    assert {row["condition"] for row in rows} == {"content_cross_seed_shared_marked"}
    assert {row["direction"] for row in rows} == {
        "0_to_1_forward", "0_to_1_reverse"}
    assert len({row["item_id"] for row in rows}) == len(rows)


def test_physical_order_overall_group_coverage_keeps_assay_cells_separate():
    records, rows = [], []
    for cell in ("marked_f0", "format_f0"):
        condition = f"physical_order_order_{cell}"
        for recipient in range(3):
            item_id = f"{cell}:{recipient}"
            records.append({
                "component": "order", "condition": condition,
                "logical_group_id": "g0", "recipient": recipient,
                "item_id": item_id, "oracle_valid": True,
                "oracle_target": 20 + recipient, "oracle_kind": "f_slot",
                "oracle_semantic_collision": False,
                "oracle_structurally_valid": True, "surface": cell,
                "assignment": "f0",
            })
            rows.append({"condition": condition, "item_id": item_id,
                         "prediction": 20 + recipient})
    score = confirm.score_order_shortcut(rows, records, "order")["overall"]
    assert score["registered_groups"] == 2
    assert score["eligible_groups"] == 2
    assert score["group_accuracy"] == 1.0
