"""Focused checks for the independent Stage-D confirmation auditor."""

from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from voynich.workspace.teacher13_geometry import (
    deranged_blocked_bases,
    haar_random_bases,
    orthogonal_factor_geometry,
)
from voynich.workspace.teacher13_tasks import (
    counterfactual_suite,
    physical_order_shortcut_oracle,
    semantic_layout,
)


ROOT = Path(__file__).parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "teacher0013_subspace_confirmation_audit_test_dependency",
        ROOT / "scripts/teacher0013_subspace_confirmation_audit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(item_id, condition, edited, clean, *, render_id="render-0"):
    probability, clean_probability = edited.softmax(-1), clean.softmax(-1)
    target, prediction, clean_prediction = 16, int(edited.argmax()) + 16, int(clean.argmax()) + 16
    return {
        "item_id": item_id, "condition": condition, "direction": "forward",
        "logical_group_id": "group-0", "group_id": "group-0:forward", "recipient": 0,
        "target": target, "prediction": prediction, "clean_prediction": clean_prediction,
        "base_answer": 17, "fixed_donor_answer": 18, "candidate_member": True,
        "wrong_destination": "target" if prediction == target else "other_legal_candidate",
        "target_probability": float(probability[0]),
        "base_target_probability": float(clean_probability[0]),
        "target_logit": float(edited[0]), "base_target_logit": float(clean[0]),
        "prediction_logit": float(edited[prediction - 16]), "base_render_id": render_id,
        "donor_render_id": "donor-0",
    }


def _write_shard(directory, index, row, edited, clean):
    path = directory / f"{index:05d}.pt"
    record = {
        "condition": row["condition"], "direction": row["direction"],
        "logical_group_ids": (row["logical_group_id"],),
        "recipients": (row["recipient"],), "targets": (row["target"],),
        "item_ids": (row["item_id"],),
        "edited_symbol_logits": edited.unsqueeze(0).float(),
        "clean_symbol_logits": clean.unsqueeze(0).float(),
    }
    torch.save({"format": "TEACH-0013-symbol-logit-shard-v1", "record": record}, path)
    return {"path": path.name, "sha256": _digest(path), "bytes": path.stat().st_size,
            "rows": 1}


def test_sharded_logits_verify_exact_values_and_identical_clean_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    clean = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    edited_a = torch.tensor([2.0, 0.0, -1.0], dtype=torch.float32)
    edited_b = torch.tensor([3.0, -2.0, 0.0], dtype=torch.float32)
    rows = [_row("item-a", "content_selected_marked", edited_a, clean),
            _row("item-b", "content_haar_00_marked", edited_b, clean)]
    shards = [_write_shard(tmp_path, 0, rows[0], edited_a, clean),
              _write_shard(tmp_path, 1, rows[1], edited_b, clean)]
    metadata = {"format": "TEACH-0013-sharded-symbol-logits-v1", "shards": shards,
                "records": 2, "rows": 2}
    assert audit.verify_sharded_logits(metadata, rows)["rows"] == 2

    bad_clean = clean.clone()
    bad_clean[1] += 0.25
    rows[1] = _row("item-b", "content_haar_00_marked", edited_b, bad_clean)
    shards[1] = _write_shard(tmp_path, 1, rows[1], edited_b, bad_clean)
    with pytest.raises(audit.AuditError, match="Clean logits changed"):
        audit.verify_sharded_logits(metadata, rows)


def test_core_grid_reconstructs_every_registered_current_condition():
    group = asdict(counterfactual_suite(
        75501, discovery_groups=1, confirmation_groups=1)["confirmation"][0])
    selections = {"content": {"selection": 1}, "binding": {"selection": None},
                  "order": {"selection": None}}
    expected = audit.expected_core_grid([group], selections)
    # 60 selected/complement + 576 matched controls + 18 necessity + 3 specificity.
    assert len(expected) == 657
    item = expected["content:content_selected_marked:forward:"
                    f"{group['group_id']}:2"]
    assert item["base_render_id"] == group["base"][2]["render_id"]
    assert item["donor_render_id"] == group["donor"][0]["render_id"]
    assert item["target"] == group["donor"][2]["answer"]


def test_independent_control_generators_match_registered_algorithms():
    torch.manual_seed(19)
    width, blocks = 7, ("a", "b", "c")
    nuisance, factor, block_labels, states = [], [], [], []
    for block in blocks:
        for nuisance_cell in (("marked", 0), ("marked", 1)):
            for level in (0, 1):
                nuisance.append(nuisance_cell)
                factor.append(level)
                block_labels.append(block)
                states.append(torch.randn(width) + level)
    states = torch.stack(states)
    expected_haar = haar_random_bases(width, 2, 4, seed=73411)
    assert all(torch.equal(left, right) for left, right in zip(
        audit._haar(width, 2, 4, 73411), expected_haar, strict=True))

    expected = deranged_blocked_bases(
        states, factor, nuisance, block_labels, count=3, seed=73411)
    # Supply all raw factors so the auditor also exercises registered weighted
    # orthogonalization after independently rebuilding the deranged covariance.
    native = audit._blocked_basis(states, factor, nuisance, block_labels)
    raw = {name: native[0] for name in ("content", "binding", "order")}
    spectra = {name: native[1] for name in raw}
    fitted = orthogonal_factor_geometry(
        raw["content"], raw["binding"], raw["order"], eigenvalues=spectra)
    geometry = {
        "panels": {"content": {"states": states, "factor": tuple(factor),
                                "nuisance": tuple(nuisance), "blocks": tuple(block_labels)}},
        "geometry": {name: {"basis": raw[name], "eigenvalues": spectra[name]}
                     for name in raw},
        "orthogonal": {"forward": fitted.forward},
    }
    actual = audit._deranged(geometry, "content", 3, 73411)
    for independent, registered in zip(actual, expected, strict=True):
        assert independent[1] == registered.donor_blocks
        expected_raw = dict(raw)
        expected_spectra = dict(spectra)
        expected_raw["content"] = registered.basis
        expected_spectra["content"] = registered.eigenvalues
        expected_fitted = orthogonal_factor_geometry(
            expected_raw["content"], expected_raw["binding"], expected_raw["order"],
            eigenvalues=expected_spectra).forward["content"]
        assert audit._same_subspace(independent[0], expected_fitted)


def test_physical_order_oracle_is_reconstructed_without_importing_generator_code():
    native = counterfactual_suite(
        75503, discovery_groups=1, confirmation_groups=1)["confirmation"][0]
    group = asdict(native)
    group["semantic_layouts"] = {
        name: [asdict(semantic_layout(episode)) for episode in getattr(native, name)]
        for name in ("base", "reordered_base", "donor", "reordered_donor")}
    expected = physical_order_shortcut_oracle(native.base[0], native.reordered_base[0])
    fields = audit._physical_oracle_fields(
        group, "base", "reordered_base", 0, "f_slot")
    assert fields == expected.compact_fields("f_slot")


def test_capability_gate_fails_closed_on_current_core_and_requires_all_extensions():
    rows = [{"condition": "content_selected_marked", "direction": "forward"}]
    controls = {seed: {"equal_energy_state_tensors_retained": False}
                for seed in ("0", "1")}
    requirements = audit.capability_requirements(
        {"status": "stage_d_confirmation_core_complete"}, {"0": rows, "1": rows}, controls)
    assert not any(requirements.values())

    extended = [
        {"condition": "content_cross_seed_shared", "direction": "seed0_to_seed1"},
        {"condition": "content_cross_seed_shared", "direction": "seed1_to_seed0"},
        {"condition": "content_cross_seed_deranged", "direction": "seed0_to_seed1"},
        {"condition": "content_cross_seed_deranged", "direction": "seed1_to_seed0"},
        {"condition": "physical_order_order_marked_f0", "direction": "forward"},
    ]
    report = {"status": "stage_d_confirmation_complete_pending_audit",
              "decisions": {
                  "status": "candidate_decisions_pending_independent_artifact_audit",
                  "candidate_labels": {name: "placeholder" for name in (
                      "content", "binding", "order", "cross_seed_content")}}}
    controls = {seed: {"equal_energy_state_tensors_retained": True}
                for seed in ("0", "1")}
    assert all(audit.capability_requirements(
        report, {"0": extended, "1": extended}, controls,
        decision_match=True, restoration_exact=True).values())


def test_factor_scores_are_recomputed_from_item_rows():
    rows = []
    for condition in ("content_selected_marked", "content_haar_00_marked"):
        for recipient in range(3):
            rows.append({
                "condition": condition, "direction": "forward", "group_id": "g:forward",
                "recipient": recipient, "prediction": 20 + recipient, "target": 20 + recipient,
                "fixed_donor_answer": 20, "target_probability": 0.8,
                "base_target_probability": 0.2,
            })
    scores = audit.score_factor(rows, "content")
    assert scores["primary"]["content_selected_marked:forward"]["item_accuracy"] == 1.0
    assert scores["controls"]["haar"]["content_haar_00_marked:forward"][
        "mean_probability_gain"] == pytest.approx(0.6)


def test_three_recipient_summary_rejects_duplicate_rows():
    rows = [{
        "group_id": "g:forward", "recipient": recipient,
        "prediction": 20 + recipient, "target": 20 + recipient,
        "fixed_donor_answer": 20, "target_probability": .8,
        "base_target_probability": .2,
    } for recipient in (0, 1, 2)]
    assert audit.summarize(rows)["group_accuracy"] == 1.0
    with pytest.raises(audit.AuditError, match="Incomplete three-recipient"):
        audit.summarize(rows + [{**rows[0]}])
