"""Sealing, scoring and exact-logit artifact tests for TEACH-0013 confirmation."""

import importlib.util
import json
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_confirm_runner",
    Path(__file__).parents[1] / "scripts/teacher0013_confirm.py")
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_confirmation_audit",
    Path(__file__).parents[1] / "scripts/teacher0013_confirmation_audit.py")
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
auditor = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(auditor)


def _triple(condition, direction, *, correct=True):
    rows = []
    for recipient in range(3):
        target = 100 + recipient
        prediction = target if correct else 999
        rows.append({
            "condition": condition, "direction": direction,
            "logical_group_id": "group", "group_id": f"group:{direction}",
            "recipient": recipient, "target": target, "prediction": prediction,
            "base_answer": target, "clean_prediction": target,
            "fixed_donor_answer": 100, "target_probability": .8 if correct else .1,
            "base_target_probability": .1, "corruption_changed_clean_prediction": not correct,
        })
    return rows


def test_stage_c_scoring_uses_worst_direction_and_all_four_controls():
    rows = []
    for render in ("marked", "marker_free"):
        for direction in ("forward", "reverse"):
            rows += _triple(f"sufficiency_{render}", direction)
    for condition in runner.CONTROL_CONDITIONS:
        for direction in ("forward", "reverse"):
            rows += _triple(condition, direction, correct=False)
    for direction in ("forward", "reverse"):
        rows += _triple("norm_matched_corruption", direction, correct=False)
        rows += _triple("native_state_rescue", direction)
    for condition in ("clean_base", "clean_donor_input_f_replacement",
                      "same_key_format", "same_key_order", "same_key_distractor",
                      "final_state_fixed_answer_injection"):
        rows += _triple(condition, "aux")
    for task in ("direct", "copy"):
        for family in ("format", "order", "distractor"):
            rows += _triple(f"{task}_same_key_{family}", "specificity")
    scores = runner.score_confirmation_rows(rows)
    assert scores["sufficiency_items_forward"] == 1.0
    assert scores["minimum_control_advantage"] == 1.0
    assert scores["rescue"] == 1.0 and scores["rescue_changed_cases"] == 6
    assert scores["direct_loss"] == scores["copy_loss"] == 0.0
    assert auditor.stage_c_scores(rows) == json.loads(json.dumps(scores))


def test_logit_capture_retains_exact_float_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(auditor, "ROOT", tmp_path)
    capture = runner.LogitCapture()
    edited = torch.randn(3, 17)
    clean = torch.randn(3, 17)
    metadata = tuple({"logical_group_id": "g", "recipient": index, "target": 20 + index}
                     for index in range(3))
    capture("condition", "forward", metadata, edited, clean)
    path = tmp_path / "logits.pt"
    record = capture.write(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert record["records"] == 1 and record["rows"] == 3
    assert torch.equal(payload["records"][0]["edited_symbol_logits"], edited)
    assert torch.equal(payload["records"][0]["clean_symbol_logits"], clean)
    probability, clean_probability = edited.softmax(-1), clean.softmax(-1)
    rows = []
    for index in range(3):
        prediction = int(edited[index].argmax().item() + 16)
        target = metadata[index]["target"]
        rows.append({
            "condition": "condition", "direction": "forward",
            "logical_group_id": "g", "group_id": "g:forward", "recipient": index,
            "target": target, "prediction": prediction, "base_answer": target,
            "clean_prediction": int(clean[index].argmax().item() + 16),
            "fixed_donor_answer": target, "candidate_member": True,
            "wrong_destination": "target",
            "target_probability": float(probability[index, target - 16].item()),
            "base_target_probability": float(clean_probability[index, target - 16].item()),
            "target_logit": float(edited[index, target - 16].item()),
            "base_target_logit": float(clean[index, target - 16].item()),
            "prediction_logit": float(edited[index, prediction - 16].item()),
        })
    assert auditor.verify_logit_archive(record, rows)["rows"] == 3


def test_single_site_prerequisite_requires_matching_independent_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(auditor, "ROOT", tmp_path)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    residual_path = result_dir / "discovery-decision.json"
    residual = {
        "status": "stage_b_single_site_complete", "suite_gzip_sha256": "suite",
        "campaign_source_sha256": {}, "elapsed_seconds": 10,
        "materialized_activation_bytes": 20,
        "residual_selection": {"selection": {
            "cut_index": 2, "semantic_label": "query", "qualified": True}},
    }
    residual_path.write_text(json.dumps(residual))
    audit = {
        "audit": "pass", "scope": "residual_discovery",
        "discovery_decision_sha256": runner.sha_file(residual_path),
        "primary_selection": residual["residual_selection"]["selection"],
    }
    (result_dir / "discovery-audit.json").write_text(json.dumps(audit))
    spec, provenance = runner.load_discovery_prerequisite(
        result_dir, {"suite_gzip_sha256": "suite", "source_sha256": {}})
    assert spec.kind == "single" and spec.site == "blocks.1.resid_post"
    assert provenance["stage_b_materialized_bytes"] == 20
    auditor.verify_discovery_prerequisite({
        "mediator": {
            "kind": "single", "site": "blocks.1.resid_post", "label": "query",
            "early_site": None, "source_label": None, "late_site": None,
            "destination_label": None,
        },
        "discovery_prerequisite": provenance,
    })

    audit["primary_selection"] = None
    (result_dir / "discovery-audit.json").write_text(json.dumps(audit))
    try:
        runner.load_discovery_prerequisite(
            result_dir, {"suite_gzip_sha256": "suite", "source_sha256": {}})
    except runner.ConfirmationError as exc:
        assert "selection mismatch" in str(exc)
    else:
        raise AssertionError("Mismatched independent selection was accepted")
