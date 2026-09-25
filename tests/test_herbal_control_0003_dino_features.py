"""Full-shape synthetic provenance and evaluation-lock controls for DINO features."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("PIL")

from scripts import herbal_control_0003_dino_features as dino
from scripts import herbal_control_0003_dino_feature_audit as dino_audit
from PIL import Image
from transformers import BitImageProcessor


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_manual_dino_preprocessing_matches_saved_processor_recipe(tmp_path) -> None:
    values = (np.arange(37 * 29 * 3, dtype=np.int32) % 256).astype(np.uint8)
    image = Image.fromarray(values.reshape(37, 29, 3))
    path = tmp_path / "synthetic.png"
    image.save(path)
    processor = BitImageProcessor(
        do_resize=True, size={"height": 224, "width": 224}, resample=3,
        do_center_crop=False, do_rescale=True, do_normalize=True,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    library = processor(images=image, return_tensors="pt").pixel_values.numpy()
    manual = dino_audit.manual_tensor(path).numpy()
    assert np.max(np.abs(library - manual)) < 1e-6


def test_dino_preflight_refuses_incomplete_and_tampered_evaluation(tmp_path,
                                                                    monkeypatch) -> None:
    source_panel = json.loads(dino.PANEL.read_text())
    monkeypatch.setattr(dino, "ROOT", tmp_path)
    panel_path = tmp_path / "panel.json"
    _save(panel_path, source_panel)
    monkeypatch.setattr(dino, "PANEL", panel_path)
    monkeypatch.setattr(dino, "PANEL_SHA256", _sha(panel_path))
    source_path = tmp_path / "sources.json"
    _save(source_path, {"synthetic_sources": 216})
    monkeypatch.setattr(dino, "SOURCES", source_path)
    freeze_path = tmp_path / "freeze.json"
    monkeypatch.setattr(dino, "FREEZE", freeze_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_files = {}
    for filename in dino.MODEL_FILES:
        path = model_dir / filename
        path.write_bytes(f"synthetic {filename}".encode())
        model_files[filename] = _sha(path)
    monkeypatch.setattr(dino, "MODEL_DIR", model_dir)
    monkeypatch.setattr(dino, "MODEL_FILES", model_files)

    image_shas = {}
    for split in ("development", "evaluation"):
        rows = []
        for role in (f"{split}_known", f"{split}_unknown"):
            for chapter in source_panel["roles"][role]:
                for manuscript in dino.MANUSCRIPTS:
                    pageid = chapter["pages"][manuscript]["pageid"]
                    crop_name = f"crop_{pageid}.png"
                    crop = tmp_path / crop_name
                    crop.write_bytes(b"synthetic crop " + str(pageid).encode())
                    rows.append({"role": role, "chapter_class": chapter["chapter"],
                                 "manuscript": manuscript, "source_pageid": pageid,
                                 "crop_file": crop_name, "crop_sha256": _sha(crop)})
        assert len(rows) == 108
        image_path = dino.image_manifest(split)
        _save(image_path, {"id": f"HERBAL-CONTROL-0003-{split}",
                           "status": f"complete-pre-score-{split}-crops",
                           "panel_manifest_sha256": dino.PANEL_SHA256,
                           "rendered_crops": 108, "rows": rows})
        image_shas[split] = _sha(image_path)

    with pytest.raises(ValueError, match="freeze"):
        dino.preflight("development")
    _save(freeze_path, {
        "id": "HERBAL-CONTROL-0003-input-freeze",
        "status": "complete-pre-score-inputs",
        "panel_manifest_sha256": dino.PANEL_SHA256,
        "source_pages": 216, "crop_pages": 216,
        "sources_manifest_sha256": _sha(source_path),
        "development_images_sha256": image_shas["development"],
        "evaluation_images_sha256": image_shas["evaluation"],
    })
    rows, image_sha, freeze_sha = dino.preflight("development")
    assert len(rows["rows"]) == 108
    assert image_sha == image_shas["development"]
    assert freeze_sha == _sha(freeze_path)
    with pytest.raises(ValueError, match="requires audited"):
        dino.preflight("evaluation")

    selection_path = tmp_path / "selection.json"
    audit_path = tmp_path / "audit.json"
    development_features_path = tmp_path / "development_features.json"
    feature_audit_path = tmp_path / "feature_audit.json"
    _save(development_features_path, {"synthetic_development_feature": True})
    development_features_sha = _sha(development_features_path)
    _save(selection_path, {
        "id": "HERBAL-CONTROL-0003-DINO-development-selection",
        "status": "development-score-produced-awaiting-audit-and-push",
        "panel_manifest_sha256": dino.PANEL_SHA256,
        "full_input_freeze_sha256": freeze_sha,
        "feature_method": dino.FEATURE_METHOD,
        "development_features_sha256": development_features_sha,
        "primary_method": "two_mean",
    })
    _save(audit_path, {
        "id": "HERBAL-CONTROL-0003-DINO-independent-development-score-audit",
        "status": "pass", "score_sha256": _sha(selection_path),
        "full_input_freeze_sha256": freeze_sha,
        "feature_sha256": development_features_sha,
        "primary_method": "two_mean",
    })
    _save(feature_audit_path, {
        "id": "HERBAL-CONTROL-0003-DINO-development-feature-audit",
        "status": "pass", "feature_sha256": development_features_sha,
        "input_manifest_sha256": image_shas["development"],
        "full_input_freeze_sha256": freeze_sha,
        "model_sha256": model_files["model.safetensors"],
        "compared_distance_cells": 108 * 108,
    })
    with pytest.raises(ValueError, match="requires audited"):
        dino.preflight("evaluation", selection_path, audit_path)
    evaluation, _, _ = dino.preflight(
        "evaluation", selection_path, audit_path,
        development_features_path, feature_audit_path)
    assert len(evaluation["rows"]) == 108
    feature_audit = json.loads(feature_audit_path.read_text())
    feature_audit["compared_distance_cells"] -= 1
    _save(feature_audit_path, feature_audit)
    with pytest.raises(ValueError, match="selection/audit"):
        dino.preflight("evaluation", selection_path, audit_path,
                       development_features_path, feature_audit_path)
    feature_audit["compared_distance_cells"] += 1
    _save(feature_audit_path, feature_audit)
    bad = json.loads(audit_path.read_text())
    bad["score_sha256"] = "wrong"
    _save(audit_path, bad)
    with pytest.raises(ValueError, match="selection/audit"):
        dino.preflight("evaluation", selection_path, audit_path,
                       development_features_path, feature_audit_path)
    bad["score_sha256"] = _sha(selection_path)
    _save(audit_path, bad)
    crop = tmp_path / evaluation["rows"][0]["crop_file"]
    crop.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="crop changed"):
        dino.preflight("evaluation", selection_path, audit_path,
                       development_features_path, feature_audit_path)
