"""Synthetic affine-folio and frozen-evaluation controls."""

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.herbal_control_0003_folio_feature_audit import replay_maps, replay_matrix
from scripts import herbal_control_0003_folio_feature_audit as audit_module
from scripts import herbal_control_0003_folio_features as feature_module
from voynich.herbal_folio import (
    directed_matrix,
    fit_development_maps,
    folio_side,
    make_direction_panels,
)


def _title(position: int) -> str:
    return f"File:synthetic, f.{position // 2:03d}{'v' if position % 2 else 'r'}.jpg"


def _panel() -> dict:
    roles = {}
    for start, role in ((0, "development_known"), (24, "development_unknown"),
                        (36, "evaluation_known"), (60, "evaluation_unknown")):
        count = 24 if role.endswith("known") and not role.endswith("unknown") else 12
        roles[role] = [{
            "chapter": f"synthetic-{index}",
            "pages": {"bnf": {"title": _title(20 + 2 * index), "pageid": 100000 + index},
                      "egerton": {"title": _title(40 + 4 * index), "pageid": 101000 + index},
                      "casanatense": {"title": _title(60 + 6 * index), "pageid": 102000 + index}},
        } for index in range(start, start + count)]
    return {"roles": roles}


def test_affine_maps_use_development_known_only_and_evaluation_is_frozen() -> None:
    panel = _panel()
    fits = fit_development_maps(panel)
    assert fits["bnf->egerton"] == {"slope": 2.0, "intercept": 0.0}
    assert fits["bnf->casanatense"] == {"slope": 3.0, "intercept": 0.0}
    first, second, truths, scales = make_direction_panels(panel, "development", fits)
    assert len(first["bnf"]) == len(second["bnf"]) == 36
    assert len(first["bnf"][0]) == 24
    assert truths["bnf"] == list(range(24)) + [None] * 12
    assert first["bnf"][0][0] == 0
    evaluation, _, _, returned = make_direction_panels(panel, "evaluation", fits, scales)
    assert returned == scales and evaluation["bnf"][0][0] == 0
    changed = copy.deepcopy(panel)
    changed["roles"]["evaluation_known"][0]["pages"]["bnf"]["title"] = _title(199)
    assert fit_development_maps(changed) == fits
    assert make_direction_panels(changed, "development", fits)[3] == scales
    assert make_direction_panels(changed, "evaluation", fits, scales)[0] != evaluation
    with pytest.raises(ValueError, match="frozen development normalizers"):
        make_direction_panels(panel, "evaluation", fits)
    rows = [{"role": role, "chapter_class": item["chapter"], "manuscript": manuscript}
            for role in ("development_known", "development_unknown")
            for item in panel["roles"][role]
            for manuscript in ("bnf", "egerton", "casanatense")]
    matrix = directed_matrix(panel, "development", rows, fits)
    assert len(matrix) == len(matrix[0]) == 108
    assert matrix[0][1] == 0 and matrix[1][0] == 0
    assert matrix[0][4] == 4 and matrix[4][0] == 2
    assert replay_maps(panel) == fits
    assert np.array_equal(replay_matrix(panel, "development", rows, fits), matrix)
    with pytest.raises(ValueError, match="rows differ"):
        directed_matrix(panel, "development", rows[:-1], fits)


def test_folio_parser_rejects_missing_or_ambiguous_coordinate() -> None:
    assert folio_side("File:A f.013v.jpg") == 27
    with pytest.raises(ValueError, match="one folio"):
        folio_side("File:A f.013r and f.013v.jpg")
    with pytest.raises(ValueError, match="one folio"):
        folio_side("File:A, unnumbered.jpg")


def _save(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_folio_full_freeze_and_both_development_gates(tmp_path, monkeypatch) -> None:
    panel = _panel()
    panel_path = tmp_path / "panel.json"
    _save(panel_path, panel)
    monkeypatch.setattr(feature_module, "ROOT", tmp_path)
    monkeypatch.setattr(feature_module, "PANEL", panel_path)
    monkeypatch.setattr(feature_module, "PANEL_SHA256", _sha(panel_path))
    source_path = tmp_path / "sources.json"
    _save(source_path, {"synthetic_source_count": 216})
    monkeypatch.setattr(feature_module, "SOURCES", source_path)
    freeze_path = tmp_path / "freeze.json"
    monkeypatch.setattr(feature_module, "FREEZE", freeze_path)
    image_shas = {}
    for split in ("development", "evaluation"):
        rows = []
        for role in (f"{split}_known", f"{split}_unknown"):
            for item in panel["roles"][role]:
                for manuscript in ("bnf", "egerton", "casanatense"):
                    pageid = item["pages"][manuscript]["pageid"]
                    crop = tmp_path / f"crop_{pageid}.png"
                    crop.write_bytes(str(pageid).encode())
                    rows.append({"role": role, "chapter_class": item["chapter"],
                                 "manuscript": manuscript, "source_pageid": pageid,
                                 "crop_file": crop.name, "crop_sha256": _sha(crop)})
        image_path = feature_module.images_path(split)
        _save(image_path, {"id": f"HERBAL-CONTROL-0003-{split}",
                           "status": f"complete-pre-score-{split}-crops",
                           "panel_manifest_sha256": feature_module.PANEL_SHA256,
                           "rendered_crops": 108, "rows": rows})
        image_shas[split] = _sha(image_path)
    with pytest.raises(ValueError, match="freeze"):
        feature_module.preflight("development")
    _save(freeze_path, {"id": "HERBAL-CONTROL-0003-input-freeze",
                        "status": "complete-pre-score-inputs",
                        "panel_manifest_sha256": feature_module.PANEL_SHA256,
                        "source_pages": 216, "crop_pages": 216,
                        "sources_manifest_sha256": _sha(source_path),
                        "development_images_sha256": image_shas["development"],
                        "evaluation_images_sha256": image_shas["evaluation"]})
    development = feature_module.extract("development")
    development_path = tmp_path / "folio_dev.json"
    _save(development_path, development)
    development_feature_audit = audit_module.audit("development", development_path)
    assert development_feature_audit["status"] == "pass"
    development_feature_audit_path = tmp_path / "folio_feature_audit.json"
    _save(development_feature_audit_path, development_feature_audit)
    with pytest.raises(ValueError, match="requires audited"):
        feature_module.extract("evaluation")

    folio_selection = tmp_path / "folio_selection.json"
    folio_audit = tmp_path / "folio_audit.json"
    apple_selection = tmp_path / "apple_selection.json"
    apple_audit = tmp_path / "apple_audit.json"
    for path, selection_id, audit_path, audit_id, method in (
            (folio_selection, "HERBAL-CONTROL-0003-FOLIO-development-selection",
             folio_audit, "HERBAL-CONTROL-0003-FOLIO-independent-development-score-audit",
             feature_module.FEATURE_METHOD),
            (apple_selection, "HERBAL-CONTROL-0003-development-selection",
             apple_audit, "HERBAL-CONTROL-0003-independent-development-score-audit", None)):
        score = {"id": selection_id,
                 "status": "development-score-produced-awaiting-audit-and-push",
                 "panel_manifest_sha256": feature_module.PANEL_SHA256,
                 "full_input_freeze_sha256": _sha(freeze_path),
                 "development_features_sha256": _sha(development_path) if method else "apple-feature",
                 "primary_method": "two_mean"}
        if method:
            score["feature_method"] = method
        _save(path, score)
        _save(audit_path, {"id": audit_id, "status": "pass",
                           "score_sha256": _sha(path),
                           "full_input_freeze_sha256": _sha(freeze_path),
                           "feature_sha256": score["development_features_sha256"],
                           "primary_method": "two_mean"})
    kwargs = {"folio_features_path": development_path,
              "folio_feature_audit_path": development_feature_audit_path,
              "folio_selection_path": folio_selection, "folio_audit_path": folio_audit,
              "image_selection_path": apple_selection, "image_audit_path": apple_audit}
    evaluation = feature_module.extract("evaluation", **kwargs)
    evaluation_path = tmp_path / "folio_eval.json"
    _save(evaluation_path, evaluation)
    assert audit_module.audit("evaluation", evaluation_path, **kwargs)["status"] == "pass"
    feature_module.verify_evaluation_feature_gate(evaluation_path, **kwargs)
    bad_feature_audit = json.loads(development_feature_audit_path.read_text())
    bad_feature_audit["feature_sha256"] = "wrong"
    _save(development_feature_audit_path, bad_feature_audit)
    with pytest.raises(ValueError, match="feature audit"):
        feature_module.preflight("evaluation", **kwargs)
    bad_feature_audit["feature_sha256"] = _sha(development_path)
    _save(development_feature_audit_path, bad_feature_audit)
    bad = json.loads(apple_audit.read_text())
    bad["score_sha256"] = "wrong"
    _save(apple_audit, bad)
    with pytest.raises(ValueError, match="selection/audit"):
        feature_module.preflight("evaluation", **kwargs)
    bad["score_sha256"] = _sha(apple_selection)
    _save(apple_audit, bad)
    crop = tmp_path / json.loads(feature_module.images_path("evaluation").read_text())["rows"][0]["crop_file"]
    crop.write_bytes(b"changed")
    with pytest.raises(ValueError, match="crop changed"):
        feature_module.preflight("evaluation", **kwargs)
