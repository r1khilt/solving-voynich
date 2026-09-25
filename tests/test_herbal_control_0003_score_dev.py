"""End-to-end synthetic control for the development-only score gate."""

import hashlib
import json

import pytest

from scripts import herbal_control_0003_score_dev as scoring


def _save(path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_score_uses_only_frozen_inputs_and_rejects_changed_holdout(tmp_path,
                                                                               monkeypatch) -> None:
    panel_path = tmp_path / "panel.json"
    sources_path = tmp_path / "sources.json"
    dev_path = tmp_path / "dev.json"
    eval_path = tmp_path / "eval.json"
    freeze_path = tmp_path / "freeze.json"
    feature_path = tmp_path / "feature.json"
    known = [f"known_{i:02d}" for i in range(24)]
    unknown = [f"unknown_{i:02d}" for i in range(12)]
    panel_hash = _save(panel_path, {"roles": {
        "development_known": [{"chapter": name} for name in known],
        "development_unknown": [{"chapter": name} for name in unknown],
    }})
    source_hash = _save(sources_path, {"synthetic": True})
    eval_hash = _save(eval_path, {"holdout_crop_hash_only": True})
    rows = [{"role": f"development_{role}", "chapter_class": name,
             "manuscript": manuscript, "crop_file": f"{name}_{manuscript}.png",
             "crop_sha256": f"hash_{name}_{manuscript}"}
            for role, chapters in (("known", known), ("unknown", unknown))
            for name in chapters for manuscript in scoring.MANUSCRIPTS]
    dev_hash = _save(dev_path, {
        "id": "HERBAL-CONTROL-0003-development", "status": "complete-pre-score-development-crops",
        "panel_manifest_sha256": panel_hash, "rendered_crops": 108, "rows": rows,
    })
    freeze_hash = _save(freeze_path, {
        "id": "HERBAL-CONTROL-0003-input-freeze", "status": "complete-pre-score-inputs",
        "panel_manifest_sha256": panel_hash, "sources_manifest_sha256": source_hash,
        "development_images_sha256": dev_hash, "evaluation_images_sha256": eval_hash,
        "source_pages": 216, "crop_pages": 216,
    })
    distances = [[0.0 if i == j else 0.1 if left["chapter_class"] == right["chapter_class"]
                  else 1.0 for j, right in enumerate(rows)] for i, left in enumerate(rows)]
    _save(feature_path, {
        "id": "HERBAL-CONTROL-0003-development",
        "feature_method": "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit",
        "input_manifest_sha256": dev_hash,
        "full_input_freeze_sha256": freeze_hash,
        "row_order": rows, "distance_matrix": distances,
    })
    for name, value in (("PANEL", panel_path), ("SOURCES", sources_path),
                        ("DEV_IMAGES", dev_path), ("EVAL_IMAGES", eval_path), ("FREEZE", freeze_path),
                        ("PANEL_SHA256", panel_hash)):
        monkeypatch.setattr(scoring, name, value)
    result = scoring.score(feature_path)
    assert result["primary_method"] == "single_first"
    assert all(item["batch_macro_ba"] == 1.0 for item in result["methods"].values())
    assert result["full_input_freeze_sha256"] == freeze_hash
    eval_path.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="input freeze"):
        scoring.score(feature_path)
