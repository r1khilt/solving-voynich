"""Full-shape synthetic control for frozen evaluation decisions."""

from hashlib import sha256
import json

import pytest

from scripts import herbal_control_0003_score_eval as evaluation
from scripts import herbal_control_0003_eval_audit as independent_audit


def _save(path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def test_evaluation_uses_audited_development_thresholds_and_rejects_tamper(
    tmp_path, monkeypatch,
) -> None:
    panel_path = tmp_path / "panel.json"
    sources_path = tmp_path / "sources.json"
    development_path = tmp_path / "development.json"
    evaluation_path = tmp_path / "evaluation.json"
    freeze_path = tmp_path / "freeze.json"
    selection_path = tmp_path / "selection.json"
    audit_path = tmp_path / "audit.json"
    features_path = tmp_path / "features.json"
    score_path = tmp_path / "score.json"
    known = [f"known_{i:02d}" for i in range(24)]
    unknown = [f"unknown_{i:02d}" for i in range(12)]
    panel_hash = _save(panel_path, {"roles": {
        "evaluation_known": [{"chapter": name} for name in known],
        "evaluation_unknown": [{"chapter": name} for name in unknown],
    }})
    sources_hash = _save(sources_path, {"synthetic_source_count": 216})
    development_hash = _save(development_path, {"synthetic_development_crop_count": 108})
    rows = []
    for kind, names in (("known", known), ("unknown", unknown)):
        for name in names:
            for manuscript in evaluation.MANUSCRIPTS:
                filename = f"{name}_{manuscript}.png"
                crop = tmp_path / filename
                crop.write_bytes(f"synthetic evaluation crop {name} {manuscript}".encode())
                rows.append({
                    "role": f"evaluation_{kind}", "chapter_class": name,
                    "manuscript": manuscript, "crop_file": filename,
                    "crop_sha256": sha256(crop.read_bytes()).hexdigest(),
                })
    evaluation_hash = _save(evaluation_path, {
        "id": "HERBAL-CONTROL-0003-evaluation",
        "status": "complete-pre-score-evaluation-crops",
        "panel_manifest_sha256": panel_hash,
        "rendered_crops": 108, "rows": rows,
    })
    freeze_hash = _save(freeze_path, {
        "id": "HERBAL-CONTROL-0003-input-freeze",
        "status": "complete-pre-score-inputs",
        "panel_manifest_sha256": panel_hash,
        "sources_manifest_sha256": sources_hash,
        "development_images_sha256": development_hash,
        "evaluation_images_sha256": evaluation_hash,
        "source_pages": 216, "crop_pages": 216,
    })
    normalizers = {f"{query}->{source}": 1.0 for query in evaluation.MANUSCRIPTS
                   for source in evaluation.MANUSCRIPTS if query != source}
    methods = {method: {"batch_threshold": 0.25,
                        "independent_threshold": 0.25,
                        "batch_macro_ba": 0.5}
               for method in evaluation.METHODS}
    selection_hash = _save(selection_path, {
        "id": "HERBAL-CONTROL-0003-development-selection",
        "status": "development-score-produced-awaiting-audit-and-push",
        "panel_manifest_sha256": panel_hash,
        "full_input_freeze_sha256": freeze_hash,
        "development_features_sha256": "synthetic-feature-hash",
        "normalization_medians": normalizers,
        "primary_method": "single_first", "methods": methods,
    })
    audit_hash = _save(audit_path, {
        "id": "HERBAL-CONTROL-0003-independent-development-score-audit",
        "status": "pass", "score_sha256": selection_hash,
        "full_input_freeze_sha256": freeze_hash,
        "feature_sha256": "synthetic-feature-hash",
        "primary_method": "single_first",
        "batch_macro_ba": {method: 0.5 for method in evaluation.METHODS},
    })
    distance = [[0.0 if i == j else 0.1 if left["chapter_class"] == right["chapter_class"]
                 else 1.0 for j, right in enumerate(rows)] for i, left in enumerate(rows)]
    _save(features_path, {
        "id": "HERBAL-CONTROL-0003-evaluation",
        "feature_method": evaluation.FEATURE_METHOD,
        "input_manifest_sha256": evaluation_hash,
        "full_input_freeze_sha256": freeze_hash,
        "development_selection_sha256": selection_hash,
        "development_audit_sha256": audit_hash,
        "row_order": rows, "distance_matrix": distance,
    })
    for module in (evaluation, independent_audit):
        for key, value in (
            ("ROOT", tmp_path), ("PANEL", panel_path), ("SOURCES", sources_path),
            ("DEVELOPMENT_IMAGES", development_path), ("EVALUATION_IMAGES", evaluation_path),
            ("FREEZE", freeze_path), ("PANEL_SHA256", panel_hash),
        ):
            monkeypatch.setattr(module, key, value)
    result = evaluation.score(features_path, selection_path, audit_path)
    assert result["primary_method"] == "single_first"
    assert result["gates"] == {
        "historical_image_open_set_feasibility": True,
        "two_source_benefit": False,
        "one_to_one_benefit": False,
    }
    assert all(row["batch_threshold"] == 0.25 for row in result["methods"].values())
    assert all(row["batch_macro_ba"] == 1.0 for row in result["methods"].values())
    assert result["primary_macro_ba_ci_95"] == [1.0, 1.0]
    _save(score_path, result)
    independent_result = independent_audit.audit(features_path, selection_path,
                                                 audit_path, score_path)
    assert independent_result["status"] == "pass"
    assert independent_result["gates"] == result["gates"]
    tampered_result = json.loads(score_path.read_text())
    tampered_result["gates"]["historical_image_open_set_feasibility"] = False
    _save(score_path, tampered_result)
    with pytest.raises(ValueError, match="gate decision differs"):
        independent_audit.audit(features_path, selection_path, audit_path, score_path)
    # A nonperfect panel exercises class-bootstrap arithmetic beyond [1, 1].
    features = json.loads(features_path.read_text())
    for j, row in enumerate(rows):
        if row["manuscript"] != "bnf":
            features["distance_matrix"][0][j] = 1.0
            features["distance_matrix"][j][0] = 1.0
    _save(features_path, features)
    imperfect = evaluation.score(features_path, selection_path, audit_path)
    assert imperfect["methods"]["single_first"]["batch_by_direction"]["bnf"]["known_correct"] == 23
    assert imperfect["primary_macro_ba_ci_95"][0] < 1.0
    _save(score_path, imperfect)
    assert independent_audit.audit(features_path, selection_path,
                                   audit_path, score_path)["status"] == "pass"
    selection = json.loads(selection_path.read_text())
    selection["methods"]["single_first"]["batch_threshold"] = 0.0
    _save(selection_path, selection)
    with pytest.raises(ValueError, match="audited development selection"):
        evaluation.score(features_path, selection_path, audit_path)
