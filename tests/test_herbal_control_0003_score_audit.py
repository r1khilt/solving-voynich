"""Independent matching and full-shape development-score audit controls."""

from hashlib import sha256
from itertools import product
import json
import random

import pytest

from scripts import herbal_control_0003_score_audit as auditor
from scripts import herbal_control_0003_score_dev as scorer
from voynich.herbal_folio import FEATURE_METHOD as FOLIO_METHOD


def _save(path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def test_partial_matching_agrees_with_small_exhaustive_search() -> None:
    rng = random.Random(3003)
    for _ in range(20):
        costs = [[rng.uniform(-1, 2) for _ in range(3)] for _ in range(4)]
        tau = rng.uniform(-0.5, 1.5)
        predictions, objective = auditor.partial_matching(costs, tau)
        patterns = {
            candidate: sum(tau if col is None else costs[i][col]
                           for i, col in enumerate(candidate))
            for candidate in product((None, 0, 1, 2), repeat=4)
            if len([col for col in candidate if col is not None])
            == len(set(col for col in candidate if col is not None))
        }
        assert objective == pytest.approx(min(patterns.values()), abs=1e-10)
        runner_up = min(value for pattern, value in patterns.items()
                        if pattern != tuple(predictions))
        assert auditor.next_pattern_cost(costs, tau, predictions) == pytest.approx(
            runner_up, abs=1e-10)
        assert len(set(col for col in predictions if col is not None)) == sum(
            col is not None for col in predictions)


def test_independent_development_audit_replays_synthetic_score_and_rejects_tamper(
    tmp_path, monkeypatch,
) -> None:
    panel_path = tmp_path / "panel.json"
    sources_path = tmp_path / "sources.json"
    development_path = tmp_path / "development.json"
    evaluation_path = tmp_path / "evaluation.json"
    freeze_path = tmp_path / "freeze.json"
    features_path = tmp_path / "features.json"
    score_path = tmp_path / "score.json"
    known = [f"known_{i:02d}" for i in range(24)]
    unknown = [f"unknown_{i:02d}" for i in range(12)]
    panel_hash = _save(panel_path, {"roles": {
        "development_known": [{"chapter": name} for name in known],
        "development_unknown": [{"chapter": name} for name in unknown],
    }})
    source_hash = _save(sources_path, {"synthetic": True})
    evaluation_hash = _save(evaluation_path, {"unopened_holdout": True})
    rows = []
    for kind, names in (("known", known), ("unknown", unknown)):
        for name in names:
            for manuscript in auditor.MANUSCRIPTS:
                filename = f"{name}_{manuscript}.png"
                crop = tmp_path / filename
                crop.write_bytes(f"synthetic crop {name} {manuscript}".encode())
                rows.append({
                    "role": f"development_{kind}", "chapter_class": name,
                    "manuscript": manuscript, "crop_file": filename,
                    "crop_sha256": sha256(crop.read_bytes()).hexdigest(),
                })
    development_hash = _save(development_path, {
        "id": "HERBAL-CONTROL-0003-development",
        "status": "complete-pre-score-development-crops",
        "panel_manifest_sha256": panel_hash, "rendered_crops": 108, "rows": rows,
    })
    freeze_hash = _save(freeze_path, {
        "id": "HERBAL-CONTROL-0003-input-freeze",
        "status": "complete-pre-score-inputs", "panel_manifest_sha256": panel_hash,
        "sources_manifest_sha256": source_hash,
        "development_images_sha256": development_hash,
        "evaluation_images_sha256": evaluation_hash,
        "source_pages": 216, "crop_pages": 216,
    })
    distances = []
    for i, left in enumerate(rows):
        line = []
        for j, right in enumerate(rows):
            if i == j:
                value = 0.0
            elif left["chapter_class"] == right["chapter_class"]:
                value = 0.1 + ((min(i, j) * 17 + max(i, j)) % 97) / 100_000
            else:
                value = 1.0 + ((min(i, j) * 29 + max(i, j)) % 101) / 10_000
            line.append(value)
        distances.append(line)
    _save(features_path, {
        "id": "HERBAL-CONTROL-0003-development",
        "feature_method": "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit",
        "input_manifest_sha256": development_hash,
        "full_input_freeze_sha256": freeze_hash,
        "row_order": rows, "distance_matrix": distances,
    })
    for module in (scorer, auditor):
        for key, value in (
            ("PANEL", panel_path), ("SOURCES", sources_path),
            ("DEVELOPMENT" if module is auditor else "DEV_IMAGES", development_path),
            ("EVALUATION" if module is auditor else "EVAL_IMAGES", evaluation_path),
            ("FREEZE", freeze_path), ("PANEL_SHA256", panel_hash),
        ):
            monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(auditor, "ROOT", tmp_path)
    score = scorer.score(features_path)
    _save(score_path, score)
    result = auditor.audit(features_path, score_path)
    assert result["status"] == "pass"
    assert result["primary_method"] == score["primary_method"]
    assert all(value == 1.0 for value in result["batch_macro_ba"].values())
    dino_features_path = tmp_path / "dino_features.json"
    dino_score_path = tmp_path / "dino_score.json"
    dino_method = "DINOv2-with-registers-base final CLS 224 direct resize L2 cosine"
    dino_features = json.loads(features_path.read_text())
    dino_features["feature_method"] = dino_method
    _save(dino_features_path, dino_features)
    dino_score = scorer.score(
        dino_features_path, feature_method=dino_method,
        result_id="HERBAL-CONTROL-0003-DINO-development-selection")
    _save(dino_score_path, dino_score)
    dino_audit = auditor.audit(
        dino_features_path, dino_score_path, feature_method=dino_method,
        score_id="HERBAL-CONTROL-0003-DINO-development-selection",
        audit_id="HERBAL-CONTROL-0003-DINO-independent-development-score-audit")
    assert dino_audit["id"] == "HERBAL-CONTROL-0003-DINO-independent-development-score-audit"
    assert dino_audit["primary_method"] == dino_score["primary_method"]
    folio_features_path = tmp_path / "folio_features.json"
    folio_score_path = tmp_path / "folio_score.json"
    folio_features = json.loads(features_path.read_text())
    folio_features["feature_method"] = FOLIO_METHOD
    folio_features["distance_matrix"][0][4] += 0.2
    _save(folio_features_path, folio_features)
    folio_score = scorer.score(
        folio_features_path, feature_method=FOLIO_METHOD,
        result_id="HERBAL-CONTROL-0003-FOLIO-development-selection")
    _save(folio_score_path, folio_score)
    folio_replay = auditor.audit(
        folio_features_path, folio_score_path, feature_method=FOLIO_METHOD,
        score_id="HERBAL-CONTROL-0003-FOLIO-development-selection",
        audit_id="HERBAL-CONTROL-0003-FOLIO-independent-development-score-audit")
    assert folio_replay["status"] == "pass"
    assert folio_replay["primary_method"] == folio_score["primary_method"]
    dino_asymmetric_path = tmp_path / "dino_asymmetric.json"
    folio_features["feature_method"] = dino_method
    _save(dino_asymmetric_path, folio_features)
    with pytest.raises(ValueError, match="asymmetric"):
        scorer.score(dino_asymmetric_path, feature_method=dino_method)
    with pytest.raises(ValueError, match="feature extraction inputs"):
        auditor.audit(dino_features_path, dino_score_path)
    score["normalization_medians"]["bnf->egerton"] *= 2
    _save(score_path, score)
    with pytest.raises(ValueError, match="bnf->egerton differs"):
        auditor.audit(features_path, score_path)
