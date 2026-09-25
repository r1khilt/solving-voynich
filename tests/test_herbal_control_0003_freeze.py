"""Synthetic full-panel integrity and corruption controls for the input freeze."""

import hashlib
import json

import pytest

from scripts import herbal_control_0003_freeze as freeze


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_full_input_freeze_requires_every_hashed_source_box_and_crop(tmp_path,
                                                                      monkeypatch) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in (
        "PANEL", "SOURCES", "DEVELOPMENT", "EVALUATION", "DEVELOPMENT_BOXES",
        "EVALUATION_BOXES", "OUT")}
    for name, path in paths.items():
        monkeypatch.setattr(freeze, name, path)
    roles = ("development_known", "development_unknown",
             "evaluation_known", "evaluation_unknown")
    panel = {"roles": {role: [] for role in roles}}
    source_rows = []
    boxes_by_split = {"development": [], "evaluation": []}
    images_by_split = {"development": [], "evaluation": []}
    next_pageid = 1
    for role in roles:
        count = 24 if role.endswith("known") and not role.endswith("unknown") else 12
        for index in range(count):
            chapter = f"{role}_{index:02d}"
            item = {"chapter": chapter, "pages": {}}
            for manuscript in freeze.MANUSCRIPTS:
                pageid = next_pageid
                next_pageid += 1
                item["pages"][manuscript] = {"pageid": pageid}
                source_path = tmp_path / f"source_{pageid}.jpg"
                source_path.write_bytes(b"\xff\xd8\xff" + pageid.to_bytes(4, "big"))
                crop_path = tmp_path / f"crop_{pageid}.png"
                crop_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
                                      + (512).to_bytes(4, "big") * 2 + pageid.to_bytes(4, "big"))
                source_rows.append({"role": role, "chapter": chapter,
                                    "manuscript": manuscript, "pageid": pageid,
                                    "file": str(source_path), "sha256": _sha(source_path),
                                    "bytes": source_path.stat().st_size})
                box = {"role": role, "chapter": chapter, "manuscript": manuscript,
                       "pageid": pageid, "source_sha256": _sha(source_path),
                       "crop_xywh": [1, 2, 3, 4], "depiction_note": "synthetic control"}
                split = role.split("_")[0]
                boxes_by_split[split].append(box)
                images_by_split[split].append({
                    "role": role, "chapter_class": chapter, "manuscript": manuscript,
                    "source_pageid": pageid, "source_file": str(source_path),
                    "source_sha256": _sha(source_path), "crop_file": str(crop_path),
                    "crop_sha256": _sha(crop_path), "crop_xywh": box["crop_xywh"],
                    "depiction_note": box["depiction_note"],
                })
            panel["roles"][role].append(item)
    assert next_pageid == 217
    _json(paths["PANEL"], panel)
    monkeypatch.setattr(freeze, "PANEL_SHA256", _sha(paths["PANEL"]))
    _json(paths["SOURCES"], {
        "status": "all-fixed-source-pages-downloaded",
        "panel_manifest_sha256": freeze.PANEL_SHA256,
        "requested_source_pages": 216, "recorded_source_pages": 216,
        "recorded_total_bytes": sum(row["bytes"] for row in source_rows),
        "sources": source_rows,
    })
    for split in ("development", "evaluation"):
        boxes_path = paths[f"{split.upper()}_BOXES"]
        images_path = paths[split.upper()]
        _json(boxes_path, {
            "id": f"HERBAL-CONTROL-0003-{split}-boxes",
            "status": "complete-pre-score-manual-review",
            "panel_manifest_sha256": freeze.PANEL_SHA256,
            "boxes": boxes_by_split[split],
        })
        _json(images_path, {
            "id": f"HERBAL-CONTROL-0003-{split}",
            "status": f"complete-pre-score-{split}-crops",
            "panel_manifest_sha256": freeze.PANEL_SHA256,
            "rendered_crops": 108, "target_crops": 108,
            "boxes_manifest_sha256": _sha(boxes_path),
            "rows": images_by_split[split],
        })
    result = freeze.run()
    assert (result["source_pages"], result["crop_pages"]) == (216, 216)
    assert json.loads(paths["OUT"].read_text()) == result
    (tmp_path / "crop_1.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="registered 512-square PNG"):
        freeze.run()
