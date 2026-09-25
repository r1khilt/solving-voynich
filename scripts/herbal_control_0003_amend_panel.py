"""Apply the first pre-score image-eligibility exclusion to the frozen panel.

The BnF page tagged ``gariofilis`` shows that name in a chapter index, but no
locatable drawing for that chapter. The registered rule replaces the whole
class in the same role with the first unused reserve, without looking at model
features or Voynich text. The original panel remains untouched.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/manifests/herbal_control_0003_panel.json"
AMENDED = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
ORIGINAL_SHA256 = "fc2bd19a8eb25d16d8f8387a26b551e42c1f221a048d8f8b07a3f20f02c4b725"
REJECTED_SOURCE_SHA256 = "392795d579a6386ae99e7ddda13585336193b9c8c13b3cc59a70aa0a03a9242f"
ROLES = ("development_known", "development_unknown", "evaluation_known", "evaluation_unknown")
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def run() -> dict:
    original_bytes = ORIGINAL.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("original metadata-only panel changed")
    panel = deepcopy(json.loads(original_bytes))
    role = "development_unknown"
    selected = next((item for item in panel["roles"][role]
                     if item["chapter"] == "gariofilis"), None)
    if selected is None or selected["pages"]["bnf"]["pageid"] != 107529796:
        raise ValueError("registered exclusion not present in its original role")
    reserve = panel["roles"]["reserve"].pop(0)
    if reserve["chapter"] != "branca ursina":
        raise ValueError("first reserve changed")
    index = panel["roles"][role].index(selected)
    panel["roles"][role][index] = reserve
    panel["role_counts"]["reserve"] -= 1
    panel["page_disjoint_classes"] -= 1
    panel["status"] = "pre-score-image-eligibility-amendment-1"
    panel["original_panel_manifest_sha256"] = ORIGINAL_SHA256
    panel["pre_score_exclusions"] = [{
        "chapter": "gariofilis",
        "role": role,
        "source_manuscript": "bnf",
        "source_pageid": 107529796,
        "source_sha256": REJECTED_SOURCE_SHA256,
        "reason": "The BnF folio carries Gariofilis in a chapter index but has no locatable Gariofilis drawing; its two prominent plants belong to other entries.",
        "replacement_chapter": reserve["chapter"],
        "replacement_rule": "first unused reserve in stored order, same role, before any image feature or Voynich text score",
    }]
    actual = [(role, item["chapter"], manuscript, item["pages"][manuscript]["pageid"])
              for role in ROLES for item in panel["roles"][role]
              for manuscript in MANUSCRIPTS]
    if len(actual) != 216 or len({entry[3] for entry in actual}) != 216:
        raise ValueError("amended primary panel is not 216 distinct pages")
    AMENDED.write_text(json.dumps(panel, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return {
        "active_panel": str(AMENDED.relative_to(ROOT) if AMENDED.is_relative_to(ROOT)
                            else AMENDED),
        "sha256": hashlib.sha256(AMENDED.read_bytes()).hexdigest(),
        "excluded": selected["chapter"],
        "replacement": reserve["chapter"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
