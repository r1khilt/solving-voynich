import copy
import hashlib
import json

from PIL import Image
import pytest

from scripts import audit_borg_exception_review004 as check


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def write(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    raw = b"#page 0003r\nnnnnnnnn Z\nhhhhhhhh\n"
    source = tmp_path / check.RAW.relative_to(check.ROOT)
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    monkeypatch.setattr(check, "PINNED_SHA256", sha(raw))
    start = raw.index(b"Z")
    target = {"codepoint": "U+005A", "source_line": 2,
              "start_byte": start, "end_byte": start + 1, "slice_sha256": sha(b"Z")}
    coverage = write("results/BORG-GLYPH-PREP-002/coverage.json", {"atlas_single_codepoint_labels": ["n", "h"]})
    catalogue = write("data/raw/borg-resource-review/vatican-manifest.json", {})
    selected = {"block": 0, "canvas": "fixture", "catalogue_label": "3r", "image_url": "fixture-url",
                "target_occurrences": {"U+005A": [{k: v for k, v in target.items() if k != "codepoint"}]}}
    plan = {"coverage_sha256": sha(coverage.read_bytes()), "catalogue_sha256": sha(catalogue.read_bytes()), "panel": [selected]}
    monkeypatch.setattr(check, "make_plan", lambda *_: copy.deepcopy(plan))
    plan_path = write("data/manifests/borg_exception_review004_plan.json", plan)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1600, 40), "white").save(image_path)
    crop_path = tmp_path / "crop.png"
    Image.open(image_path).crop([10, 10, 30, 30]).save(crop_path)
    crop = {"path": "crop.png", "box_xyxy": [10, 10, 30, 30], "sha256": sha(crop_path.read_bytes())}
    image = {"block": 0, "canvas": "fixture", "label": "3r", "url": "fixture-url", "path": "image.png",
             "returncode": 0, "bytes": len(image_path.read_bytes()), "sha256": sha(image_path.read_bytes())}
    acquisition = write("data/manifests/borg_exception_review004_acquisition.json",
                        {"plan_sha256": sha(plan_path.read_bytes()), "images": [image]})
    checks = []
    for line, content in [(2, b"nnnnnnnn"), (3, b"hhhhhhhh")]:
        begin = raw.index(content)
        checks.append({"source_line": line, "start_byte": begin, "end_byte": begin + 8,
                       "slice_sha256": sha(content), "glyph_occurrences": 8, "crop": crop})
    page = {"block": 0, "canvas": "fixture", "label": "3r", "image_path": "image.png",
            "image_sha256": image["sha256"], "page_identity_checks": checks,
            "targets": [{**target, "status": "unresolved_individual_correspondence",
                         "observation": "Artificial unresolved target", "crop": crop}]}
    ledger = {"plan_sha256": sha(plan_path.read_bytes()), "acquisition_sha256": sha(acquisition.read_bytes()),
              "pages": [page], "solver_records": 0}
    path = write("results/BORG-EXCEPTION-REVIEW-004/visual_ledger.json", ledger)
    return tmp_path, path, ledger


def test_artifact_replay_accepts_explicitly_unresolved_target(fixture):
    root, _, _ = fixture
    result = check.audit(root)
    assert result["target_occurrences"] == 1
    assert result["page_identity_line_checks"] == 2
    assert result["unique_crops_replayed"] == 1
    assert result["review_status_counts"] == {"unresolved_individual_correspondence": 1}


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "same_line", "wrong_line", "crop", "certainty", "solver"])
def test_artifact_replay_rejects_missing_changed_or_overclaimed_evidence(fixture, mutation):
    root, path, ledger = fixture
    page = ledger["pages"][0]
    if mutation == "omit":
        page["targets"] = []
    elif mutation == "duplicate":
        page["targets"] *= 2
    elif mutation == "same_line":
        first = page["page_identity_checks"][0]
        page["page_identity_checks"] = [first, copy.deepcopy(first)]
    elif mutation == "wrong_line":
        page["page_identity_checks"][0]["source_line"] = 7
    elif mutation == "crop":
        (root / "crop.png").write_bytes(b"altered")
    elif mutation == "certainty":
        page["targets"][0]["status"] = "globally_proved_alias"
    else:
        ledger["solver_records"] = 1
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        check.audit(root)
