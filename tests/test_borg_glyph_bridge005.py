from copy import deepcopy
import json

from PIL import Image
import pytest

from scripts import borg_glyph_bridge005 as bridge


def image_range(block, start, end, canvas):
    return {"block": block, "start_line": start, "end_line": end, "canvas": canvas,
            "image_path": "image.png", "image_sha256": "fixture"}


def test_selection_preserves_annotations_case_unicode_and_missing():
    raw = "#page 1r\nM m [M] é M\nM? m\n#page 2r\nm é M\n".encode()
    ranges = [image_range(0, 1, 3, "one"), image_range(1, 4, 5, "two")]
    plan = bridge.make_plan(raw, {"M", "m", "é", "cl", "z"}, ranges)
    rows = {r["symbol"]: r for r in plan["targets"]}
    assert rows["z"]["missing_slots"] == 2
    assert rows["M"]["available_distinct_canvases"] == 2
    assert len(rows["M"]["occurrences"]) == 2
    assert rows["M"]["occurrences"][0]["start_byte"] == raw.index(b"M")
    for symbol in ("M", "m", "é"):
        assert [o["source_line"] for o in rows[symbol]["occurrences"]] == [2, 5]
        for item in rows[symbol]["occurrences"]:
            assert raw[item["start_byte"]:item["end_byte"]].decode() == symbol
    assert "cl" not in rows  # Atlas directory names are not legacy tokens.
    assert rows[","]["role"] == "punctuation_candidate_no_alias"
    assert plan["solver_records"] == 0


def test_selection_refuses_overlapping_or_wrong_block_ranges():
    raw = b"#page 1r\nM\n#page 2r\nM\n"
    with pytest.raises(ValueError, match="Overlapping"):
        bridge.make_plan(raw, {"M"}, [image_range(0, 1, 2, "one"), image_range(0, 2, 2, "two")])
    with pytest.raises(ValueError, match="leaves source block"):
        bridge.make_plan(raw, {"M"}, [image_range(0, 1, 4, "one")])


def test_distinct_canvas_not_duplicate_record_is_selection_unit():
    raw = b"#page 1r\nM\n#page 2r\nM\n"
    plan = bridge.make_plan(raw, {"M"}, [image_range(0, 1, 2, "same"), image_range(1, 3, 4, "same")])
    row = next(r for r in plan["targets"] if r["symbol"] == "M")
    assert row["missing_slots"] == 1
    assert row["available_distinct_canvases"] == 1


def test_path_escape_refused(tmp_path):
    with pytest.raises(ValueError, match="leaves repository"):
        bridge.local_file(tmp_path, "../outside")


@pytest.fixture
def review_fixture(tmp_path, monkeypatch):
    Image.new("RGB", (10, 10), "white").save(tmp_path / "image.png")
    Image.open(tmp_path / "image.png").crop((0, 0, 5, 5)).save(tmp_path / "crop.png")
    plan = bridge.make_plan(b"#page 1r\nM\n", {"M"}, [image_range(0, 1, 2, "one")])
    p = tmp_path / bridge.PLAN
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(plan))
    ledger = {"plan_sha256": bridge.digest(p.read_bytes()), "solver_records": 0,
        "observations": [{"target": deepcopy(plan["targets"][-1]["occurrences"][0]),
            "shape_status": "uncertain_correspondence", "context_status": "uncertain_context",
            "observation": "Artificial fixture, not actual handwriting.",
            "crop": {"path": "crop.png", "box_xyxy": [0, 0, 5, 5],
                     "sha256": bridge.digest((tmp_path / "crop.png").read_bytes())}}]}
    monkeypatch.setattr(bridge, "plan_from_root", lambda root: deepcopy(plan))
    lp = tmp_path / bridge.LEDGER
    lp.parent.mkdir(parents=True)
    lp.write_text(json.dumps(ledger))
    return tmp_path, lp, ledger


def test_audit_accepts_uncertainty_and_retains_missing_slots(review_fixture):
    root, _, _ = review_fixture
    result = bridge.audit(root)
    assert result["targets"] == 1
    assert result["missing_slots"] == 7
    assert result["shape_statuses"] == {"uncertain_correspondence": 1}


@pytest.mark.parametrize("change,match", [("omission", "Target count"), ("replacement", "Target omitted"),
                                         ("certainty", "Unsupported shape"), ("pixels", "Crop pixels")])
def test_audit_refuses_replacement_certainty_and_corrupted_pixels(review_fixture, change, match):
    root, path, ledger = review_fixture
    if change == "omission":
        ledger["observations"] = []
    elif change == "replacement":
        ledger["observations"][0]["target"]["start_byte"] += 1
    elif change == "certainty":
        ledger["observations"][0]["shape_status"] = "globally_verified_cipher_glyph"
    else:
        Image.new("RGB", (5, 5), "black").save(root / "crop.png")
        ledger["observations"][0]["crop"]["sha256"] = bridge.digest((root / "crop.png").read_bytes())
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match=match):
        bridge.audit(root)
