"""Small counterexamples for case loss, label merging and resource corruption."""
import hashlib
import json

import pytest

from scripts.audit_borg_glyph_resources import atlas_labels, checked_bytes, coverage, verify_tree_inventory


def entry(path, content, label):
    return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "git_blob": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest(),
            "source_path": f"gpu/few_shot_train/alphabet/borg/{label}/1.jpg"}


def test_case_distinct_assets_and_labels(tmp_path):
    a, b = b"upper", b"lower"
    (tmp_path / "a").write_bytes(a)
    (tmp_path / "b").write_bytes(b)
    manifest = {"artifacts": [entry("a", a, "M"), entry("b", b, "m")]}
    assert atlas_labels(tmp_path, manifest) == {"M", "m"}


def test_multichar_names_are_not_automatic_tokens_or_cross_boundary_matches():
    raw = b"#page 0002r\nM m cm c[note]m c m dt cl ,.: Z\n"
    got = coverage(raw, {"M", "m", "c", "cm", "dt", "cl"})
    assert got["literal_single_label_occurrences"] == 9
    assert got["literal_multi_name_substrings_not_glyph_tokens"] == {"cl": 1, "cm": 1, "dt": 1}
    assert got["remaining_codepoints"] == {"U+005A": 1, "U+0064": 1, "U+006C": 1, "U+0074": 1}
    assert got["solver_records"] == 0


def test_markers_cleartext_and_annotations_remain_quarantined():
    raw = b"#page 0001r\n<CLEARTEXT-LA>\nmmmm\n#page 0002r\nM?mmm\nm[note] M\n"
    got = coverage(raw, {"M", "m"})
    assert got["unresolved_codepoints"] == got["literal_single_label_occurrences"] == 2


def test_corrupted_bytes_and_git_identity_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    e = entry("a", b"y", "M")
    with pytest.raises(ValueError, match="bytes differ"):
        checked_bytes(tmp_path, e)
    e = entry("a", b"x", "M")
    e["git_blob"] = "0" * 40
    with pytest.raises(ValueError, match="Git blob"):
        checked_bytes(tmp_path, e)


def test_duplicate_label_and_path_escape_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    e = entry("a", b"x", "M")
    with pytest.raises(ValueError, match="duplicate"):
        atlas_labels(tmp_path, {"artifacts": [e, e]})
    with pytest.raises(ValueError, match="leaves repository"):
        checked_bytes(tmp_path, dict(e, path="../outside"))


def test_labels_bound_to_complete_pinned_tree(tmp_path):
    e = entry("a", b"upper", "M")
    f = entry("b", b"lower", "m")
    tree = {"truncated": False, "tree": [
        {"path": x["source_path"], "type": "blob", "size": x["bytes"], "sha": x["git_blob"]}
        for x in [e, f]]}
    raw = json.dumps(tree).encode()
    (tmp_path / "tree").write_bytes(raw)
    m = {"source_tree": entry("tree", raw, "unused"), "artifacts": [e, f]}
    verify_tree_inventory(tmp_path, m)
    with pytest.raises(ValueError, match="incomplete"):
        verify_tree_inventory(tmp_path, dict(m, artifacts=[e]))
    with pytest.raises(ValueError, match="pinned tree"):
        verify_tree_inventory(tmp_path, dict(m, artifacts=[dict(e, git_blob=f["git_blob"]), f]))
