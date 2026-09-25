from scripts.build_boundary_geometry_train import align_folio, leaf_id, parse_line


def test_parse_line_preserves_original_separator_and_excludes_annotation() -> None:
    tokens = parse_line("<f1r.1,@P0> dada.@168;oto.qokeedy,chol")
    assert tokens == [("dada", "X"), ("qokeedy", ","), ("chol", "L")]


def test_alignment_requires_same_physical_line_and_known_separator() -> None:
    words = [("choky", "."), ("daiin", ","), ("qokey", "L")]
    boxes = [{"word": "choky", "x": 10, "y": 0, "w": 15, "h": 12, "line": 0},
             {"word": "daiin", "x": 28, "y": 0, "w": 10, "h": 12, "line": 0},
             {"word": "qokey", "x": 5, "y": 20, "w": 15, "h": 12, "line": 1}]
    rows, counts = align_folio("f89r1", words, boxes)
    assert len(rows) == 1
    assert rows[0]["gap_px"] == 3
    assert rows[0]["label"] == "."
    assert rows[0]["leaf"] == "f89"
    assert counts["matched_tokens"] == 3
    assert leaf_id("fRos") == "fRos"
