"""Artificial fixtures only: no real cleartext, historical key, or cipher fit."""
from dataclasses import replace
import json

import pytest

from scripts.borg_control_parser import digest, parse_bytes, redacted_summary, validate_partition


def parts(blob, parsed, kind):
    return [blob[x.start_byte:x.end_byte].decode() for x in parsed.spans if x.kind == kind]


def test_complete_byte_roundtrip_with_mixed_newlines_and_unicode():
    raw = "#source artificial\r\n#page 0002r\rα🙂 9x\nβ\r\n#page 2v\nx0".encode()
    parsed = parse_bytes(raw, expected_sha256=digest(raw))
    assert b"".join(raw[x.start_byte:x.end_byte] for x in parsed.spans) == raw
    assert parsed.source_chars == len(raw.decode())
    assert parsed.source_bytes > parsed.source_chars
    assert parsed.physical_lines == 6
    assert parts(raw, parsed, "unresolved_body") == ["α🙂", "9x", "β", "x0"]
    assert parsed.pages[1].normalized_label_only == "0002v"


def test_cleartext_whole_page_is_excluded_even_before_marker():
    raw = b"#page 1r\nFAKE_FRONT\n<CLEARTEXT-LA>\nFAKE_REST\n#page 2r\nx98\n"
    parsed = parse_bytes(raw)
    assert parsed.pages[0].explicit_cleartext
    assert parts(raw, parsed, "unresolved_body") == ["x98"]
    assert "FAKE_FRONT\n" in parts(raw, parsed, "explicit_cleartext_page")
    summary = json.dumps(redacted_summary(raw, parsed))
    assert "FAKE" not in summary
    assert "x98" not in summary


@pytest.mark.parametrize("marker", ["<CLEARTEXT-LA>", "<CLEARTEXT-AR>"])
def test_both_explicit_cleartext_languages_quarantine(marker):
    raw = f"#page 1r\n{marker}\nFAKE\n".encode()
    parsed = parse_bytes(raw)
    assert not parts(raw, parsed, "unresolved_body")
    assert parsed.pages[0].explicit_cleartext


def test_nested_multiline_annotations_do_not_join_neighbors():
    raw = b"#page 2r\nxx[FAKE\r\n[EDITOR]]zz <NOTE\rBODY> yy\n"
    parsed = parse_bytes(raw)
    assert parts(raw, parsed, "unresolved_body") == ["xx", "zz", "yy"]
    assert [x.kind for x in parsed.annotations] == ["square_annotation", "angle_annotation"]
    assert all(x.start_line != x.end_line for x in parsed.annotations)
    assert redacted_summary(raw, parsed)["multiline_annotations"] == 2


@pytest.mark.parametrize("body", ["x? y", "x??y", "x/y zz", "* x y", "x??? y"])
def test_unknown_marker_scope_quarantines_entire_line(body):
    raw = f"#page 2r\n{body}\nq8\n".encode()
    parsed = parse_bytes(raw)
    assert parts(raw, parsed, "unresolved_body") == ["q8"]
    assert parsed.marker_lines == (2,)
    assert parts(raw, parsed, "unresolved_marker_line") == [body + "\n"]


def test_annotation_question_mark_does_not_claim_uncertain_ciphertext():
    raw = b"#page 2r\nx [FAKE?] y\n"
    parsed = parse_bytes(raw)
    assert parsed.marker_lines == ()
    assert parts(raw, parsed, "unresolved_body") == ["x", "y"]


def test_comments_and_headers_are_not_cipher_units():
    raw = b"#meta ? [ignored\n#page 2r\n#comment <ignored\nx9\n"
    parsed = parse_bytes(raw)
    assert parts(raw, parsed, "unresolved_body") == ["x9"]
    assert parsed.marker_lines == ()
    assert len(parts(raw, parsed, "hash_comment")) == 2


def test_duplicate_suffix_and_typo_are_retained_not_repaired():
    raw = b"#page 49v\nx\n#page 0049v\ny\n#pahe 150r.01\nz\n"
    parsed = parse_bytes(raw)
    summary = redacted_summary(raw, parsed)
    assert [x.raw_label for x in parsed.pages] == ["49v", "0049v", "150r.01"]
    assert [x.block_index for x in parsed.pages] == [0, 1, 2]
    assert summary["duplicate_normalized_labels"] == {"0049v": 2}
    assert summary["suffixed_block_indices"] == [2]
    assert summary["typo_header_block_indices"] == [2]


def test_literal_case_digits_and_punctuation_are_unresolved_not_normalized():
    raw = b"#page 2r\nM m 08 . , : ;\n"
    parsed = parse_bytes(raw)
    assert parts(raw, parsed, "unresolved_body") == ["M", "m", "08", ".", ",", ":", ";"]
    summary = redacted_summary(raw, parsed)
    assert summary["unresolved_body"]["codepoint_types"] == 8
    assert summary["solver_records_produced"] == 0
    assert summary["glyph_unit_semantics_verified"] is False
    assert summary["remaining_body_proven_ciphertext_only"] is False


@pytest.mark.parametrize("raw", [b"", b"no pages", b"#page bad\nx", b" #page 2r\nx",
                                   b"#pahe 4x\nx", b"#page 2r\nx\xff"])
def test_malformed_headers_missing_pages_and_encoding_fail_closed(raw):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        parse_bytes(raw)


@pytest.mark.parametrize("body", ["[open", "]", "<open", ">", "[<cross]>" ])
def test_unbalanced_or_crossed_annotations_fail_closed(body):
    with pytest.raises(ValueError, match="annotation"):
        parse_bytes(f"#page 2r\n{body}\n".encode())


def test_annotation_cannot_cross_page_or_comment():
    for raw in (b"#page 2r\n[x\n#page 2v\ny]\n", b"#page 2r\n[x\n#comment\ny]\n"):
        with pytest.raises(ValueError, match="crosses"):
            parse_bytes(raw)


def test_pin_and_partition_tampering_are_detected():
    raw = b"#page 2r\nx9\n"
    with pytest.raises(ValueError, match="SHA-256"):
        parse_bytes(raw, expected_sha256="0" * 64)
    parsed = parse_bytes(raw)
    bad = replace(parsed.spans[0], end_byte=parsed.spans[0].end_byte - 1)
    with pytest.raises(ValueError):
        validate_partition(raw, replace(parsed, spans=(bad,) + parsed.spans[1:]))
    with pytest.raises(ValueError):
        validate_partition(raw, replace(parsed, spans=parsed.spans[1:]))
    with pytest.raises(ValueError):
        validate_partition(raw + b"x", parsed)


@pytest.mark.parametrize("separator", ["\v", "\f", "\x85", "\u2028", "\u2029"])
def test_undocumented_line_separators_are_rejected(separator):
    with pytest.raises(ValueError, match="separator"):
        parse_bytes(f"#page 2r\nx{separator}y".encode())
