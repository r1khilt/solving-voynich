import hashlib

import pytest

from scripts.build_blind_channel_development_corpora import (
    body_intervals, bracket_intervals, extract, normalize_segment, overlap_counts,
)


def test_normalization_excludes_apparatus_and_retains_source_offsets():
    text = "Jūlius [nota [altera]] vīvit; cælum 1o 12."
    clean, tokens, excluded = normalize_segment(text, 7)
    assert clean == "iuliusuiuitcaelum"
    assert text[tokens[0]["raw_character_start"] - 7:tokens[0]["raw_character_end"] - 7] == "Jūlius"
    assert [item["reason"] for item in excluded] == [
        "square_bracket_span", "numeric_or_mixed_token", "numeric_or_mixed_token"]
    assert clean[tokens[-1]["text_start"]:tokens[-1]["text_end"]] == "caelum"


@pytest.mark.parametrize("text", ["a[", "a]", "a[b]]"])
def test_unbalanced_apparatus_rejected(text):
    with pytest.raises(ValueError):
        bracket_intervals(text)


def test_unknown_letters_fail_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_segment("arma λογος")


def test_editorial_summary_and_headers_are_outside_authorial_spans():
    text = ("Project Gutenberg License\nHEADER\nARGUMENTUM\neditor\n"
            "Arma bona\nHEADER\neditor\nUrbs magna\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST\n")
    spec = {"heading": "^HEADER$", "expected_bodies": 2,
            "openings": ["Arma", "Urbs"]}
    bodies = [text[a:b] for a, b in body_intervals(text, spec)]
    assert bodies == ["Arma bona\n", "Urbs magna\n"]
    raw = text.encode()
    spec.update(sha256=hashlib.sha256(raw).hexdigest(), role="D", author="fixture", path="fixture", pg=0)
    payload, summary = extract(raw, spec, cap=10)
    assert payload["text"] == "armabonaur"
    assert payload["body_boundaries"] == [8, 10]
    assert payload["tokens"][-1]["text_end"] == 12
    assert payload["tokens"][-1]["selected_end"] == 10
    assert summary["eligible_characters"] == 17
    for row in summary["body_spans"]:
        selected = raw[row["raw_byte_start"]:row["raw_byte_end"]]
        assert hashlib.sha256(selected).hexdigest() == row["raw_sha256"]
    with pytest.raises(ValueError, match="pinned"):
        extract(raw + b"changed", spec, cap=10)


def test_exact_overlap_screen_counts_repetitions_without_returning_text():
    result = overlap_counts({"a": "abcabc", "b": "zzabc", "c": "xyzyz"}, width=3)
    assert result[0] == {"first": "a", "second": "b", "window": 3,
                         "shared_unique_windows": 1, "first_positions": 2, "second_positions": 1}
    assert result[1]["shared_unique_windows"] == result[2]["shared_unique_windows"] == 0
