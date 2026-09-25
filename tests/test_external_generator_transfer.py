"""Pre-score structural checks for the external-generator transfer assay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voynich.external_generator_transfer import (
    lexical_stats, make_stream, stream_sha, summarize, wrap_stream,
)


def test_validation_skeleton_transfers_only_layout_and_context():
    source = [
        {"leaf": "f1", "locus": "f1r.1", "context": ("H", "1"),
         "words": [("a",)] * 4},
        {"leaf": "f2", "locus": "f2v.3", "context": ("P", "2"),
         "words": [("b",)] * 5},
    ]
    stream = ["ch", "sh", "qok", "ar", "or", "she", "cth", "dy", "a"]
    result = wrap_stream(stream, source)
    assert [(row["leaf"], row["locus"], row["context"], len(row["words"]))
            for row in result] == [("f1", "f1r.1", ("H", "1"), 4),
                                   ("f2", "f2v.3", ("P", "2"), 5)]
    assert ["".join(word) for row in result for word in row["words"]] == stream
    assert result[0]["words"][0] == ("ch",)
    with pytest.raises(ValueError, match="count"):
        wrap_stream(stream[:-1], source)


def test_external_timm_emits_one_fewer_visible_word_and_is_compensated():
    class FakeExternal:
        @staticmethod
        def generate_timm(n, _profile, *, seed):
            assert seed == 400001
            return " ".join(["a"] * (n - 1))

    profile = SimpleNamespace(seed_words=["a"], seed_w=[1])
    words = make_stream("timm_faithful", 7, profile, "a", 400001, FakeExternal)
    assert words == ["a"] * 7
    assert stream_sha(words) == stream_sha(["a"] * 7)
    with pytest.raises(ValueError, match="invalid word stream"):
        make_stream("stroke", 2, profile, "a", 400001,
                    SimpleNamespace(generate_stroke=lambda *args, **kwargs: "a ?"))


def test_lexical_counts_distinguish_token_novelty_from_type_hapax():
    stats = lexical_stats(["a", "a", "a", "b", "c"], {"a", "b"})
    assert stats == {"novel_token_fraction": .2,
                     "hapax_type_fraction": 2 / 3,
                     "type_token_ratio": 3 / 5}


def test_decision_uses_full_seed_range_and_existing_reference_interval():
    base = {"last_gain": 0.0, "shuffle_last_mean": 0.0,
            "last_minus_shuffle": 0.05, "last_minus_first": 0.0,
            "novel_token_fraction": 0.3, "hapax_type_fraction": 0.7,
            "type_token_ratio": 0.2}
    ref = {"bootstrap95": {"last_minus_null": [0.1221108346, 0.2683944521]},
           "lexical": {"novel_token_fraction": 0.3, "hapax_type_fraction": 0.7}}
    rows = [dict(base) for _ in range(16)]
    assert summarize(rows, ref)["edge_decision"] == "edge_below_reference"
    rows[-1]["last_minus_shuffle"] = 0.12
    assert summarize(rows, ref)["edge_decision"] == "not_falsified_by_this_edge_assay"
    rows = [dict(base, last_minus_shuffle=0.32) for _ in range(16)]
    assert summarize(rows, ref)["edge_decision"] == "edge_above_reference"
