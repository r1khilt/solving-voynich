import math
from collections import Counter

import pytest

from scripts.build_blind_channel_dev001 import (
    encode_records, estimate_source, make_channel, segments, shuffled_records, source_bits,
)
from scripts.evaluate_blind_channel_dev001 import evaluate_channel
from scripts.run_blind_channel_dev001 import baseline_log_likelihood, fit_glyph_baseline
from voynich.finite_state_channel import Channel, Emission, SourceModel


def test_source_uses_separate_body_contexts_and_fractional_backoff():
    source = estimate_source(["abba"], ("a", "b"), 1, 2)
    assert source.probabilities["a"]["b"] == pytest.approx(2 / 3)
    assert source.probabilities["b"]["b"] == .5
    assert source_bits(source, ["abba"]) == pytest.approx(3 - math.log2(2 / 3))
    assert source_bits(source, ["a", "b"]) == 2
    assert segments({"text": "abba", "body_boundaries": [1, 4]}) == ["a", "bba"]


def test_generated_controls_have_declared_unknown_unit_structure():
    alphabet = tuple("abcdefghiklmnopqrstuxyz")
    assert len(alphabet) == 23
    for family in ("A", "B"):
        channel = make_channel(alphabet, family, 17)
        assert channel.to_dict() == make_channel(alphabet, family, 17).to_dict()
        units = [channel.rows["s0", char][0].glyphs for char in alphabet]
        assert len(set(units)) == 23
        assert sorted(map(len, units)) == ([1] * 23 if family == "A" else [1] * 6 + [2] * 17)
        assert encode_records(["ab", "ba"], channel) == [units[0] + units[1], units[1] + units[0]]


def test_null_preserves_each_record_length_and_glyph_counts():
    import random
    records = ["AAAAABBCD", "CCDDEEFF"]
    shuffled = shuffled_records(records, random.Random(14))
    assert [Counter(row) for row in records] == [Counter(row) for row in shuffled]
    assert records != shuffled


def test_null_code_and_normalized_geometric_probability():
    model = fit_glyph_baseline(["AB", "BA"], ("A", "B"))
    assert model["counts"] == [128, 128]
    assert model["model_bits"] == len(model["model_code"]) == 21
    rho = model["stop_count"] / 4096
    assert math.exp(baseline_log_likelihood(model, ["AB"])) == pytest.approx(rho * (1 - rho) ** 2 / 4)
    assert math.exp(baseline_log_likelihood(model, [""])) == pytest.approx(rho)
    assert sum(fit_glyph_baseline(["AAAA"], ("A", "B"))["counts"]) == 256
    assert min(fit_glyph_baseline(["AAAA"], ("A", "B"))["counts"]) >= 1


def test_literal_recovery_and_abstention_accounting():
    source = SourceModel(("a", "b"), 0, {"": {"a": .75, "b": .25}})
    channel = Channel(("s",), ("A", "B"), {"s": 1.},
                      {("s", "a"): (Emission("s", "A", 1.),),
                       ("s", "b"): (Emission("s", "B", 1.),)}, .5)
    evaluated = evaluate_channel(source, channel, ["AB", "BA"], ["ab", "bb"])
    assert evaluated["edits"] == 1
    assert evaluated["exact_records"] == 1
    assert evaluated["decoded_characters"] == evaluated["gold_characters"] == 4
    assert evaluated["log_likelihood"] == pytest.approx(2 * math.log(.5 ** 3 * .75 * .25))
    abstained = evaluate_channel(source, None, ["AB"], ["ab"])
    assert abstained["edits"] == 2 and abstained["log_likelihood"] is None
    assert abstained["decoded_characters"] == abstained["exact_records"] == 0
    null = evaluate_channel(source, channel, ["AB"])
    assert null["gold_characters"] is null["edits"] is null["exact_records"] is None
