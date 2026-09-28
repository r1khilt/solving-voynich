import itertools
import math

import pytest

from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import (
    CodingContext, _composition_rank, channel_code_bits, encode_channel, fit_em,
    quantize_channel, two_part_score,
)


def models(probability=.5):
    source = SourceModel(("a",), 0, {"": {"a": 1.}})
    channel = Channel(("s",), ("x",), {"s": 1.},
                      {("s", "a"): (Emission("s", "x", probability),
                                       Emission("s", "xx", 1 - probability))}, .5)
    return source, channel


def context(**kwargs):
    return CodingContext(**({"source_alphabet": ("a",), "glyph_alphabet": ("x",),
                             "denominator": 4, "max_states": 2, "max_emission_length": 3,
                             "max_alternatives": 2, "stop_probability": .5} | kwargs))


def test_em_improves_marginal_likelihood_and_preserves_observations_and_support():
    source, channel = models()
    before = channel.to_dict()
    result = fit_em(source, channel, ["xx"], max_iterations=10)
    assert result.initial_log_likelihood == pytest.approx(math.log(5 / 32))
    assert result.log_likelihood > result.initial_log_likelihood
    assert all(row["gain"] >= -1e-12 for row in result.trace)
    assert [(item.next_state, item.glyphs) for item in result.channel.rows[("s", "a")]] == [("s", "x"), ("s", "xx")]
    assert channel.to_dict() == before
    assert result.iterations == len(result.trace)
    assert fit_em(source, channel, ["xx"], max_iterations=0).channel.to_dict() == before


def test_composition_ranks_are_a_bijection_on_both_weight_spaces():
    for positive in (False, True):
        for size in range(1, 5):
            for total in range(size if positive else 1, 7):
                compositions = [row for row in itertools.product(range(int(positive), total + 1), repeat=size)
                                if sum(row) == total]
                ranks = [_composition_rank(row, total, positive) for row in compositions]
                assert ranks == [(i, len(compositions)) for i in range(len(compositions))]


def test_code_is_prefix_free_for_complete_small_conditional_model_family():
    # Shared singleton alphabets, one state, lengths1..2, at most2 alternatives,
    # denominator3; enumerate all distinct supports/orders and positive weights.
    ctx = context(denominator=3, max_states=1, max_emission_length=2)
    channels = []
    for size in (1, 2):
        for units in itertools.permutations(("x", "xx"), size):
            for counts in itertools.product(range(1, 4), repeat=size):
                if sum(counts) != 3:
                    continue
                row = tuple(Emission("s", unit, count / 3) for unit, count in zip(units, counts, strict=True))
                channels.append(Channel(("s",), ("x",), {"s": 1.}, {("s", "a"): row}, .5))
    codes = [encode_channel(channel, ctx) for channel in channels]
    assert len(set(codes)) == len(codes)
    assert all(not right.startswith(left) for i, left in enumerate(codes) for j, right in enumerate(codes) if i != j)
    assert sum(2**-len(code) for code in codes) <= 1


def test_quantization_is_explicit_and_model_description_charges_more_support():
    source, channel = models(.6)
    with pytest.raises(ValueError, match="grid"):
        channel_code_bits(channel, context())
    rounded = quantize_channel(channel, 4)
    assert [item.probability for item in rounded.rows[("s", "a")]] == [.5, .5]
    score = two_part_score(source, rounded, ["xx"], context())
    assert score["model_bits"] == len(encode_channel(rounded, context()))
    assert score["data_bits"] == pytest.approx(-math.log2(5 / 32))
    simple = Channel(("s",), ("x",), {"s": 1.}, {("s", "a"): (Emission("s", "x", 1.),)}, .5)
    assert channel_code_bits(simple, context()) < channel_code_bits(rounded, context())
    assert score["total_bits"] == score["model_bits"] + score["data_bits"]


def test_quantizer_merges_duplicates_and_drops_only_exact_zero_alternatives():
    channel = Channel(("s",), ("x",), {"s": 1.},
                      {("s", "a"): (Emission("s", "x", .25), Emission("s", "x", .25),
                                       Emission("s", "xx", .5), Emission("s", "xxx", 0.))}, .5)
    rounded = quantize_channel(channel, 4)
    assert rounded.rows[("s", "a")] == (Emission("s", "x", .5), Emission("s", "xx", .5))
    with pytest.raises(ValueError, match="coarse"):
        quantize_channel(channel, 1)


@pytest.mark.parametrize("records", [[], "xx", [5], ["y"]])
def test_invalid_or_impossible_training_records_fail(records):
    source, channel = models()
    with pytest.raises(ValueError):
        fit_em(source, channel, records, max_iterations=1)
