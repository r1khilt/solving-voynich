"""Independent tiny-fixture checks of observation-only emission-weight EM.

Closed-form rational examples check the M step. A separate exhaustive path sum
checks observed likelihoods without using forward/backward or decoder outputs.
There are no corpora, reference plaintexts, supplied segmentations, or answers.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction as F

import pytest

from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich import finite_state_channel_fit as fit_module
from voynich.finite_state_channel_fit import (
    CodingContext, channel_code_bits, encode_channel, fit_em, quantize_channel, two_part_score,
)


def one_letter_channel(x_weight=0.5):
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    channel = Channel(
        states=("s",), glyph_alphabet=("x",), initial={"s": 1.0},
        rows={("s", "a"): (Emission("s", "x", x_weight), Emission("s", "xx", 1 - x_weight))},
        stop_probability=0.5, max_emission_length=2,
    )
    return source, channel


def row_weights(channel, key=("s", "a")):
    return tuple(emission.probability for emission in channel.rows[key])


def exhaustive_probability(source, channel, observation):
    """Enumerate all complete paths directly, using rational arithmetic.

    The public probability values define the model, but none of the inference
    or fitting implementation is called. Each recursive edge consumes glyphs.
    """
    stop = F.from_float(channel.stop_probability)

    def visit(offset, context, state):
        if offset == len(observation):
            return stop
        total = F(0)
        for letter, source_probability in source.probabilities[context].items():
            next_context = letter if source.order else ""
            for event in channel.rows[(state, letter)]:
                if observation.startswith(event.glyphs, offset):
                    total += ((1 - stop) * F.from_float(source_probability)
                              * F.from_float(event.probability)
                              * visit(offset + len(event.glyphs), next_context, event.next_state))
        return total

    return sum((F.from_float(probability) * visit(0, "", state)
                for state, probability in channel.initial.items()), F(0))


def exhaustive_log_likelihood(source, channel, observations):
    probabilities = [exhaustive_probability(source, channel, observed) for observed in observations]
    assert all(probabilities), "All hand fixtures used for fitting must have nonzero likelihood"
    return math.fsum(math.log(p.numerator) - math.log(p.denominator) for p in probabilities)


def assert_fixed_structure(before, after):
    assert before.states == after.states
    assert before.glyph_alphabet == after.glyph_alphabet
    assert before.initial == after.initial
    assert before.stop_probability == after.stop_probability
    assert before.max_emission_length == after.max_emission_length
    assert before.rows.keys() == after.rows.keys()
    for key, original in before.rows.items():
        assert [(event.next_state, event.glyphs) for event in original] == [
            (event.next_state, event.glyphs) for event in after.rows[key]
        ]
        assert math.fsum(event.probability for event in after.rows[key]) == pytest.approx(1.0, abs=1e-13)


def test_one_step_integrates_unknown_number_and_boundaries_of_emissions():
    source, channel = one_letter_channel()
    # For xx, the one-event path has mass 1/8 and the two-event path 1/32.
    # Their posterior masses are 4/5 and 1/5, hence counts (2/5, 4/5).
    before = channel.to_dict()
    result = fit_em(source, channel, ("xx",), max_iterations=1, tolerance=0)
    assert row_weights(result.channel) == pytest.approx((F(1, 3), F(2, 3)), abs=2e-13)
    assert result.initial_log_likelihood == pytest.approx(math.log(F(5, 32)), abs=2e-13)
    assert result.log_likelihood == pytest.approx(math.log(F(13, 72)), abs=2e-13)
    assert result.iterations == 1
    assert result.log_likelihood > result.initial_log_likelihood
    assert channel.to_dict() == before
    assert_fixed_structure(channel, result.channel)


def test_independent_records_sum_counts_before_normalizing_each_row():
    source, channel = one_letter_channel()
    # x contributes (1, 0); xx contributes (2/5, 4/5). Averaging their
    # separately normalized rows would incorrectly produce (2/3, 1/3).
    result = fit_em(source, channel, ("x", "xx"), max_iterations=1, tolerance=0)
    assert row_weights(result.channel) == pytest.approx((F(7, 11), F(4, 11)), abs=2e-13)
    assert result.initial_log_likelihood == pytest.approx(math.log(F(5, 256)), abs=2e-13)
    assert result.log_likelihood == pytest.approx(math.log(F(959, 42592)), abs=2e-13)


def test_each_source_conditioned_row_has_its_own_expected_occupancy():
    source = SourceModel(("a", "b"), 0, {"": {"a": 0.75, "b": 0.25}})
    channel = Channel(
        ("s",), ("x", "y"), {"s": 1.0},
        {("s", "a"): (Emission("s", "x", 0.25), Emission("s", "y", 0.75)),
         ("s", "b"): (Emission("s", "x", 0.75), Emission("s", "y", 0.25))},
        0.5,
    )
    original_source = source.to_dict()
    # Each x assigns source posterior (1/2, 1/2), and y assigns (9/10, 1/10).
    # Two x records and one y give row counts a=(1,9/10), b=(1,1/10).
    result = fit_em(source, channel, ("x", "x", "y"), max_iterations=1, tolerance=0)
    assert row_weights(result.channel, ("s", "a")) == pytest.approx((F(10, 19), F(9, 19)), abs=2e-13)
    assert row_weights(result.channel, ("s", "b")) == pytest.approx((F(10, 11), F(1, 11)), abs=2e-13)
    assert source.to_dict() == original_source
    assert_fixed_structure(channel, result.channel)


def test_empty_records_contribute_stop_mass_but_no_emission_counts():
    source, channel = one_letter_channel()
    result = fit_em(source, channel, ("", ""), max_iterations=3, tolerance=0)
    assert result.channel.to_dict() == channel.to_dict()
    assert result.initial_log_likelihood == pytest.approx(math.log(F(1, 4)), abs=2e-13)
    assert result.log_likelihood == pytest.approx(math.log(F(1, 4)), abs=2e-13)
    mixed = fit_em(source, channel, ("", "xx", ""), max_iterations=1, tolerance=0)
    assert row_weights(mixed.channel) == pytest.approx((F(1, 3), F(2, 3)), abs=2e-13)
    assert mixed.log_likelihood == pytest.approx(math.log(F(13, 288)), abs=2e-13)


def test_unoccupied_state_and_zero_source_rows_are_preserved_without_smoothing():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1.0, "b": 0.0}})
    channel = Channel(
        ("live", "unused"), ("x", "y"), {"live": 1.0, "unused": 0.0},
        {("live", "a"): (Emission("live", "x", 0.25), Emission("live", "y", 0.75)),
         ("live", "b"): (Emission("live", "x", 0.5), Emission("unused", "y", 0.5)),
         ("unused", "a"): (Emission("unused", "x", 0.75), Emission("live", "y", 0.25)),
         ("unused", "b"): (Emission("live", "x", 0.125), Emission("unused", "y", 0.875))},
        0.5,
    )
    result = fit_em(source, channel, ("x",), max_iterations=1, tolerance=0)
    assert row_weights(result.channel, ("live", "a")) == (1.0, 0.0)
    for key in (("live", "b"), ("unused", "a"), ("unused", "b")):
        assert result.channel.rows[key] == channel.rows[key]
    assert_fixed_structure(channel, result.channel)


def test_zero_weight_alternatives_cannot_be_revived_by_em():
    source, channel = one_letter_channel(x_weight=1.0)
    # xx can be explained by two x events, but the zero-weight xx alternative
    # has zero posterior responsibility and stays zero, despite its better fit.
    result = fit_em(source, channel, ("xx",), max_iterations=3, tolerance=0)
    assert row_weights(result.channel) == (1.0, 0.0)
    assert result.log_likelihood == pytest.approx(math.log(F(1, 8)), abs=2e-13)
    assert_fixed_structure(channel, result.channel)


def test_duplicate_latent_alternatives_retain_their_separate_responsibilities():
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    channel = Channel(
        ("s",), ("x",), {"s": 1.0},
        {("s", "a"): (Emission("s", "x", 0.25), Emission("s", "x", 0.5),
                      Emission("s", "xx", 0.25))},
        0.5, max_emission_length=2,
    )
    result = fit_em(source, channel, ("x",), max_iterations=1, tolerance=0)
    assert row_weights(result.channel) == pytest.approx((F(1, 3), F(2, 3), 0.0), abs=2e-13)
    assert result.log_likelihood == pytest.approx(math.log(F(1, 4)), abs=2e-13)
    assert_fixed_structure(channel, result.channel)


def stateful_channel():
    source = SourceModel(
        ("a", "b"), 1,
        {"": {"a": 0.75, "b": 0.25}, "a": {"a": 0.25, "b": 0.75}, "b": {"a": 0.5, "b": 0.5}},
    )
    channel = Channel(
        ("u", "v"), ("x", "y"), {"u": 0.25, "v": 0.75},
        {("u", "a"): (Emission("u", "x", 0.5), Emission("v", "xy", 0.25), Emission("u", "y", 0.25)),
         ("u", "b"): (Emission("v", "y", 0.75), Emission("u", "x", 0.25)),
         ("v", "a"): (Emission("u", "xx", 0.5), Emission("v", "y", 0.5)),
         ("v", "b"): (Emission("u", "xy", 0.25), Emission("v", "x", 0.75))},
        0.25, max_emission_length=2,
    )
    return source, channel


@pytest.mark.parametrize("fixture,records", [
    (one_letter_channel, ("", "x", "xx", "xxx", "xxxx")),
    (stateful_channel, ("", "x", "xy", "y", "xx", "yx", "xyx")),
])
def test_every_em_step_improves_independently_summed_observed_likelihood(fixture, records):
    source, initial = fixture()
    source_before = source.to_dict()
    current = initial
    previous = exhaustive_log_likelihood(source, current, records)
    for _ in range(6):
        fitted = fit_em(source, current, records, max_iterations=1, tolerance=0)
        verified = exhaustive_log_likelihood(source, fitted.channel, records)
        assert fitted.initial_log_likelihood == pytest.approx(previous, abs=3e-12)
        assert fitted.log_likelihood == pytest.approx(verified, abs=3e-12)
        assert verified >= previous - 3e-12
        assert_fixed_structure(initial, fitted.channel)
        current, previous = fitted.channel, verified
    # A multi-iteration call must follow the same updates as six separate calls.
    combined = fit_em(source, initial, records, max_iterations=6, tolerance=0)
    assert combined.log_likelihood == pytest.approx(previous, abs=3e-12)
    for key in initial.rows:
        assert row_weights(combined.channel, key) == pytest.approx(row_weights(current, key), abs=3e-12)
    assert source.to_dict() == source_before


def test_impossible_record_is_rejected_instead_of_silently_excluded():
    source, channel = one_letter_channel()
    with pytest.raises(ValueError):
        fit_em(source, channel, ("xx", "z"), max_iterations=1)
    # This string uses a declared glyph but is outside the supported lengths.
    _, only_pairs = one_letter_channel(x_weight=0.0)
    with pytest.raises(ValueError):
        fit_em(source, only_pairs, ("xx", "x"), max_iterations=1)


def test_finite_description_pays_for_source_support_glyphs_and_positive_weight_rank():
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    channel = Channel(
        ("s",), ("x", "y"), {"s": 1.0},
        {("s", "a"): (Emission("s", "x", 0.25), Emission("s", "xy", 0.75))},
        0.5, max_emission_length=2,
    )
    context = CodingContext(("a",), ("x", "y"), 4, 2, 2, 3, source_count=3, stop_probability=0.5)
    # Language index 2: 10. One of <=2 states: 0. Singleton initial row: no bits.
    # Two of <=3 alternatives: 01. Their length/glyph fields: 00 and 101.
    # Positive D=4 two-cell compositions are (1,3),(2,2),(3,1); rank zero: 00.
    expected = "10" + "0" + "01" + "00" + "101" + "00"
    assert encode_channel(channel, context, source_index=2) == expected
    assert channel_code_bits(channel, context, source_index=2) == 12
    # Only the whole xy emission can account for this record, with mass 3/16.
    score = two_part_score(source, channel, ("xy",), context, source_index=2)
    assert score["model_bits"] == 12
    assert score["data_bits"] == pytest.approx(4 - math.log2(3), abs=2e-13)
    assert score["total_bits"] == pytest.approx(16 - math.log2(3), abs=2e-13)


@pytest.mark.parametrize("initial,initial_bits", [
    ({"u": 0.0, "v": 1.0}, "00"),
    ({"u": 0.5, "v": 0.5}, "01"),
    ({"u": 1.0, "v": 0.0}, "10"),
])
def test_initial_distribution_uses_weak_compositions_with_zero_states_allowed(initial, initial_bits):
    channel = Channel(
        ("u", "v"), ("x",), initial,
        {("u", "a"): (Emission("v", "x", 1.0),),
         ("v", "a"): (Emission("u", "x", 1.0),)},
        0.5, max_emission_length=1,
    )
    context = CodingContext(("a",), ("x",), 2, 2, 1, 1, stop_probability=0.5)
    # Weak D=2 two-state compositions: (0,2),(1,1),(2,0). Following the
    # state-count and initial fields, only the two destination bits cost space.
    assert encode_channel(channel, context) == "1" + initial_bits + "1" + "0"
    assert channel_code_bits(channel, context) == 5


@pytest.mark.parametrize("times,completed", [
    ([0.0, 0.1, 0.2, 0.3, 1.1, 1.2], 0),  # Stop partway through candidate scoring.
    ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2], 1),  # Stop in the next E step.
])
def test_deadline_returns_last_fully_scored_model_without_partial_update(monkeypatch, times, completed):
    source, channel = one_letter_channel()
    moments = iter(times)
    monkeypatch.setattr(fit_module.time, "monotonic", lambda: next(moments))
    result = fit_em(source, channel, ("x", "xx"), max_iterations=3, tolerance=0, max_seconds=1)
    expected = (F(7, 11), F(4, 11)) if completed else (F(1, 2), F(1, 2))
    assert row_weights(result.channel) == pytest.approx(expected, abs=2e-13)
    assert result.stop_reason == "time_limit"
    assert result.iterations == len(result.trace) == completed
    assert result.log_likelihood == pytest.approx(
        exhaustive_log_likelihood(source, result.channel, ("x", "xx")), abs=2e-13,
    )


def test_quantization_can_undo_likelihood_gain_and_must_be_scored_separately():
    source, channel = one_letter_channel()
    fitted = fit_em(source, channel, ("xx",), max_iterations=1, tolerance=0)
    context = CodingContext(("a",), ("x",), 2, 1, 2, 2, stop_probability=0.5)
    with pytest.raises(ValueError, match="grid"):
        encode_channel(fitted.channel, context)
    quantized = quantize_channel(fitted.channel, 2)
    assert row_weights(quantized) == (0.5, 0.5)
    assert exhaustive_probability(source, quantized, "xx") == F(5, 32)
    assert exhaustive_log_likelihood(source, quantized, ("xx",)) < fitted.log_likelihood
    assert encode_channel(quantized, context)


@pytest.mark.parametrize("change", [
    {"source_alphabet": ("b",)}, {"glyph_alphabet": ("y",)}, {"stop_probability": 0.25},
    {"max_emission_length": 1}, {"max_alternatives": 1},
])
def test_model_cannot_be_encoded_using_incompatible_or_insufficient_shared_context(change):
    _, channel = one_letter_channel()
    context = CodingContext(("a",), ("x",), 2, 1, 2, 2, stop_probability=0.5)
    with pytest.raises(ValueError):
        encode_channel(channel, replace(context, **change))
