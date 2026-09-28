"""Tiny independent checks of bounded, observation-only channel search.

The oracle below enumerates complete latent paths and counts model-description
fields directly. It never calls production likelihood, decoder or code helpers.
No corpora, generated recovery benchmark, segmentation or answer files are used.
"""

from __future__ import annotations

import math
import json
from dataclasses import replace
from fractions import Fraction as F

import pytest

from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich import finite_state_channel_search as search_module
from voynich.finite_state_channel_search import SearchConfig, observed_units, search_channel


def independent_mass(source, channel, observation):
    """Rational path enumeration from model primitives, without lattice helpers."""
    rho = F.from_float(channel.stop_probability)

    def expand(offset, state, context):
        if offset == len(observation):
            return rho
        mass = F(0)
        for letter, weight in source.probabilities[context].items():
            next_context = letter if source.order else ""
            for alternative in channel.rows[state, letter]:
                if observation.startswith(alternative.glyphs, offset):
                    mass += ((1 - rho) * F.from_float(weight) * F.from_float(alternative.probability)
                             * expand(offset + len(alternative.glyphs), alternative.next_state, next_context))
        return mass

    return sum((F.from_float(weight) * expand(0, state, "")
                for state, weight in channel.initial.items()), F(0))


def independent_model_bits(channel, context):
    """Count each encoded field from its independently specified cardinality."""
    def width(possibilities):
        return (possibilities - 1).bit_length()

    states = len(channel.states)
    bits = width(context.source_count) + width(context.max_states)
    bits += width(math.comb(context.denominator + states - 1, states - 1))
    for row in channel.rows.values():
        bits += width(context.max_alternatives)
        for alternative in row:
            bits += width(states) + width(context.max_emission_length)
            bits += len(alternative.glyphs) * width(len(context.glyph_alphabet))
        bits += width(math.comb(context.denominator - 1, len(row) - 1))
    return bits


def assert_independent_score(source, channel, records, context, score):
    masses = [independent_mass(source, channel, record) for record in records]
    assert all(masses)
    log_probability = math.fsum(math.log(p.numerator) - math.log(p.denominator) for p in masses)
    model_bits = independent_model_bits(channel, context)
    assert score["model_bits"] == model_bits
    assert score["log_likelihood"] == pytest.approx(log_probability, abs=3e-11)
    assert score["data_bits"] == pytest.approx(-log_probability / math.log(2), abs=3e-11)
    assert score["total_bits"] == pytest.approx(model_bits - log_probability / math.log(2), abs=3e-11)


def assert_coded_channel(channel, context, records):
    assert len(channel.states) <= context.max_states
    assert channel.source_alphabet == set(context.source_alphabet)
    assert channel.glyph_alphabet == context.glyph_alphabet
    assert channel.stop_probability == context.stop_probability
    assert math.fsum(channel.initial.values()) == pytest.approx(1.0, abs=1e-13)
    for weight in channel.initial.values():
        assert weight * context.denominator == pytest.approx(round(weight * context.denominator), abs=1e-13)
    allowed_units = set(context.glyph_alphabet) | {
        record[start:start + length] for record in records for start in range(len(record))
        for length in range(2, min(context.max_emission_length, len(record) - start) + 1)
    }
    for row in channel.rows.values():
        assert 1 <= len(row) <= min(context.max_alternatives, context.denominator)
        assert math.fsum(item.probability for item in row) == pytest.approx(1.0, abs=1e-13)
        assert len({(item.next_state, item.glyphs) for item in row}) == len(row)
        for item in row:
            assert item.next_state in channel.states
            assert item.glyphs in allowed_units
            assert 0 < item.probability <= 1
            assert item.probability * context.denominator == pytest.approx(
                round(item.probability * context.denominator), abs=1e-13,
            )


def small_source():
    return SourceModel(
        ("a", "b"), 1,
        {"": {"a": 0.75, "b": 0.25}, "a": {"a": 0.25, "b": 0.75}, "b": {"a": 0.5, "b": 0.5}},
    )


def small_context():
    return CodingContext(("a", "b"), ("x", "y"), 4, 2, 2, 3, stop_probability=0.5)


def test_unit_pool_keeps_declared_singletons_counts_overlaps_and_never_crosses_records():
    context = replace(small_context(), max_emission_length=3)
    # xx occurs twice by overlapping in xxx; xy and yx each occur only once.
    assert observed_units(("xxx", "xy", "yx"), context, max_units=5) == ("x", "y", "xx", "xy", "yx")
    # Concatenating records would incorrectly propose xy; empty records add none.
    assert observed_units(("x", "", "y"), context) == ("x", "y")
    assert observed_units(("xxx",), context) == ("x", "y", "xx", "xxx")
    assert observed_units(("", ""), context) == ("x", "y")
    with pytest.raises(ValueError, match="singleton"):
        observed_units(("xy",), context, max_units=1)


def test_every_search_stage_uses_full_observed_likelihood_and_actual_coded_weights():
    source, context = small_source(), replace(small_context(), source_count=3)
    records = ("", "x", "xy", "yx", "xxx")
    before = source.to_dict()
    config = SearchConfig(seed=51, state_counts=(1, 2), restarts_per_state=2, proposals_per_restart=12,
                          em_iterations=2, max_seconds=60, max_units=6)
    result = search_channel(source, records, context, config=config, source_index=2)
    scored = []
    for event in result.trace:
        stage_scores = []
        for stage in event["stages"]:
            channel = Channel.from_dict(stage["channel"])
            assert_coded_channel(channel, context, records)
            if stage["status"] == "scored":
                assert stage["records_scored"] == len(records)
                assert_independent_score(source, channel, records, context, stage["score"])
                stage_scores.append(stage["score"]["total_bits"])
                scored.append(stage["score"]["total_bits"])
            else:
                assert stage["score"] is None
        if stage_scores:
            assert event["selected_score"]["total_bits"] == min(stage_scores)
            selected = Channel.from_dict(event["selected_channel"])
            assert_independent_score(source, selected, records, context, event["selected_score"])
        if event["event"] == "initialization":
            # The initializer's raw channel covers all declared singletons at
            # every state, before any probabilities or structures are refined.
            initial = Channel.from_dict(event["stages"][0]["channel"])
            for state in initial.states:
                alternatives = [item for letter in source.alphabet for item in initial.rows[state, letter]]
                assert {item.glyphs for item in alternatives} == set(context.glyph_alphabet)
                assert all(item.next_state == state for item in alternatives)
        elif event["accepted"]:
            assert (event["parent_score"]["total_bits"] - event["selected_score"]["total_bits"]
                    > config.improvement_tolerance_bits)
        if scored:
            assert event["best_total_bits"] == min(scored)
    assert result.proposals == 2 * 2 * 12
    assert result.completed_candidates == len(scored)
    assert result.stop_reason == "budget_exhausted"
    assert result.score["total_bits"] == min(scored)
    assert_independent_score(source, result.channel, records, context, result.score)
    assert source.to_dict() == before
    assert any(stage["phase"] == "refined" for event in result.trace for stage in event["stages"])
    json.dumps(result.to_dict(), allow_nan=False)


def test_seed_and_count_budget_reproduce_every_candidate_and_decision_without_deadline():
    source, context = small_source(), small_context()
    config = SearchConfig(seed=310, state_counts=(2, 1), restarts_per_state=2, proposals_per_restart=10,
                          em_iterations=1, max_seconds=60)
    first = search_channel(source, ("x", "xy", "yx"), context, config=config).to_dict()
    second = search_channel(source, ("x", "xy", "yx"), context, config=config).to_dict()
    first.pop("seconds")
    second.pop("seconds")
    assert first == second
    initializations = [event for event in first["trace"] if event["event"] == "initialization"]
    assert [(event["restart"], event["state_count"]) for event in initializations] == [(0, 2), (0, 1), (1, 2), (1, 1)]


def test_singleton_initializer_capacity_failure_does_not_claim_channel_nonexistence():
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    context = CodingContext(("a",), ("x", "y"), 1, 1, 2, 1, stop_probability=0.5)
    with pytest.raises(ValueError, match="Singleton initialization capacity"):
        search_channel(source, ("xy",), context, config=SearchConfig(state_counts=(1,)))
    # The absent y singleton still consumes required initializer capacity.
    with pytest.raises(ValueError, match="Singleton initialization capacity"):
        search_channel(source, ("x",), context, config=SearchConfig(state_counts=(1,)))
    # A deterministic xy emission would be valid under exactly these grammar
    # bounds. Rejection is caused by the singleton initialization strategy.


def test_zero_source_support_exhausts_declared_attempts_without_hidden_fallback():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1.0, "b": 0.0}})
    context = CodingContext(("a", "b"), ("x", "y"), 1, 1, 1, 1, stop_probability=0.5)
    config = SearchConfig(seed=9, state_counts=(1,), restarts_per_state=2, initialization_attempts=3,
                          proposals_per_restart=8, em_iterations=0)
    result = search_channel(source, ("xy",), context, config=config)
    assert result.channel is None and result.score is None
    assert result.stop_reason == "no_supported_initialization"
    assert result.proposals == result.completed_candidates == 0
    assert len(result.trace) == 6
    assert all(event["rejection_reason"] == "unsupported" for event in result.trace)
    assert all(not event["accepted"] for event in result.trace)
    json.dumps(result.to_dict(), allow_nan=False)


def test_partial_initial_score_is_never_returned_as_a_completed_candidate(monkeypatch):
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    context = CodingContext(("a",), ("x",), 2, 1, 2, 2, stop_probability=0.5)
    moments = iter((0.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3))
    monkeypatch.setattr(search_module.time, "monotonic", lambda: next(moments))
    config = SearchConfig(state_counts=(1,), restarts_per_state=1, initialization_attempts=1,
                          proposals_per_restart=0, em_iterations=0, max_seconds=1)
    result = search_channel(source, ("x", "x"), context, config=config)
    assert result.channel is None and result.score is None
    assert result.stop_reason == "time_limit"
    assert result.completed_candidates == result.proposals == 0
    assert result.trace[0]["stages"][0]["records_scored"] == 1
    assert result.trace[0]["stages"][0]["score"] is None


def test_partial_proposal_score_cannot_replace_completed_incumbent(monkeypatch):
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    context = CodingContext(("a",), ("x",), 2, 1, 2, 2, stop_probability=0.5)
    proposed = Channel(("s0",), ("x",), {"s0": 1.0},
                       {("s0", "a"): (Emission("s0", "xx", 1.0),)}, 0.5, 2)
    # Force a legal better proposal so the clock test is independent of random
    # move selection. Interrupt after one of its two record scores completes.
    monkeypatch.setattr(search_module, "_propose", lambda *args: (proposed, {"kind": "test_fixture"}))
    current_time = [0.0]
    monkeypatch.setattr(search_module.time, "monotonic", lambda: current_time[0])
    original = search_module.forward_log_probability
    calls = [0]

    def score_and_expire(*args):
        calls[0] += 1
        result = original(*args)
        if calls[0] == 3:
            current_time[0] = 2.0
        return result

    monkeypatch.setattr(search_module, "forward_log_probability", score_and_expire)
    config = SearchConfig(state_counts=(1,), restarts_per_state=1, proposals_per_restart=1,
                          em_iterations=0, max_seconds=1)
    result = search_channel(source, ("xx", "xx"), context, config=config)
    assert result.channel.rows[("s0", "a")][0].glyphs == "x"
    assert result.stop_reason == "time_limit"
    assert result.completed_candidates == result.proposals == 1
    assert result.trace[-1]["stages"][0]["status"] == "time_limit"
    assert result.trace[-1]["stages"][0]["records_scored"] == 1
    assert not result.trace[-1]["accepted"]
    assert_independent_score(source, result.channel, ("xx", "xx"), context, result.score)


def test_fully_scored_raw_candidate_survives_interrupted_refinement(monkeypatch):
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    context = CodingContext(("a",), ("x", "y"), 4, 1, 1, 2, stop_probability=0.5)
    records = ("x", "x", "x", "y")
    # Force the clock to expire only after the real EM refinement has finished.
    # The refined probabilities differ, but their explicit re-score is blocked.
    current_time = [0.0]
    monkeypatch.setattr(search_module.time, "monotonic", lambda: current_time[0])
    original = search_module.fit_em

    def fit_then_expire(*args, **kwargs):
        fitted = original(*args, **kwargs)
        current_time[0] = 2.0
        return fitted

    monkeypatch.setattr(search_module, "fit_em", fit_then_expire)
    config = SearchConfig(state_counts=(1,), restarts_per_state=1, proposals_per_restart=0,
                          em_iterations=1, max_seconds=1)
    result = search_channel(source, records, context, config=config)
    assert result.stop_reason == "time_limit"
    assert result.completed_candidates == 1
    assert [stage["status"] for stage in result.trace[0]["stages"]] == ["scored", "time_limit"]
    assert [item.probability for item in result.channel.rows[("s0", "a")]] == [0.5, 0.5]
    assert result.trace[0]["accepted"]
    assert_independent_score(source, result.channel, records, context, result.score)


def test_declared_but_unobserved_glyph_remains_in_search_pool_and_initial_support():
    source, context = small_source(), small_context()
    config = SearchConfig(state_counts=(1, 2), restarts_per_state=1, proposals_per_restart=0, em_iterations=0)
    result = search_channel(source, ("xx",), context, config=config)
    assert result.unit_pool == ("x", "y", "xx")
    for event in result.trace:
        initial = Channel.from_dict(event["stages"][0]["channel"])
        assert_coded_channel(initial, context, ("xx",))
        assert independent_mass(source, initial, "y") > 0
        for state in initial.states:
            assert {item.glyphs for letter in source.alphabet for item in initial.rows[state, letter]} == {"x", "y"}


def test_empty_only_records_have_stop_likelihood_and_can_use_declared_glyph_support():
    source, context = small_source(), small_context()
    config = SearchConfig(state_counts=(1,), restarts_per_state=1, proposals_per_restart=0, em_iterations=1)
    result = search_channel(source, ("", ""), context, config=config)
    assert result.unit_pool == ("x", "y")
    assert result.score["data_bits"] == 2
    assert result.score["log_likelihood"] == pytest.approx(math.log(F(1, 4)), abs=2e-13)
    assert_independent_score(source, result.channel, ("", ""), context, result.score)


def test_frequency_initializer_uses_source_context_and_fit_counts_without_fitting_source():
    source = SourceModel(("a", "b"), 1,
                         {"": {"a": 1.0, "b": 0.0}, "a": {"a": 0.0, "b": 1.0},
                          "b": {"a": 0.0, "b": 1.0}})
    context = CodingContext(("a", "b"), ("x", "y"), 4, 1, 1, 1, stop_probability=0.25)
    # The source starts with a then remains b. Conditional on a source step,
    # survival-weighted letter proportions approach (1/4,3/4); the 256-term
    # normalized values are a=1/sum(k=0..255)(3/4)^k and b=1-a.
    total = sum((F(3, 4) ** k for k in range(256)), F(0))
    expected = {"a": float(1 / total), "b": float(1 - 1 / total)}
    assert search_module._source_frequencies(source, 0.25) == pytest.approx(expected, abs=2e-13)
    before = source.to_dict()
    config = SearchConfig(seed=19, state_counts=(1,), restarts_per_state=2, proposals_per_restart=0,
                          em_iterations=0, initialization_attempts=1)
    result = search_channel(source, ("yxxx",), context, config=config)
    first = result.trace[0]
    assert first["method"] == "frequency"
    channel = Channel.from_dict(first["stages"][0]["channel"])
    assert channel.rows[("s0", "b")][0].glyphs == "x"
    assert channel.rows[("s0", "a")][0].glyphs == "y"
    assert all(event["method"] == "random" for event in result.trace[1:])
    assert source.to_dict() == before
    random_only = search_channel(source, ("yxxx",), context,
                                 config=replace(config, frequency_initialization=False))
    assert all(event["method"] == "random" for event in random_only.trace)


def test_frequency_ties_use_declared_order_and_fixed_rows_skip_redundant_em(monkeypatch):
    source = SourceModel(("b", "a"), 0, {"": {"b": 0.5, "a": 0.5}})
    context = CodingContext(("b", "a"), ("y", "x"), 4, 1, 1, 1, stop_probability=0.5)

    def redundant_fit(*args, **kwargs):
        pytest.fail("EM cannot change singleton rows; the redundant fit should be skipped")

    monkeypatch.setattr(search_module, "fit_em", redundant_fit)
    config = SearchConfig(state_counts=(1,), restarts_per_state=1, proposals_per_restart=0, em_iterations=2)
    result = search_channel(source, ("xy",), context, config=config)
    assert result.channel.rows[("s0", "b")][0].glyphs == "y"
    assert result.channel.rows[("s0", "a")][0].glyphs == "x"
    assert "em" not in result.trace[0]
    assert_independent_score(source, result.channel, ("xy",), context, result.score)
