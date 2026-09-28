"""Artificial independent controls for deterministic literal-unit search.

Expected likelihoods use a backward Fraction recurrence. Expected description
lengths count the fixed binary fields directly; neither reference uses the
production search, batched likelihood, or finite-state score calculation.
"""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction as F
from functools import lru_cache

import pytest

from voynich import unit_channel_search as search_module
from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import CodingContext, encode_channel
from voynich.unit_channel_search import (
    UnitSearchConfig, channel_from_units, literal_unit_pool, search_unit_channel,
    unit_neighbors,
)


def exact_record_probability(source, units, record, rho):
    stop = F(rho)

    @lru_cache(None)
    def suffix(offset, context):
        if offset == len(record):
            return stop
        value = F(0)
        for letter, unit in zip(source.alphabet, units, strict=True):
            if record.startswith(unit, offset):
                next_context = letter if source.order else ""
                value += ((1 - stop) * F(source.probabilities[context][letter])
                          * suffix(offset + len(unit), next_context))
        return value

    return suffix(0, "")


def literal_model_bits(units, context):
    def width(possibilities):
        return (possibilities - 1).bit_length()

    # One state, one deterministic alternative per row: initial-state,
    # destination-state and probability-composition fields have zero width.
    return (width(context.source_count) + width(context.max_states)
            + len(units) * (width(context.max_alternatives) + width(context.max_emission_length))
            + width(len(context.glyph_alphabet)) * sum(map(len, units)))


def exact_score(source, units, records, context):
    probability = F(1)
    for record in records:
        probability *= exact_record_probability(source, units, record, context.stop_probability)
    likelihood = math.log(probability.numerator) - math.log(probability.denominator) if probability else -math.inf
    model = literal_model_bits(units, context)
    data = -likelihood / math.log(2)
    return {"log_likelihood": likelihood, "model_bits": model, "data_bits": data, "total_bits": model + data}


def complete_pool(glyphs, maximum_length):
    return tuple("".join(chars) for length in range(1, maximum_length + 1)
                 for chars in itertools.product(glyphs, repeat=length))


def distinct_neighbors(units, pool):
    candidates = set()
    for index in range(len(units)):
        for unit in pool:
            changed = list(units)
            changed[index] = unit
            if tuple(changed) != tuple(units):
                candidates.add(tuple(changed))
    for left, right in itertools.combinations(range(len(units)), 2):
        changed = list(units)
        changed[left], changed[right] = changed[right], changed[left]
        if tuple(changed) != tuple(units):
            candidates.add(tuple(changed))
    return candidates


def fixture(order=1):
    rows = {"": {"a": .75, "b": .25}}
    if order:
        rows.update(a={"a": .125, "b": .875}, b={"a": .625, "b": .375})
    source = SourceModel(("a", "b"), order, rows)
    context = CodingContext(source.alphabet, ("x", "y"), 8, 3, 2, 3,
                            source_count=3, stop_probability=.25)
    return source, context, ("xxxy", "xyx", "", "yy")


def test_variable_length_model_cost_matches_every_literal_binary_field():
    source, _, _ = fixture()
    context = CodingContext(source.alphabet, ("x", "λ", "\0"), 8, 3, 2, 3,
                            source_count=3, stop_probability=.25)
    pool = complete_pool(context.glyph_alphabet, 2)
    for units in itertools.product(pool, repeat=2):
        channel = Channel(("s",), context.glyph_alphabet, {"s": 1.},
                          {("s", letter): (Emission("s", unit, 1.),)
                           for letter, unit in zip(source.alphabet, units, strict=True)},
                          context.stop_probability, 2)
        # Source index2 -> 10, one state -> 00. Each row has alternative count
        # 00, length bit, then two bits per literal glyph. All weight and
        # destination fields have a single possible value and use zero bits.
        expected = "1000" + "".join(
            "00" + str(len(unit) - 1)
            + "".join(format(context.glyph_alphabet.index(glyph), "02b") for glyph in unit)
            for unit in units)
        actual = encode_channel(channel, context, source_index=2)
        assert actual == expected
        assert len(actual) == literal_model_bits(units, context)


def assert_score(actual, expected):
    if expected["log_likelihood"] == -math.inf:
        assert actual is None
    else:
        assert actual is not None
        assert actual["model_bits"] == expected["model_bits"]
        for key in ("log_likelihood", "data_bits", "total_bits"):
            assert actual[key] == pytest.approx(expected[key], rel=0, abs=6e-13)


def test_every_tiny_neighborhood_is_complete_unique_and_contains_literal_unseen_units():
    source, context, _ = fixture()
    pool = complete_pool(context.glyph_alphabet, 2)
    assert literal_unit_pool(context, max_units=6) == pool == ("x", "y", "xx", "xy", "yx", "yy")
    for units in itertools.product(pool, repeat=2):
        neighbors = list(unit_neighbors(units, (*pool, "xy", "x")))
        candidates = [candidate for _, candidate in neighbors]
        assert len(candidates) == len(set(candidates))
        assert set(candidates) == distinct_neighbors(units, pool)
        assert len(candidates) == 10 + (units[0] != units[1])
        for move, candidate in neighbors:
            if move["kind"] == "swap":
                assert move["rows"] == [0, 1] and candidate == units[::-1]
            else:
                changed = list(units)
                changed[move["row"]] = move["unit"]
                assert tuple(changed) == candidate
            channel = channel_from_units(source, context, candidate)
            assert len(channel.states) == 1 and channel.initial["s0"] == 1.
            for letter, unit in zip(source.alphabet, candidate, strict=True):
                assert channel.rows[("s0", letter)] == (Emission("s0", unit, 1.),)


@pytest.mark.parametrize("order", [0, 1])
def test_every_scored_search_neighbor_matches_rational_likelihood_and_literal_cost(order):
    source, context, records = fixture(order)
    original = source.to_dict(), tuple(records)
    config = UnitSearchConfig(seed=17, restarts=3, max_sweeps=40, batch_size=3, max_units=6)
    result = search_unit_channel(source, records, context, config=config, source_index=2)
    found, examined, complete, local, initializations = [], 0, 0, 0, 0
    for event in result.trace:
        if event["event"] == "restart":
            expected = exact_score(source, event["units"], records, context)
            assert_score(event["score"], expected)
            found.append(expected["total_bits"])
            initializations += 1
            continue
        parent = exact_score(source, event["parent_units"], records, context)
        assert_score(event["parent_score"], parent)
        expected_neighbors = distinct_neighbors(event["parent_units"], result.unit_pool)
        actual_neighbors = [tuple(neighbor["units"]) for neighbor in event["neighbors"]]
        assert event["expected_neighbors"] == len(expected_neighbors)
        assert event["evaluated_neighbors"] == len(actual_neighbors)
        assert event["complete"]
        assert len(actual_neighbors) == len(set(actual_neighbors))
        assert set(actual_neighbors) == expected_neighbors
        available = [parent["total_bits"]]
        for neighbor in event["neighbors"]:
            expected = exact_score(source, neighbor["units"], records, context)
            assert_score(neighbor["score"], expected)
            assert neighbor["status"] == ("unsupported" if not math.isfinite(expected["total_bits"]) else "scored")
            available.append(expected["total_bits"])
            found.append(expected["total_bits"])
        selected = exact_score(source, event["selected_units"], records, context)
        assert_score(event["selected_score"], selected)
        assert selected["total_bits"] == pytest.approx(min(available), rel=0, abs=6e-13)
        assert event["accepted"] == (parent["total_bits"] - selected["total_bits"] > config.improvement_tolerance_bits)
        first = 0
        for batch in event["batches"]:
            assert batch["first_neighbor"] == first
            assert 1 <= batch["count"] <= config.batch_size
            assert batch["kernel_diagnostics"]["candidate_count"] == batch["count"]
            first += batch["count"]
        assert first == len(actual_neighbors)
        examined += len(actual_neighbors)
        complete += 1
        local += not event["accepted"]
    assert result.evaluated_neighbors == examined
    assert result.completed_sweeps == complete
    assert result.local_optima == local > 0
    assert result.scored_initializations == initializations == 3
    final = exact_score(source, result.units, records, context)
    assert_score(result.score, final)
    assert final["total_bits"] == pytest.approx(min(found), rel=0, abs=6e-13)
    certified = any(event["event"] == "sweep" and event["complete"] and not event["accepted"]
                    and tuple(event["parent_units"]) == result.units for event in result.trace)
    assert result.best_is_certified_local_optimum == certified
    assert (source.to_dict(), tuple(records)) == original
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("order", [0, 1])
def test_exhaustive_artificial_starts_and_their_scored_neighbors_have_correct_scores(monkeypatch, order):
    source, context, _ = fixture(order)
    records = ("xxxx", "")
    assignments = tuple(itertools.product(complete_pool(context.glyph_alphabet, 2), repeat=2))

    def prescribed_start(_source, _records, _context, _pool, restart, _rng):
        return assignments[restart], "exhaustive_artificial_fixture"

    # Inject starts solely to expose every tiny coded assignment to the public
    # evaluator. Production singleton-coverage initialization is tested above.
    monkeypatch.setattr(search_module, "_initial_units", prescribed_start)
    result = search_unit_channel(source, records, context,
                                 config=UnitSearchConfig(restarts=36, max_sweeps=1, batch_size=4,
                                                         max_units=6), source_index=2)
    starts, all_values = [], []
    for event in result.trace:
        if event["event"] == "restart":
            starts.append(tuple(event["units"]))
            expected = exact_score(source, event["units"], records, context)
            assert_score(event["score"], expected)
            all_values.append(expected["total_bits"])
        else:
            assert event["complete"]
            assert {tuple(row["units"]) for row in event["neighbors"]} == distinct_neighbors(event["parent_units"], result.unit_pool)
            for row in event["neighbors"]:
                expected = exact_score(source, row["units"], records, context)
                assert_score(row["score"], expected)
                all_values.append(expected["total_bits"])
    assert starts == list(assignments) and result.scored_initializations == 36
    assert result.score["total_bits"] == pytest.approx(min(all_values), rel=0, abs=6e-13)


def test_initialization_covers_absent_declared_glyphs_and_seeds_are_reproducible():
    source = SourceModel(tuple("abcd"), 0, {"": {"a": .5, "b": .25, "c": .125, "d": .125}})
    context = CodingContext(source.alphabet, ("x", "y", "z"), 8, 1, 2, 1, stop_probability=.5)
    config = UnitSearchConfig(seed=123, restarts=8, max_sweeps=0, max_units=12)
    first = search_unit_channel(source, ("xxxxxxy", ""), context, config=config)
    second = search_unit_channel(source, ("xxxxxxy", ""), context, config=config)
    assert first.trace == second.trace and first.units == second.units and first.score == second.score
    assert first.unit_pool == complete_pool(context.glyph_alphabet, 2)
    assert first.trace[0]["units"] == ["x", "y", "z", "x"]
    for event in first.trace:
        assert event["event"] == "restart"
        assert set(context.glyph_alphabet) <= set(event["units"])
        assert all(unit in first.unit_pool for unit in event["units"])
    assert first.scored_initializations == 8
    assert first.completed_sweeps == first.evaluated_neighbors == first.local_optima == 0
    assert not first.best_is_certified_local_optimum


def test_a_better_subtolerance_neighbor_is_retained_without_its_parents_certificate():
    source, context, _ = fixture(0)
    records = ("x" * 12, "y")
    config = UnitSearchConfig(restarts=1, max_sweeps=3, batch_size=3, max_units=6,
                             improvement_tolerance_bits=1000.)
    result = search_unit_channel(source, records, context, config=config)
    initial, sweep = result.trace
    assert sweep["complete"] and not sweep["accepted"]
    assert sweep["stop_reason"] == "local_optimum"
    assert result.local_optima == result.completed_sweeps == 1
    assert result.score["total_bits"] < initial["score"]["total_bits"]
    assert result.units != tuple(initial["units"])
    assert not result.best_is_certified_local_optimum
    assert_score(result.score, exact_score(source, result.units, records, context))


@pytest.mark.parametrize("finish_whole_neighborhood", [False, True])
def test_completed_batches_survive_timeout_and_only_complete_neighborhoods_certify(monkeypatch, finish_whole_neighborhood):
    source, _, _ = fixture(0)
    context = CodingContext(source.alphabet, ("x", "y"), 8, 1, 1, 1, stop_probability=.25)
    records = ("", "")
    clock = [0.]
    original_kernel, original_final = search_module.batch_log_likelihood, search_module.two_part_score
    calls, final_calls = [], []

    def timed_kernel(*args, **kwargs):
        value = original_kernel(*args, **kwargs)
        calls.append(tuple(args[1]))
        if len(calls) == 2:
            clock[0] = 2.
        return value

    def final_engine(*args, **kwargs):
        final_calls.append(args)
        return original_final(*args, **kwargs)

    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_module, "batch_log_likelihood", timed_kernel)
    monkeypatch.setattr(search_module, "two_part_score", final_engine)
    config = UnitSearchConfig(restarts=2, max_sweeps=4, max_seconds=1., max_units=2,
                             batch_size=3 if finish_whole_neighborhood else 2)
    result = search_unit_channel(source, records, context, config=config)
    initial, sweep = result.trace
    assert result.stop_reason == "time_limit"
    assert result.scored_initializations == 1
    assert len(sweep["neighbors"]) == result.evaluated_neighbors == config.batch_size
    assert sweep["complete"] == finish_whole_neighborhood
    assert result.completed_sweeps == result.local_optima == int(finish_whole_neighborhood)
    assert result.best_is_certified_local_optimum == finish_whole_neighborhood
    assert len(final_calls) == 1 and final_calls[0][1] == result.channel
    assert_score(result.score, exact_score(source, result.units, records, context))
    assert result.units == tuple(initial["units"])


def test_partial_sweep_retains_its_completed_improvement_and_has_no_certificate(monkeypatch):
    source, context, _ = fixture(0)
    records = ("x" * 12, "y")
    clock, batches = [0.], []
    original = search_module.batch_log_likelihood

    def timed_kernel(*args, **kwargs):
        value = original(*args, **kwargs)
        batches.append(tuple(args[1]))
        if len(batches) == 2:
            clock[0] = 2.
        return value

    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_module, "batch_log_likelihood", timed_kernel)
    result = search_unit_channel(source, records, context,
                                 config=UnitSearchConfig(restarts=2, max_sweeps=4, max_seconds=1.,
                                                         batch_size=3, max_units=6))
    initial, sweep = result.trace
    # Ordered batch: swap, replace a by y, replace a by xx. The third candidate
    # improves the total code score by shortening the latent source path.
    assert sweep["accepted"] and not sweep["complete"]
    assert result.units == ("xx", "y")
    assert result.score["total_bits"] < initial["score"]["total_bits"]
    assert result.evaluated_neighbors == 3
    assert result.completed_sweeps == result.local_optima == 0
    assert not result.best_is_certified_local_optimum
    values = [exact_score(source, units, records, context)["total_bits"] for batch in batches for units in batch]
    assert result.score["total_bits"] == pytest.approx(min(values), rel=0, abs=6e-13)


def test_timeout_after_neighbor_enumeration_does_not_count_or_select_unscored_candidates(monkeypatch):
    source, context, _ = fixture(0)
    records = ("x" * 12, "y")
    clock = [0.]
    original = search_module.unit_neighbors

    def enumerate_then_expire(*args):
        for move, units in original(*args):
            clock[0] = 2.
            yield move, units

    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_module, "unit_neighbors", enumerate_then_expire)
    result = search_unit_channel(source, records, context,
                                 config=UnitSearchConfig(restarts=2, max_sweeps=4, max_seconds=1.,
                                                         batch_size=3, max_units=6))
    initial, sweep = result.trace
    assert result.stop_reason == "time_limit"
    assert result.units == tuple(initial["units"])
    assert sweep["neighbors"] == sweep["batches"] == []
    assert not sweep["complete"] and not sweep["accepted"]
    assert result.evaluated_neighbors == result.completed_sweeps == result.local_optima == 0
    assert not result.best_is_certified_local_optimum
    assert_score(result.score, exact_score(source, result.units, records, context))


def test_zero_probability_initializations_do_not_imply_the_family_is_unsupported():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    context = CodingContext(source.alphabet, ("x", "y"), 8, 1, 2, 1, stop_probability=.5)
    result = search_unit_channel(source, ("xy",), context,
                                 config=UnitSearchConfig(restarts=5, max_sweeps=5, max_units=6))
    assert result.channel is result.score is result.units is None
    assert result.stop_reason == "no_supported_initialization"
    assert all(event["status"] == "unsupported" and event["score"] is None for event in result.trace)
    assert result.evaluated_neighbors == 0 and result.scored_initializations == 5
    assert not result.best_is_certified_local_optimum
    # The skipped-start result is not a proof of impossibility in the family.
    assert exact_record_probability(source, ("xy", "x"), "xy", .5) == F(1, 4)
    json.dumps(result.to_dict(), allow_nan=False)


def test_combined_neighborhood_local_optimum_can_be_worse_than_a_two_row_change():
    source, context, _ = fixture(1)
    records, local, better = ("xxxx",), ("xx", "xx"), ("x", "x")
    local_score = exact_score(source, local, records, context)["total_bits"]
    assert exact_record_probability(source, local, records[0], .25) == F(9, 64)
    assert exact_record_probability(source, better, records[0], .25) == F(81, 1024)
    for _, neighbor in unit_neighbors(local, literal_unit_pool(context, 6)):
        assert exact_score(source, neighbor, records, context)["total_bits"] > local_score
    better_score = exact_score(source, better, records, context)["total_bits"]
    assert local_score - better_score == pytest.approx(2 - math.log2(F(16, 9)), rel=0, abs=1e-13)


def test_complete_pool_and_singleton_coverage_limits_are_explicit_rejections():
    source, context, records = fixture()
    with pytest.raises(ValueError, match="Complete literal unit pool"):
        search_unit_channel(source, records, context, config=UnitSearchConfig(max_units=5))
    more_glyphs = CodingContext(source.alphabet, ("x", "y", "z"), 8, 1, 2, 1)
    with pytest.raises(ValueError, match="Singleton-coverage initialization"):
        search_unit_channel(source, ("x",), more_glyphs)
    with pytest.raises(ValueError, match="at least one"):
        search_unit_channel(source, (), context)
