import itertools
import json
import math

import pytest

import voynich.unit_channel_search as search
from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext, encode_channel, two_part_score


def fixture(order=1, **overrides):
    probabilities = {"": {"a": .75, "b": .25}}
    if order:
        probabilities |= {"a": {"a": .125, "b": .875}, "b": {"a": .625, "b": .375}}
    source = SourceModel(("a", "b"), order, probabilities)
    options = dict(source_alphabet=source.alphabet, glyph_alphabet=("x", "y"), denominator=3,
                   max_states=3, max_emission_length=2, max_alternatives=5, source_count=3,
                   stop_probability=.25)
    options.update(overrides)
    return source, CodingContext(**options)


def original(source, context, units, records=("xxxx", "xy", ""), source_index=2):
    return two_part_score(source, search.channel_from_units(source, context, units), records, context, source_index)


def completed_scores(result):
    for event in result.trace:
        if event["event"] == "restart":
            yield event["units"], event["score"]
        else:
            yield from ((candidate["units"], candidate["score"]) for candidate in event["neighbors"])


def test_pool_contains_unseen_literals_in_length_then_declared_glyph_order_and_caps_before_product(monkeypatch):
    _, context = fixture(glyph_alphabet=("y", "x"))
    assert search.literal_unit_pool(context, 6) == ("y", "x", "yy", "yx", "xy", "xx")
    with pytest.raises(ValueError, match="exceeds"):
        search.literal_unit_pool(context, 5)
    _, giant = fixture(max_emission_length=1000000000)
    monkeypatch.setattr(search.itertools, "product", lambda *args, **kwargs: pytest.fail("materialized huge pool"))
    with pytest.raises(ValueError, match="exceeds"):
        search.literal_unit_pool(giant, 256)


@pytest.mark.parametrize("order", [0, 1])
def test_every_small_candidate_and_trace_score_matches_original_engine_and_literal_code(order):
    source, context = fixture(order)
    pool = search.literal_unit_pool(context)
    width = (len(context.glyph_alphabet) - 1).bit_length()
    fixed = (context.source_count - 1).bit_length() + (context.max_states - 1).bit_length()
    fixed += len(source.alphabet) * ((context.max_alternatives - 1).bit_length()
                                    + (context.max_emission_length - 1).bit_length())
    for units in itertools.product(pool, repeat=2):
        channel = search.channel_from_units(source, context, units)
        assert len(encode_channel(channel, context, 2)) == fixed + width * sum(map(len, units))
        assert all(row[0].probability == 1. for row in channel.rows.values())
    result = search.search_unit_channel(source, ["xxxx", "xy", ""], context,
                                       config=search.UnitSearchConfig(seed=7, restarts=8, max_sweeps=5), source_index=2)
    for units, score in completed_scores(result):
        expected = original(source, context, units)
        if score is None:
            assert expected["log_likelihood"] == -math.inf
        else:
            assert score == pytest.approx(expected, abs=1e-11)
    assert result.score == original(source, context, result.units)
    best = min(score["total_bits"] for _, score in completed_scores(result) if score is not None)
    assert result.score["total_bits"] == pytest.approx(best, abs=1e-11)


def test_exhaustive_neighbor_set_contains_every_one_and_two_position_operation_once():
    pool = ("x", "y", "xx")
    for units in itertools.product(pool, repeat=3):
        expected = set()
        for i in range(3):
            for j in range(i + 1, 3):
                changed = list(units)
                changed[i], changed[j] = changed[j], changed[i]
                if tuple(changed) != units:
                    expected.add(tuple(changed))
            for replacement in pool:
                if replacement != units[i]:
                    expected.add(units[:i] + (replacement,) + units[i + 1:])
        proposals = list(search.unit_neighbors(units, (*pool, "x")))
        assert {candidate for _, candidate in proposals} == expected
        assert len(proposals) == len(expected)
        assert proposals == list(search.unit_neighbors(units, pool))


def test_complete_sweep_certificates_are_for_the_scanned_parent_and_replay_full_objective():
    source, context = fixture()
    result = search.search_unit_channel(source, ["xxxx", "xy"], context,
                                       config=search.UnitSearchConfig(seed=43, restarts=5, max_sweeps=10))
    assert result.completed_sweeps > 0
    certified = []
    for event in result.trace:
        if event["event"] != "sweep":
            continue
        expected = list(search.unit_neighbors(event["parent_units"], result.unit_pool))
        assert event["complete"]
        assert event["evaluated_neighbors"] == event["expected_neighbors"] == len(expected)
        assert [tuple(row["units"]) for row in event["neighbors"]] == [units for _, units in expected]
        if not event["accepted"]:
            threshold = event["parent_score"]["total_bits"] - result.config.improvement_tolerance_bits
            assert all(row["score"] is None or row["score"]["total_bits"] >= threshold
                       for row in event["neighbors"])
            certified.append(tuple(event["parent_units"]))
    assert result.best_is_certified_local_optimum == (result.units in certified)


def test_strict_combined_local_optimum_can_be_worse_than_two_row_change():
    source, context = fixture()
    local = ("xx", "xx")
    value = original(source, context, local, ["xxxx"])["total_bits"]
    assert all(original(source, context, units, ["xxxx"])["total_bits"] > value
               for _, units in search.unit_neighbors(local, search.literal_unit_pool(context)))
    global_witness = original(source, context, ("x", "x"), ["xxxx"])["total_bits"]
    assert value - global_witness == pytest.approx(1.1699250014423126)


def test_tolerance_does_not_give_unscanned_better_returned_candidate_a_certificate():
    source, context = fixture(0)
    result = search.search_unit_channel(source, ["xxxx"], context,
                                       config=search.UnitSearchConfig(restarts=1, improvement_tolerance_bits=1000))
    assert result.local_optima == 1
    sweep = result.trace[1]
    assert sweep["complete"] and not sweep["accepted"]
    assert result.score["total_bits"] < sweep["parent_score"]["total_bits"]
    assert result.units != tuple(sweep["parent_units"])
    assert not result.best_is_certified_local_optimum


def test_same_seed_count_budgets_and_different_batch_sizes_preserve_moves_and_best():
    source, context = fixture()
    config = dict(seed=12, restarts=4, max_sweeps=3)
    one = search.search_unit_channel(source, ["xxxx", "xy"], context, config=search.UnitSearchConfig(**config))
    two = search.search_unit_channel(source, ["xxxx", "xy"], context, config=search.UnitSearchConfig(**config))
    assert one.trace == two.trace
    small = search.search_unit_channel(source, ["xxxx", "xy"], context,
                                      config=search.UnitSearchConfig(**config, batch_size=2))
    assert one.units == small.units
    assert one.evaluated_neighbors == small.evaluated_neighbors
    for (units, score), (other_units, other_score) in zip(completed_scores(one), completed_scores(small), strict=True):
        assert units == other_units
        assert score == pytest.approx(other_score) if score else other_score is None
    json.dumps(one.to_dict(), allow_nan=False)


def test_initializers_cover_absent_declared_glyphs_and_random_rows_can_duplicate_units():
    source = SourceModel(tuple("abcd"), 0, {"": {"a": .1, "b": .4, "c": .3, "d": .2}})
    context = CodingContext(source.alphabet, ("y", "x"), 3, 2, 2, 3, stop_probability=.25)
    result = search.search_unit_channel(source, ["xxxx"], context,
                                       config=search.UnitSearchConfig(seed=3, restarts=30, max_sweeps=0))
    assert result.trace[0]["units"] == ["y", "x", "y", "x"]
    assert result.trace[0]["method"] == "source_start_and_glyph_frequency_singletons"
    for event in result.trace:
        assert set(context.glyph_alphabet) <= set(event["units"])
        assert set(event["units"]) <= set(result.unit_pool)
    assert any(len(unit) == 2 for event in result.trace[1:] for unit in event["units"])
    assert any(len(set(event["units"])) < 4 for event in result.trace[1:])
    assert result.scored_initializations == 30
    assert result.evaluated_neighbors == result.completed_sweeps == 0
    assert not result.best_is_certified_local_optimum


def test_tied_initial_frequencies_use_declared_order_and_all_empty_records_are_valid():
    source = SourceModel(("b", "a"), 0, {"": {"a": .5, "b": .5}})
    context = CodingContext(source.alphabet, ("y", "x"), 1, 1, 2, 1, stop_probability=.25)
    result = search.search_unit_channel(source, ["", ""], context, config=search.UnitSearchConfig(restarts=1))
    assert result.trace[0]["units"] == ["y", "x"]
    assert result.score["log_likelihood"] == 2 * math.log(.25)
    assert result.best_is_certified_local_optimum


def test_one_glyph_one_unit_has_vacuous_complete_neighborhood():
    source = SourceModel(("a",), 0, {"": {"a": 1.}})
    context = CodingContext(source.alphabet, ("x",), 1, 1, 1, 1, stop_probability=.25)
    result = search.search_unit_channel(source, ["xxx"], context, config=search.UnitSearchConfig(restarts=1))
    assert result.evaluated_neighbors == 0 and result.completed_sweeps == 1
    assert result.best_is_certified_local_optimum


def test_zero_source_support_is_reported_without_fake_score_or_oracle_repair():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    _, context = fixture()
    result = search.search_unit_channel(source, ["xy"], context,
                                       config=search.UnitSearchConfig(restarts=3, max_sweeps=2))
    assert result.channel is result.score is result.units is None
    assert result.stop_reason == "no_supported_initialization"
    assert all(event["status"] == "unsupported" for event in result.trace)
    json.dumps(result.to_dict(), allow_nan=False)


def test_partial_timeout_keeps_best_completed_batch_and_runs_original_check(monkeypatch):
    source, context = fixture(0)
    clock = [0.]
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])
    real_batch, real_final = search.batch_log_likelihood, search.two_part_score
    calls, final_calls = [], []

    def timed_batch(*args, **kwargs):
        result = real_batch(*args, **kwargs)
        calls.append(len(args[1]))
        clock[0] += .1 if len(calls) == 1 else 2.
        return result

    def final(*args, **kwargs):
        final_calls.append(True)
        return real_final(*args, **kwargs)

    monkeypatch.setattr(search, "batch_log_likelihood", timed_batch)
    monkeypatch.setattr(search, "two_part_score", final)
    result = search.search_unit_channel(source, ["xxxx"], context,
                                       config=search.UnitSearchConfig(max_seconds=1, batch_size=3))
    assert calls == [1, 3] and final_calls == [True]
    assert result.stop_reason == "time_limit"
    assert result.completed_sweeps == 0 and result.evaluated_neighbors == 3
    assert not result.trace[-1]["complete"] and not result.best_is_certified_local_optimum
    available = [(tuple(units), score) for units, score in completed_scores(result) if score]
    best_units, best_score = min(available, key=lambda item: item[1]["total_bits"])
    assert result.units == best_units
    assert result.score == pytest.approx(best_score)
    assert result.score["total_bits"] < result.trace[0]["score"]["total_bits"]


def test_timeout_before_scoring_returns_no_incomplete_initialization(monkeypatch):
    source, context = fixture()
    clock = [0.]
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])
    real_initialize = search._initial_units

    def initialize(*args):
        result = real_initialize(*args)
        clock[0] = 2.
        return result

    monkeypatch.setattr(search, "_initial_units", initialize)
    result = search.search_unit_channel(source, ["xxxx"], context,
                                       config=search.UnitSearchConfig(max_seconds=1))
    assert result.channel is None and result.score is None and result.trace == ()
    assert result.stop_reason == "time_limit" and result.scored_initializations == 0


def test_fallback_diagnostics_are_aggregated_and_final_replay_detects_incorrect_kernel(monkeypatch):
    source = SourceModel(("a", "b"), 0, {"": {"a": 1e-320, "b": 1.}})
    _, context = fixture()
    result = search.search_unit_channel(source, ["xy"], context,
                                       config=search.UnitSearchConfig(restarts=1, max_sweeps=0))
    assert result.kernel_fallbacks == 1
    assert result.trace[0]["kernel_diagnostics"]["log_domain_fallbacks"] == 1
    real = search.batch_log_likelihood

    def wrong(*args, **kwargs):
        return real(*args, **kwargs) + 1.

    monkeypatch.setattr(search, "batch_log_likelihood", wrong)
    with pytest.raises(ArithmeticError, match="disagree"):
        search.search_unit_channel(source, ["xy"], context,
                                   config=search.UnitSearchConfig(restarts=1, max_sweeps=0))


@pytest.mark.parametrize("options", [dict(restarts=0), dict(seed=True), dict(max_sweeps=-1),
                                   dict(batch_size=0), dict(max_units=0), dict(max_seconds=0),
                                   dict(improvement_tolerance_bits=math.nan)])
def test_invalid_configs_rejected(options):
    with pytest.raises(ValueError):
        search.UnitSearchConfig(**options)


def test_invalid_input_boundaries_rejected():
    source, context = fixture()
    for records in ([], "xxxx", [None], ["z"]):
        with pytest.raises(ValueError):
            search.search_unit_channel(source, records, context)
    with pytest.raises(ValueError, match="Source index"):
        search.search_unit_channel(source, ["x"], context, source_index=3)
    _, too_many_glyphs = fixture(glyph_alphabet=("x", "y", "z"))
    with pytest.raises(ValueError, match="Singleton-coverage"):
        search.search_unit_channel(source, ["x"], too_many_glyphs)
    reversed_source = SourceModel(("b", "a"), 0, {"": {"a": .5, "b": .5}})
    with pytest.raises(ValueError, match="ordered"):
        search.search_unit_channel(reversed_source, ["x"], context)
    for units in ("xy", ("x",), ("x", ""), ("x", "xyz"), ("x", "z")):
        with pytest.raises(ValueError):
            search.channel_from_units(source, context, units)
