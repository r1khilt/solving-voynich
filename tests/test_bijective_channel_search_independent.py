"""Independent rational sequential scores for hand-specified bijective cases.

No corpus, pilot ciphertext, fitted pilot model or answer artifact is opened.
The reference multiplies source probabilities along each unique decoded path;
it does not reuse production sufficient statistics, swap deltas or inference.
"""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction as F

import pytest

from voynich import bijective_channel_search as search_module
from voynich.bijective_channel_search import (
    BijectiveSearchConfig, channel_from_mapping, ciphertext_statistics, mapping_log_likelihood,
    search_bijective_channel, swap_log_likelihood_delta,
)
from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext, channel_code_bits


def fixture(order=1):
    rows = {"": {"a": .5, "b": .25, "c": .125, "d": .125}}
    if order:
        rows.update(a={"a": .125, "b": .5, "c": .25, "d": .125},
                    b={"a": .375, "b": .125, "c": .125, "d": .375},
                    c={"a": .25, "b": .125, "c": .5, "d": .125},
                    d={"a": .125, "b": .125, "c": .25, "d": .5})
    source = SourceModel(("a", "b", "c", "d"), order, rows)
    context = CodingContext(source.alphabet, ("X", "Y", "Z", "W"), 16, 2, 2, 3,
                            source_count=3, stop_probability=.25)
    records = ("XXXXYXY", "YYX", "YZZ", "", "Z", "")
    return source, context, records


def rational_probability(source, records, mapping, rho):
    total = F(1)
    for record in records:
        context = ""
        mass = F(rho)
        for glyph in record:
            letter = mapping[glyph]
            mass *= (1 - F(rho)) * F(source.probabilities[context][letter])
            context = letter if source.order else ""
        total *= mass
    return total


def rational_log(source, records, mapping, rho):
    value = rational_probability(source, records, mapping, rho)
    return math.log(value.numerator) - math.log(value.denominator) if value else -math.inf


def swapped(mapping, left, right):
    answer = dict(mapping)
    answer[left], answer[right] = answer[right], answer[left]
    return answer


@pytest.mark.parametrize("order", [0, 1])
def test_every_permutation_and_swap_matches_independent_sequential_probability(order):
    source, context, records = fixture(order)
    statistics = ciphertext_statistics(records, context.glyph_alphabet)
    assert statistics.glyph_counts[-1] == statistics.start_counts[-1] == 0
    assert statistics.records == 6 and statistics.glyphs == 14
    for letters in itertools.permutations(source.alphabet):
        mapping = dict(zip(context.glyph_alphabet, letters, strict=True))
        expected = rational_log(source, records, mapping, .25)
        assert mapping_log_likelihood(source, statistics, mapping, .25) == pytest.approx(expected, abs=3e-13)
        for left, right in itertools.combinations(context.glyph_alphabet, 2):
            changed = swapped(mapping, left, right)
            ratio = rational_probability(source, records, changed, .25) / rational_probability(source, records, mapping, .25)
            expected_delta = math.log(ratio.numerator) - math.log(ratio.denominator)
            assert swap_log_likelihood_delta(source, statistics, mapping, left, right) == pytest.approx(expected_delta, abs=3e-13)
            # The stop factors cancel for the same records under every swap.
            second_difference = rational_log(source, records, changed, .5) - rational_log(source, records, mapping, .5)
            assert expected_delta == pytest.approx(second_difference, abs=3e-13)


def test_statistics_separate_starts_self_pairs_directed_pairs_and_empty_records():
    stats = ciphertext_statistics(("XXY", "YXX", "", "Y", ""), ("X", "Y", "Z"))
    assert stats.glyph_counts == (4, 3, 0)
    assert stats.start_counts == (1, 2, 0)
    assert stats.pair_counts == ((2, 1, 0), (1, 0, 0), (0, 0, 0))
    assert stats.records == 5 and stats.glyphs == 7


def test_family_description_cost_is_constant_for_all_bijections_and_charges_absent_glyphs():
    source, context, _ = fixture()
    for letters in itertools.permutations(source.alphabet):
        mapping = dict(zip(context.glyph_alphabet, letters, strict=True))
        channel = channel_from_mapping(source, context, mapping)
        # Source-choice header2 + state-count header1 + four rows each paying
        # alternative-count2 + length1 + literal-glyph2; other fields singleton.
        assert channel_code_bits(channel, context, source_index=2) == 23
        assert len(channel.rows) == 4
        assert {event.glyphs for row in channel.rows.values() for event in row} == set(context.glyph_alphabet)


def test_full_search_trace_replays_every_neighbor_and_only_complete_local_certificates():
    source, context, records = fixture()
    before = source.to_dict()
    config = BijectiveSearchConfig(seed=28, restarts=3, max_sweeps=15, max_seconds=60.)
    result = search_bijective_channel(source, records, context, config=config, source_index=2)
    all_scores, examined, complete, optima = [], 0, 0, 0
    all_pairs = list(itertools.combinations(context.glyph_alphabet, 2))
    for event in result.trace:
        if event["event"] == "restart":
            expected = rational_log(source, records, event["mapping"], .25)
            assert event["log_likelihood"] == pytest.approx(expected, abs=3e-13)
            all_scores.append(expected)
            continue
        parent = rational_log(source, records, event["parent_mapping"], .25)
        assert event["parent_log_likelihood"] == pytest.approx(parent, abs=3e-13)
        pairs = [tuple(neighbor["pair"]) for neighbor in event["neighbors"]]
        assert pairs == all_pairs[:len(pairs)]
        assert event["complete"] == (len(pairs) == len(all_pairs))
        scores = [parent]
        for neighbor in event["neighbors"]:
            mapping = swapped(event["parent_mapping"], *neighbor["pair"])
            value = rational_log(source, records, mapping, .25)
            assert neighbor["log_likelihood"] == pytest.approx(value, abs=3e-13)
            scores.append(value)
            all_scores.append(value)
        best = max(scores)
        assert event["best_neighbor_log_likelihood"] == pytest.approx(best, abs=3e-13)
        if event["accepted"]:
            assert best - parent > config.improvement_tolerance_bits * math.log(2)
        if event["stop_reason"] == "pair_local_optimum":
            assert event["complete"] and not event["accepted"]
            assert best - parent <= config.improvement_tolerance_bits * math.log(2) + 3e-13
            optima += 1
        complete += event["complete"]
        examined += len(pairs)
    assert result.evaluated_neighbors == examined
    assert result.completed_sweeps == complete
    assert result.pair_local_optima == optima > 0
    certified_return = any(event["event"] == "sweep" and event["complete"] and not event["accepted"]
                           and event["parent_mapping"] == result.mapping for event in result.trace)
    assert result.best_is_certified_pair_local_optimum == certified_return
    expected = rational_log(source, records, result.mapping, .25)
    assert expected == pytest.approx(max(all_scores), abs=3e-13)
    assert result.score["log_likelihood"] == pytest.approx(expected, abs=3e-13)
    assert result.score["model_bits"] == 23
    assert result.score["total_bits"] == pytest.approx(23 - expected / math.log(2), abs=3e-13)
    assert source.to_dict() == before
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_sweep_budget_and_empty_only_data_have_honest_certificates():
    source, context, _ = fixture()
    result = search_bijective_channel(source, ("", ""), context,
                                     config=BijectiveSearchConfig(restarts=1, max_sweeps=0))
    assert result.completed_sweeps == result.evaluated_neighbors == result.pair_local_optima == 0
    assert not result.best_is_certified_pair_local_optimum
    assert result.score["log_likelihood"] == pytest.approx(math.log(F(1, 16)), abs=1e-13)
    complete = search_bijective_channel(source, ("", ""), context,
                                       config=BijectiveSearchConfig(restarts=1, max_sweeps=1))
    assert complete.pair_local_optima == complete.completed_sweeps == 1
    assert complete.evaluated_neighbors == 6
    assert complete.best_is_certified_pair_local_optimum


def test_supported_to_impossible_swap_reports_negative_infinity_not_nan():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    stats = ciphertext_statistics(("X", ""), ("X", "Y"))
    mapping = {"X": "a", "Y": "b"}
    assert swap_log_likelihood_delta(source, stats, mapping, "X", "Y") == -math.inf
    assert mapping_log_likelihood(source, stats, swapped(mapping, "X", "Y"), .5) == -math.inf


@pytest.mark.parametrize("frequency_initialization", [False, True])
def test_partial_sweep_retains_best_finished_key_without_local_optimum_certificate(monkeypatch, frequency_initialization):
    source = SourceModel(("a", "b", "c"), 0, {"": {"a": .125, "b": .25, "c": .625}})
    context = CodingContext(source.alphabet, ("X", "Y", "Z"), 8, 1, 1, 1, stop_probability=.5)
    records = ("XXXXXY", "Z", "")
    clock = [0.]
    monkeypatch.setattr(search_module.time, "monotonic", lambda: clock[0])
    original_delta, original_score = search_module._delta, search_module.two_part_score
    final_calls = []

    def one_neighbor_then_timeout(*args):
        answer = original_delta(*args)
        clock[0] = 2.
        return answer

    def record_final_engine_call(*args):
        final_calls.append(args)
        return original_score(*args)

    monkeypatch.setattr(search_module, "_delta", one_neighbor_then_timeout)
    monkeypatch.setattr(search_module, "two_part_score", record_final_engine_call)
    config = BijectiveSearchConfig(seed=1, restarts=2, max_sweeps=5, max_seconds=1,
                                   frequency_initialization=frequency_initialization)
    result = search_bijective_channel(source, records, context, config=config)
    initial, sweep = result.trace
    assert initial["event"] == "restart" and sweep["event"] == "sweep"
    assert len(sweep["neighbors"]) == result.evaluated_neighbors == 1
    assert not sweep["complete"]
    assert result.completed_sweeps == result.pair_local_optima == 0
    assert not result.best_is_certified_pair_local_optimum
    assert result.stop_reason == "time_limit"
    first_changed = swapped(initial["mapping"], "X", "Y")
    expected = max(rational_log(source, records, initial["mapping"], .5),
                   rational_log(source, records, first_changed, .5))
    assert result.score["log_likelihood"] == pytest.approx(expected, abs=2e-13)
    assert rational_log(source, records, result.mapping, .5) == pytest.approx(expected, abs=2e-13)
    assert len(final_calls) == 1 and final_calls[0][1] == result.channel
    if not frequency_initialization:
        assert sweep["accepted"]
        assert result.mapping == first_changed
    else:
        assert not sweep["accepted"] and sweep["stop_reason"] == "time_limit"


def test_best_completed_subtolerance_neighbor_is_retained_even_when_sweep_rejects_move():
    source = SourceModel(("a", "b", "c"), 0, {"": {"a": .125, "b": .25, "c": .625}})
    context = CodingContext(source.alphabet, ("X", "Y", "Z"), 8, 1, 1, 1, stop_probability=.5)
    records = ("XXXXXY", "Z", "")
    config = BijectiveSearchConfig(seed=1, restarts=1, max_sweeps=3, frequency_initialization=False,
                                   improvement_tolerance_bits=10.)
    result = search_bijective_channel(source, records, context, config=config)
    initial, sweep = result.trace
    assert sweep["complete"] and not sweep["accepted"]
    assert sweep["stop_reason"] == "pair_local_optimum"
    assert result.pair_local_optima == 1
    assert not result.best_is_certified_pair_local_optimum
    assert result.score["log_likelihood"] > initial["log_likelihood"]
    assert result.score["log_likelihood"] == pytest.approx(max(item["log_likelihood"] for item in sweep["neighbors"]))


def test_order_one_pair_local_optimum_is_not_a_global_permutation_optimum():
    source = SourceModel(("a", "b", "c"), 1,
                         {"": {"a": .25, "b": .5, "c": .25},
                          "a": {"a": .125, "b": .75, "c": .125},
                          "b": {"a": .125, "b": .125, "c": .75},
                          "c": {"a": .75, "b": .125, "c": .125}})
    records, glyphs = ("XYZ",), ("X", "Y", "Z")
    stats = ciphertext_statistics(records, glyphs)
    local = {"X": "a", "Y": "b", "Z": "c"}
    better = {"X": "b", "Y": "c", "Z": "a"}
    assert rational_probability(source, records, local, .5) == F(9, 1024)
    assert rational_probability(source, records, better, .5) == F(9, 512)
    for left, right in itertools.combinations(glyphs, 2):
        assert swap_log_likelihood_delta(source, stats, local, left, right) < 0
        assert rational_probability(source, records, swapped(local, left, right), .5) < F(9, 1024)
