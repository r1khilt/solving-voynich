import itertools
import json
import math

import pytest

import voynich.bijective_channel_search as search
from voynich.bijective_channel_search import (
    BijectiveSearchConfig, channel_from_mapping, ciphertext_statistics,
    mapping_log_likelihood, search_bijective_channel, swap_log_likelihood_delta,
)
from voynich.finite_state_channel import SourceModel, forward_log_probability
from voynich.finite_state_channel_fit import CodingContext, channel_code_bits


def source_model(order=1):
    probabilities = {"": {"a": .5, "b": .3, "c": .2}}
    if order:
        probabilities |= {"a": {"a": .1, "b": .2, "c": .7},
                          "b": {"a": .6, "b": .3, "c": .1},
                          "c": {"a": .2, "b": .5, "c": .3}}
    return SourceModel(("a", "b", "c"), order, probabilities)


def context(source=None, **changes):
    source = source or source_model()
    return CodingContext(**({"source_alphabet": source.alphabet, "glyph_alphabet": ("x", "y", "z"),
                             "denominator": 32, "max_states": 2, "max_emission_length": 3,
                             "max_alternatives": 3, "stop_probability": .31} | changes))


def mappings(source, ctx):
    for permutation in itertools.permutations(source.alphabet):
        yield dict(zip(ctx.glyph_alphabet, permutation, strict=True))


def test_statistics_preserve_empty_records_starts_and_directed_self_pairs():
    stats = ciphertext_statistics(["xyx", "zzy", "", "yxxy"], ("x", "y", "z"))
    assert stats.records == 4
    assert stats.glyphs == 10
    assert stats.glyph_counts == (4, 4, 2)
    assert stats.start_counts == (1, 1, 1)
    assert stats.pair_counts == ((1, 2, 0), (2, 0, 0), (0, 1, 1))
    assert sum(map(sum, stats.pair_counts)) == stats.glyphs - sum(stats.start_counts)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("records", [["xyx", "zzy", "", "yxxy"], ["xxyx", "xy", ""], ["", ""]])
def test_all_small_keys_and_swap_deltas_equal_full_engine(order, records):
    source = source_model(order)
    ctx = context(source)
    stats = ciphertext_statistics(records, ctx.glyph_alphabet)
    code_sizes = set()
    for mapping in mappings(source, ctx):
        channel = channel_from_mapping(source, ctx, mapping)
        expected = math.fsum(forward_log_probability(source, channel, record) for record in records)
        actual = mapping_log_likelihood(source, stats, mapping, ctx.stop_probability)
        assert actual == pytest.approx(expected, abs=1e-12)
        code_sizes.add(channel_code_bits(channel, ctx))
        for left, right in itertools.combinations(ctx.glyph_alphabet, 2):
            swapped = dict(mapping)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            swapped_channel = channel_from_mapping(source, ctx, swapped)
            expected_new = math.fsum(forward_log_probability(source, swapped_channel, record) for record in records)
            assert swap_log_likelihood_delta(source, stats, mapping, left, right) == pytest.approx(
                expected_new - expected, abs=1e-12)
    assert len(code_sizes) == 1


def test_order_zero_frequency_start_attains_exhaustive_global_optimum_in_restricted_family():
    source = source_model(0)
    ctx = context(source)
    records = ["zyzyzzyx", "zzzy"]
    result = search_bijective_channel(source, records, ctx,
                                      config=BijectiveSearchConfig(seed=12, restarts=1, max_sweeps=10))
    scores = [mapping_log_likelihood(source, result.statistics, mapping, ctx.stop_probability)
              for mapping in mappings(source, ctx)]
    assert result.score["log_likelihood"] == pytest.approx(max(scores))
    assert result.mapping == {"x": "c", "y": "b", "z": "a"}
    assert result.pair_local_optima == result.completed_sweeps == 1
    assert result.best_is_certified_pair_local_optimum
    assert result.evaluated_neighbors == 3
    assert result.trace[-1]["stop_reason"] == "pair_local_optimum"


def test_all_253_pairs_are_checked_once_before_23_letter_local_certificate():
    alphabet = tuple("abcdefghijklmnopqrstuvw")
    glyphs = tuple("ABCDEFGHIJKLMNOPQRSTUVW")
    source = SourceModel(alphabet, 0, {"": dict.fromkeys(alphabet, 1 / 23)})
    ctx = context(source, glyph_alphabet=glyphs)
    result = search_bijective_channel(source, [""], ctx, config=BijectiveSearchConfig(restarts=1))
    neighbors = result.trace[-1]["neighbors"]
    assert len(neighbors) == result.evaluated_neighbors == 253
    assert {tuple(item["pair"]) for item in neighbors} == set(itertools.combinations(glyphs, 2))
    assert result.pair_local_optima == 1
    assert result.score["log_likelihood"] == pytest.approx(math.log(ctx.stop_probability))


def test_pair_local_optimum_need_not_be_global_even_on_a_positive_source():
    # A reverse 5-cycle is locally stable: fixing the orientation requires
    # coordinated changes. This is a mathematical counterexample, not a corpus.
    letters, glyphs = tuple("abcde"), tuple("xyzwv")
    probabilities = {"": dict.fromkeys(letters, .2)}
    probabilities |= {letter: {other: (.5 if j == (i + 1) % 5 else .4 if j == (i - 1) % 5 else .1 / 3)
                               for j, other in enumerate(letters)} for i, letter in enumerate(letters)}
    source = SourceModel(letters, 1, probabilities)
    stats = ciphertext_statistics(["xyzwvx"], glyphs)
    reverse = dict(zip(glyphs, ("a", "e", "d", "c", "b"), strict=True))
    forward = dict(zip(glyphs, letters, strict=True))
    assert all(swap_log_likelihood_delta(source, stats, reverse, left, right) < 0
               for left, right in itertools.combinations(glyphs, 2))
    assert mapping_log_likelihood(source, stats, forward, .1) > mapping_log_likelihood(source, stats, reverse, .1)


def test_search_trace_replays_every_neighbor_and_is_seed_reproducible():
    source, ctx = source_model(), context()
    records = ["xyzzxyyxz", "zyyxx", ""]
    config = BijectiveSearchConfig(seed=190, restarts=4, max_sweeps=8)
    result = search_bijective_channel(source, records, ctx, config=config)
    again = search_bijective_channel(source, records, ctx, config=config)
    assert result.trace == again.trace
    assert result.mapping == again.mapping
    assert result.score == again.score
    scored = []
    for event in result.trace:
        if event["event"] == "restart":
            scored.append(event["log_likelihood"])
            continue
        assert len(event["neighbors"]) == 3
        for neighbor in event["neighbors"]:
            mapping = dict(event["parent_mapping"])
            left, right = neighbor["pair"]
            mapping[left], mapping[right] = mapping[right], mapping[left]
            expected = mapping_log_likelihood(source, result.statistics, mapping, ctx.stop_probability)
            assert neighbor["log_likelihood"] == pytest.approx(expected, abs=1e-12)
            scored.append(expected)
    assert result.score["log_likelihood"] == pytest.approx(max(scored), abs=1e-12)
    assert result.evaluated_neighbors == sum(len(event["neighbors"]) for event in result.trace if event["event"] == "sweep")
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_probability_keys_are_explicit_and_do_not_produce_json_infinities():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    ctx = context(source, glyph_alphabet=("x", "y"))
    one = ciphertext_statistics(["x"], ctx.glyph_alphabet)
    assert swap_log_likelihood_delta(source, one, {"x": "a", "y": "b"}, "x", "y") == -math.inf
    two = ciphertext_statistics(["xy"], ctx.glyph_alphabet)
    with pytest.raises(ValueError, match="undefined"):
        swap_log_likelihood_delta(source, two, {"x": "a", "y": "b"}, "x", "y")
    result = search_bijective_channel(source, ["xy"], ctx, config=BijectiveSearchConfig(restarts=3))
    assert result.channel is result.score is result.mapping is None
    assert result.stop_reason == "no_supported_initialization"
    assert len(result.trace) == 3
    assert result.evaluated_neighbors == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_deadline_during_sweep_keeps_best_finished_key_without_local_certificate(monkeypatch):
    source, ctx = source_model(), context()
    now = [0.]
    original_delta = search._delta
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])

    def slow_delta(*args):
        result = original_delta(*args)
        now[0] += 1
        return result

    monkeypatch.setattr(search, "_delta", slow_delta)
    result = search_bijective_channel(source, ["xzyxzyxzy"], ctx,
                                      config=BijectiveSearchConfig(seed=9, restarts=4, max_seconds=2,
                                                                   frequency_initialization=False))
    assert result.stop_reason == "time_limit"
    assert result.evaluated_neighbors == 2
    assert result.completed_sweeps == result.pair_local_optima == 0
    assert not result.best_is_certified_pair_local_optimum
    event = result.trace[-1]
    assert not event["complete"]
    assert event["stop_reason"] != "pair_local_optimum"
    finished = [result.trace[0]["log_likelihood"], *(item["log_likelihood"] for item in event["neighbors"])]
    assert result.score["log_likelihood"] == pytest.approx(max(finished))


def test_deadline_before_first_mapping_reports_no_result(monkeypatch):
    clock = iter([0., 2., 2., 2.])
    monkeypatch.setattr(search.time, "monotonic", lambda: next(clock))
    result = search_bijective_channel(source_model(), ["xyz"], context(),
                                      config=BijectiveSearchConfig(max_seconds=1))
    assert result.channel is result.score is result.mapping is None
    assert result.stop_reason == "time_limit"
    assert result.trace == ()


def test_invalid_alphabets_records_and_mappings_are_rejected():
    with pytest.raises(ValueError, match="equal"):
        search_bijective_channel(source_model(), ["x"], context(glyph_alphabet=("x", "y")))
    with pytest.raises(ValueError, match="outside"):
        search_bijective_channel(source_model(), ["w"], context())
    with pytest.raises(ValueError, match="separate"):
        ciphertext_statistics("xy", ("x", "y"))
    with pytest.raises(ValueError, match="bijection"):
        channel_from_mapping(source_model(), context(), {"x": "a", "y": "a", "z": "b"})


@pytest.mark.parametrize("changes", [{"max_seconds": 0}, {"max_seconds": math.inf}, {"restarts": 0},
                                     {"max_sweeps": -1}, {"seed": True}, {"frequency_initialization": 1}])
def test_invalid_search_budgets_are_rejected(changes):
    with pytest.raises(ValueError):
        BijectiveSearchConfig(**changes)
