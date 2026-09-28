import json
import math

import pytest

import voynich.finite_state_channel_search as search
from voynich.finite_state_channel import Channel, Emission, SourceModel, forward_log_probability
from voynich.finite_state_channel_fit import CodingContext, EMResult, two_part_score
from voynich.finite_state_channel_search import SearchConfig, observed_units, search_channel


def singleton_context(**changes):
    return CodingContext(**({"source_alphabet": ("a",), "glyph_alphabet": ("x",), "denominator": 4,
                             "max_states": 2, "max_emission_length": 2, "max_alternatives": 2,
                             "stop_probability": .5} | changes))


def singleton_source():
    return SourceModel(("a",), 0, {"": {"a": 1.}})


def test_visible_unit_inventory_retains_singletons_and_never_crosses_records():
    context = singleton_context(glyph_alphabet=("y", "x"), max_emission_length=3)
    assert observed_units(["xy", "yx"], context) == ("y", "x", "yx", "xy")
    assert observed_units(["xyxy"], context, 3) == ("y", "x", "xy")
    assert observed_units(["xx"], context) == ("y", "x", "xx")
    with pytest.raises(ValueError, match="singleton"):
        observed_units(["xy"], context, 1)
    with pytest.raises(ValueError, match="outside"):
        observed_units(["z"], context)


def test_exact_score_on_a_single_possible_path_includes_stopping_and_empty_records():
    context = singleton_context(max_states=1, max_emission_length=1, max_alternatives=1)
    result = search_channel(singleton_source(), ["", "xx"], context,
                            config=SearchConfig(state_counts=(1,), restarts_per_state=1,
                                                proposals_per_restart=0, em_iterations=0))
    # p(empty)=1/2, p(xx)=(1/2)^3; all code fields here have cardinality one.
    assert result.score == pytest.approx({"model_bits": 0, "data_bits": 4, "total_bits": 4,
                                         "log_likelihood": -4 * math.log(2)})
    assert result.completed_candidates == 1
    assert result.stop_reason == "budget_exhausted"
    json.dumps(result.to_dict(), allow_nan=False)


def test_actual_structural_move_can_improve_complete_quantized_objective():
    context = singleton_context(max_states=1)
    result = search_channel(singleton_source(), ["xx"] * 12, context,
                            config=SearchConfig(seed=7, state_counts=(1,), restarts_per_state=1,
                                                proposals_per_restart=100, em_iterations=0))
    assert result.score["total_bits"] < result.trace[0]["selected_score"]["total_bits"]
    assert any(event["event"] == "proposal" and event["accepted"] for event in result.trace)
    assert result.channel.rows[("s0", "a")] == (Emission("s0", "xx", 1.),)
    assert result.score == two_part_score(singleton_source(), result.channel, ["xx"] * 12, context)
    assert result.proposals == 100
    assert len(result.trace) == 101


def test_all_completed_proposals_are_valid_normalized_grid_channels_and_best_is_retained():
    source = SourceModel(("a", "b"), 1, {"": {"a": .7, "b": .3},
                                            "a": {"a": .2, "b": .8},
                                            "b": {"a": .6, "b": .4}})
    context = singleton_context(source_alphabet=("a", "b"), glyph_alphabet=("x", "y", "z"),
                                max_emission_length=3)
    records = ["xyzxy", "yzzx"]
    before = source.to_dict()
    result = search_channel(source, records, context,
                            config=SearchConfig(seed=91, state_counts=(1, 2), restarts_per_state=1,
                                                proposals_per_restart=100, em_iterations=1))
    assert source.to_dict() == before
    stages = [stage for event in result.trace for stage in event["stages"] if stage["status"] == "scored"]
    assert result.completed_candidates == len(stages)
    assert {event["state_count"] for event in result.trace} == {1, 2}
    assert {event["move"]["kind"] for event in result.trace if event["event"] == "proposal"} == set(search.MOVES)
    assert result.score["total_bits"] == min(stage["score"]["total_bits"] for stage in stages)
    for stage in stages:
        channel = Channel.from_dict(stage["channel"])
        for row in channel.rows.values():
            assert sum(item.probability for item in row) == pytest.approx(1)
            assert len(row) <= context.max_alternatives
            assert all(item.glyphs in result.unit_pool for item in row)
            assert all(item.probability > 0 and item.probability * 4 == round(item.probability * 4) for item in row)
        assert stage["score"] == two_part_score(source, channel, records, context)
    json.dumps(result.to_dict(), allow_nan=False)


def test_restart_schedule_and_count_budget_are_reproducible():
    config = SearchConfig(seed=81, state_counts=(1, 2), restarts_per_state=2,
                          proposals_per_restart=15, em_iterations=0)
    first = search_channel(singleton_source(), ["xxx", "xxxx"], singleton_context(), config=config)
    second = search_channel(singleton_source(), ["xxx", "xxxx"], singleton_context(), config=config)
    assert first.channel == second.channel
    assert first.score == second.score
    assert first.trace == second.trace
    assert first.proposals == 60
    assert [(event["restart"], event["state_count"]) for event in first.trace
            if event["event"] == "initialization"] == [(0, 1), (0, 2), (1, 1), (1, 2)]


def test_partial_candidate_score_cannot_replace_finished_incumbent(monkeypatch):
    now = [0.]
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])

    def slow_score(source, channel, record):
        value = forward_log_probability(source, channel, record)
        now[0] += 1
        return value

    def fixed_proposal(channel, context, units, rng):
        candidate = Channel(channel.states, channel.glyph_alphabet, channel.initial,
                            {("s0", "a"): (Emission("s0", "x", .5), Emission("s0", "xx", .5))}, .5, 2)
        return candidate, {"kind": "add_emission"}

    monkeypatch.setattr(search, "forward_log_probability", slow_score)
    monkeypatch.setattr(search, "_propose", fixed_proposal)
    result = search_channel(singleton_source(), ["x", "x"], singleton_context(),
                            config=SearchConfig(state_counts=(1,), restarts_per_state=1,
                                                em_iterations=0, max_seconds=3))
    assert result.stop_reason == "time_limit"
    assert result.channel.to_dict() == result.trace[0]["selected_channel"]
    assert result.score == result.trace[0]["selected_score"]
    last = result.trace[-1]
    assert last["rejection_reason"] == "time_limit"
    assert last["stages"][0]["records_scored"] == 1
    assert last["stages"][0]["score"] is None
    assert result.completed_candidates == 1


def test_refinement_deadline_retains_already_finished_raw_candidate(monkeypatch):
    now = [0.]
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])

    def interrupted_fit(source, channel, records, **kwargs):
        now[0] = 2.
        fitted = Channel(channel.states, channel.glyph_alphabet, channel.initial,
                         {("s0", "a"): (Emission("s0", "x", .75), Emission("s0", "y", .25))}, .5, 2)
        return EMResult(fitted, -1., -1., [], 0, "time_limit", 2.)

    monkeypatch.setattr(search, "fit_em", interrupted_fit)
    result = search_channel(singleton_source(), ["xx"], singleton_context(glyph_alphabet=("x", "y")),
                            config=SearchConfig(state_counts=(1,), restarts_per_state=1,
                                                em_iterations=1, max_seconds=1))
    assert result.stop_reason == "time_limit"
    assert result.completed_candidates == 1
    assert result.channel.rows[("s0", "a")] == (Emission("s0", "x", .5), Emission("s0", "y", .5))
    assert result.trace[0]["stages"][1]["status"] == "time_limit"
    assert result.trace[0]["stages"][1]["records_scored"] == 0


def test_zero_transition_source_reports_failed_initialization_without_fabricating_model():
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    context = singleton_context(source_alphabet=("a", "b"), glyph_alphabet=("x", "y"))
    result = search_channel(source, ["xy"], context,
                            config=SearchConfig(state_counts=(1,), restarts_per_state=1,
                                                initialization_attempts=3, em_iterations=0))
    assert result.stop_reason == "no_supported_initialization"
    assert result.channel is None and result.score is None
    assert result.proposals == result.completed_candidates == 0
    assert len(result.trace) == 3
    assert all(event["rejection_reason"] == "unsupported" for event in result.trace)
    json.dumps(result.to_dict(), allow_nan=False)


def test_initializer_capacity_and_shared_context_are_explicit_restrictions():
    with pytest.raises(ValueError, match="capacity"):
        search_channel(singleton_source(), ["xyz"], singleton_context(glyph_alphabet=("x", "y", "z")))
    with pytest.raises(ValueError, match="state count"):
        search_channel(singleton_source(), ["x"], singleton_context(max_states=1))
    with pytest.raises(ValueError, match="Source index"):
        search_channel(singleton_source(), ["x"], singleton_context(), source_index=1)
    with pytest.raises(ValueError, match="capacity"):
        search_channel(singleton_source(), ["x"], singleton_context(glyph_alphabet=("x", "y", "z")))


def test_frequency_seed_uses_fixed_source_and_ciphertext_counts_only():
    source = SourceModel(("a", "b"), 0, {"": {"a": .2, "b": .8}})
    context = singleton_context(source_alphabet=("a", "b"), glyph_alphabet=("x", "y"), max_states=1)
    result = search_channel(source, ["xxxy"], context,
                            config=SearchConfig(state_counts=(1,), restarts_per_state=2,
                                                proposals_per_restart=0, em_iterations=0))
    first = result.trace[0]
    assert first["method"] == "frequency"
    channel = Channel.from_dict(first["selected_channel"])
    assert channel.rows[("s0", "b")] == (Emission("s0", "x", 1.),)
    assert channel.rows[("s0", "a")] == (Emission("s0", "y", 1.),)
    assert result.trace[1]["method"] == "random"


def test_empty_records_are_valid_with_disclosed_alphabet_and_deterministic_em_is_skipped(monkeypatch):
    def unexpected_em(*args, **kwargs):
        raise AssertionError("One-alternative rows have no EM parameters")

    monkeypatch.setattr(search, "fit_em", unexpected_em)
    result = search_channel(singleton_source(), [""], singleton_context(),
                            config=SearchConfig(state_counts=(1,), restarts_per_state=1,
                                                proposals_per_restart=0, em_iterations=5))
    assert result.score["data_bits"] == 1
    assert result.unit_pool == ("x",)


@pytest.mark.parametrize("changes", [{"max_seconds": 0}, {"max_seconds": math.inf}, {"state_counts": ()},
                                     {"state_counts": (1, 1)}, {"seed": True}, {"max_units": 0},
                                     {"em_iterations": -1}, {"initialization_attempts": 0}])
def test_invalid_search_budgets_are_rejected(changes):
    with pytest.raises(ValueError):
        SearchConfig(**changes)
