"""Exact-engine unit tests; no corpus, fitted model, or recovery benchmark."""

import itertools
import json
import math

import pytest

from voynich.finite_state_channel import (
    Channel,
    Emission,
    SourceModel,
    forward_log_probability,
    posterior_expected_counts,
    viterbi_decode,
)


def simple_models():
    source = SourceModel(("a", "b"), 0, {"": {"a": 0.6, "b": 0.4}})
    channel = Channel(("s",), ("x", "y"), {"s": 1.0},
                      {("s", "a"): (Emission("s", "x", 1.0),),
                       ("s", "b"): (Emission("s", "y", 1.0),)}, 0.2)
    return source, channel


def test_probability_includes_initial_non_stop_and_final_stop():
    source, channel = simple_models()
    assert math.exp(forward_log_probability(source, channel, "xyx")) == pytest.approx(
        0.2 * 0.8**3 * 0.6**2 * 0.4)
    result = viterbi_decode(source, channel, "xyx")
    assert result.plaintext == "aba"
    assert result.states == ("s", "s", "s", "s")
    assert result.glyph_chunks == ("x", "y", "x")
    assert result.emission_indices == (0, 0, 0)
    posterior = posterior_expected_counts(source, channel, "xyx")
    assert posterior.expected_source_length == pytest.approx(3)
    assert posterior.expected_emissions == pytest.approx({("s", "a", 0): 2, ("s", "b", 0): 1})


def test_complete_glyph_strings_and_geometric_tail_normalize():
    source, channel = simple_models()
    mass = sum(math.exp(forward_log_probability(source, channel, "".join(glyphs)))
               for length in range(7) for glyphs in itertools.product("xy", repeat=length))
    assert mass + 0.8**7 == pytest.approx(1.0)


def test_variable_length_shared_prefix_sums_distinct_segmentations():
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    channel = Channel(("s",), ("x",), {"s": 1.0},
                      {("s", "a"): (Emission("s", "x", 0.5), Emission("s", "xx", 0.5))}, 0.5)
    # 'xx': one emission has mass 1/8, two one-glyph emissions have mass 1/32.
    assert math.exp(forward_log_probability(source, channel, "xx")) == pytest.approx(5 / 32)
    best = viterbi_decode(source, channel, "xx")
    assert best.plaintext == "a"
    assert best.glyph_chunks == ("xx",)
    assert math.exp(best.log_probability) == pytest.approx(1 / 8)
    posterior = posterior_expected_counts(source, channel, "xx")
    assert posterior.expected_emissions == pytest.approx({("s", "a", 0): 0.4, ("s", "a", 1): 0.8})
    assert posterior.expected_source_length == pytest.approx(1.2)
    # All paths with >= 5 glyphs require >= 3 source steps; this is a tail bound,
    # not a false assertion that a glyph-length truncation equals a source tail.
    partial = sum(math.exp(forward_log_probability(source, channel, "x" * size)) for size in range(5))
    assert 0 <= 1 - partial <= 0.5**3


def test_bigram_context_and_nonuniform_initial_states_are_used():
    source = SourceModel(("a", "b"), 1,
                         {"": {"a": 1.0, "b": 0.0}, "a": {"a": 0.0, "b": 1.0},
                          "b": {"a": 1.0, "b": 0.0}})
    rows = {(state, letter): (Emission(state, "x" if letter == "a" else "y", 1.0),)
            for state in ("one", "two") for letter in "ab"}
    channel = Channel(("one", "two"), ("x", "y"), {"one": 0.25, "two": 0.75}, rows, 0.4)
    assert math.exp(forward_log_probability(source, channel, "xyx")) == pytest.approx(0.6**3 * 0.4)
    best = viterbi_decode(source, channel, "xyx")
    assert best.plaintext == "aba"
    assert best.states == ("two",) * 4
    assert math.exp(best.log_probability) == pytest.approx(0.75 * 0.6**3 * 0.4)
    assert forward_log_probability(source, channel, "xx") == -math.inf


def test_joint_viterbi_need_not_choose_marginal_map_plaintext():
    source = SourceModel(("a", "b"), 0, {"": {"a": 0.6, "b": 0.4}})
    channel = Channel(("s",), ("x",), {"s": 1.0},
                      {("s", "a"): (Emission("s", "x", 0.5), Emission("s", "x", 0.5)),
                       ("s", "b"): (Emission("s", "x", 1.0),)}, 0.5)
    # a has total mass .15, but each of its paths has .075; b's path has .10.
    best = viterbi_decode(source, channel, "x")
    assert best.plaintext == "b"
    assert math.exp(best.log_probability) == pytest.approx(0.1)
    assert math.exp(forward_log_probability(source, channel, "x")) == pytest.approx(0.25)
    assert posterior_expected_counts(source, channel, "x").expected_emissions == pytest.approx(
        {("s", "a", 0): 0.3, ("s", "a", 1): 0.3, ("s", "b", 0): 0.4})


def test_empty_and_impossible_observations_have_distinct_results():
    source = SourceModel(("a",), 0, {"": {"a": 1.0}})
    channel = Channel(("one", "two"), ("x", "y"), {"one": 0.3, "two": 0.7},
                      {(state, "a"): (Emission(state, "xx", 1.0),) for state in ("one", "two")}, 0.2)
    assert math.exp(forward_log_probability(source, channel, "")) == pytest.approx(0.2)
    best = viterbi_decode(source, channel, "")
    assert best.plaintext == ""
    assert best.states == ("two",)
    assert math.exp(best.log_probability) == pytest.approx(0.14)
    empty = posterior_expected_counts(source, channel, "")
    assert empty.expected_source_length == 0
    assert all(value == 0 for value in empty.expected_emissions.values())
    for impossible in ("x", "y", "z"):
        assert forward_log_probability(source, channel, impossible) == -math.inf
        assert viterbi_decode(source, channel, impossible).plaintext is None
        posterior = posterior_expected_counts(source, channel, impossible)
        assert posterior.log_likelihood == -math.inf
        assert posterior.expected_source_length is None
        assert posterior.expected_emissions == {}


def test_log_space_keeps_tiny_long_path_finite():
    source, channel = simple_models()
    score = forward_log_probability(source, channel, "y" * 2000)
    assert math.isfinite(score)
    assert score == pytest.approx(math.log(0.2) + 2000 * math.log(0.8 * 0.4))


def test_renaming_state_and_glyph_labels_preserves_all_scores():
    source, channel = simple_models()
    renamed = Channel(("renamed",), ("Ω", "λ"), {"renamed": 1.0},
                      {("renamed", "a"): (Emission("renamed", "Ω", 1.0),),
                       ("renamed", "b"): (Emission("renamed", "λ", 1.0),)}, 0.2)
    assert forward_log_probability(source, channel, "xyx") == forward_log_probability(source, renamed, "ΩλΩ")
    assert viterbi_decode(source, channel, "xyx").log_probability == viterbi_decode(
        source, renamed, "ΩλΩ").log_probability
    original_counts = posterior_expected_counts(source, channel, "xyx").expected_emissions
    renamed_counts = posterior_expected_counts(source, renamed, "ΩλΩ").expected_emissions
    assert {("renamed", a, i): count for (_, a, i), count in original_counts.items()} == renamed_counts


def test_serialization_roundtrip_and_input_mutation_do_not_change_models():
    source, channel = simple_models()
    source_json = json.loads(json.dumps(source.to_dict()))
    channel_json = json.loads(json.dumps(channel.to_dict()))
    source_copy = SourceModel.from_dict(source_json)
    channel_copy = Channel.from_dict(channel_json)
    assert source_copy.to_dict() == source.to_dict()
    assert channel_copy.to_dict() == channel.to_dict()
    source_json["probabilities"][""]["a"] = 0.9
    channel_json["initial"]["s"] = 0.1
    assert source_copy.probabilities[""]["a"] == 0.6
    assert channel_copy.initial["s"] == 1.0
    with pytest.raises(TypeError):
        source_copy.probabilities[""]["a"] = 0.8
    with pytest.raises(TypeError):
        channel_copy.rows[("s", "a")] = ()


@pytest.mark.parametrize("defect", ["order", "context", "source_sum", "probability", "extra_letter",
                                   "epsilon", "long", "destination", "glyph", "row_sum", "missing_row",
                                   "initial", "stop_zero", "stop_one", "boolean"])
def test_invalid_probability_models_are_rejected(defect):
    source, channel = simple_models()
    source_data, channel_data = source.to_dict(), channel.to_dict()
    if defect == "order":
        source_data["order"] = 2
    elif defect == "context":
        source_data["probabilities"]["a"] = {"a": 0.6, "b": 0.4}
    elif defect == "source_sum":
        source_data["probabilities"][""]["a"] = 0.5
    elif defect == "probability":
        source_data["probabilities"][""]["a"] = math.nan
    elif defect == "extra_letter":
        source_data["probabilities"][""]["c"] = 0
    elif defect == "epsilon":
        channel_data["rows"][0]["emissions"][0]["glyphs"] = ""
    elif defect == "long":
        channel_data["rows"][0]["emissions"][0]["glyphs"] = "x" * 5
    elif defect == "destination":
        channel_data["rows"][0]["emissions"][0]["next_state"] = "missing"
    elif defect == "glyph":
        channel_data["rows"][0]["emissions"][0]["glyphs"] = "unknown"
    elif defect == "row_sum":
        channel_data["rows"][0]["emissions"][0]["probability"] = 0.5
    elif defect == "missing_row":
        channel_data["states"].append("new_state")
    elif defect == "initial":
        channel_data["initial"]["s"] = 0.8
    elif defect == "stop_zero":
        channel_data["stop_probability"] = 0
    elif defect == "stop_one":
        channel_data["stop_probability"] = 1
    else:
        source_data["probabilities"][""]["a"] = True
    with pytest.raises(ValueError):
        SourceModel.from_dict(source_data)
        Channel.from_dict(channel_data)


def test_serialized_duplicate_rows_and_incompatible_models_are_rejected():
    source, channel = simple_models()
    data = channel.to_dict()
    data["rows"].append(data["rows"][0])
    with pytest.raises(ValueError):
        Channel.from_dict(data)
    other = SourceModel(("c",), 0, {"": {"c": 1.0}})
    for function in (forward_log_probability, viterbi_decode, posterior_expected_counts):
        with pytest.raises(ValueError):
            function(other, channel, "")
        with pytest.raises(TypeError):
            function(source, channel, ["x"])
