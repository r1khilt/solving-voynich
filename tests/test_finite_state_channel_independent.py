"""Exact rational path oracle, independent of the channel engine's algorithms.

Every fixture is hand specified. Enumeration includes stopping and initial-state
mass, never conditions emission rows on whether they match the observation,
and terminates only because each emitted chunk consumes a positive length.
There is no training, corpus access, random benchmark generation, or gold file.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction as F

import pytest

from voynich.finite_state_channel import (
    Channel,
    Emission,
    SourceModel,
    forward_log_probability,
    posterior_expected_counts,
    viterbi_decode,
)


@dataclass(frozen=True)
class ExactModel:
    alphabet: tuple[str, ...]
    states: tuple[str, ...]
    glyphs: tuple[str, ...]
    order: int
    source: dict[str, dict[str, F]]
    initial: dict[str, F]
    rows: dict[tuple[str, str], tuple[tuple[str, str, F], ...]]
    stop: F


@dataclass(frozen=True)
class ExactPath:
    plaintext: str
    ciphertext: str
    states: tuple[str, ...]
    chunks: tuple[str, ...]
    indices: tuple[int, ...]
    events: tuple[tuple[str, str, int], ...]
    probability: F


def exact_paths(model: ExactModel, maximum_glyphs: int) -> list[ExactPath]:
    """Exhaust all terminated latent paths within an output-length bound."""
    assert maximum_glyphs >= 0
    assert sum(model.initial.values(), F(0)) == 1
    assert all(sum(row.values(), F(0)) == 1 for row in model.source.values())
    assert all(sum((event[2] for event in row), F(0)) == 1 for row in model.rows.values())
    assert 0 < model.stop <= 1
    assert all(chunk and weight >= 0 for row in model.rows.values() for _, chunk, weight in row)
    completed = []

    def visit(plain, cipher, states, chunks, indices, events, mass):
        completed.append(ExactPath(plain, cipher, states, chunks, indices, events, mass * model.stop))
        if len(cipher) == maximum_glyphs or model.stop == 1:
            return
        history = plain[-1:] if model.order else ""
        for letter, source_probability in model.source[history].items():
            for index, (destination, chunk, channel_probability) in enumerate(model.rows[states[-1], letter]):
                if len(cipher + chunk) > maximum_glyphs:
                    continue
                updated = mass * (1 - model.stop) * source_probability * channel_probability
                if updated:
                    visit(plain + letter, cipher + chunk, (*states, destination), (*chunks, chunk),
                          (*indices, index), (*events, (states[-1], letter, index)), updated)

    for state, probability in model.initial.items():
        if probability:
            visit("", "", (state,), (), (), (), probability)
    return completed


def engine(model: ExactModel) -> tuple[SourceModel, Channel]:
    source = SourceModel(alphabet=model.alphabet, order=model.order,
                         probabilities={history: {letter: float(value) for letter, value in row.items()}
                                        for history, row in model.source.items()})
    channel = Channel(states=model.states, glyph_alphabet=model.glyphs,
                      initial={state: float(value) for state, value in model.initial.items()},
                      rows={key: tuple(Emission(destination, chunk, float(weight))
                                       for destination, chunk, weight in row) for key, row in model.rows.items()},
                      stop_probability=float(model.stop), max_emission_length=4)
    return source, channel


def log_fraction(value: F) -> float:
    return math.log(value.numerator) - math.log(value.denominator) if value else -math.inf


def strings(alphabet: tuple[str, ...], bound: int):
    for length in range(bound + 1):
        for letters in itertools.product(alphabet, repeat=length):
            yield "".join(letters)


def stateful_fixture() -> ExactModel:
    return ExactModel(
        alphabet=("a", "b"), states=("left", "right"), glyphs=("x", "y"), order=1,
        source={"": {"a": F(2, 3), "b": F(1, 3)},
                "a": {"a": F(1, 4), "b": F(3, 4)}, "b": {"a": F(4, 5), "b": F(1, 5)}},
        initial={"left": F(1, 3), "right": F(2, 3)}, stop=F(1, 3),
        rows={("left", "a"): (("left", "x", F(1, 2)), ("right", "xy", F(1, 3)),
                                ("left", "y", F(1, 6))),
              ("left", "b"): (("right", "y", F(2, 5)), ("left", "x", F(3, 5))),
              ("right", "a"): (("left", "x", F(1, 4)), ("right", "y", F(3, 4))),
              ("right", "b"): (("left", "xy", F(1, 2)), ("right", "x", F(1, 2)))})


def split_path_fixture() -> ExactModel:
    # A has the largest summed plaintext mass, but its two equal channel paths
    # each have less joint mass than B's single path.
    return ExactModel(alphabet=("a", "b"), states=("s",), glyphs=("x", "y"), order=0,
                      source={"": {"a": F(3, 5), "b": F(2, 5)}}, initial={"s": F(1)}, stop=F(1, 2),
                      rows={("s", "a"): (("s", "x", F(1, 2)), ("s", "x", F(1, 2))),
                            ("s", "b"): (("s", "x", F(1)),)})


def compare_all_inference(model: ExactModel, bound: int) -> int:
    source, channel = engine(model)
    by_cipher = defaultdict(list)
    for path in exact_paths(model, bound):
        by_cipher[path.ciphertext].append(path)
    valid_events = {(*key, index) for key, row in model.rows.items() for index in range(len(row))}
    checked = 0
    for cipher in strings(model.glyphs, bound):
        paths = by_cipher[cipher]
        likelihood = sum((path.probability for path in paths), F(0))
        result = forward_log_probability(source, channel, cipher)
        best = viterbi_decode(source, channel, cipher)
        posterior = posterior_expected_counts(source, channel, cipher)
        if not likelihood:
            assert result == best.log_probability == posterior.log_likelihood == -math.inf
            assert best.plaintext is None
            assert posterior.expected_source_length is None
            assert posterior.expected_emissions == {}
            continue
        assert result == pytest.approx(log_fraction(likelihood), abs=2e-12)
        assert posterior.log_likelihood == pytest.approx(result, abs=2e-12)
        optimum = max(path.probability for path in paths)
        assert best.log_probability == pytest.approx(log_fraction(optimum), abs=2e-12)
        matching = [path for path in paths if path.plaintext == best.plaintext
                    and path.states == tuple(best.states) and path.chunks == tuple(best.glyph_chunks)
                    and path.indices == tuple(best.emission_indices)]
        assert len(matching) == 1 and matching[0].probability == optimum
        expected_counts = defaultdict(F)
        expected_length = F(0)
        for path in paths:
            weight = path.probability / likelihood
            expected_length += len(path.plaintext) * weight
            for event, count in Counter(path.events).items():
                expected_counts[event] += count * weight
        assert set(posterior.expected_emissions) <= valid_events
        for event in set(posterior.expected_emissions) | set(expected_counts):
            assert posterior.expected_emissions.get(event, 0.) == pytest.approx(float(expected_counts[event]), abs=2e-12)
        assert posterior.expected_source_length == pytest.approx(float(expected_length), abs=2e-12)
        assert sum(posterior.expected_emissions.values()) == pytest.approx(float(expected_length), abs=2e-12)
        checked += 1
    return checked


def test_exact_fraction_oracle_matches_stateful_variable_length_inference():
    assert compare_all_inference(stateful_fixture(), 4) == 31


def test_duplicate_channel_paths_are_summed_and_joint_map_is_not_plaintext_map():
    model = split_path_fixture()
    assert compare_all_inference(model, 4) == 5
    paths = [path for path in exact_paths(model, 1) if path.ciphertext == "x"]
    plaintext_mass = defaultdict(F)
    for path in paths:
        plaintext_mass[path.plaintext] += path.probability
    assert plaintext_mass == {"a": F(3, 20), "b": F(1, 10)}
    assert max(plaintext_mass, key=plaintext_mass.get) == "a"
    source, channel = engine(model)
    assert viterbi_decode(source, channel, "x").plaintext == "b"
    assert math.exp(forward_log_probability(source, channel, "x")) == pytest.approx(.25)
    posterior = posterior_expected_counts(source, channel, "x")
    assert posterior.expected_emissions[("s", "a", 0)] == pytest.approx(.3)
    assert posterior.expected_emissions[("s", "a", 1)] == pytest.approx(.3)
    assert posterior.expected_emissions[("s", "b", 0)] == pytest.approx(.4)


def test_empty_observation_marginalizes_initial_state_but_joint_viterbi_does_not():
    model = stateful_fixture()
    source, channel = engine(model)
    assert forward_log_probability(source, channel, "") == pytest.approx(log_fraction(model.stop))
    best = viterbi_decode(source, channel, "")
    assert best.log_probability == pytest.approx(log_fraction(model.stop * max(model.initial.values())))
    assert best.plaintext == "" and best.glyph_chunks == () and best.emission_indices == ()
    assert tuple(best.states) == ("right",)
    posterior = posterior_expected_counts(source, channel, "")
    assert posterior.expected_source_length == 0.
    assert not any(posterior.expected_emissions.values())


def test_matching_emissions_are_not_renormalized_and_zero_mass_paths_stay_zero():
    model = ExactModel(alphabet=("a", "b"), states=("live", "dead"), glyphs=("x", "y"), order=0,
                       source={"": {"a": F(1), "b": F(0)}}, initial={"live": F(1), "dead": F(0)},
                       stop=F(1, 3), rows={
                           ("live", "a"): (("live", "x", F(1, 4)), ("live", "y", F(3, 4)),
                                           ("dead", "x", F(0))),
                           ("live", "b"): (("live", "y", F(1)),),
                           ("dead", "a"): (("dead", "x", F(1)),),
                           ("dead", "b"): (("dead", "y", F(1)),)})
    assert compare_all_inference(model, 3) == 15
    source, channel = engine(model)
    assert forward_log_probability(source, channel, "x") == pytest.approx(log_fraction(F(1, 18)))
    posterior = posterior_expected_counts(source, channel, "x")
    assert posterior.expected_emissions[("live", "a", 0)] == pytest.approx(1.)
    assert all(value == 0 for key, value in posterior.expected_emissions.items() if key != ("live", "a", 0))


def test_posterior_source_length_marginalizes_ambiguous_segmentation():
    model = ExactModel(alphabet=("a",), states=("s",), glyphs=("x",), order=0,
                       source={"": {"a": F(1)}}, initial={"s": F(1)}, stop=F(1, 2),
                       rows={("s", "a"): (("s", "x", F(1, 2)), ("s", "xx", F(1, 2)))})
    assert compare_all_inference(model, 5) == 6
    source, channel = engine(model)
    assert forward_log_probability(source, channel, "xx") == pytest.approx(log_fraction(F(5, 32)))
    assert viterbi_decode(source, channel, "xx").plaintext == "a"
    posterior = posterior_expected_counts(source, channel, "xx")
    assert posterior.expected_source_length == pytest.approx(6 / 5)
    assert posterior.expected_emissions[("s", "a", 0)] == pytest.approx(2 / 5)
    assert posterior.expected_emissions[("s", "a", 1)] == pytest.approx(4 / 5)


def test_one_glyph_emissions_have_exact_geometric_length_mass_and_tail():
    model = replace(stateful_fixture(), rows={key: tuple((state, chunk[0], weight) for state, chunk, weight in row)
                                             for key, row in stateful_fixture().rows.items()})
    source, channel = engine(model)
    paths = exact_paths(model, 5)
    for length in range(6):
        exact_mass = sum((path.probability for path in paths if len(path.ciphertext) == length), F(0))
        expected = model.stop * (1 - model.stop) ** length
        assert exact_mass == expected
        engine_mass = math.fsum(math.exp(forward_log_probability(source, channel, "".join(chars)))
                                for chars in itertools.product(model.glyphs, repeat=length))
        assert engine_mass == pytest.approx(float(expected), abs=2e-12)
    cumulative = sum((path.probability for path in paths), F(0))
    assert cumulative + (1 - model.stop) ** 6 == 1


def test_variable_length_truncation_bounds_and_whole_emission_consumption():
    model = stateful_fixture()
    for limit in range(5):
        mass = sum((path.probability for path in exact_paths(model, limit)), F(0))
        assert 1 - (1 - model.stop) ** (limit // 2 + 1) <= mass <= 1 - (1 - model.stop) ** (limit + 1)
    doubled = ExactModel(alphabet=("a",), states=("s",), glyphs=("x",), order=0,
                         source={"": {"a": F(1)}}, initial={"s": F(1)}, stop=F(1, 3),
                         rows={("s", "a"): (("s", "xx", F(1)),)})
    assert compare_all_inference(doubled, 8) == 5
    source, channel = engine(doubled)
    for length in range(9):
        result = forward_log_probability(source, channel, "x" * length)
        if length % 2:
            assert result == -math.inf
        else:
            assert result == pytest.approx(log_fraction(doubled.stop * (1 - doubled.stop) ** (length // 2)), abs=2e-12)


def test_state_renaming_and_reordering_preserve_likelihood_and_posterior_events():
    model = stateful_fixture()
    names = {"left": "q7", "right": "q2"}
    renamed = replace(model, states=tuple(names[state] for state in reversed(model.states)),
                      initial={names[state]: mass for state, mass in reversed(tuple(model.initial.items()))},
                      rows={(names[state], letter): tuple((names[dest], chunk, weight) for dest, chunk, weight in row)
                            for (state, letter), row in reversed(tuple(model.rows.items()))})
    first_source, first_channel = engine(model)
    second_source, second_channel = engine(renamed)
    for cipher in strings(model.glyphs, 3):
        assert forward_log_probability(first_source, first_channel, cipher) == pytest.approx(
            forward_log_probability(second_source, second_channel, cipher), abs=2e-12)
        assert viterbi_decode(first_source, first_channel, cipher).log_probability == pytest.approx(
            viterbi_decode(second_source, second_channel, cipher).log_probability, abs=2e-12)
        first = posterior_expected_counts(first_source, first_channel, cipher)
        second = posterior_expected_counts(second_source, second_channel, cipher)
        assert first.expected_source_length == pytest.approx(second.expected_source_length, abs=2e-12)
        for (state, letter, index), count in first.expected_emissions.items():
            assert count == pytest.approx(second.expected_emissions.get((names[state], letter, index), 0.), abs=2e-12)


def test_observational_equivalence_does_not_identify_plaintext():
    identity = ExactModel(alphabet=("a", "b"), states=("s",), glyphs=("0", "1"), order=0,
                          source={"": {"a": F(1, 2), "b": F(1, 2)}}, initial={"s": F(1)}, stop=F(2, 5),
                          rows={("s", "a"): (("s", "0", F(1)),), ("s", "b"): (("s", "1", F(1)),)})
    swapped = replace(identity, rows={("s", "a"): (("s", "1", F(1)),),
                                      ("s", "b"): (("s", "0", F(1)),)})
    first_source, first_channel = engine(identity)
    second_source, second_channel = engine(swapped)
    for cipher in strings(identity.glyphs, 6):
        probability = identity.stop * ((1 - identity.stop) / 2) ** len(cipher)
        first = forward_log_probability(first_source, first_channel, cipher)
        second = forward_log_probability(second_source, second_channel, cipher)
        assert first == pytest.approx(log_fraction(probability), abs=2e-12)
        assert second == pytest.approx(first, abs=2e-12)
        first_plain = viterbi_decode(first_source, first_channel, cipher).plaintext
        second_plain = viterbi_decode(second_source, second_channel, cipher).plaintext
        assert first_plain == cipher.translate(str.maketrans("01", "ab"))
        assert second_plain == cipher.translate(str.maketrans("01", "ba"))
        if cipher:
            assert first_plain != second_plain


def test_source_and_channel_serialization_preserve_all_latent_path_likelihoods():
    source, channel = engine(stateful_fixture())
    restored_source = SourceModel.from_dict(source.to_dict())
    restored_channel = Channel.from_dict(channel.to_dict())
    for cipher in strings(("x", "y"), 3):
        assert forward_log_probability(restored_source, restored_channel, cipher) == pytest.approx(
            forward_log_probability(source, channel, cipher), abs=2e-12)


def test_log_forward_does_not_underflow_on_a_long_unambiguous_stream():
    model = ExactModel(alphabet=("a",), states=("s",), glyphs=("x",), order=0,
                       source={"": {"a": F(1)}}, initial={"s": F(1)}, stop=F(1, 2),
                       rows={("s", "a"): (("s", "x", F(1)),)})
    source, channel = engine(model)
    length = 2000
    expected = -(length + 1) * math.log(2.)
    assert forward_log_probability(source, channel, "x" * length) == pytest.approx(expected, abs=1e-9)
