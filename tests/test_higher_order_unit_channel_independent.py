"""Independent rational source-count and full plaintext enumeration controls.

Artificial strings only. References use direct substring counts and complete
source strings, not the production estimator or its offset/context recurrence.
"""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction as F

import pytest

from voynich.higher_order_unit_channel import MarkovSource, decode_units, estimate_source, source_bits


def words(alphabet, maximum):
    return ("".join(chars) for length in range(maximum + 1)
            for chars in itertools.product(alphabet, repeat=length))


def reference_source(records, alphabet, order, tau):
    # Count each complete context+next-letter window independently, preventing
    # any record-boundary pair or higher-order n-gram from being invented.
    total = sum(map(len, records))
    rows = {"": {char: F(2 * sum(record.count(char) for record in records) + 1,
                         2 * total + len(alphabet)) for char in alphabet}}
    for context in words(alphabet, order):
        if not context:
            continue
        counts = {char: sum(record.startswith(context + char, start)
                            for record in records for start in range(max(0, len(record) - len(context))))
                  for char in alphabet}
        amount = sum(counts.values())
        rows[context] = {char: (counts[char] + F(tau) * rows[context[1:]][char]) / (amount + F(tau))
                         for char in alphabet}
    return rows


def explicit_source(order):
    rows = {}
    for context in words("ab", order):
        # Dyadic, asymmetric, position-sensitive rows, including short starts.
        numerator = 2 * ((len(context) + sum((i + 1) * (char == "b") for i, char in enumerate(context))) % 4) + 1
        rows[context] = {"a": numerator / 8, "b": 1 - numerator / 8}
    return MarkovSource(("a", "b"), order, rows)


def path_mass(source, plain, rho):
    value = F(rho)
    for index, char in enumerate(plain):
        history = plain[max(0, index - source.order):index] if source.order else ""
        value *= (1 - F(rho)) * F(source.probabilities[history][char])
    return value


def full_plaintext_reference(source, units, maximum_observed, rho):
    found = {}
    for plain in words(source.alphabet, maximum_observed):
        observed = "".join(units[source.alphabet.index(char)] for char in plain)
        if len(observed) > maximum_observed:
            continue
        probability = path_mass(source, plain, rho)
        if probability:
            found.setdefault(observed, {})[plain] = probability
    return found


def rational_log(value):
    # These bounded enumeration fixtures stay representable. Taking one log
    # avoids cancellation from subtracting logs of large rational integers.
    return math.log(float(value)) if value else -math.inf


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("tau", [.25, 1., 4., 64.])
def test_every_estimated_row_matches_independent_window_counts_and_suffix_interpolation(order, tau):
    records, alphabet = ("aabb", "baba", "a", "baaab"), ("a", "b", "c")
    expected = reference_source(records, alphabet, order, tau)
    actual = estimate_source(records, alphabet, order, tau)
    assert set(actual.probabilities) == set(expected)
    for context, row in expected.items():
        assert sum(row.values()) == 1
        assert all(value > 0 for value in row.values())
        for char, value in row.items():
            assert actual.probabilities[context][char] == pytest.approx(float(value), rel=0, abs=3e-16)
        assert math.fsum(actual.probabilities[context].values()) == pytest.approx(1., rel=0, abs=3e-16)
    # A never-observed context backs off recursively, not to an unrelated row.
    if order >= 2:
        assert actual.probabilities["ac"] == actual.probabilities["c"]
    if order >= 3:
        assert actual.probabilities["bac"] == actual.probabilities["ac"]


def test_record_boundaries_and_global_unigram_start_are_counted_as_declared():
    separate = estimate_source(("aaa", "bb"), ("a", "b"), 3, 2.)
    joined = estimate_source(("aaabb",), ("a", "b"), 3, 2.)
    assert separate.probabilities[""] == {"a": 7 / 12, "b": 5 / 12}
    assert separate.probabilities["a"]["b"] == pytest.approx(5 / 24, rel=0, abs=1e-16)
    assert separate.probabilities["aa"]["b"] == pytest.approx(5 / 36, rel=0, abs=1e-16)
    assert separate.probabilities["a"] != joined.probabilities["a"]
    assert separate.probabilities["aab"] == separate.probabilities["ab"]
    # Initial source row is the smoothed global unigram, not empirical starts.
    expected_bits = -math.log2(F(7, 12)) - math.log2(F(5, 12))
    assert source_bits(separate, ("a", "b", "")) == pytest.approx(expected_bits, rel=0, abs=2e-15)
    assert source_bits(separate, ()) == 0.


def test_order_one_tau64_reproduces_legacy_count_arithmetic_bit_for_bit():
    records, alphabet = ("abbaba", "b", "aaaabbb", "ab"), tuple("abc")
    unigrams = {char: sum(row.count(char) for row in records) for char in alphabet}
    denominator = sum(unigrams.values()) + .5 * len(alphabet)
    expected = {"": {char: (unigrams[char] + .5) / denominator for char in alphabet}}
    for left in alphabet:
        counts = {right: sum(record[start:start + 2] == left + right
                              for record in records for start in range(len(record) - 1)) for right in alphabet}
        expected[left] = {right: (counts[right] + 64. * expected[""][right]) / (sum(counts.values()) + 64.)
                          for right in alphabet}
    assert estimate_source(records, alphabet, 1, 64.).to_dict()["probabilities"] == expected


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_all_tiny_unit_maps_and_observations_match_full_rational_plaintext_enumeration(order):
    source = explicit_source(order)
    vocabulary = ("x", "y", "xx", "xy", "yx", "yy")
    for units in itertools.product(vocabulary, repeat=2):
        expected = full_plaintext_reference(source, units, 4, .25)
        for observed in words("xy", 4):
            decoded = decode_units(source, units, observed, .25)
            paths = expected.get(observed, {})
            if not paths:
                assert decoded.plaintext is None
                assert decoded.log_likelihood == decoded.joint_log_probability == -math.inf
                assert decoded.to_dict()["log_likelihood"] is None
            else:
                total, best = sum(paths.values(), F(0)), max(paths.values())
                assert decoded.log_likelihood == pytest.approx(rational_log(total), rel=0, abs=3e-14)
                assert decoded.joint_log_probability == pytest.approx(rational_log(best), rel=0, abs=3e-14)
                assert paths[decoded.plaintext] == best
                assert decoded.joint_log_probability <= decoded.log_likelihood + 1e-14
            json.dumps(decoded.to_dict(), allow_nan=False)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("units", [("x", "y"), ("x", "x"), ("xx", "yy")])
def test_geometric_stopping_mass_is_normalized_even_when_letters_alias(order, units):
    source, rho = explicit_source(order), .5
    # Equal unit lengths let an observed-length cutoff correspond exactly to
    # at most three emitted source letters, including the empty record.
    length = 3 * len(units[0])
    total = math.fsum(math.exp(decode_units(source, units, observed, rho).log_likelihood)
                      for observed in words("xy", length))
    assert total == pytest.approx(1 - (1 - rho) ** 4, rel=0, abs=3e-15)
    empty = decode_units(source, units, "", rho)
    assert empty.plaintext == "" and empty.reachable_nodes == 1 and empty.edges == 0
    assert empty.log_likelihood == empty.joint_log_probability == math.log(rho)


def test_growing_start_context_order_three_zeros_and_exact_node_cap():
    rows = {history: {"a": .5, "b": .5} for history in words("ab", 3)}
    for history, char in (("", "a"), ("a", "b"), ("ab", "b"), ("abb", "a"), ("bba", "b")):
        rows[history] = {"a": float(char == "a"), "b": float(char == "b")}
    source = MarkovSource(("a", "b"), 3, rows)
    result = decode_units(source, ("x", "yy"), "xyyyyxyy", .5)
    assert result.plaintext == "abbab"
    assert result.log_likelihood == result.joint_log_probability == pytest.approx(math.log(F(1, 64)), rel=0, abs=2e-15)
    assert decode_units(source, ("x", "yy"), "xx", .5).plaintext is None
    assert decode_units(source, ("x", "yy"), "xyyyyxyy", .5, max_nodes=result.reachable_nodes) == result
    with pytest.raises(RuntimeError, match="no pruning"):
        decode_units(source, ("x", "yy"), "xyyyyxyy", .5, max_nodes=result.reachable_nodes - 1)


def test_log_space_preserves_an_early_rare_order_three_history():
    rows = {history: {"a": 1., "b": 0.} for history in words("ab", 3)}
    rows[""] = {"a": 1e-200, "b": 1.}
    for history in ("b", "bb", "bbb"):
        rows[history] = {"a": 0., "b": 1.}
    source = MarkovSource(("a", "b"), 3, rows)
    observed = "x" * 1200
    result = decode_units(source, ("x", "y"), observed, .25)
    expected = math.log(1e-200) + 1200 * math.log(.75) + math.log(.25)
    assert result.plaintext == "a" * 1200
    assert result.log_likelihood == pytest.approx(expected, rel=0, abs=1e-10)
    assert result.joint_log_probability == result.log_likelihood


def test_source_storage_roundtrip_is_immutable_and_unicode_units_are_literal():
    reference = reference_source(("αββα", "β"), ("α", "β"), 3, .25)
    raw = {history: {char: float(value) for char, value in row.items()} for history, row in reference.items()}
    source = MarkovSource(("α", "β"), 3, raw)
    saved = source.to_dict()
    raw[""]["α"] = 0.
    assert source.to_dict() == saved
    restored = MarkovSource.from_dict(saved)
    saved["probabilities"][""]["α"] = 0.
    assert restored.to_dict() == source.to_dict()
    with pytest.raises(TypeError):
        source.probabilities[""]["α"] = .4
    result = decode_units(source, ("雪\0", "雪"), "雪\0雪雪\0", .5)
    assert result.plaintext == "αβα"
    assert result.log_likelihood == pytest.approx(rational_log(path_mass(source, "αβα", .5)), rel=0, abs=1e-14)
