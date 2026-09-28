"""Independent exact and log-space controls for the batched unit likelihood.

Fixtures are artificial literal strings only. Expected values come from a
backward Fraction recurrence or closed-form source-path probabilities, never
from the production matrix/scaling kernel or existing channel engine.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction as F
from functools import lru_cache

import numpy as np
import pytest

from voynich.batched_unit_channel import batch_log_likelihood
from voynich.finite_state_channel import SourceModel


def exact_record_probability(source, units, record, rho):
    stop = F(rho)

    @lru_cache(None)
    def suffix(offset, context):
        if offset == len(record):
            return stop
        probability = F(0)
        for letter, unit in zip(source.alphabet, units, strict=True):
            if record.startswith(unit, offset):
                next_context = letter if source.order else ""
                probability += ((1 - stop) * F(source.probabilities[context][letter])
                                * suffix(offset + len(unit), next_context))
        return probability

    return suffix(0, "")


def exact_corpus_log(source, units, records, rho):
    product = F(1)
    for record in records:
        product *= exact_record_probability(source, units, record, rho)
    return math.log(product.numerator) - math.log(product.denominator) if product else -math.inf


def source_fixture(order):
    rows = {"": {"a": .75, "b": .25}}
    if order:
        rows.update(a={"a": .125, "b": .875}, b={"a": .625, "b": .375})
    return SourceModel(("a", "b"), order, rows)


@pytest.mark.parametrize("order", [0, 1])
def test_every_tiny_candidate_and_observation_matches_backward_fraction_sum(order):
    source = source_fixture(order)
    vocabulary = ("x", "y", "xx", "xy", "yx", "yy", "xxx")
    candidates = tuple(itertools.product(vocabulary, repeat=2))
    for length in range(6):
        for chars in itertools.product("xy", repeat=length):
            record = "".join(chars)
            expected = [exact_corpus_log(source, units, (record,), .25) for units in candidates]
            actual = batch_log_likelihood(source, candidates, (record,), .25, max_emission_length=3)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-13)


def test_shared_scale_preserves_mass_at_different_pending_offsets():
    source = SourceModel(("a", "b"), 0, {"": {"a": .5, "b": .5}})
    candidates = (("x", "xxx"), ("x", "xx"))
    diagnostics = {}
    actual = batch_log_likelihood(source, candidates, ("xxxx",), .5, max_emission_length=9,
                                  diagnostics=diagnostics)
    # x/xxx gives four a events or ab or ba. x/xx gives aaaa, aab/aba/baa, bb.
    np.testing.assert_allclose(actual, [math.log(F(33, 512)), math.log(F(29, 512))], rtol=0, atol=1e-13)
    assert diagnostics["maximum_actual_unit_length"] == 3
    assert diagnostics["rolling_buffer_bytes"] == 4 * 2 * 3 * 8


def test_duplicate_units_preserve_distinct_source_contexts_and_sum_their_paths():
    source = SourceModel(("a", "b", "c"), 1,
                         {"": {"a": .5, "b": .5, "c": 0.},
                          "a": {"a": 1., "b": 0., "c": 0.},
                          "b": {"a": 0., "b": .25, "c": .75},
                          "c": {"a": 1., "b": 0., "c": 0.}})
    units = ("x", "x", "y")
    records = ("xxy", "x", "", "xxyx")
    actual = batch_log_likelihood(source, (units,), records, .5)[0]
    expected = exact_corpus_log(source, units, records, .5)
    assert actual == pytest.approx(expected, abs=1e-13)
    # After xx, only the b context can emit the final y; merging a/b contexts
    # because their units both equal x would alter this probability.
    assert exact_record_probability(source, units, "xxy", .5) == F(3, 512)


def test_source_zeros_impossible_records_and_empty_records_are_distinguished():
    source = SourceModel(("a", "b"), 1,
                         {"": {"a": 1., "b": 0.}, "a": {"a": 0., "b": 1.}, "b": {"a": 1., "b": 0.}})
    candidates = (("x", "y"), ("xx", "x"), ("x", "x"))
    for records in (("",), ("x", "xy"), ("xx",), ("yx",), ("", "xyx", "")):
        expected = [exact_corpus_log(source, units, records, .25) for units in candidates]
        actual = batch_log_likelihood(source, candidates, records, .25)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
        assert not np.isnan(actual).any()


def test_record_lengths_batch_order_and_batch_partitions_do_not_mix_probability_mass():
    source = source_fixture(1)
    candidates = [["x", "xx"], ["xx", "x"], ["x", "x"], ["xxx", "xx"], ["y", "x"]]
    records = ["", "x", "xxx", "xxxxxx", "xx"]
    before = [row[:] for row in candidates], list(records), source.to_dict()
    expected = np.array([exact_corpus_log(source, units, records, .125) for units in candidates])
    actual = batch_log_likelihood(source, candidates, records, .125, max_emission_length=3)
    singletons = np.concatenate([batch_log_likelihood(source, [units], records, .125, max_emission_length=3)
                                for units in candidates])
    partitioned = np.concatenate([batch_log_likelihood(source, candidates[:2], records, .125, 3),
                                  batch_log_likelihood(source, candidates[2:], records, .125, 3)])
    reverse = batch_log_likelihood(source, candidates[::-1], records[::-1], .125, 3)
    for result in (actual, singletons, partitioned, reverse[::-1]):
        np.testing.assert_allclose(result, expected, rtol=0, atol=5e-13)
    assert (candidates, records, source.to_dict()) == before


def test_early_tiny_context_is_retained_when_a_late_suffix_uniquely_requires_it():
    source = SourceModel(("a", "b", "c"), 1,
                         {"": {"a": .5, "b": .5, "c": 0.},
                          "a": {"a": 1., "b": 0., "c": 0.},
                          "b": {"a": 0., "b": .5, "c": .5},
                          "c": {"a": 0., "b": 0., "c": 1.}})
    count, rho = 1800, .25
    candidates = (("x", "x", "y"), ("x", "xx", "y"), ("z", "x", "y"))
    diagnostics = {}
    actual = batch_log_likelihood(source, candidates, ("x" * count + "y",), rho,
                                  diagnostics=diagnostics)
    # The only complete source path is b repeated n times followed by c.
    # The much larger a-only prefix cannot produce y. Candidate2 groups x into
    # pairs, changing both its number of source steps and stop/continue factors.
    def expected(n):
        return math.log(rho) + (n + 1) * math.log1p(-rho) + math.log(.5) + n * math.log(.5)

    np.testing.assert_allclose(actual, [expected(count), expected(count // 2), expected(count)], rtol=0, atol=2e-10)
    assert np.isfinite(actual).all()
    assert diagnostics["log_domain_fallbacks"] >= 1
    # A safe neighbor in the same batch cannot inherit another candidate's
    # fallback, scale or normalization state.
    separate = batch_log_likelihood(source, (candidates[1],), ("x" * count + "y",), rho)[0]
    assert separate == pytest.approx(actual[1], abs=1e-12)


def test_positive_subnormal_source_probability_triggers_safe_log_rescoring():
    rare = float.fromhex("0x0.0000000000001p-1022")
    source = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": rare}})
    diagnostics = {}
    actual = batch_log_likelihood(source, (("x", "y"),), ("y",), .5, diagnostics=diagnostics)[0]
    assert actual == pytest.approx(math.log(rare) + 2 * math.log(.5), abs=1e-12)
    assert math.isfinite(actual)
    assert diagnostics["log_domain_fallbacks"] == 1


def test_unicode_null_padding_lengths_and_whole_emissions_match_literal_strings():
    source = source_fixture(1)
    candidates = (("x", "x\0"), ("x\0", "\0"), ("λ", "λ\0"), ("x\0\0", "x"))
    for record in ("", "x", "x\0", "x\0\0", "λ", "λ\0", "\0", "xx\0"):
        expected = [exact_corpus_log(source, units, (record,), .5) for units in candidates]
        actual = batch_log_likelihood(source, candidates, (record,), .5, max_emission_length=3)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


def test_empty_collections_and_boundaries_have_exact_neutral_or_stop_values():
    source = source_fixture(0)
    candidates = (("x", "xx"), ("xx", "x"))
    np.testing.assert_array_equal(batch_log_likelihood(source, candidates, (), .25), [0., 0.])
    empty = batch_log_likelihood(source, (), ("x",), .25)
    assert empty.shape == (0,) and empty.dtype == np.float64
    np.testing.assert_allclose(batch_log_likelihood(source, candidates, ("", "", ""), .25),
                               [3 * math.log(.25)] * 2, rtol=0, atol=1e-13)
    # No emitted chunk may span records: xx is legal, but separate x records
    # are impossible when every source letter emits xx.
    assert math.isfinite(batch_log_likelihood(source, (("xx", "xx"),), ("xx",), .5)[0])
    assert batch_log_likelihood(source, (("xx", "xx"),), ("x", "x"), .5)[0] == -math.inf
