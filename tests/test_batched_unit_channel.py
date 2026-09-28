import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from voynich.batched_unit_channel import batch_log_likelihood
from voynich.finite_state_channel import Channel, Emission, SourceModel, forward_log_probability


def models(order):
    rational = {"": {"a": Fraction(1, 3), "b": Fraction(2, 3)}}
    if order:
        rational |= {"a": {"a": Fraction(1, 2), "b": Fraction(1, 2)},
                     "b": {"a": Fraction(3, 4), "b": Fraction(1, 4)}}
    source = SourceModel(("a", "b"), order,
                         {context: {letter: float(value) for letter, value in row.items()}
                          for context, row in rational.items()})
    return source, rational


def exhaustive_probability(rational, order, units, record, rho=Fraction(1, 4)):
    # Independent rational path enumeration: no production recurrence, array
    # scaling, matches table, ring buffers, or source/channel engine internals.
    alphabet = tuple(rational[""])

    def walk(offset, previous):
        if offset == len(record):
            return rho
        result = Fraction(0)
        for letter, unit in zip(alphabet, units, strict=True):
            probability = rational[previous if order else ""][letter]
            if probability and record.startswith(unit, offset):
                result += (1 - rho) * probability * walk(offset + len(unit), letter)
        return result

    return walk(0, "")


def engine_score(source, units, records, rho=.25):
    channel = Channel(("s",), tuple(sorted(set("".join(units)))), {"s": 1.},
                      {("s", letter): (Emission("s", unit, 1.),)
                       for letter, unit in zip(source.alphabet, units, strict=True)}, rho, max(map(len, units)))
    return math.fsum(forward_log_probability(source, channel, record) for record in records)


@pytest.mark.parametrize("order", [0, 1])
def test_exhaustive_fraction_paths_and_existing_engine_agree_for_every_small_string(order):
    source, rational = models(order)
    candidates = list(itertools.product(("x", "xx", "y", "xy"), repeat=2))
    for length in range(6):
        for glyphs in itertools.product("xy", repeat=length):
            record = "".join(glyphs)
            actual = batch_log_likelihood(source, candidates, [record], .25)
            for index, units in enumerate(candidates):
                probability = exhaustive_probability(rational, order, units, record)
                expected = math.log(float(probability)) if probability else -math.inf
                assert actual[index] == pytest.approx(expected, abs=1e-12)
                assert actual[index] == pytest.approx(engine_score(source, units, [record]), abs=1e-12)


@pytest.mark.parametrize("order,probabilities", [
    (0, (Fraction(1, 4), Fraction(1, 16), Fraction(9, 64), Fraction(17, 256), Fraction(89, 1024))),
    (1, (Fraction(1, 4), Fraction(1, 16), Fraction(19, 128), Fraction(105, 1024))),
])
def test_ambiguous_length_paths_keep_correct_cross_offset_weights(order, probabilities):
    source, _ = models(order)
    for length, probability in enumerate(probabilities):
        result = batch_log_likelihood(source, [("x", "xx")], ["x" * length], .25)
        assert result[0] == pytest.approx(math.log(float(probability)), abs=1e-12)


@pytest.mark.parametrize("order", [0, 1])
def test_duplicate_units_remain_separate_source_paths_and_skip_empty_offsets(order):
    source, _ = models(order)
    for length in (0, 1, 2, 3, 4, 6, 27, 4000):
        result = batch_log_likelihood(source, [("xx", "xx")], ["x" * length], .25)
        expected = -math.inf if length % 2 else math.log(.25) + (length // 2) * math.log(.75)
        assert result[0] == pytest.approx(expected, abs=2e-10)


def test_independent_records_reset_start_context_and_charge_each_stop():
    source, _ = models(1)
    candidates = [("x", "xx"), ("xx", "x"), ("y", "xy")]
    records = ["x", "xx", "", "xxx"]
    actual = batch_log_likelihood(source, candidates, records, .25)
    assert np.allclose(actual, [engine_score(source, row, records) for row in candidates], atol=1e-12)
    separate = batch_log_likelihood(source, [("x", "xx")], ["x", "x"], .25)[0]
    joined = batch_log_likelihood(source, [("x", "xx")], ["xx"], .25)[0]
    assert separate != joined


def test_empty_collections_and_records_have_declared_identities():
    source, _ = models(1)
    assert batch_log_likelihood(source, [], ["xx"], .25).shape == (0,)
    assert batch_log_likelihood(source, [], [], .25).dtype == np.float64
    np.testing.assert_array_equal(batch_log_likelihood(source, [("x", "xx"), ("y", "y")], [], .25), [0., 0.])
    np.testing.assert_allclose(batch_log_likelihood(source, [("x", "xx"), ("y", "y")], ["", ""], .25),
                               [2 * math.log(.25)] * 2, atol=1e-15)


def test_batch_order_duplication_and_other_impossible_candidates_do_not_change_a_score():
    source, _ = models(1)
    candidates = [("x", "xx"), ("y", "y"), ("xx", "x"), ("x", "xx"), ("xyz", "x")]
    records = ["xxx", "xxxx", "x"]
    before = list(candidates)
    full = batch_log_likelihood(source, candidates, records, .25, 3)
    alone = np.asarray([batch_log_likelihood(source, [units], records, .25, 3)[0] for units in candidates])
    reverse = batch_log_likelihood(source, list(reversed(candidates)), records, .25, 3)
    np.testing.assert_allclose(full, alone, atol=1e-12)
    np.testing.assert_allclose(full, reverse[::-1], atol=1e-12)
    assert full[0] == full[3]
    assert full[1] == -math.inf
    assert candidates == before


def test_zeros_do_not_create_paths_from_matching_but_impossible_source_letters():
    source = SourceModel(("a", "b"), 1, {"": {"a": 1., "b": 0.},
                                         "a": {"a": 1., "b": 0.}, "b": {"a": 0., "b": 1.}})
    candidates = [("x", "y"), ("y", "x"), ("x", "x")]
    for record in ("", "x", "y", "xy", "xxx"):
        actual = batch_log_likelihood(source, candidates, [record], .5)
        expected = [engine_score(source, row, [record], .5) for row in candidates]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dynamic_range_guard_preserves_a_rare_context_that_later_alone_matches_suffix():
    source = SourceModel(("a", "b", "c"), 1,
                         {"": {"a": .5, "b": .5, "c": 0.},
                          "a": {"a": 1., "b": 0., "c": 0.},
                          "b": {"a": 0., "b": .5, "c": .5},
                          "c": {"a": 0., "b": 0., "c": 1.}})
    candidates = [("x", "x", "y"), ("x", "z", "y"), ("z", "x", "y")]
    record, rho = "x" * 1100 + "y", .25
    diagnostics = {}
    actual = batch_log_likelihood(source, candidates, [record], rho, diagnostics=diagnostics)
    expected = math.log(rho) + 1101 * (math.log1p(-rho) - math.log(2))
    assert actual[0] == pytest.approx(expected, abs=1e-10)
    assert actual[1] == -math.inf
    assert actual[2] == pytest.approx(expected, abs=1e-10)
    assert diagnostics["log_domain_fallbacks"] == 1
    np.testing.assert_allclose(actual, [engine_score(source, row, [record], rho) for row in candidates], atol=1e-10)
    # The guard also fires before a rare-context loss when the final result
    # stays finite; it is not merely a repair after observing -inf.
    finite = batch_log_likelihood(source, [candidates[0]], ["x" * 1100], rho, diagnostics=diagnostics)
    assert math.isfinite(finite[0])
    assert diagnostics["log_domain_fallbacks"] == 1


def test_subnormal_source_probabilities_use_log_path_without_losing_support():
    tiny = 1e-320
    source = SourceModel(("a", "b"), 0, {"": {"a": tiny, "b": 1.}})
    diagnostics = {}
    score = batch_log_likelihood(source, [("x", "y")], ["xxx"], .25, diagnostics=diagnostics)[0]
    assert score == pytest.approx(math.log(.25) + 3 * (math.log(.75) + math.log(tiny)), abs=1e-10)
    assert diagnostics["log_domain_fallbacks"] == 1


def test_unicode_lengths_and_trailing_null_glyphs_match_engine_semantics():
    source, _ = models(1)
    candidates = [("é", "é🙂"), ("x\0", "x"), ("\0", "\0\0"), ("🙂", "é")]
    for record in ("é🙂é", "x\0x", "\0\0\0", "🙂é", ""):
        actual = batch_log_likelihood(source, candidates, [record], .25)
        np.testing.assert_allclose(actual, [engine_score(source, row, [record]) for row in candidates], atol=1e-12)


@pytest.mark.parametrize("candidates,records,rho,bound", [
    ([("", "x")], ["x"], .25, 2), ([("xxx", "x")], ["x"], .25, 2),
    ([("x",)], ["x"], .25, 2), (["xy"], ["x"], .25, 2),
    ([("x", 2)], ["x"], .25, 2), ([("x", "y")], "xy", .25, 2),
    ([("x", "y")], [None], .25, 2), ([("x", "y")], ["x"], 0., 2),
    ([("x", "y")], ["x"], 1., 2), ([("x", "y")], ["x"], math.nan, 2),
    ([("x", "y")], ["x"], True, 2), ([("x", "y")], ["x"], .25, 0),
])
def test_invalid_inputs_rejected(candidates, records, rho, bound):
    source, _ = models(0)
    with pytest.raises(ValueError):
        batch_log_likelihood(source, candidates, records, rho, bound)
