"""Small exhaustive probability/decoding and source-estimation controls."""
import itertools
import math

import pytest

from voynich.higher_order_unit_channel import MarkovSource, decode_units, estimate_source, source_bits
from voynich.finite_state_channel import Channel, Emission, SourceModel, forward_log_probability, viterbi_decode


def words(alphabet, bound):
    return ("".join(row) for length in range(bound + 1) for row in itertools.product(alphabet, repeat=length))


def enumerated(source, units, observed, rho):
    found = {}
    for plain in words(source.alphabet, len(observed)):
        if "".join(units[source.alphabet.index(char)] for char in plain) != observed:
            continue
        probability, context = rho, ""
        for char in plain:
            probability *= (1 - rho) * source.probabilities[context][char]
            context = (context + char)[-source.order:] if source.order else ""
        if probability:
            found[plain] = probability
    return found


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("units", [("X", "YY"), ("X", "XX"), ("XY", "XY")])
def test_full_sum_and_maximum_match_exhaustive_plaintexts(order, units):
    source = estimate_source(["abbaabab", "bbab"], ("a", "b"), order, .75)
    for observed in words("XY", 5):
        expected = enumerated(source, units, observed, .2)
        actual = decode_units(source, units, observed, .2)
        if not expected:
            assert actual.plaintext is None
            assert actual.log_likelihood == -math.inf
        else:
            assert actual.log_likelihood == pytest.approx(math.log(sum(expected.values())), abs=2e-14)
            assert actual.joint_log_probability == pytest.approx(math.log(max(expected.values())), abs=2e-14)
            assert expected[actual.plaintext] == pytest.approx(max(expected.values()))


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_normalized_stopping_mass_and_record_reset(order):
    source = estimate_source(["abbaab"], ("a", "b"), order, 1.)
    mass = sum(math.exp(decode_units(source, ("X", "Y"), observed, .3).log_likelihood)
               for observed in words("XY", 4))
    assert mass == pytest.approx(1 - .7 ** 5, abs=3e-15)
    assert source_bits(source, ["a", "b"]) == pytest.approx(-math.log2(source.probabilities[""]["a"])
                                                           - math.log2(source.probabilities[""]["b"]))


def test_hierarchical_counts_never_join_records_and_unseen_suffix_backoff():
    source = estimate_source(["aa", "bb"], ("a", "b"), 3, 2.)
    assert source.probabilities["a"] == {"a": 2 / 3, "b": 1 / 3}
    assert source.probabilities["b"] == {"a": 1 / 3, "b": 2 / 3}
    assert source.probabilities["ab"] == source.probabilities["b"]
    assert source.probabilities["aba"] == source.probabilities["ba"]
    assert source.probabilities["aa"] == source.probabilities["a"]
    assert MarkovSource.from_dict(source.to_dict()).to_dict() == source.to_dict()
    with pytest.raises(TypeError):
        source.probabilities["a"]["b"] = .9


@pytest.mark.parametrize("order", [0, 1])
def test_order_one_and_zero_original_channel_agreement(order):
    source = estimate_source(["aaababbababa"], ("a", "b"), order, 4.)
    old = SourceModel(source.alphabet, order, source.probabilities)
    channel = Channel(("s",), ("X",), {"s": 1.}, {
        ("s", "a"): (Emission("s", "X", 1.),),
        ("s", "b"): (Emission("s", "XX", 1.),),
    }, .3, 2)
    for length in (0, 1, 4, 17, 300):
        observed = "X" * length
        value = decode_units(source, ("X", "XX"), observed, .3)
        assert value.log_likelihood == pytest.approx(forward_log_probability(old, channel, observed), abs=1e-11)
        assert value.joint_log_probability == pytest.approx(viterbi_decode(old, channel, observed).log_probability,
                                                            abs=1e-11)


def test_zero_probability_rare_context_and_logspace_underflow():
    source = MarkovSource(("a", "b"), 1, {"": {"a": 1e-200, "b": 1.},
                                           "a": {"a": 1., "b": 0.}, "b": {"a": 0., "b": 1.}})
    result = decode_units(source, ("X", "Y"), "X" * 700, .1)
    assert result.log_likelihood == pytest.approx(math.log(1e-200) + 700 * math.log(.9) + math.log(.1), abs=1e-10)
    assert result.plaintext == "a" * 700
    assert decode_units(source, ("X", "Y"), "XY", .1).plaintext is None


def test_invalid_and_resource_inputs_fail_closed():
    source = estimate_source(["ab"], ("a", "b"), 2, 1.)
    with pytest.raises(RuntimeError, match="no pruning"):
        decode_units(source, ("X", "XX"), "XXXX", .1, max_nodes=1)
    for units, rho in [(("", "Y"), .1), (("X",), .1), (("X", "Y"), 0), (("X", "Y"), True)]:
        with pytest.raises(ValueError):
            decode_units(source, units, "X", rho)
    for order in (-1, 4, True):
        with pytest.raises(ValueError):
            estimate_source(["ab"], ("a", "b"), order, 1.)
    with pytest.raises(ValueError):
        estimate_source(["ac"], ("a", "b"), 1, 1.)
    raw = source.to_dict()
    raw["probabilities"]["ab"]["a"] = 1.1
    with pytest.raises(ValueError):
        MarkovSource.from_dict(raw)


def test_unicode_and_nul_units_are_literal_strings():
    source = estimate_source(["αββα"], ("α", "β"), 3, 1.)
    result = decode_units(source, ("\0", "雪"), "\0雪\0", .2)
    assert result.plaintext == "αβα"
