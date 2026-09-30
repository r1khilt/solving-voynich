"""Rational full-history enumeration, not the production state recurrence."""
import itertools
import math
import random
from fractions import Fraction as F

import pytest

from voynich.higher_order_unit_channel import decode_units, estimate_source
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def words(alphabet, maximum):
    for size in range(maximum + 1):
        for chars in itertools.product(alphabet, repeat=size):
            yield "".join(chars)


def probability(records, alphabet, history, char, order, tau):
    history = history[-order:] if order else ""
    if not history:
        return F(2 * sum(r.count(char) for r in records) + 1, 2 * sum(map(len, records)) + len(alphabet))
    counts = {c: sum(r.startswith(history + c, i) for r in records
                    for i in range(max(0, len(r) - len(history)))) for c in alphabet}
    return (counts[char] + F(tau) * probability(records, alphabet, history[1:], char, order, tau)) / (sum(counts.values()) + F(tau))


@pytest.mark.parametrize("order", [0, 1, 3, 5, 8, 12])
@pytest.mark.parametrize("tau", [.25, 16.])
def test_all_short_histories_equal_rational_uncompressed_model(order, tau):
    records, alphabet = ("abbabaabbaba", "baaaa", "ab"), tuple("abc")
    source = SuffixSource(alphabet, order, tau, collect_counts(records, alphabet, order))
    for history in words(alphabet, 5):
        state = source.state(history)
        for char in alphabet:
            assert source.probabilities[state, alphabet.index(char)] == pytest.approx(
                float(probability(records, alphabet, history, char, order, tau)), rel=0, abs=4e-16)
            assert source.step(state, alphabet.index(char)) == source.state(history + char)
    assert SuffixSource.from_dict(source.to_dict()).to_dict() == source.to_dict()


@pytest.mark.parametrize("order", [0, 3, 8, 12])
def test_full_plaintext_enumeration_matches_marginal_map_and_normalization(order):
    records, alphabet, tau, rho = ("abbaabba", "bab"), tuple("ab"), 4., F(1, 4)
    source = SuffixSource(alphabet, order, tau, collect_counts(records, alphabet, order))
    for units in itertools.product(("x", "y", "xy"), repeat=2):
        groups = {}
        for plain in words(alphabet, 5):
            observed = "".join(units[alphabet.index(c)] for c in plain)
            if len(observed) > 5:
                continue
            mass = rho
            for i, char in enumerate(plain):
                mass *= (1 - rho) * probability(records, alphabet, plain[:i], char, order, tau)
            groups.setdefault(observed, {})[plain] = mass
        for observed in words("xy", 5):
            result = decode(source, units, observed, float(rho))
            paths = groups.get(observed, {})
            if not paths:
                assert result.plaintext is None and result.log_likelihood == -math.inf
            else:
                assert result.log_likelihood == pytest.approx(math.log(float(sum(paths.values()))), abs=2e-14)
                assert result.joint_log_probability == pytest.approx(math.log(float(max(paths.values()))), abs=2e-14)
                assert paths[result.plaintext] == max(paths.values())
    mass = sum(math.exp(decode(source, ('x','x'), observed, .25).log_likelihood)
               for observed in words('x', 5))
    assert mass == pytest.approx(1 - .75 ** 6, abs=2e-15)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_dense_legacy_equivalence_and_record_resets(order):
    records, alphabet = ('abbaab', 'bbbb', 'a'), tuple('abc')
    sparse = SuffixSource(alphabet, order, 64., collect_counts(records, alphabet, order))
    dense = estimate_source(records, alphabet, order, 64.)
    for observed in words('xy', 5):
        first = decode(sparse, ('x','y','xy'), observed, .2)
        second = decode_units(dense, ('x','y','xy'), observed, .2)
        assert first.log_likelihood == pytest.approx(second.log_likelihood, abs=1e-14)
        assert first.joint_log_probability == pytest.approx(second.joint_log_probability, abs=1e-14)
    assert sparse.bits(('ab','ba')) == pytest.approx(sum(sparse.bits((r,)) for r in ('ab','ba')), abs=1e-15)
    if order:
        assert collect_counts(('aa', 'bb'), alphabet, order)['a'] == {'a': 1}


def test_caps_and_malformed_models_fail_without_pruning():
    with pytest.raises(RuntimeError, match='no pruning'):
        collect_counts(('abba',), 'ab', 3, max_contexts=2)
    source = SuffixSource('ab', 3, 4., collect_counts(('abbaa',), 'ab', 3))
    actual = decode(source, ('x','xy'), 'xyxx', .25)
    assert decode(source, ('x','xy'), 'xyxx', .25, max_nodes=actual.reachable_nodes) == actual
    with pytest.raises(RuntimeError, match='no pruning'):
        decode(source, ('x','xy'), 'xyxx', .25, max_nodes=actual.reachable_nodes - 1)
    for rho in (0, 1, True, float('nan')):
        with pytest.raises(ValueError):
            decode(source, ('x','y'), 'x', rho)
    for counts in ({'': {'a':1}, 'ab': {'b':1}}, {'':{'a':True}}, {'':{'a':0}}, {'':{'z':1}}):
        with pytest.raises(ValueError):
            SuffixSource('ab', 3, 4., counts)
    with pytest.raises(ValueError):
        collect_counts(('abz',), 'ab', 3)
    with pytest.raises(ValueError):
        source.bits(('z',))
    with pytest.raises(ValueError):
        decode(source, ('','x'), 'x', .5)


@pytest.mark.parametrize('order', [5, 8, 12])
def test_long_history_state_is_sufficient_after_unseen_prefixes(order):
    records, alphabet = ('abcabbacabbacabcabbacacbbcabacabac', 'bbbabcac'), tuple('abc')
    source = SuffixSource(alphabet, order, 16., collect_counts(records, alphabet, order))
    assert max(map(len, source.contexts)) == order
    rng = random.Random(714)
    histories = [records[0][:i] for i in range(len(records[0]))]
    histories += [''.join(rng.choices(alphabet, k=30)) for _ in range(100)]
    for history in histories:
        state = 0
        for char in history:
            state = source.step(state, alphabet.index(char))
        assert state == source.state(history)
        for char in alphabet:
            assert source.step(state, alphabet.index(char)) == source.state(history + char)
            assert source.probabilities[state, alphabet.index(char)] == pytest.approx(
                float(probability(records, alphabet, history, char, order, 16.)), abs=3e-16)
