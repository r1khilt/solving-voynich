import itertools
import math
from fractions import Fraction as F

import pytest

from scripts.audit_suffix_posterior_diagnostics import replay
from voynich.calibrated_suffix_source import DepthSource
from voynich.sparse_suffix_source import collect_counts, decode
from voynich.suffix_posterior_diagnostics import diagnose


def enumerate_weights(source, units, observed, rho, length):
    weighted = []
    for n in range(len(observed) + 1):
        if length is not None and n != length:
            continue
        for letters in itertools.product(source.alphabet, repeat=n):
            text = ''.join(letters)
            if ''.join(units[source.letters[c]] for c in text) != observed:
                continue
            weight = F(rho)
            for i, char in enumerate(text):
                root = source.counts['']
                prob = (F(root.get(char, 0)) + F(1, 2)) / (sum(root.values()) + F(len(source.alphabet), 2))
                for depth in range(1, min(i, source.order) + 1):
                    row = source.counts.get(text[i - depth:i])
                    if row:
                        tau = F(source.masses[depth - 1])
                        prob = (row.get(char, 0) + tau * prob) / (sum(row.values()) + tau)
                weight *= (1 - rho) * prob
            weighted.append((text, weight))
    return weighted


@pytest.mark.parametrize('order', [0, 2])
@pytest.mark.parametrize('units,observed,length', [
    (('A', 'AA', 'B'), 'AAAA', None),
    (('A', 'AA', 'B'), 'AAAA', 3),
    (('A', 'A', 'AA'), 'AAA', None),
    (('A', 'AA', 'B'), 'AB', 2),
    (('A', 'AA', 'B'), '', 0),
])
def test_posterior_against_complete_rational_enumeration(order, units, observed, length):
    model = DepthSource('abc', [2, 5][:order], collect_counts(['aabbacabc'], 'abc', order))
    rho = F(1, 7)
    actual = diagnose(model, units, observed, float(rho), length=length)
    paths = enumerate_weights(model, units, observed, rho, length)
    total = sum(w for _, w in paths)
    probs = [w / total for _, w in paths]
    mean = sum(len(s) * p for (s, _), p in zip(paths, probs, strict=True))
    variance = sum((len(s) - mean) ** 2 * p for (s, _), p in zip(paths, probs, strict=True))
    entropy = -sum(float(p) * math.log2(p) for p in probs)
    assert actual.log_likelihood == pytest.approx(math.log(total), abs=2e-12)
    assert actual.joint_log_probability == pytest.approx(math.log(max(w for _, w in paths)), abs=2e-12)
    assert dict(paths)[actual.plaintext] == max(w for _, w in paths)
    assert actual.expected_length == pytest.approx(float(mean), abs=2e-12)
    assert actual.length_variance == pytest.approx(float(variance), abs=2e-12)
    assert actual.entropy_bits == pytest.approx(entropy, abs=2e-12)
    assert actual.path_count == len(paths)
    reference = replay(model.to_dict(), units, observed, float(rho), length=length)
    for field, expected in reference.items():
        assert getattr(actual, field) == pytest.approx(expected, abs=2e-12)
    if length is None:
        old = decode(model, units, observed, float(rho))
        assert old.log_likelihood == pytest.approx(actual.log_likelihood, abs=2e-12)
        assert old.joint_log_probability == pytest.approx(actual.joint_log_probability, abs=2e-12)


def test_fails_explicitly_at_node_cap_or_unsupported_length():
    model = DepthSource('ab', [3], collect_counts(['ababba'], 'ab', 1))
    with pytest.raises(RuntimeError, match='cap'):
        diagnose(model, ('A', 'AA'), 'AAAA', .1, max_nodes=1)
    with pytest.raises(ValueError, match='supported'):
        diagnose(model, ('A', 'AA'), 'AAAA', .1, length=1)
    with pytest.raises(ValueError, match='supported'):
        diagnose(model, ('A', 'AA'), 'Q', .1)
    with pytest.raises(ValueError, match='supported'):
        diagnose(model, ('A', 'AA'), '', .1, length=1)


@pytest.mark.parametrize('argument,value', [('rho', 0), ('rho', True), ('length', -1),
                                          ('length', True), ('max_nodes', 0)])
def test_invalid_settings(argument, value):
    model = DepthSource('ab', [], collect_counts(['abab'], 'ab', 0))
    kwargs = {'rho': .1, argument: value}
    with pytest.raises(ValueError):
        diagnose(model, ('A', 'AA'), 'AA', **kwargs)


def test_length_condition_is_a_subset_and_changes_no_source_weights():
    model = DepthSource('ab', [2], collect_counts(['abaab'], 'ab', 1))
    original = model.probabilities.copy()
    whole = diagnose(model, ('A', 'AA'), 'AAAA', .1)
    constrained = diagnose(model, ('A', 'AA'), 'AAAA', .1, length=3)
    assert constrained.log_likelihood <= whole.log_likelihood
    assert constrained.joint_log_probability <= whole.joint_log_probability
    assert constrained.expected_length == pytest.approx(3)
    assert constrained.length_variance == pytest.approx(0, abs=1e-12)
    assert (original == model.probabilities).all()
