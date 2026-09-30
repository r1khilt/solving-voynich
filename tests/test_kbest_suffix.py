import itertools
import math
import random

import pytest

from voynich.kbest_suffix import bounded_mixture, compatible_mask, key_kbest, record_kbest
from voynich.shared_key_mixture import decode_shared_keys
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def model(order=2):
    return SuffixSource('ab', order, 3., collect_counts(['aababbababaaa'], 'ab', order))


def enumerate_texts(source, key, record, rho):
    result = {}
    for n in range(len(record) + 1):
        for letters in itertools.product(range(len(key)), repeat=n):
            if ''.join(key[i] for i in letters) != record:
                continue
            state, score = source.state(''), math.log(rho) + n * math.log1p(-rho)
            for i in letters:
                p = source.probabilities[state, i]
                if p == 0:
                    score = -math.inf
                    break
                score += math.log(p)
                state = source.step(state, i)
            if score != -math.inf:
                result[''.join(source.alphabet[i] for i in letters)] = score
    return result


@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_record_order_and_evidence_match_all_strings(order):
    rng, source = random.Random(2249 + order), model(order)
    for _ in range(16):
        key = tuple(rng.choice(['x', 'y', 'xx', 'xy', 'yx', 'yy']) for _ in range(2))
        record = ''.join(rng.choice('xy') for _ in range(rng.randrange(6)))
        truth = enumerate_texts(source, key, record, .3)
        found, evidence, _, _, _ = record_kbest(source, key, record, .3, 5)
        assert len(found) == min(5, len(truth))
        assert [v for _, v in found] == pytest.approx(sorted(truth.values(), reverse=True)[:5], abs=1e-12)
        assert len({text for text, _ in found}) == len(found)
        assert all(score == pytest.approx(truth[text], abs=1e-12) for text, score in found)
        assert evidence == pytest.approx(decode(source, key, record, .3).log_likelihood, abs=1e-12)


def test_cartesian_lists_are_exact_for_multiple_records_and_empty_boundaries():
    source, key, records = model(), ('x', 'xx'), ['xxxx', '', 'xxx']
    single = [enumerate_texts(source, key, r, .2) for r in records]
    truth = {texts: sum(s[t] for s, t in zip(single, texts)) for texts in itertools.product(*single)}
    for k in (1, 3, 10, 100):
        result = key_kbest(source, key, records, .2, k)
        ordered = sorted(truth.values(), reverse=True)
        assert [v for _, v in result.readings] == pytest.approx(ordered[:k], abs=1e-12)
        assert result.next_log_probability == pytest.approx(ordered[k] if len(ordered) > k else -math.inf)
        assert all(score == pytest.approx(truth[texts], abs=1e-12) for texts, score in result.readings)


def test_truncated_list_cannot_claim_the_wrong_mixture_winner_is_certified():
    source = SuffixSource('abc', 0, 4., collect_counts(['a'*45 + 'b'*41 + 'c'*11], 'abc', 0))
    keys = [('x', 'y', 'xx'), ('y', 'x', 'xx')]
    short = bounded_mixture(source, keys, ['xx'], .2, k=1)
    exact = decode_shared_keys(source, keys, ['xx'], .2)
    assert short['plaintexts'] != exact.plaintexts
    assert not short['floating_bound_separated']
    assert short['joint_log_probability'] < exact.joint_log_probability <= short['unseen_log_probability_upper_bound'] + 1e-12
    longer = bounded_mixture(source, keys, ['xx'], .2, k=2)
    assert longer['plaintexts'] == exact.plaintexts == ('c',)
    assert longer['floating_bound_separated']
    assert longer['joint_log_probability'] == pytest.approx(exact.joint_log_probability)


@pytest.mark.parametrize('order', [0, 1, 2])
def test_mixture_bounds_cover_exact_solution_and_evidence(order):
    rng, source = random.Random(23081 + order), model(order)
    keys_pool = list(itertools.product(['x', 'y', 'xx', 'xy'], repeat=2))
    for _ in range(12):
        keys = rng.sample(keys_pool, 4)
        records = [''.join(rng.choice('xy') for _ in range(rng.randrange(4))) for _ in range(2)]
        weights = [rng.uniform(-10, 0) for _ in keys]
        exact = decode_shared_keys(source, keys, records, .25, log_weights=weights)
        result = bounded_mixture(source, keys, records, .25, k=2, log_weights=weights)
        if exact.plaintexts is None:
            assert result['plaintexts'] is None and result['log_likelihood'] == -math.inf
            continue
        lower, upper = result['joint_log_probability'], result['unseen_log_probability_upper_bound']
        assert lower <= exact.joint_log_probability + 1e-10
        assert exact.joint_log_probability <= max(lower, upper) + 1e-10
        assert result['log_likelihood'] == pytest.approx(exact.log_likelihood, abs=1e-10)
        if result['floating_bound_separated']:
            assert lower == pytest.approx(exact.joint_log_probability, abs=1e-10)
        texts = result['plaintexts']
        expected = tuple(i for i, key in enumerate(keys)
                         if all(''.join(key[source.alphabet.index(c)] for c in t) == r
                                for t, r in zip(texts, records)))
        assert result['compatible_key_indices'] == expected


def test_joint_key_cannot_switch_between_records():
    source = SuffixSource('ab', 0, 1., {'': {'a': 90, 'b': 10}})
    keys = [('x', 'y'), ('y', 'x')]
    result = bounded_mixture(source, keys, ['x', 'y'], .2, k=2)
    assert result['plaintexts'] in [('a', 'b'), ('b', 'a')]
    assert compatible_mask('ab', keys, ['x', 'y'], ['a', 'a']) == 0


@pytest.mark.parametrize('limit', ['max_nodes', 'max_edges', 'max_expanded'])
def test_each_resource_cap_fails_without_pruning(limit):
    with pytest.raises(RuntimeError, match='cap'):
        record_kbest(model(), ('x', 'xx'), 'xxxx', .2, 8, **{limit: 1})


def test_zero_source_probabilities_ties_and_zero_key_weights():
    source = SuffixSource('ab', 0, 1., {'': {'a': 1, 'b': 1}})
    # The finite-state interface also permits exact zero probabilities.
    source.probabilities = source.probabilities.copy()
    source.probabilities[0] = [1., 0.]
    rows, evidence, *_ = record_kbest(source, ('x', 'x'), 'xx', .2, 8)
    assert [s for s, _ in rows] == ['aa']
    assert evidence == pytest.approx(math.log(.2 * .8**2))
    result = bounded_mixture(source, [('x', 'x'), ('y', 'y')], ['y'], .2, log_weights=[0., -math.inf])
    assert result['plaintexts'] is None


def test_tiny_mass_and_huge_common_weight_offsets():
    source, keys = model(), [('x', 'x'), ('y', 'y')]
    tiny = bounded_mixture(source, keys, ['y'], .2, log_weights=[0., -10000.])
    assert math.isfinite(tiny['joint_log_probability'])
    first = bounded_mixture(source, keys, ['x'], .2)
    second = bounded_mixture(source, keys, ['x'], .2, log_weights=[1e200, 1e200])
    assert first == second
    with pytest.raises(ArithmeticError):
        bounded_mixture(source, keys, ['x'], .2, log_weights=[1e308, -1e308])


@pytest.mark.parametrize('options', [{'count': 0}, {'max_nodes': True}, {'rho': True}, {'units': ('', 'x')}])
def test_invalid_lattices_fail(options):
    args = {'units': ('x', 'xx'), 'observed': 'xx', 'rho': .2, 'count': 4}
    args.update(options)
    with pytest.raises(ValueError):
        record_kbest(model(), **args)
