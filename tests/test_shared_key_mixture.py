import itertools
import math
import random

import pytest

from voynich.shared_key_mixture import decode_shared_keys
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def source_model(alphabet='ab', order=2):
    text = 'a' * 45 + 'b' * 41 + ('c' * 11 if alphabet == 'abc' else 'ababa')
    return SuffixSource(alphabet, order, 4., collect_counts([text], alphabet, order))


def brute(source, keys, observed, rho, weights):
    # Enumerate literal source strings, not production compatibility states.
    possibilities = {}
    for i, key in enumerate(keys):
        if not weights[i]:
            continue
        per_record = []
        for record in observed:
            texts = []
            for n in range(len(record) + 1):
                for letters in itertools.product(source.alphabet, repeat=n):
                    if ''.join(key[source.alphabet.index(c)] for c in letters) == record:
                        texts.append(''.join(letters))
            per_record.append(texts)
        for texts in itertools.product(*per_record):
            possibilities[texts] = possibilities.get(texts, 0.) + weights[i]
    scored = {}
    for texts, mass in possibilities.items():
        probability = mass
        for text in texts:
            state = source.state('')
            probability *= rho * (1 - rho)**len(text)
            for char in text:
                letter = source.alphabet.index(char)
                probability *= source.probabilities[state, letter]
                state = source.step(state, letter)
        scored[texts] = probability
    return scored


@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_exact_shared_bank_matches_full_enumeration_and_individual_evidence(order):
    source = source_model(order=order)
    rng = random.Random(612 + order)
    pool = ['x', 'y', 'xx', 'xy', 'yx', 'yy']
    possible_keys = list(itertools.product(pool, repeat=2))
    for _ in range(12):
        keys = rng.sample(possible_keys, 3)
        records = [''.join(rng.choice('xy') for _ in range(rng.randrange(4))) for _ in range(2)]
        masses = [rng.randrange(1, 10) for _ in keys]
        weights = [m / sum(masses) for m in masses]
        result = decode_shared_keys(source, keys, records, .2, log_weights=[math.log(m) for m in masses])
        scores = brute(source, keys, records, .2, weights)
        independent_evidence = sum(w * math.prod(math.exp(decode(source, key, record, .2).log_likelihood)
                                                for record in records) for w, key in zip(weights, keys))
        if not scores:
            assert result.plaintexts is None and result.log_likelihood is None
            assert independent_evidence == 0
        else:
            assert result.log_likelihood == pytest.approx(math.log(sum(scores.values())), abs=1e-12)
            assert result.log_likelihood == pytest.approx(math.log(independent_evidence), abs=1e-12)
            assert result.joint_log_probability == pytest.approx(math.log(max(scores.values())), abs=1e-12)
            assert result.joint_log_probability == pytest.approx(math.log(scores[result.plaintexts]), abs=1e-12)
            expected = tuple(i for i, key in enumerate(keys)
                             if all(''.join(key[source.alphabet.index(c)] for c in text) == record
                                    for text, record in zip(result.plaintexts, records)))
            assert result.compatible_key_indices == expected


def test_summing_keys_finds_text_that_is_not_best_under_any_single_key():
    source = source_model('abc', order=0)
    keys = [('x', 'y', 'xx'), ('y', 'x', 'xx')]
    single = [decode(source, key, 'xx', .2).plaintext for key in keys]
    assert single == ['aa', 'bb']
    mixture = decode_shared_keys(source, keys, ['xx'], .2)
    assert mixture.plaintexts == ('c',)
    assert mixture.compatible_key_indices == (0, 1)


def test_a_shared_key_cannot_switch_between_records():
    source = SuffixSource('ab', 0, 1., {'': {'a': 90, 'b': 10}})
    keys = [('x', 'y'), ('y', 'x')]
    individual = [decode_shared_keys(source, keys, [record], .3).plaintexts[0] for record in ['x', 'y']]
    assert individual == ['a', 'a']
    joint = decode_shared_keys(source, keys, ['x', 'y'], .3)
    assert joint.plaintexts in [('a', 'b'), ('b', 'a')]
    assert len(joint.compatible_key_indices) == 1
    assert joint.log_likelihood < sum(decode_shared_keys(source, keys, [r], .3).log_likelihood for r in ['x', 'y'])


@pytest.mark.parametrize('records', [('',), ('', ''), ('xxx',), ('xxx', '', 'xx')])
def test_one_key_bank_reduces_to_original_decoder_with_resets(records):
    source = source_model()
    key = ('x', 'xx')
    mixture = decode_shared_keys(source, [key], records, .25)
    rows = [decode(source, key, record, .25) for record in records]
    assert mixture.plaintexts == tuple(r.plaintext for r in rows)
    assert mixture.log_likelihood == pytest.approx(sum(r.log_likelihood for r in rows), abs=1e-12)
    assert mixture.joint_log_probability == pytest.approx(sum(r.joint_log_probability for r in rows), abs=1e-12)


def test_log_weights_zero_mass_and_extreme_small_supported_key_are_preserved():
    source = source_model(order=0)
    keys = [('x', 'x'), ('y', 'y')]
    missing = decode_shared_keys(source, keys, ['y'], .2, log_weights=[0., -math.inf])
    assert missing.plaintexts is None
    tiny = decode_shared_keys(source, keys, ['y'], .2, log_weights=[0., -10000.])
    expected = decode(source, keys[1], 'y', .2)
    assert tiny.compatible_key_indices == (1,)
    assert tiny.log_likelihood == pytest.approx(expected.log_likelihood - 10000, abs=1e-9)


def test_node_cap_fails_without_returning_a_silent_approximation():
    with pytest.raises(RuntimeError, match='state cap'):
        decode_shared_keys(source_model(), [('x', 'xx'), ('xx', 'x')], ['xxxxx'], .2, max_nodes=1)


def test_large_common_log_weight_offset_cannot_double_the_probability_mass():
    source = source_model()
    keys = [('x', 'xx'), ('xx', 'x')]
    first = decode_shared_keys(source, keys, ['xxxx'], .2)
    second = decode_shared_keys(source, keys, ['xxxx'], .2, log_weights=[1e200, 1e200])
    assert second == first
    with pytest.raises(ArithmeticError, match='range'):
        decode_shared_keys(source, keys, ['xxxx'], .2, log_weights=[1e308, -1e308])


def test_many_keys_with_common_behavior_share_one_offset_group():
    source = source_model('abc', order=2)
    keys = [('x', 'y', str(i) + 'z') for i in range(128)]
    many = decode_shared_keys(source, keys, ['xyxyx', 'yxy'], .2)
    single = decode_shared_keys(source, [keys[0]], ['xyxyx', 'yxy'], .2)
    assert many.plaintexts == single.plaintexts
    assert many.joint_log_probability == pytest.approx(single.joint_log_probability, abs=1e-12)
    assert many.log_likelihood == pytest.approx(single.log_likelihood, abs=1e-12)
    assert many.reachable_nodes == single.reachable_nodes
    assert many.maximum_offset_groups == 1 and len(many.compatible_key_indices) == 128


@pytest.mark.parametrize('options', [
    {'keys': [('x', 'y'), ('x', 'y')]}, {'keys': [('x', '')]}, {'keys': ['xy']},
    {'log_weights': [math.nan]}, {'log_weights': [math.inf]}, {'log_weights': [-math.inf]},
    {'log_weights': [True]}, {'log_weights': []}, {'records': 'xxx'}, {'rho': True}, {'max_nodes': True},
])
def test_invalid_banks_weights_and_inputs_are_rejected(options):
    params = {'keys': [('x', 'y')], 'records': ['x'], 'rho': .2}
    params.update(options)
    with pytest.raises(ValueError):
        decode_shared_keys(source_model(), **params)
