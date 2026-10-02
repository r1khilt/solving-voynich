"""Same-observation posterior conversion checked against literal rational sums."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from voynich.posterior_key_reading import posterior_reading_weights
from voynich.shared_key_mixture import decode_shared_keys


class Source:
    alphabet = ('a','b')
    probabilities = np.array([[2/3,1/3]])

    def state(self,_):
        return 0

    def step(self,state,_):
        return state


def enumerate_law(key,records):
    compatible = []
    for record in records:
        candidates = {}
        for length in range(len(record)+1):
            for letters in itertools.product('ab',repeat=length):
                text = ''.join(letters)
                if ''.join(key['ab'.index(c)] for c in text)==record:
                    probability = F(1,3)*F(2,3)**length
                    for letter in text:
                        probability *= F(2,3) if letter=='a' else F(1,3)
                    candidates[text] = probability
        compatible.append(candidates)
    result = {}
    for terms in itertools.product(*(list(c.items()) for c in compatible)):
        texts, probabilities = zip(*terms,strict=True)
        result[texts] = math.prod(probabilities)
    return result


def test_same_record_decode_matches_exact_empirical_mixture_and_not_squared_likelihood():
    keys = (('0','1'),('1','0'),('0','0'))
    counts = (1,2,3)
    records = ('0','0')
    laws = [enumerate_law(k,records) for k in keys]
    likelihoods = [sum(p.values()) for p in laws]
    assert likelihoods==[F(16,729),F(4,729),F(4,81)]
    particles = tuple(k for k,n in zip(keys,counts,strict=True) for _ in range(n))
    scores = [math.log(float(likelihoods[keys.index(k)])) for k in particles]
    weights = posterior_reading_weights(particles,scores)
    expected = {}
    for law,likelihood,count in zip(laws,likelihoods,counts,strict=True):
        for text,p in law.items():
            expected[text] = expected.get(text,F(0))+F(count,6)*p/likelihood
    assert expected=={('a','a'):F(7,18),('b','b'):F(7,18),('a','b'):F(1,9),('b','a'):F(1,9)}
    assert weights.keys==keys and weights.multiplicities==counts
    assert np.exp(weights.decoder_log_weights)==pytest.approx([3/31,24/31,4/31])
    found = decode_shared_keys(Source(),weights.keys,records,1/3,log_weights=weights.decoder_log_weights)
    assert found.plaintexts in (('a','a'),('b','b'))
    assert found.log_likelihood==pytest.approx(weights.expected_decoder_log_evidence,abs=1e-14)
    assert math.exp(found.joint_log_probability-found.log_likelihood)==pytest.approx(7/18)
    wrong = decode_shared_keys(Source(),keys,records,1/3,log_weights=[math.log(n/6) for n in counts])
    assert wrong.plaintexts==('a','a')
    assert math.exp(wrong.joint_log_probability-wrong.log_likelihood)==pytest.approx(F(16,33))
    assert F(16,33)!=expected[wrong.plaintexts]


@pytest.mark.parametrize('records',[('',),('0',),('00',),('0','1'),('01','10')])
def test_exhaustive_six_unit_dictionaries_with_preserved_duplicates(records):
    keys = tuple(itertools.product(('0','1','00','01','10','11'),repeat=2))
    laws = {k:enumerate_law(k,records) for k in keys}
    laws = {k:v for k,v in laws.items() if v}
    # Deterministic multiplicities define an arbitrary empirical posterior;
    # they are neither new source data nor evidence of posterior calibration.
    counts = {k:1+i%3 for i,k in enumerate(laws)}
    particles = tuple(k for k,n in counts.items() for _ in range(n))
    likelihoods = {k:sum(v.values()) for k,v in laws.items()}
    weights = posterior_reading_weights(particles,[math.log(float(likelihoods[k])) for k in particles])
    expected = {}
    for key,law in laws.items():
        for text,p in law.items():
            expected[text] = expected.get(text,F(0))+F(counts[key],len(particles))*p/likelihoods[key]
    assert sum(expected.values())==1
    found = decode_shared_keys(Source(),weights.keys,records,1/3,log_weights=weights.decoder_log_weights)
    assert expected[found.plaintexts]==max(expected.values())
    assert math.exp(found.joint_log_probability-found.log_likelihood)==pytest.approx(float(max(expected.values())),abs=1e-13)
    assert found.log_likelihood==pytest.approx(weights.expected_decoder_log_evidence,abs=1e-13)
    posterior = np.exp(np.array(weights.decoder_log_weights)+np.array([math.log(float(likelihoods[k])) for k in weights.keys])-found.log_likelihood)
    assert posterior==pytest.approx([counts[k]/len(particles) for k in weights.keys],abs=1e-13)


def test_single_key_reduction_common_score_shift_and_order():
    keys = (('0','1'),('0','1'),('1','0'))
    original = posterior_reading_weights(keys,[-10.,-10.,-11.])
    shifted = posterior_reading_weights(keys,[-1010.,-1010.,-1011.])
    assert shifted.decoder_log_weights==pytest.approx(original.decoder_log_weights,abs=1e-12)
    assert shifted.expected_decoder_log_evidence==pytest.approx(original.expected_decoder_log_evidence-1000,abs=1e-12)
    single = posterior_reading_weights([keys[0]]*7,[-20.]*7)
    assert single.decoder_log_weights==(0.,) and single.expected_decoder_log_evidence==-20.
    reverse = posterior_reading_weights(list(reversed(keys)),[-11.,-10.,-10.])
    assert dict(zip(reverse.keys,reverse.decoder_log_weights,strict=True))==pytest.approx(dict(zip(original.keys,original.decoder_log_weights,strict=True)))


@pytest.mark.parametrize('keys,scores',[
    ([],[]),([('0',)],[]),([('0',)]*129,[0.]*129),
    (['01'],[0.]),([('',)],[0.]),([('0',),('0','1')],[0.,0.]),
    ([('0',)],[1.]),([('0',)],[float('-inf')]),([('0',)],[float('nan')]),([('0',)],[True]),
])
def test_rejects_incomplete_or_unbounded_population(keys,scores):
    with pytest.raises(ValueError):
        posterior_reading_weights(keys,scores)


def test_duplicate_disagreement_fails_and_large_common_scores_preserve_multiplicity():
    with pytest.raises(ArithmeticError,match='Duplicate'):
        posterior_reading_weights([('0',)]*2,[-1.,-1.0000001])
    equal = posterior_reading_weights([('0',),('1',),('1',)],[-1.7e308]*3)
    assert np.exp(equal.decoder_log_weights)==pytest.approx([1/3,2/3])
    assert equal.expected_decoder_log_evidence==-1.7e308
