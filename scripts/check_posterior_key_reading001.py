"""Bounded rational law and floating same-observation reading interface checks."""

import argparse
import itertools
import math
import signal
import time
from fractions import Fraction as F

import numpy as np

from scripts.run_source_state_systems001 import ROOT,artifact,limit_resources,resource_report,save_new
from voynich.posterior_key_reading import posterior_reading_weights
from voynich.shared_key_mixture import decode_shared_keys

EXP = 'POSTERIOR-KEY-READING-THEORY-001'
PATHS = ('src/voynich/posterior_key_reading.py','tests/test_posterior_key_reading.py',
    'src/voynich/shared_key_mixture.py','scripts/check_posterior_key_reading001.py',
    'docs/research/posterior-key-reading-2026-10-01.md')
UNITS = ('0','1','00','01','10','11')
KEYS = tuple(itertools.product(UNITS,repeat=2))
RECORDS = tuple(''.join(c) for n in range(4) for c in itertools.product('01',repeat=n))


class Source:
    alphabet = ('a','b')

    def __init__(self,contextual):
        self.rows = ((F(2,3),F(1,3)),(F(1,4),F(3,4)),(F(3,4),F(1,4))) if contextual else ((F(2,3),F(1,3)),)*3
        self.probabilities = np.array([[float(p) for p in row] for row in self.rows])

    def state(self,_):
        return 0

    def step(self,_,letter):
        return 1+letter

    def probability(self,text):
        result,state = F(1,3)*F(2,3)**len(text),0
        for letter in text:
            result *= self.rows[state][self.alphabet.index(letter)]
            state = self.step(state,self.alphabet.index(letter))
        return result


def literal_record_law(source,key,record):
    result = {}
    for n in range(len(record)+1):
        for letters in itertools.product(source.alphabet,repeat=n):
            text = ''.join(letters)
            if ''.join(key[source.alphabet.index(c)] for c in text)==record:
                result[text] = source.probability(text)
    return result


def tv(left,right):
    return sum((abs(left.get(k,F(0))-right.get(k,F(0))) for k in left.keys()|right.keys()),F(0))/2


def check():
    wall,cpu = time.monotonic(),time.process_time()
    limit_resources(120,100)
    cases,complete_keys,text_entries,wrong_cases = 0,0,0,0
    maximum_probability_delta,maximum_evidence_delta = 0.,0.
    maximum_naive_key_tv,maximum_key_tv,maximum_text_tv = F(0),F(0),F(0)
    for contextual in (False,True):
        source = Source(contextual)
        cached = {(k,c):literal_record_law(source,k,c) for k in KEYS for c in RECORDS}
        for records in itertools.product(RECORDS,repeat=2):
            laws = {}
            for key in KEYS:
                law = {texts:math.prod(probabilities) for terms in itertools.product(*(list(cached[key,c].items()) for c in records))
                    for texts,probabilities in [zip(*terms,strict=True)]}
                if law:
                    laws[key] = law
            assert laws
            likelihoods = {k:sum(v.values(),F(0)) for k,v in laws.items()}
            counts = {k:1+(i+cases)%3 for i,k in enumerate(laws)}
            population = tuple(k for k,n in counts.items() for _ in range(n))
            mu = {k:F(n,len(population)) for k,n in counts.items()}
            expected = {}
            true_text = {}
            z = sum(likelihoods.values(),F(0))
            target = {k:v/z for k,v in likelihoods.items()}
            for key,law in laws.items():
                for text,probability in law.items():
                    expected[text] = expected.get(text,F(0))+mu[key]*probability/likelihoods[key]
                    true_text[text] = true_text.get(text,F(0))+probability/z
            assert sum(expected.values(),F(0))==sum(true_text.values(),F(0))==1
            inverse = {k:mu[k]/likelihoods[k] for k in laws}
            inverse_z = sum(inverse.values(),F(0))
            assert sum((v/inverse_z*likelihoods[k] for k,v in inverse.items()),F(0))==1/inverse_z
            assert {k:v/inverse_z*likelihoods[k]*inverse_z for k,v in inverse.items()}==mu
            weights = posterior_reading_weights(population,[math.log(float(likelihoods[k])) for k in population])
            assert weights.keys==tuple(laws) and weights.multiplicities==tuple(counts.values())
            reading = decode_shared_keys(source,weights.keys,records,1/3,log_weights=weights.decoder_log_weights,max_nodes=100_000)
            assert reading.plaintexts in expected and expected[reading.plaintexts]==max(expected.values())
            probability_delta = abs(math.exp(reading.joint_log_probability-reading.log_likelihood)-float(max(expected.values())))
            evidence_delta = abs(reading.log_likelihood-weights.expected_decoder_log_evidence)
            assert probability_delta<=1e-12 and evidence_delta<=1e-12
            # Data processing holds for the actual joint-record conditional law.
            key_tv,text_tv = tv(mu,target),tv(expected,true_text)
            assert text_tv<=key_tv
            naive_z = sum((mu[k]*likelihoods[k] for k in laws),F(0))
            naive = {k:mu[k]*likelihoods[k]/naive_z for k in laws}
            naive_tv = tv(mu,naive)
            wrong_cases += int(naive_tv>0)
            maximum_naive_key_tv = max(maximum_naive_key_tv,naive_tv)
            maximum_key_tv,maximum_text_tv = max(maximum_key_tv,key_tv),max(maximum_text_tv,text_tv)
            maximum_probability_delta = max(maximum_probability_delta,probability_delta)
            maximum_evidence_delta = max(maximum_evidence_delta,evidence_delta)
            cases += 1
            complete_keys += len(laws)
            text_entries += len(expected)
            if cases%75==0:
                print(f'Finite posterior interface {cases}/450',flush=True)
            if resource_report(wall,cpu)['peak_rss_bytes']>512*1024**2:
                raise MemoryError('Finite theory512MiB cap')
    assert cases==450 and wrong_cases>0
    return {'experiment':EXP,'status':'PASS_rational_laws_and_float_reading_interface',
        'inputs':[artifact(ROOT/p) for p in PATHS],'cases':cases,'supported_key_instances':complete_keys,
        'distinct_joint_text_instances':text_entries,'two_context_modes':True,'all_36_keys_enumerated_per_case':True,
        'rational_reweighting_and_probability_conservation':True,'rational_tv_contraction':True,
        'naive_reconditioning_changes_key_law_cases':wrong_cases,'maximum_naive_key_tv':str(maximum_naive_key_tv),
        'maximum_empirical_to_exact_key_tv':str(maximum_key_tv),'maximum_empirical_to_exact_text_tv':str(maximum_text_tv),
        'maximum_float_probability_delta':maximum_probability_delta,'maximum_float_evidence_delta':maximum_evidence_delta,
        'resources':resource_report(wall,cpu),'paid_spend_usd':0,'no_original_source_or_empirical_data_access':True,
        'no_posterior_calibration_or_recovery_claim':True}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    try:
        result = check()
        if args.save:
            print(save_new(ROOT/'results'/EXP/'result.json',result))
    finally:
        signal.alarm(0)
