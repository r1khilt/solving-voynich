"""Finite posterior-path and coupled-key-swap law qualification, no corpus."""

import argparse
import itertools
import math
import signal
import time
from fractions import Fraction as F

from scripts.run_source_state_systems001 import ROOT,artifact,limit_resources,resource_report,save_new
from tests.test_suffix_posterior_paths import Source,graph_law,law
from voynich.auxiliary_key_swap import auxiliary_swap_log_acceptance,swap_joint_state
from voynich.suffix_posterior_paths import SuffixPosteriorPaths

EXP = 'AUXILIARY-KEY-SWAP-THEORY-001'
UNITS = ('0','1','00','01','10','11')
KEYS = tuple(itertools.product(UNITS,repeat=2))
RECORDS = tuple(''.join(c) for n in range(4) for c in itertools.product('01',repeat=n))
PATHS = ('src/voynich/suffix_posterior_paths.py','tests/test_suffix_posterior_paths.py',
    'src/voynich/auxiliary_key_swap.py','tests/test_auxiliary_key_swap.py',
    'src/voynich/tempered_inventory.py','scripts/check_auxiliary_key_swap001.py',
    'docs/research/auxiliary-key-swap-2026-10-01.md')


def encode(key,texts):
    return tuple(''.join(key['ab'.index(c)] for c in text) for text in texts)


def check():
    wall,cpu = time.monotonic(),time.process_time()
    limit_resources(120,100)
    lattice_cases,joint_cases,path_draws,balance,strict_dominance = 0,0,0,0,0
    wrong_intermediate = {0:0,F(1,2):0}
    weighted_prior_failures,path_temper_zero_failures = 0,0
    max_probability_delta,max_log_delta,max_accept_delta = 0.,0.,0.
    for contextual in (False,True):
        source = Source(contextual)
        cached = {}
        for key,record in itertools.product(KEYS,RECORDS):
            expected = law(source,key,record)
            cached[key,record] = expected
            lattice = SuffixPosteriorPaths(source,key,record,1/3)
            assert lattice.probabilities is source.probabilities and lattice.transitions is source.transitions
            if not expected:
                assert lattice.log_likelihood==-math.inf and lattice.sample_paths(draws=7,seed=37)==()
            else:
                total = sum(expected.values(),F(0))
                actual = graph_law(lattice)
                assert set(actual)==set(expected)
                delta = max(abs(actual[k]-float(p/total)) for k,p in expected.items())
                assert delta<=1e-12
                max_probability_delta = max(max_probability_delta,delta)
                log_delta = abs(lattice.log_likelihood-math.log(float(total)))
                assert log_delta<=1e-12
                max_log_delta = max(max_log_delta,log_delta)
                sampled = lattice.sample_paths(draws=7,seed=37)
                assert sampled==lattice.sample_paths(draws=7,seed=37)
                for path in sampled:
                    assert encode(key,(path.text,))==(record,)
                    assert abs(path.joint_log_probability-math.log(float(expected[path.text])))<=1e-12
                    assert abs(path.conditional_log_probability-math.log(float(expected[path.text]/total)))<=1e-12
                path_draws += len(sampled)
            lattice_cases += 1
        for records in itertools.product(RECORDS,repeat=2):
            laws = {}
            for key in KEYS:
                terms = itertools.product(*(list(cached[key,c].items()) for c in records))
                full = {texts:math.prod(probabilities) for row in terms for texts,probabilities in [zip(*row,strict=True)]}
                if full:
                    laws[key] = full
            likelihoods = {k:sum(v.values(),F(0)) for k,v in laws.items()}
            averaged = {}
            for key,full in laws.items():
                changed_key = tuple(reversed(key))
                assert changed_key in laws
                total = F(0)
                for texts,q in full.items():
                    candidate,new_texts = swap_joint_state(key,texts,source.alphabet,0,1)
                    assert candidate==changed_key and encode(candidate,new_texts)==records
                    assert swap_joint_state(candidate,new_texts,source.alphabet,0,1)==(key,texts)
                    new_q = laws[candidate][new_texts]
                    acceptance = min(F(1),new_q/q)
                    reverse = min(F(1),q/new_q)
                    assert q*acceptance==new_q*reverse
                    actual = math.exp(auxiliary_swap_log_acceptance(math.log(float(q)),math.log(float(new_q)),beta=1,uniform_prior=True))
                    delta = abs(actual-float(acceptance))
                    assert delta<=1e-12
                    max_accept_delta = max(max_accept_delta,delta)
                    total += q/likelihoods[key]*acceptance
                    # Abstract position-dependent prior, NOT implemented.
                    prior = F((1+UNITS.index(key[0]))*(6-UNITS.index(key[1])))
                    new_prior = F((1+UNITS.index(candidate[0]))*(6-UNITS.index(candidate[1])))
                    weighted_prior_failures += int(prior*q*acceptance!=new_prior*new_q*reverse)
                    balance += 1
                    if records[0]==records[1]:
                        g = sum(cached[key,records[0]].values(),F(0))
                        new_g = sum(cached[candidate,records[0]].values(),F(0))
                        for beta in wrong_intermediate:
                            factor = likelihoods[key]/likelihoods[candidate] if beta==0 else g/new_g
                            target = q/likelihoods[key] if beta==0 else q/g
                            new_target = new_q/likelihoods[candidate] if beta==0 else new_q/new_g
                            correct = min(F(1),new_q/q*factor)
                            back = min(F(1),q/new_q/factor)
                            assert target*correct==new_target*back
                            wrong_intermediate[beta] += int(target*acceptance!=new_target*reverse)
                averaged[key] = total
                collapsed = min(F(1),likelihoods[changed_key]/likelihoods[key])
                assert total<=collapsed
                strict_dominance += int(total<collapsed)
            for key,acceptance in averaged.items():
                candidate = tuple(reversed(key))
                assert likelihoods[key]*acceptance==likelihoods[candidate]*averaged[candidate]
                assert likelihoods[candidate]*averaged[candidate]+likelihoods[key]*(1-acceptance)==likelihoods[key]
            if len({len(full) for full in laws.values()})>1:
                path_temper_zero_failures += 1
            joint_cases += 1
            if joint_cases%75==0:
                print(f'Finite auxiliary law {joint_cases}/450',flush=True)
            if resource_report(wall,cpu)['peak_rss_bytes']>512*1024**2:
                raise MemoryError('Finite auxiliary512MiB cap')
    assert lattice_cases==1080 and joint_cases==450 and all(wrong_intermediate.values())
    assert weighted_prior_failures and path_temper_zero_failures and strict_dominance
    return {'experiment':EXP,'status':'PASS_finite_posterior_paths_and_terminal_auxiliary_swap',
        'inputs':[artifact(ROOT/p) for p in PATHS],'lattice_cases':lattice_cases,'joint_record_cases':joint_cases,
        'checked_seeded_path_draws':path_draws,'rational_joint_balance_checks':balance,
        'rational_gibbs_refreshed_key_balance_and_stationarity':True,
        'collapsed_swap_acceptance_dominates_auxiliary_in_all_cases':True,'strict_acceptance_dominance_key_cases':strict_dominance,
        'wrong_point_only_beta0_witnesses':wrong_intermediate[0],'wrong_point_only_beta_half_witnesses':wrong_intermediate[F(1,2)],
        'omitted_nonexchangeable_prior_ratio_witnesses':weighted_prior_failures,
        'path_tempered_beta0_wrong_marginal_panels':path_temper_zero_failures,
        'maximum_float_path_probability_delta':max_probability_delta,'maximum_float_log_likelihood_delta':max_log_delta,
        'maximum_float_auxiliary_acceptance_delta':max_accept_delta,'resources':resource_report(wall,cpu),
        'same_inventory_only':True,'no_original_source_or_empirical_data_access':True,'no_speed_mixing_or_recovery_claim':True,
        'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    try:
        value = check()
        if args.save:
            print(save_new(ROOT/'results'/EXP/'result.json',value))
    finally:
        signal.alarm(0)
