"""Exhaustive permutation ceilings and exact known-reading bank inequalities."""

import itertools
import math
from fractions import Fraction as F

import pytest

from scripts.check_posterior_key_reading001 import KEYS,RECORDS,Source,literal_record_law
from voynich.key_bank_diagnostics import full_key_bank_bounds,inventory_ceiling


def test_all_three_row_inventory_ceilings_equal_exhaustive_best_permutation():
    keys = tuple(itertools.product(range(3),repeat=3))
    for known,key,mask in itertools.product(keys,keys,range(8)):
        used = tuple(i for i in range(3) if mask>>i&1)
        result = inventory_ceiling(key,known,used,pool_size=3)
        best = max(sum(k[i]==known[i] for i in used) for k in itertools.permutations(key))
        assert result['permutation_used_match_ceiling']==best
        assert result['minimum_inventory_replacements']==len(used)-best
        assert result['current_used_matches']<=best


@pytest.mark.parametrize('contextual',[False,True])
def test_all_tiny_known_readings_bank_bounds_against_complete_rational_posterior(contextual):
    source = Source(contextual)
    cache = {(k,c):literal_record_law(source,k,c) for k in KEYS for c in RECORDS}
    for records in itertools.product(RECORDS,repeat=2):
        laws = {}
        for key in KEYS:
            law = {texts:math.prod(ps) for terms in itertools.product(*(list(cache[key,c].items()) for c in records))
                   for texts,ps in [zip(*terms,strict=True)]}
            if law:
                laws[key] = law
        values = {k:sum(v.values(),F(0)) for k,v in laws.items()}
        z = sum(values.values(),F(0))
        events = {x:(k,q) for k,law in laws.items() for x,q in law.items()}
        codes = {k:tuple(('0','1','00','01','10','11').index(u) for u in k) for k in laws}
        candidates = list(values)
        for texts,(known,q) in events.items():
            used = tuple(sorted({source.alphabet.index(a) for text in texts for a in text}))
            for bank in (candidates[:1],candidates[-1:],candidates[::2]):
                mass = sum((values[k] for k in bank),F(0))
                lower = max(q*6**(2-len(used)),values[known])
                assert lower<=z
                expected = min(F(1),mass/lower)
                result = full_key_bank_bounds([codes[k] for k in bank],
                    [math.log(float(values[k])) for k in bank],math.log(float(q)),used,
                    pool_size=6,known_key_log_likelihood=math.log(float(values[known])))
                assert result['full_key_bank_log_posterior_mass_upper']==pytest.approx(math.log(float(expected)),abs=1e-12)
                assert mass/z<=expected
                mu = {k:F(1,len(bank)) for k in bank}
                tv = sum((abs(mu.get(k,F(0))-v/z) for k,v in values.items()),F(0))/2
                assert tv>=1-expected
                kl = sum(float(p)*math.log(float(p/(values[k]/z))) for k,p in mu.items())
                assert kl+1e-12>=result['bank_supported_law_forward_kl_lower_nats']


def test_duplicate_mass_not_added_and_underflow_retains_log_bound():
    a = full_key_bank_bounds([(0,1),(1,0)],[-1000.,-1001.],-10.,(0,),pool_size=2)
    b = full_key_bank_bounds([(0,1),(0,1),(1,0)],[-1000.,-1000.,-1001.],-10.,(0,),pool_size=2)
    assert a['full_key_bank_log_posterior_mass_upper']==b['full_key_bank_log_posterior_mass_upper']
    assert b['distinct_full_keys']==2 and b['particles']==3
    assert b['full_key_bank_log_posterior_mass_upper']<-990
    assert b['bank_supported_law_tv_lower_float']==1. # Floating rounding; log remains authoritative.
    with pytest.raises(ArithmeticError,match='Duplicate'):
        full_key_bank_bounds([(0,1)]*2,[-10.,-11.],-10.,(0,),pool_size=2)


@pytest.mark.parametrize('key,known,used,pool',[
    ([],[],(),2),((0,),(-1,),(0,),2),((0,),(0,),(0,0),2),
    ((0,),(0,),(True,),2),((0,),(0,),(1,),2),((0,),(0,),(),True),
])
def test_invalid_or_unbounded_inventory_inputs(key,known,used,pool):
    with pytest.raises(ValueError):
        inventory_ceiling(key,known,used,pool_size=pool)


@pytest.mark.parametrize('keys,scores,q',[
    ([],[],-1.),([(0,)]*129,[-1.]*129,-1.),([(0,)],[],-1.),
    ([(0,)],[float('nan')],-1.),([(0,)],[float('-inf')],-1.),
    ([(0,)],[1.],-1.),([(0,)],[True],-1.),([(0,)],[-1.],0.1),
])
def test_invalid_bank_or_incomplete_scores_refused(keys,scores,q):
    with pytest.raises(ValueError):
        full_key_bank_bounds(keys,scores,q,(),pool_size=2)


def test_known_marginal_cannot_be_smaller_than_its_compatible_path():
    with pytest.raises(ArithmeticError,match='exceeds'):
        full_key_bank_bounds([(0,)],[-2.],-1.,(0,),pool_size=2,known_key_log_likelihood=-2.)
