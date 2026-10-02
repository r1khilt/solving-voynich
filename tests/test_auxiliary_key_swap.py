import math
from collections import Counter

import pytest

from voynich.auxiliary_key_swap import auxiliary_swap_log_acceptance,swap_joint_state


def encode(key,texts,alphabet):
    return tuple(''.join(key[alphabet.index(c)] for c in t) for t in texts)


def test_joint_involution_preserves_all_record_ciphertexts_and_inventory():
    alphabet,key,texts = ('a','b','c'),('01','0','01'),('abcac','bc','')
    for a,b in ((0,1),(0,2),(1,2)):
        new_key,new_texts = swap_joint_state(key,texts,alphabet,a,b)
        assert Counter(new_key)==Counter(key)
        assert encode(new_key,new_texts,alphabet)==encode(key,texts,alphabet)
        assert tuple(map(len,new_texts))==tuple(map(len,texts))
        assert swap_joint_state(new_key,new_texts,alphabet,a,b)==(key,texts)
        assert key==('01','0','01') and texts==('abcac','bc','')


def test_point_score_acceptance_and_intermediate_temperature_refusal():
    value = auxiliary_swap_log_acceptance(math.log(.2),math.log(.1),beta=1,uniform_prior=True)
    assert math.exp(value)==pytest.approx(.5)
    assert auxiliary_swap_log_acceptance(-3.,-2.,beta=1.,uniform_prior=True)==0.
    for beta,prior in ((0,True),(.5,True),(True,True),(1,False),(1,1)):
        with pytest.raises(ValueError):
            auxiliary_swap_log_acceptance(-3.,-2.,beta=beta,uniform_prior=prior)
    for old,new in ((float('-inf'),-1.),(-1.,float('nan')),(-1.,1.),(True,-1.)):
        with pytest.raises(ValueError):
            auxiliary_swap_log_acceptance(old,new,beta=1,uniform_prior=True)


@pytest.mark.parametrize('key,texts,alphabet,a,b',[
    (('0','1'),('a',),('a','b'),0,0),
    (('0','1'),('c',),('a','b'),0,1),
    (('0',),('a',),('a','b'),0,1),
    (('','1'),('a',),('a','b'),0,1),
    (('0','1'),('a',),('a','a'),0,1),
    (('0','1'),('a',),('a','b'),True,1),
    (('0','1'),'a',('a','b'),0,1),
])
def test_invalid_joint_states_are_refused(key,texts,alphabet,a,b):
    with pytest.raises(ValueError):
        swap_joint_state(key,texts,alphabet,a,b)
