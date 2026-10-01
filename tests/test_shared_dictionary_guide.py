"""Independent full source-string references and shared-parameter controls."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from tests.test_source_bellman_guide import full_future
from tests.test_source_state_lattice import source
from voynich.shared_dictionary_guide import SharedDictionaryIid
from voynich.source_state_lattice import State


def all_keys():
    return np.array(list(itertools.product(range(6),repeat=2)),dtype=np.int32)


def test_all36_pairs_and49_partial_bindings_equal_independent_iid_future():
    p,goto = source(iid=True)
    observed = [(0,),(1,)]+list(itertools.product(range(2),repeat=2))
    for cipher in itertools.product(observed,repeat=2):
        guide = SharedDictionaryIid(cipher,p[0],glyphs=2,rho=.25,epsilon=0,max_tables=100000)
        for key in itertools.product(range(-1,6),repeat=2):
            state = State((0,0),(0,0),key)
            expected = float(full_future(state,cipher,p,goto))
            result = guide.estimate(key,(0,0),all_keys())
            actual = 0 if result["log_mean"] is None else math.exp(result["log_mean"])
            assert actual==pytest.approx(expected,abs=2e-15)


def test_partial_offsets_and_paid_eos_completion():
    p,goto = source(iid=True)
    cipher = ((0,1,0),(0,0))
    guide = SharedDictionaryIid(cipher,p[0],glyphs=2,rho=.25,epsilon=0)
    for offsets in itertools.product(range(4),range(3)):
        state = State(offsets,(0,0),(0,-1))
        expected = float(full_future(state,cipher,p,goto))
        result = guide.estimate(state.key,offsets,all_keys())
        actual = 0 if result["log_mean"] is None else math.exp(result["log_mean"])
        assert actual==pytest.approx(expected,abs=2e-15)
    assert guide.estimate((-1,-1),(3,2),all_keys())["log_mean"]==0


def test_one_row_conditional_integration_is_exact_and_reduces_variance_at_fixed_base_count():
    guide = SharedDictionaryIid(((0,0),(0,1)),[1/3,2/3],glyphs=2,rho=.25,epsilon=0)
    raw = guide.estimate((-1,-1),(0,0),all_keys())
    rb = guide.estimate((-1,-1),(0,0),all_keys(),integrate_row=1)
    assert rb["log_mean"]==pytest.approx(raw["log_mean"],abs=1e-14)
    raw_values = np.array([0 if v is None else math.exp(v) for v in raw["base_log_likelihoods"]])
    rb_values = np.array([0 if v is None else math.exp(v) for v in rb["base_log_likelihoods"]])
    assert rb_values.var()<=raw_values.var()+1e-30
    for code in range(6):
        bank = np.array([[code,5]],dtype=np.int32)
        r = guide.estimate((-1,-1),(0,0),bank,integrate_row=1)
        brute = guide.estimate((code,-1),(0,0),all_keys())
        assert r["log_mean"]==pytest.approx(brute["log_mean"],abs=1e-14)


def test_cross_record_correlation_matches_rational_witness_not_product_of_marginals():
    bank = all_keys()
    joint = SharedDictionaryIid(((0,),(0,)),[1/3,2/3],glyphs=2,rho=.25,epsilon=0)
    result = math.exp(joint.estimate((-1,-1),(0,0),bank)["log_mean"])
    single = SharedDictionaryIid(((0,),),[1/3,2/3],glyphs=2,rho=.25,epsilon=0)
    marginal = math.exp(single.estimate((-1,-1),(0,),bank)["log_mean"])
    assert result==pytest.approx(float(F(17,4608)),abs=1e-16)
    assert result>marginal**2


def test_sampling_deterministic_and_known_binding_overrides_complete_bank():
    bank = np.random.default_rng(81301).integers(6,size=(16,2),dtype=np.int32)
    guide = SharedDictionaryIid(((0,1),(1,0)),[1/3,2/3],glyphs=2,rho=.25)
    assert guide.estimate((0,1),(0,0),bank)==guide.estimate((0,1),(0,0),bank[::-1])
    assert np.array_equal(bank,np.random.default_rng(81301).integers(6,size=(16,2),dtype=np.int32))


def test_zero_bank_likelihood_does_not_invent_floor_support_and_hard_work_guard():
    guide = SharedDictionaryIid(((0,1,0),),[1.],glyphs=2,rho=.25,epsilon=0,max_tables=2)
    r = guide.estimate((0,),(0,),np.array([[0]],dtype=np.int32))
    assert r["log_mean"] is None and r["likelihood_contribution_ess"]==0
    with pytest.raises(RuntimeError):
        guide.estimate((-1,),(0,),np.array([[0]],dtype=np.int32),integrate_row=0)
    assert guide.tables==1


@pytest.mark.parametrize("invalid",[(-2,-1),(6,-1),(True,False)])
def test_invalid_keys(invalid):
    guide = SharedDictionaryIid(((0,),),[1/3,2/3],glyphs=2,rho=.25)
    with pytest.raises(ValueError):
        guide.estimate(invalid,(0,),all_keys())


def test_invalid_bank_offsets_integrated_row_and_memory_guard():
    guide = SharedDictionaryIid(((0,),),[1/3,2/3],glyphs=2,rho=.25)
    for bank,offsets,row in ((np.array([[6,0]]),(0,),None),(all_keys(),(2,),None),(all_keys(),(0,),True)):
        with pytest.raises(ValueError):
            guide.estimate((-1,-1),offsets,bank,integrate_row=row)
    with pytest.raises(ValueError):
        guide.estimate((0,-1),(0,),all_keys(),integrate_row=0)
    small = SharedDictionaryIid(((0,),),[1/3,2/3],glyphs=2,max_work_bytes=1)
    with pytest.raises(MemoryError):
        small.estimate((-1,-1),(0,),all_keys())
