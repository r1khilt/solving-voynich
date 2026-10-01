"""Independent source-string sums, transition matrices and stochastic laws."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_prefix_bridge001 import CONTEXTS, enumerated
from tests.test_source_state_lattice import source
from voynich.dictionary_smc import FixedKeyBridge, bridge_schedule, propose, run_dictionary_smc


KEYS = np.array(list(itertools.product(range(6),repeat=2)),dtype=np.int32)


def bridge(cipher=((0,1),(1,0)), *, iid=False, **kwargs):
    p,t = source(iid=iid)
    if iid:
        p,t = p[:1].copy(),np.zeros((1,2),dtype=np.uint32)
    return FixedKeyBridge(cipher,p,t,glyphs=2,rho=.25,**kwargs)


def reference(cipher, key, contexts, rows=CONTEXTS, closed=False):
    return math.prod(enumerated(c,key,ctx,rows,closed=closed) for c,ctx in zip(cipher,contexts,strict=True))


@pytest.mark.parametrize("iid",[False,True])
def test_every_short_key_context_prefix_and_closed_observation_matches_rational_strings(iid):
    rows = (CONTEXTS[0],)*3 if iid else CONTEXTS
    for ctx in range(1 if iid else 3):
        for n in range(1,4):
            for cipher in itertools.product(range(2),repeat=n):
                b = bridge((cipher,),iid=iid,contexts=(ctx,))
                for cut in range(n+1):
                    actual = b.log_values(KEYS,(cut,))
                    expected = [float(enumerated(cipher[:cut],key,ctx,rows)) for key in KEYS]
                    np.testing.assert_allclose(np.exp(actual),expected,rtol=0,atol=2e-15)
                actual = b.log_values(KEYS,(n,),closed=True)
                expected = [float(enumerated(cipher,key,ctx,rows,closed=True)) for key in KEYS]
                np.testing.assert_allclose(np.exp(actual),expected,rtol=0,atol=2e-15)


@pytest.mark.parametrize("iid",[False,True])
def test_two_records_same_key_conditional_offsets_contexts_and_paid_eos(iid):
    cipher = ((0,1,0),(1,0))
    rows = (CONTEXTS[0],)*3 if iid else CONTEXTS
    contexts = (0,0) if iid else (1,2)
    for offsets in itertools.product(range(4),range(3)):
        b = bridge(cipher,iid=iid,offsets=offsets,contexts=contexts)
        for cuts,closed in bridge_schedule(b.lengths):
            actual = b.log_values(KEYS,cuts,closed=closed)
            expected = []
            for key in KEYS:
                value = F(1)
                for c,o,n,ctx in zip(cipher,offsets,cuts,contexts,strict=True):
                    value *= F(1) if o==len(c) else enumerated(c[o:o+n],key,ctx,rows,closed=closed)
                expected.append(float(value))
            np.testing.assert_allclose(np.exp(actual),expected,rtol=0,atol=2e-15)


def test_mid_unit_prefix_never_forces_premature_eos():
    b = bridge(((0,0),))
    key = np.array([[2,2]],dtype=np.int32)
    assert math.exp(b.log_values(key,(1,))[0])==pytest.approx(.75)
    assert math.exp(b.log_values(key,(2,),closed=True)[0])==pytest.approx(3/16)
    assert np.isneginf(bridge(((0,),)).log_values(key,(1,),closed=True)[0])


def symmetric_q(keys, free, kernel):
    """Exact proposal probabilities; independent enumeration, no RNG code."""
    matrix = np.zeros((len(keys),len(keys)))
    pieces = [(1,F(1))] if kernel=="row" else [(1,F(3,4)),(min(2,len(free)),F(3,16)),(len(free),F(1,16))]
    for i,a in enumerate(keys):
        for size,mix in pieces:
            subsets = list(itertools.combinations(free,size))
            for subset in subsets:
                for j,b in enumerate(keys):
                    if all(a[r]==b[r] for r in range(2) if r not in subset):
                        matrix[i,j] += float(mix/F(len(subsets)*6**size))
    return matrix


def mh_matrix(q, values):
    transition = np.zeros_like(q)
    for i,a in enumerate(values):
        if a==0:
            transition[i,i] = 1
            continue
        for j,b in enumerate(values):
            if i!=j:
                transition[i,j] = q[i,j]*min(1,b/a)
        transition[i,i] = 1-transition[i].sum()
    return transition


@pytest.mark.parametrize("kernel",["row","joint"])
def test_all_tiny_conditional_targets_mh_detailed_balance_and_normalizer_transport(kernel):
    # All 49 partial dictionaries, zero targets included; every bridge stage.
    b = bridge()
    all_stage = [np.ones(36)]+[np.exp(b.log_values(KEYS,c,closed=d)) for c,d in bridge_schedule(b.lengths)]
    for partial in itertools.product(range(-1,6),repeat=2):
        free = [r for r,k in enumerate(partial) if k<0]
        allowed = np.array([all(k<0 or key[r]==k for r,k in enumerate(partial)) for key in KEYS])
        keys = KEYS[allowed]
        q = symmetric_q(keys,free,kernel) if free else np.eye(1)
        np.testing.assert_allclose(q,q.T,rtol=0,atol=1e-15)
        np.testing.assert_allclose(q.sum(axis=1),1,rtol=0,atol=1e-15)
        previous = np.ones(len(keys))
        for full in all_stage[1:]:
            values = full[allowed]
            transition = mh_matrix(q,values)
            flows = values[:,None]*transition
            np.testing.assert_allclose(flows,flows.T,rtol=0,atol=1e-15)
            np.testing.assert_allclose(values@transition,values,rtol=0,atol=1e-15)
            # Weight at the OLD key, then mutate to current invariant target.
            ratio = np.divide(values,previous,out=np.zeros_like(values),where=previous>0)
            transported = (previous*ratio)@transition
            np.testing.assert_allclose(transported,values,rtol=0,atol=1e-15)
            assert np.all(values[previous==0]==0)
            previous = values


def test_joint_full_refresh_has_positive_routes_between_every_supported_tiny_key():
    values = np.exp(bridge().log_values(KEYS,(2,2),closed=True))
    transition = mh_matrix(symmetric_q(KEYS,[0,1],"joint"),values)
    good = values>0
    assert np.all(transition[np.ix_(good,good)]>0)
    # This verifies reachability, not a bound on practical mixing time.


@pytest.mark.parametrize("kernel",["row","joint"])
def test_implemented_proposals_respect_known_rows_and_exact_conditional_symmetry(kernel):
    rng = np.random.default_rng(81901)
    keys = np.tile([0,1],(4096,1)).astype(np.int32)
    result = propose(keys,np.array([1]),6,rng,kernel)
    assert np.all(result[:,0]==0)
    counts = np.bincount(result[:,1],minlength=6)
    assert np.all(np.abs(counts/4096-1/6)<.025)  # Fixed RNG transport smoke check, not mixing claim.


@pytest.mark.parametrize("kernel",["row","joint"])
def test_actual_smc_reproducibility_positive_final_likelihood_and_all_bindings(kernel):
    b = bridge()
    result = run_dictionary_smc(b,(0,-1),particles=64,seed=81902,kernel=kernel)
    other = run_dictionary_smc(bridge(),(0,-1),particles=64,seed=81902,kernel=kernel)
    assert result["status"]=="complete_particles"
    assert result["trace"]==other["trace"]
    assert result["log_evidence_estimate"]==other["log_evidence_estimate"]
    assert np.array_equal(result["keys"],other["keys"])
    assert np.all(result["keys"][:,0]==0)
    assert np.all(np.isfinite(result["key_log_likelihoods"]))
    expected = [math.log(float(reference(((0,1),(1,0)),key,(0,0),closed=True))) for key in result["keys"]]
    np.testing.assert_allclose(result["key_log_likelihoods"],expected,rtol=0,atol=2e-14)
    assert result["tables"]==64*5*3


def test_fully_bound_smc_telescopes_exact_target_without_mutations():
    key = (0,1)
    result = run_dictionary_smc(bridge(),key,particles=16,seed=81903,mutations=64)
    expected = math.log(float(reference(((0,1),(1,0)),key,(0,0),closed=True)))
    assert result["log_evidence_estimate"]==pytest.approx(expected,abs=1e-14)
    assert all(r["proposals"]==0 for r in result["trace"])
    assert result["tables"]==16*5


def test_literal_zero_is_extinction_no_refill_no_floor():
    result = run_dictionary_smc(bridge(((0,1,0),)),(2,2),particles=8,seed=81904)
    assert result["status"]=="extinct" and result["log_evidence_estimate"] is None
    assert result["trace"][-1]["extinct"]
    assert result["tables"]==16


def test_already_closed_records_factor_one_and_zero_probability_source_support():
    b = bridge(offsets=(2,2))
    result = run_dictionary_smc(b,(-1,-1),particles=16,seed=81905)
    assert result["log_evidence_estimate"]==0
    assert np.all(result["key_log_likelihoods"]==0)
    p,t = source(zero=True)
    b = FixedKeyBridge(((1,),),p,t,glyphs=2,rho=.25)
    values = b.log_values(KEYS,(1,),closed=True)
    expected = [float(enumerated((1,),key,0,((F(1),F(0)),)*3,closed=True)) for key in KEYS]
    np.testing.assert_allclose(np.exp(values),expected,rtol=0,atol=1e-15)


def test_invalid_stage_key_population_configuration_and_work_guards():
    b = bridge()
    for cuts,closed in (((3,0),False),((1,2),True),((False,0),False)):
        with pytest.raises(ValueError):
            b.log_values(KEYS,cuts,closed=closed)
    for invalid in (np.array([0,1]),np.array([[6,0]]),np.array([[True,False]])):
        with pytest.raises(ValueError):
            b.log_values(invalid,(1,0))
    with pytest.raises(RuntimeError):
        bridge(max_tables=1).log_values(KEYS,(1,0))
    with pytest.raises(RuntimeError):
        bridge(max_edges=1).log_values(KEYS,(1,0))
    with pytest.raises(MemoryError):
        bridge(max_states_per_table=1).log_values(np.array([[0,1]]),(2,2),closed=True)
    with pytest.raises(MemoryError):
        bridge(max_work_bytes=1)
    with pytest.raises(MemoryError):
        run_dictionary_smc(b,(-1,-1),max_particle_bytes=1)
    with pytest.raises(ValueError):
        run_dictionary_smc(b,(-1,-1),mutations=True)
    with pytest.raises(ValueError):
        bridge_schedule((2,2),stride=False)


def test_source_tables_are_copied_and_inputs_not_mutated():
    p,t = source()
    b = FixedKeyBridge(((0,1),),p,t,glyphs=2,rho=.25)
    expected = b.log_values(KEYS,(2,),closed=True)
    p[:] = [1,0]
    t[:] = 0
    np.testing.assert_array_equal(b.log_values(KEYS,(2,),closed=True),expected)
    np.testing.assert_array_equal(KEYS,np.array(list(itertools.product(range(6),repeat=2))))
