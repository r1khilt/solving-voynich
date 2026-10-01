"""Exact finite laws, support barriers, and actual sampler transport."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_inventory_barrier001 import components
from scripts.check_prefix_bridge001 import CONTEXTS,enumerated
from voynich.coordinated_dictionary import (
    OPERATION_TICKETS,PAIR_OPERATIONS,completion_count,covered_completion,
    initialization,pair_involution,proposals,randbelow,run_coordinated_smc,
)
from voynich.dictionary_smc import FixedKeyBridge,bridge_schedule


def pool(g):
    return tuple(u for n in (1,2) for u in itertools.product(range(g),repeat=n))


@pytest.mark.parametrize('glyphs',[1,2,6])
def test_all_pair_operations_are_actual_involutions(glyphs):
    units = pool(glyphs)
    for a,b in itertools.product(range(len(units)),repeat=2):
        for operation in PAIR_OPERATIONS:
            aa,bb = pair_involution(a,b,units,operation)
            assert pair_involution(aa,bb,units,operation)==(a,b)


@pytest.mark.parametrize('u',[2,3,6])
def test_completion_counts_equal_exhaustive_suffixes(u):
    for n in range(5):
        for m in range(min(3,u)+1):
            expected = sum(set(range(m))<=set(k) for k in itertools.product(range(u),repeat=n))
            assert completion_count(n,m,u)==expected
            inclusion = sum((-1)**j*math.comb(m,j)*(u-j)**n for j in range(m+1))
            assert inclusion==expected


def proposal_matrix(keys,units):
    index = {k:i for i,k in enumerate(keys)}
    q = [[F(0) for _ in keys] for _ in keys]
    rows = len(keys[0])
    for i,key in enumerate(keys):
        for operation in OPERATION_TICKETS:
            if operation=='refresh':
                for j in range(len(keys)):
                    q[i][j] += F(1,32*len(keys))
            elif operation in ('row','pair'):
                size = 1 if operation=='row' else min(2,rows)
                for selected in itertools.permutations(range(rows),size):
                    for codes in itertools.product(range(len(units)),repeat=size):
                        k = list(key)
                        for row,code in zip(selected,codes,strict=True):
                            k[row] = code
                        q[i][index[tuple(k)]] += F(1,32*math.perm(rows,size)*len(units)**size)
            elif rows<2:
                q[i][i] += F(1,32)
            else:
                for a,b in itertools.permutations(range(rows),2):
                    k = list(key)
                    k[a],k[b] = pair_involution(k[a],k[b],units,operation)
                    q[i][index[tuple(k)]] += F(1,32*rows*(rows-1))
    assert all(sum(row)==1 for row in q)
    assert all(q[i][j]==q[j][i] for i in range(len(keys)) for j in range(len(keys)))
    return q


def checked_laws():
    keys,units = list(itertools.product(range(6),repeat=2)),pool(2)
    q = proposal_matrix(keys,units)
    event = [set(range(2))<=set(k) for k in keys]
    pa = F(sum(event),len(keys))
    init = [F(1,len(keys))*(F(1,2)+F(int(a),2)/pa) for a in event]
    correction = [F(1,2)+(F(1,2)/pa if a else 0) for a in event]
    assert sum(init)==1 and all(v>0 for v in init)
    assert all(init[i]/correction[i]==F(1,len(keys)) for i in range(len(keys)))
    stages = bridge_schedule((2,2),1)
    masses = [F(1,len(keys)) for _ in keys]
    previous = [F(1) for _ in keys]
    balance = 0
    for stage,(cuts,closed) in enumerate(stages):
        values = [enumerated((0,1)[:cuts[0]],k,1,CONTEXTS,closed=closed)*
                  enumerated((1,0)[:cuts[1]],k,2,CONTEXTS,closed=closed) for k in keys]
        # For N=1 resampling is identity; track unnormalized law including Zhat.
        weighted = [((init[i]/correction[i] if stage==0 else masses[i]/previous[i])
                     *values[i] if previous[i] else F(0)) for i in range(len(keys))]
        transition = [[F(0) for _ in keys] for _ in keys]
        for i,vi in enumerate(values):
            if not vi:
                transition[i][i] = 1
                continue
            for j,vj in enumerate(values):
                if i!=j:
                    transition[i][j] = q[i][j]*min(F(1),vj/vi)
            transition[i][i] = 1-sum(transition[i])
        for i in range(len(keys)):
            for j in range(len(keys)):
                assert values[i]*transition[i][j]==values[j]*transition[j][i]
                balance += 1
        masses = [sum(weighted[i]*transition[i][j] for i in range(len(keys))) for j in range(len(keys))]
        assert masses==[v/len(keys) for v in values]
        previous = values
    positive = [i for i,v in enumerate(previous) if v]
    restricted = [[q[i][j]*min(F(1),previous[j]/previous[i]) for j in positive] for i in positive]
    assert len(components(restricted))==1
    motif = [[F(int(pair_involution(*keys[i],units,'motif')==keys[j])) for j in positive] for i in positive]
    assert len(components(motif))==2  # motif crosses inventory; swaps connect labels.
    motif_swap = [[motif[i][j]+F(int(keys[positive[i]][::-1]==keys[positive[j]]))
                   for j in range(len(positive))] for i in range(len(positive))]
    assert len(components(motif_swap))==1
    incorrect_z = sum(init[i]*previous[i] for i in range(len(keys)))
    correct_z = sum(previous)/len(keys)
    assert incorrect_z!=correct_z  # Dropping prior/q changes even exact evidence.
    return {'keys':len(keys),'stages':len(stages),'balance_pairs':balance,
        'proposal_symmetry_pairs':len(keys)**2,'positive_support_components':1,
        'motif_and_swap_support_components_without_refresh':1,
        'correct_evidence':str(correct_z),'uncorrected_evidence':str(incorrect_z)}


def test_full_discrete_proposal_symmetry_balance_and_initial_weight_law():
    checked_laws()
    # Ordered row-pair choices, duplicate codes, and one-free-row identities.
    for rows in (1,3):
        proposal_matrix(list(itertools.product(range(2),repeat=rows)),pool(1))


def test_coverage_sampler_known_bindings_and_exact_big_integer_weights():
    partial = np.array([-1]*23,dtype=np.int32)
    rng = np.random.default_rng(912)
    assert completion_count(23,6,42)>2**64
    for _ in range(40):
        key = covered_completion(partial,6,42,rng)
        assert set(range(6))<=set(key)
    partial[:3] = [0,1,41]
    keys,correction,info = initialization(partial,32,6,42,rng,True)
    assert np.all(keys[:,:3]==partial[:3])
    pa = F(info['coverage_probability'])
    for key,logw in zip(keys,correction,strict=True):
        denominator = F(1,2)+(F(1,2)/pa if set(range(6))<=set(key) else 0)
        assert math.exp(logw)==pytest.approx(float(1/denominator))
    a = initialization(partial,32,6,42,np.random.default_rng(2),True)
    b = initialization(partial,32,6,42,np.random.default_rng(2),True)
    np.testing.assert_array_equal(a[0],b[0])
    with pytest.raises(ValueError,match='zero prior'):
        covered_completion(np.array([-1]),2,6,rng)


def test_every_conditional_sampler_path_has_uniform_exact_probability():
    for n in range(2,5):
        for key in itertools.product(range(3),repeat=n):
            missing,probability = {0,1},F(1)
            for i,code in enumerate(key):
                denominator = completion_count(n-i,len(missing),3)
                if not denominator:
                    probability = 0
                    break
                probability *= F(completion_count(n-i-1,len(missing)-int(code in missing),3),denominator)
                missing.discard(code)
            assert probability==(F(1,completion_count(n,2,3)) if not missing else 0)


def test_unidentifiable_label_null_has_equal_likelihood_despite_different_keys():
    bridge = FixedKeyBridge(((0,1),(1,0)),np.array([[.5,.5]]),np.array([[0,0]]),glyphs=2)
    values = bridge.log_values(np.array([[0,1],[1,0]]),(2,2),closed=True)
    assert values[0]==values[1] and np.isfinite(values).all()


def test_exact_integer_draw_bounds_and_explicit_no_biased_fallback():
    class MaximumBytes:
        def bytes(self,n):
            return b'\xff'*n
    with pytest.raises(RuntimeError,match='rejection budget'):
        randbelow(MaximumBytes(),2**65+1)
    assert randbelow(MaximumBytes(),1)==0
    assert randbelow(MaximumBytes(),2**65)==2**65-1
    for bound in (0,True,42**23+1):
        with pytest.raises(ValueError):
            randbelow(MaximumBytes(),bound)


@pytest.mark.parametrize('covered,kernel',[(False,'row'),(True,'row'),(True,'coordinated')])
def test_actual_smc_targets_counters_and_fixed_bindings(covered,kernel):
    p = np.array([[.7,.3]])
    goto = np.array([[0,0]])
    bridge = FixedKeyBridge(((0,1),(1,0)),p,goto,glyphs=2,rho=.25)
    value = run_coordinated_smc(bridge,np.array([-1,-1]),particles=32,seed=9,
        stride=1,mutations=4,covered=covered,kernel=kernel)
    assert value['status']=='complete_particles'
    assert value['tables']==32*5*5
    assert all(sum(t['operation_counts'].values())==128 for t in value['trace'])
    for k,v in zip(value['keys'],value['key_log_likelihoods'],strict=True):
        ref = FixedKeyBridge(((0,1),(1,0)),p,goto,glyphs=2,rho=.25)
        assert ref.log_values(k[None,:],(2,2),closed=True)[0]==pytest.approx(v)
    keys = np.array([[0,1,3]]*32)
    proposed,_ = proposals(keys,np.array([1,2]),pool(2),np.random.default_rng(8),'coordinated')
    assert np.all(proposed[:,0]==0)


def test_actual_empty_target_is_extinction_not_refill():
    bridge = FixedKeyBridge(((0,1),),np.array([[1.]]),np.array([[0]]),glyphs=2)
    value = run_coordinated_smc(bridge,np.array([-1]),particles=1,seed=1,stride=2,mutations=0)
    assert value['status']=='extinct'
    assert value['log_evidence_estimate'] is None
