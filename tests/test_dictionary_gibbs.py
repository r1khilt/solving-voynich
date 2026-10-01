"""Independent conditional matrices and actual finite-source Gibbs transport."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from tests.test_dictionary_smc import KEYS,bridge,reference
from voynich.dictionary_gibbs import gibbs_update,run_dictionary_gibbs
from voynich.dictionary_smc import bridge_schedule,run_dictionary_smc


def test_all_conditional_row_kernels_have_exact_rational_detailed_balance():
    matrices = pairs = 0
    for partial in itertools.product(range(-1,6),repeat=2):
        ids = [i for i,k in enumerate(KEYS) if all(p<0 or k[r]==p for r,p in enumerate(partial))]
        keys = KEYS[ids]
        free = [r for r,p in enumerate(partial) if p<0]
        weighted = [F(1,len(keys)) for _ in keys]
        previous = [F(1) for _ in keys]
        for cuts,closed in bridge_schedule((2,2)):
            values = [reference(((0,1)[:cuts[0]],(1,0)[:cuts[1]]),k,(1,2),closed=closed) for k in keys]
            if not sum(values):
                continue
            matrix = [[F(0) for _ in keys] for _ in keys]
            for i,key in enumerate(keys):
                if not values[i] or not free:
                    matrix[i][i] = F(1)
                    continue
                for row in free:
                    compatible = [j for j,k in enumerate(keys) if all(k[r]==key[r] for r in range(2) if r!=row)]
                    total = sum(values[j] for j in compatible)
                    for j in compatible:
                        matrix[i][j] += values[j]/total/len(free)
            assert all(sum(row)==1 for row in matrix)
            for i in range(len(keys)):
                for j in range(len(keys)):
                    assert values[i]*matrix[i][j]==values[j]*matrix[j][i]
                    pairs += 1
            transported = [sum(weighted[i]*values[i]/previous[i]*matrix[i][j]
                for i in range(len(keys)) if previous[i]) for j in range(len(keys))]
            assert transported==[v/len(keys) for v in values]
            weighted,previous = transported,values
            matrices += 1
    assert matrices==124 and pairs==8349


def test_actual_gibbs_candidate_probabilities_match_independent_source_strings():
    b = bridge(contexts=(1,2),max_tables=10000)
    keys = np.array([[0,1],[2,1],[0,1]],dtype=np.int32)
    values = b.log_values(keys,(1,1))
    class ObservedRng:
        def choice(self,free,size):
            assert list(free)==[0,1] and size==3
            return np.array([0,1,0])
        def random(self,n):
            assert n==3
            return np.array([0.,.25,.999999])
    updated,logs,stats = gibbs_update(b,keys,values,np.array([0,1]),(1,1),False,ObservedRng())
    expected = []
    for key,row,uniform in zip(keys,[0,1,0],[0.,.25,.999999],strict=True):
        candidates = [tuple(u if r==row else key[r] for r in range(2)) for u in range(6)]
        mass = [reference(((0,),(1,)),k,(1,2)) for k in candidates]
        threshold = uniform*float(sum(mass))
        acc = 0.
        chosen = None
        for k,p in zip(candidates,mass,strict=True):
            acc += float(p)
            if acc>threshold:
                chosen = k
                break
        expected.append(chosen)
    assert updated.tolist()==[list(k) for k in expected]
    np.testing.assert_allclose(np.exp(logs),[float(reference(((0,),(1,)),k,(1,2))) for k in expected],rtol=0,atol=1e-14)
    assert stats['conditional_target_tables']==18 and stats['accepted_proposals']==3


def test_same_initial_bank_smc_invariance_counts_and_full_bound_key_telescoping():
    for partial in ([0,-1],[0,1]):
        b = bridge(((0,0),(0,0)),contexts=(1,2),max_tables=100000)
        value = run_dictionary_gibbs(b,partial,particles=16,seed=91203,stride=1)
        assert value['status']=='complete_particles'
        assert all(all(p<0 or k[r]==p for r,p in enumerate(partial)) for k in value['keys'])
        free = any(p<0 for p in partial)
        assert b.tables==16*len(value['trace'])*(7 if free else 1)
        np.testing.assert_allclose(np.exp(value['key_log_likelihoods']),
            [float(reference(((0,0),(0,0)),k,(1,2),closed=True)) for k in value['keys']],rtol=0,atol=1e-14)
        if not free:
            assert value['log_evidence_estimate']==pytest.approx(math.log(float(reference(((0,0),(0,0)),partial,(1,2),closed=True))))
    # Same prior bank and first incremental weights before either kernel runs.
    a = run_dictionary_gibbs(bridge(contexts=(1,2)),[-1,-1],particles=16,seed=91204)
    b = run_dictionary_smc(bridge(contexts=(1,2)),[-1,-1],particles=16,seed=91204,stride=16,mutations=6,kernel='row')
    assert a['trace'][0]['increment_log_mean']==b['trace'][0]['increment_log_mean']
    assert a['trace'][0]['maximum_incremental_weight']==b['trace'][0]['maximum_incremental_weight']


def test_extinction_and_work_caps_remain_errors_or_literal_zero():
    value = run_dictionary_gibbs(bridge(((1,),)),[0,0],particles=1)
    assert value['status']=='extinct' and value['log_evidence_estimate'] is None
    with pytest.raises(RuntimeError,match='table cap'):
        run_dictionary_gibbs(bridge(max_tables=1),[-1,-1],particles=1)
    with pytest.raises(ValueError):
        run_dictionary_gibbs(bridge(),[-1,-1],particles=700)
    with pytest.raises(MemoryError):
        run_dictionary_gibbs(bridge(),[-1,-1],max_particle_bytes=1)


def test_row_gibbs_can_be_reducible_despite_exact_conditionals():
    b = bridge(((0,),(1,)),contexts=(1,2),max_tables=1000)
    values = b.log_values(KEYS,(1,1),closed=True)
    positive = KEYS[np.isfinite(values)]
    assert positive.tolist()==[[0,1],[1,0]]
    keys = np.array([[0,1],[1,0]],dtype=np.int32)
    values = b.log_values(keys,(1,1),closed=True)
    updated,_,stats = gibbs_update(b,keys,values,np.array([0,1]),(1,1),True,np.random.default_rng(91301))
    assert updated.tolist()==keys.tolist() and stats['minimum_positive_codes']==1
