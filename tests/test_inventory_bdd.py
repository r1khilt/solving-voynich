"""Independent word break, exact occupancy laws, ranking, and cap controls."""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_prefix_bridge001 import CONTEXTS,enumerated
from scripts.check_observation_initialization001 import support_counts
from tests.test_coordinated_dictionary import proposal_matrix
from voynich.inventory_bdd import InventoryBDD,InventoryBudgetExceeded,InventoryProfile


def literal_support(records,units,codes,closed=True):
    for record in records:
        record = tuple(record)
        reach = {0}
        for pos in range(len(record)):
            if pos not in reach:
                continue
            for code in codes:
                unit = units[code]
                end = pos+len(unit)
                visible = min(len(unit),len(record)-pos)
                if (not closed or end<=len(record)) and unit[:visible]==record[pos:pos+visible]:
                    reach.add(min(end,len(record)))
        if len(record) not in reach:
            return False
    return True


def finite_checks():
    masks_checked,profiles_checked,ranks_checked,keys_checked = 0,0,0,0
    records = [(),*((tuple(x),) for n in range(1,4) for x in itertools.product(range(2),repeat=n))]
    records[0] = ((),)
    records += [((0,1),(1,0)),((0,0),(1,1)),((0,),(1,0,1))]
    for observations in records:
        for closed in (False,True):
            for order in (tuple(range(6)),tuple(range(5,-1,-1))):
                bdd = InventoryBDD(observations,glyphs=2,order=order,closed=closed)
                for mask in range(64):
                    codes = {c for c in range(6) if mask&(1<<c)}
                    assert bdd.accepts(codes)==literal_support(observations,bdd.units,codes,closed)
                    masks_checked += 1
                for partial in ((-1,-1),(0,-1),(3,-1),(0,1)):
                    profile = InventoryProfile(bdd,np.array(partial))
                    known = set(c for c in partial if c>=0)
                    unknown = [c for c in range(6) if c not in known]
                    expected = []
                    for mask in range(1<<len(unknown)):
                        codes = {unknown[i] for i in range(len(unknown)) if mask&(1<<i)}
                        if literal_support(observations,bdd.units,known|codes,closed):
                            expected.append(codes)
                    coefficients = [sum(len(s)==k for s in expected) for k in range(len(unknown)+1)]
                    assert list(profile.coefficients)==coefficients
                    ranked = []
                    for k,b in enumerate(coefficients):
                        for rank in range(b):
                            ranked.append(profile.unrank(k,rank))
                            ranks_checked += 1
                    assert {frozenset(s) for s in ranked}=={frozenset(s) for s in expected}
                    assert len(ranked)==len(expected)
                    free = partial.count(-1)
                    count = 0
                    for completion in itertools.product(range(6),repeat=free):
                        remaining = iter(completion)
                        key = tuple(next(remaining) if c<0 else c for c in partial)
                        valid = literal_support(observations,bdd.units,set(key),closed)
                        count += int(valid)
                        new = set(completion)-known
                        if valid:
                            k = len(new)
                            assignments = profile.onto_count(free,k,len(known|new))
                            q = F(profile.cardinality_weights[k],profile.total)/profile.coefficients[k]/assignments
                            assert q==F(1,profile.total)
                        keys_checked += 1
                    assert profile.total==count and profile.prior_event_probability==F(count,6**free)
                    profiles_checked += 1
    return {'boolean_mask_checks':masks_checked,'known_binding_profiles':profiles_checked,
        'subset_rank_checks':ranks_checked,'dictionary_factorization_checks':keys_checked}


def test_exhaustive_boolean_and_conditional_distribution_laws():
    finite_checks()


def full_support_smc_law():
    keys = list(itertools.product(range(6),repeat=2))
    units = ((0,),(1,),(0,0),(0,1),(1,0),(1,1))
    q = proposal_matrix(keys,units)
    bdd = InventoryBDD(((0,1),(1,0)),glyphs=2)
    profile = InventoryProfile(bdd,np.array([-1,-1]))
    supported = [bdd.accepts(k) for k in keys]
    assert profile.total==4 and profile.prior_event_probability==F(1,9)
    stages = (((1,0),False),((2,1),False),((2,2),False),((2,2),True))
    previous = [F(1) if s else F(0) for s in supported]
    masses = [F(int(s),36) for s in supported]
    checks = 0
    for cuts,closed in stages:
        values = [enumerated((0,1)[:cuts[0]],k,1,CONTEXTS,closed=closed)*
                  enumerated((1,0)[:cuts[1]],k,2,CONTEXTS,closed=closed)*s
                  for k,s in zip(keys,supported,strict=True)]
        weighted = [m*v/old if old else F(0) for m,v,old in zip(masses,values,previous,strict=True)]
        transition = [[F(0) for _ in keys] for _ in keys]
        for i,vi in enumerate(values):
            if not vi:
                transition[i][i] = 1
            else:
                for j,vj in enumerate(values):
                    if i!=j:
                        transition[i][j] = q[i][j]*min(F(1),vj/vi)
                transition[i][i] = 1-sum(transition[i])
        masses = [sum(weighted[i]*transition[i][j] for i in range(36)) for j in range(36)]
        assert masses==[v/36 for v in values]
        previous = values
        checks += 36**2
    correct_z = sum(previous)/36
    # Incorrectly unrestricted intermediate moves after support-conditioned init.
    first = [enumerated((0,),k,1,CONTEXTS) for k in keys]
    unrestricted = [[F(0) for _ in keys] for _ in keys]
    for i,vi in enumerate(first):
        if not vi:
            unrestricted[i][i] = 1
        else:
            for j,vj in enumerate(first):
                if i!=j:
                    unrestricted[i][j] = q[i][j]*min(F(1),vj/vi)
            unrestricted[i][i] = 1-sum(unrestricted[i])
    moved = [sum(F(int(supported[i]),36)*first[i]*unrestricted[i][j] for i in range(36)) for j in range(36)]
    bad_z = sum(m*v/old if old else F(0) for m,v,old in zip(moved,previous,first,strict=True))
    assert bad_z!=correct_z
    return {'full_support_smc_exact_pairs':checks,'exact_closed_evidence':str(correct_z),
        'incorrect_unrestricted_intermediate_evidence':str(bad_z)}


def test_whole_record_conditioning_needs_restricted_intermediate_targets():
    full_support_smc_law()


@pytest.mark.parametrize('partial',[[-1]*23,[0,1]+[-1]*21,[0,1,3,4]])
def test_actual_sampler_validity_known_bindings_repeatability_and_integer_law(partial):
    bdd = InventoryBDD(((0,1),(1,0)),glyphs=2)
    profile = InventoryProfile(bdd,np.array(partial))
    a,b = np.random.default_rng(82701),np.random.default_rng(82701)
    for _ in range(32):
        x,y = profile.sample(a),profile.sample(b)
        np.testing.assert_array_equal(x,y)
        assert bdd.accepts(list(map(int,x)))
        for i,c in enumerate(partial):
            if c>=0:
                assert x[i]==c


def test_symmetric_label_null_support_is_not_plaintext_identification():
    bdd = InventoryBDD(((0,1),(1,0)),glyphs=2)
    assert bdd.accepts([0,1]) and bdd.accepts([1,0])
    profile = InventoryProfile(bdd,np.array([2,2]))
    assert profile.total==0
    with pytest.raises(ValueError,match='No supporting'):
        profile.sample(np.random.default_rng(0))


def generic_checks():
    pool = tuple(u for n in (1,2) for u in itertools.product(range(6),repeat=n))
    checked = 0
    counts = []
    for order in (tuple(range(42)),tuple(range(41,-1,-1))):
        bdd = InventoryBDD(((0,1,2,3),),glyphs=6,closed=False,order=order)
        for known in ((),(0,1),(3,4)):
            partial = np.array([*known,*([-1]*(23-len(known)))])
            profile = InventoryProfile(bdd,partial)
            reference,_,start = support_counts((0,1,2,3),pool,len(profile.free),known)
            assert profile.total==reference[-1][start]
            checked += 1
            counts.append(str(profile.total))
            rng = np.random.default_rng(82703)
            for _ in range(16):
                key = profile.sample(rng)
                assert literal_support(((0,1,2,3),),pool,set(map(int,key)),False)
    assert counts[:3]==counts[3:]
    return {'generic_six_glyph_known_profile_checks':checked,'generic_full_free_prefix_count':counts[0]}


def test_six_glyph_big_integer_counts_skipped_variables_and_known_rows():
    generic_checks()


@pytest.mark.parametrize('settings',[{'max_nodes':2},{'max_apply_steps':1},
    {'max_apply_entries':1},{'max_work_bytes':1}])
def test_boolean_limits_never_return_partial_graph(settings):
    with pytest.raises(InventoryBudgetExceeded):
        InventoryBDD(((0,1,0,1),),glyphs=2,**settings)


@pytest.mark.parametrize('settings',[{'max_polynomial_cells':1},{'max_polynomial_work':1}])
def test_polynomial_limits_never_return_partial_count(settings):
    bdd = InventoryBDD(((0,1,0,1),),glyphs=2,**settings)
    with pytest.raises(InventoryBudgetExceeded):
        InventoryProfile(bdd,np.array([-1,-1]))


def test_invalid_orders_partial_rows_and_mid_emission_behavior():
    with pytest.raises(ValueError,match='permutation'):
        InventoryBDD(((0,),),glyphs=2,order=(0,)*6)
    with pytest.raises(ValueError):
        InventoryProfile(InventoryBDD(((0,),),glyphs=2),np.array([-.5]))
    prefix = InventoryBDD(((0,),),glyphs=2,closed=False)
    closed = InventoryBDD(((0,),),glyphs=2,closed=True)
    assert prefix.accepts([3]) and not closed.accepts([3])
