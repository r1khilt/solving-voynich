"""Exact rational transport, resampling and incorrect-order/support controls."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.check_prefix_bridge001 import CONTEXTS,enumerated
from tests.test_coordinated_dictionary import proposal_matrix
from tests.test_inventory_bdd import literal_support
from tests.test_native_censored_bridge import native as native
from voynich.dictionary_smc import FixedKeyBridge
from voynich.inventory_bdd import InventoryBDD,InventoryProfile
from voynich.strict_censored_bridge import StrictCensoredNativeBridge
from voynich.tempered_inventory import run_tempered_inventory_smc,supported_proposals


def mh(q,values):
    matrix = [[F(0) for _ in values] for _ in values]
    for i,v in enumerate(values):
        if v:
            for j,w in enumerate(values):
                if i!=j:
                    matrix[i][j] = q[i][j]*min(F(1),w/v)
        matrix[i][i] = 1-sum(matrix[i])
    assert all(sum(row)==1 and min(row)>=0 for row in matrix)
    return matrix


def move(masses,matrix):
    return [sum(m*row[j] for m,row in zip(masses,matrix,strict=True)) for j in range(len(masses))]


def finite_tempering_checks():
    keys = list(itertools.product(range(6),repeat=2))
    units = ((0,),(1,),(0,0),(0,1),(1,0),(1,1))
    records = ((0,1),(0,1))
    base = proposal_matrix(keys,units)
    support = [literal_support(records,units,set(k)) for k in keys]
    bdd = InventoryBDD(records,glyphs=2)
    profile = InventoryProfile(bdd,np.array([-1,-1]))
    assert sum(support)==profile.total==13
    assert profile.prior_event_probability==F(13,36)
    refreshed = [[base[i][j]-F(1,32*36)+F(int(support[j]),32*13)
                  for j in range(36)] for i in range(36)]
    assert all(sum(row)==1 and min(row)>=0 for row in refreshed)
    assert all(refreshed[i][j]==refreshed[j][i] for i in range(36) if support[i]
               for j in range(36) if support[j])
    balance,transport,resampling = 0,0,0
    cases,witnesses = [],[]
    priors = [('uniform',[F(1,36)]*36),
              ('weighted_abstract_only',[F((a+1)*(b+1),21**2) for a,b in keys])]
    sources = [('contextual',CONTEXTS),('equal_label_null',((F(1,2),F(1,2)),)*3)]
    for source_name,source in sources:
        # Identical closed records/context: L = G^2, so beta=1/2 is rational.
        g = [enumerated((0,1),k,1,source,closed=True) for k in keys]
        assert [bool(v) for v in g]==support
        for prior_name,prior in priors:
            assert sum(prior)==1 and min(prior)>0
            pb = sum(p for p,s in zip(prior,support,strict=True) if s)
            q0 = [p/pb if s else F(0) for p,s in zip(prior,support,strict=True)]
            for kernel_name,q in [('coordinated',base),('supported_refresh',refreshed)]:
                masses = [pb*p for p in q0]
                previous = [p if s else F(0) for p,s in zip(prior,support,strict=True)]
                for power in (0,1,2):
                    values = [p*v**power if s else F(0)
                              for p,v,s in zip(prior,g,support,strict=True)]
                    matrix = mh(q,values)
                    for i in range(36):
                        for j in range(36):
                            assert values[i]*matrix[i][j]==values[j]*matrix[j][i]
                            balance += 1
                    weighted = [m*v/old if old else F(0)
                                for m,v,old in zip(masses,values,previous,strict=True)]
                    masses = move(weighted,matrix)
                    assert masses==values
                    previous = values
                    transport += 36
                    if power==1:
                        # Literal ordered two-particle multinomial resampling,
                        # followed by independent mutation. Enumerate ancestors;
                        # integrate each particle's mutation by its exact row.
                        expected = [F(0)]*36
                        positive = [i for i,s in enumerate(support) if s]
                        for a,b in itertools.product(positive,repeat=2):
                            start_probability = q0[a]*q0[b]
                            total = g[a]+g[b]
                            ancestor_weights = (g[a]/total,g[b]/total)
                            zhat = pb*total/2
                            for x,y in itertools.product(range(2),repeat=2):
                                ix,iy = (a,b)[x],(a,b)[y]
                                factor = start_probability*ancestor_weights[x]*ancestor_weights[y]*zhat/2
                                for j in positive:
                                    expected[j] += factor*(matrix[ix][j]+matrix[iy][j])
                                resampling += 1
                        assert expected==values
                        first_matrix = matrix
                z = sum(masses)
                assert z==sum(p*v*v for p,v in zip(prior,g,strict=True))
                without_initial_z = z/pb
                assert without_initial_z!=z
                initial = [p if s else F(0) for p,s in zip(prior,support,strict=True)]
                moved_first = move(initial,first_matrix)
                bad_order = [m*v for m,v in zip(moved_first,g,strict=True)]
                assert bad_order!=[p*v for p,v in zip(initial,g,strict=True)]
                # Treating L^0 as1 outside support and using unrestricted beta0
                # mutation loses some initial unnormalized mass from support.
                unsafe_zero = mh(q,prior)
                leaked = move(initial,unsafe_zero)
                assert any(v>0 for v,s in zip(leaked,support,strict=True) if not s)
                assert sum(v for v,s in zip(leaked,support,strict=True) if s)<pb
                cases.append({'source':source_name,'prior':prior_name,'kernel':kernel_name,
                    'support_probability':str(pb),'closed_evidence':str(z),
                    'omitting_initial_normalizer_evidence':str(without_initial_z)})
                witnesses.append({'mutate_before_weight_total':str(sum(bad_order)),
                    'correct_half_temperature_total':str(sum(p*v for p,v in zip(initial,g,strict=True)))})
            if prior_name!='uniform':
                wrong = mh(base,[v*v for v in g])
                target = [p*v*v for p,v in zip(prior,g,strict=True)]
                assert move(target,wrong)!=target  # Future weighted priors need pi-ratio.
    # The equal-label source retains at least the row-swap ambiguity at beta1.
    null = [enumerated((0,1),k,1,sources[1][1],closed=True) for k in keys]
    assert null[keys.index((0,1))]==null[keys.index((1,0))]>0
    assert null[keys.index((3,0))]==null[keys.index((0,3))]>0
    return {'balance_pairs':balance,'unnormalized_transport_state_checks':transport,
        'two_particle_first_increment_ancestor_paths':resampling,'cases':cases,
        'wrong_order_witnesses':witnesses,'supported_dictionary_count':13,
        'temperature_schedule':['0','1/2','1'],'equal_label_null_remains_ambiguous':True,
        'weighted_prior_cases_are_abstract_not_implemented_sampler':True}


def test_exact_rational_tempering_transport_resampling_and_negative_controls():
    finite_tempering_checks()


def environment(partial=None):
    records = ((0,1),(0,1))
    bridge = FixedKeyBridge(records,np.array([[.7,.3]]),np.array([[0,0]]),glyphs=2,rho=.25)
    profile = InventoryProfile(InventoryBDD(records,glyphs=2),
        np.array([-1,-1] if partial is None else partial))
    return bridge,profile


@pytest.mark.parametrize('kernel',['row','coordinated','supported_refresh'])
def test_actual_fixed_schedule_support_reference_scores_and_repeatability(kernel):
    a,profile = environment([3,-1])
    b,other = environment([3,-1])
    kwargs = dict(betas=(0,.125,.5,1),particles=32,seed=14,mutations=4,kernel=kernel)
    x,y = run_tempered_inventory_smc(a,profile,**kwargs),run_tempered_inventory_smc(b,other,**kwargs)
    assert x['trace']==y['trace'] and x['initialization']==y['initialization']
    np.testing.assert_array_equal(x['keys'],y['keys'])
    np.testing.assert_array_equal(x['key_log_likelihoods'],y['key_log_likelihoods'])
    assert np.all(x['keys'][:,0]==3)
    reference,_ = environment()
    expected = reference.log_values(x['keys'],reference.lengths,closed=True)
    np.testing.assert_allclose(x['key_log_likelihoods'],expected,atol=1e-12,rtol=0)
    assert all(profile.bdd.accepts(list(map(int,k))) for k in x['keys'])
    assert all(1<=row['incremental_ess']<=32.000000001 for row in x['trace'])
    assert all(sum(row['operation_counts'].values())==row['proposals']==128 for row in x['trace'])
    initial = math.log(float(profile.prior_event_probability))
    assert x['log_evidence_estimate']==pytest.approx(initial+sum(t['increment_log_mean'] for t in x['trace']))


@pytest.mark.parametrize('kernel',['row','coordinated','supported_refresh'])
def test_real_native_and_python_full_trajectory_agree_on_contextual_fixture(native,kernel):
    records = ((0,1),(1,0,1))
    bridge = StrictCensoredNativeBridge(records,native,glyphs=2,rho=.25)
    reference = FixedKeyBridge(records,native.probabilities,native.transitions,glyphs=2,rho=.25)
    profile = InventoryProfile(InventoryBDD(records,glyphs=2),np.array([-1,-1]))
    kwargs = dict(betas=(0,.03125,.125,.5,1),particles=32,seed=143,mutations=4,kernel=kernel)
    a,b = run_tempered_inventory_smc(bridge,profile,**kwargs),run_tempered_inventory_smc(reference,profile,**kwargs)
    np.testing.assert_array_equal(a['keys'],b['keys'])
    np.testing.assert_allclose(a['key_log_likelihoods'],b['key_log_likelihoods'],rtol=0,atol=2e-14)
    assert a['log_evidence_estimate']==pytest.approx(b['log_evidence_estimate'],abs=2e-14)
    for x,y in zip(a['trace'],b['trace'],strict=True):
        for field in ('parents','key_bank_sha256','accepted_proposals','changed_proposals',
                      'support_rejected_without_scoring','operation_counts','supported_refresh_draws'):
            assert x[field]==y[field]


def test_structural_rejection_never_calls_source_scorer_on_invalid_key(monkeypatch):
    bridge,profile = environment()
    original = bridge.log_values
    def scorer(keys,*args,**kwargs):
        assert all(literal_support(bridge.cipher,bridge.units,set(map(int,k))) for k in keys)
        return original(keys,*args,**kwargs)
    monkeypatch.setattr(bridge,'log_values',scorer)
    x = run_tempered_inventory_smc(bridge,profile,betas=(0,.5,1),particles=32,seed=12,kernel='row')
    assert sum(row['support_rejected_without_scoring'] for row in x['trace'])>0


def test_all_fixed_rows_exact_evidence_and_no_mutation_or_prior_loss():
    bridge,profile = environment([0,1])
    x = run_tempered_inventory_smc(bridge,profile,betas=(0,.2,.5,1),particles=1)
    assert x['log_evidence_estimate']==pytest.approx(float(x['key_log_likelihoods'][0]))
    assert x['initialization']['prior_full_support_probability']=='1'
    assert all(row['proposals']==0 for row in x['trace'])


def test_supported_refresh_ticket_is_actual_conditional_draw():
    _,profile = environment()
    class RefreshOnly:
        def __init__(self):
            self.inner = np.random.default_rng(5)
        def integers(self,high,**kwargs):
            assert high==32 and not kwargs
            return 31
        def bytes(self,n):
            return self.inner.bytes(n)
    keys = np.array([[0,1]]*64)
    candidates,counts,draws = supported_proposals(keys,profile,RefreshOnly())
    assert draws==counts['refresh']==64
    assert all(profile.bdd.accepts(list(map(int,k))) for k in candidates)


@pytest.mark.parametrize('betas',[(0,1,1),(.1,1),(0,.5),(0,True,1),(0,float('nan'),1),(0,2,1)])
def test_malformed_schedule_rejected_before_sampling(betas):
    bridge,profile = environment()
    with pytest.raises(ValueError):
        run_tempered_inventory_smc(bridge,profile,betas=betas)
    assert bridge.tables==0


def test_wrong_observation_and_source_zeros_rejected_before_sampling():
    bridge,profile = environment()
    wrong = InventoryProfile(InventoryBDD(((1,0),(1,0)),glyphs=2),np.array([-1,-1]))
    with pytest.raises(ValueError,match='matching'):
        run_tempered_inventory_smc(bridge,wrong,betas=(0,1))
    zero = FixedKeyBridge(bridge.cipher,np.array([[1.,0.]]),np.array([[0,0]]),glyphs=2)
    with pytest.raises(ValueError,match='positive source'):
        run_tempered_inventory_smc(zero,profile,betas=(0,1))
    assert bridge.tables==zero.tables==0


def test_work_failure_and_integer_failure_propagate_without_repair(monkeypatch):
    bridge,profile = environment()
    bridge.max_tables = 1
    with pytest.raises(RuntimeError,match='table'):
        run_tempered_inventory_smc(bridge,profile,betas=(0,1),particles=2)
    bridge,profile = environment()
    def fail(_):
        raise RuntimeError('Exact integer sampler rejection budget exhausted')
    monkeypatch.setattr(profile,'sample',fail)
    with pytest.raises(RuntimeError,match='integer sampler'):
        run_tempered_inventory_smc(bridge,profile,betas=(0,1))
    assert bridge.tables==0


def test_infinite_supported_proposal_score_aborts_not_rejects(monkeypatch):
    bridge,profile = environment()
    original = bridge.log_values
    calls = []
    def scorer(*args,**kwargs):
        calls.append(1)
        values = original(*args,**kwargs)
        return values if len(calls)==1 else np.full(len(values),-math.inf)
    monkeypatch.setattr(bridge,'log_values',scorer)
    with pytest.raises(ArithmeticError,match='proposed key'):
        run_tempered_inventory_smc(bridge,profile,betas=(0,1),seed=7)
