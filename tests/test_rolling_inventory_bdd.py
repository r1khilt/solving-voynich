"""Exact root preservation and frozen occupancy/sampling laws after collection."""

import itertools

import numpy as np
import pytest

from tests import test_inventory_bdd as frozen_checks
from voynich.inventory_bdd import InventoryBDD,InventoryBudgetExceeded,InventoryProfile
from voynich.rolling_inventory_bdd import RollingInventoryBDD


def signature(nodes,node):
    if node<=1:
        return node
    var,lo,hi = nodes[node]
    return var,signature(nodes,lo),signature(nodes,hi)


class VerifiedCollector(RollingInventoryBDD):
    def _collect(self,roots):
        original = [signature(self.nodes,n) for n in roots]
        mapped = super()._collect(roots)
        assert original==[signature(self.nodes,n) for n in mapped]
        assert not self.applied
        return mapped


def finite_checks(interval=16):
    original = frozen_checks.InventoryBDD
    class BoundCollector(VerifiedCollector):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,collection_interval=interval,**kwargs)
    try:
        frozen_checks.InventoryBDD = BoundCollector
        return {**frozen_checks.finite_checks(),**frozen_checks.full_support_smc_law()}
    finally:
        frozen_checks.InventoryBDD = original


@pytest.mark.parametrize('interval',[1,16])
def test_all_finite_roots_and_exact_occupancy_transport(interval):
    finite_checks(interval)


def six_glyph_checks():
    checks = 0
    for closed in (False,True):
        for order in (tuple(range(42)),tuple(range(41,-1,-1))):
            records = [(0,1,2,3),(3,2,1,0)]
            old = InventoryBDD(records,order=order,closed=closed)
            new = RollingInventoryBDD(records,order=order,closed=closed,collection_interval=1)
            assert signature(old.nodes,old.root)==signature(new.nodes,new.root)
            for partial in ([-1]*23,[0,1]+[-1]*21,[3,4]+[-1]*21):
                p,q = InventoryProfile(old,np.array(partial)),InventoryProfile(new,np.array(partial))
                assert p.coefficients==q.coefficients and p.total==q.total
                a,b = np.random.default_rng(82801),np.random.default_rng(82801)
                for _ in range(32):
                    assert np.array_equal(p.sample(a),q.sample(b))
                    checks += 1
            assert new.stats()['collections']>=3
    return {'six_glyph_identical_seeded_draws':checks}


def test_six_glyph_counts_and_actual_seeded_draws_equal():
    six_glyph_checks()


@pytest.mark.parametrize('interval',[1,16])
def test_longer_two_suffix_recurrence_and_dead_node_removal(interval):
    observations = ((0,1,1,0,0,1)*5,(1,0,0,1,0,1)*4)
    for closed,order in itertools.product((False,True),(tuple(range(6)),tuple(range(5,-1,-1)))):
        old = InventoryBDD(observations,glyphs=2,order=order,closed=closed)
        new = RollingInventoryBDD(observations,glyphs=2,order=order,closed=closed,collection_interval=interval)
        assert signature(old.nodes,old.root)==signature(new.nodes,new.root)
        assert new.collected_nodes>0 and len(new.nodes)<len(old.nodes)
        for mask in range(64):
            codes = {i for i in range(6) if mask&(1<<i)}
            assert new.accepts(codes)==frozen_checks.literal_support(observations,new.units,codes,closed)


def test_collection_cap_and_sealed_profile_boundary():
    with pytest.raises(InventoryBudgetExceeded,match='collection-work'):
        RollingInventoryBDD(((0,1),),glyphs=2,max_collection_work=1)
    bdd = RollingInventoryBDD(((0,1),),glyphs=2)
    with pytest.raises(ValueError,match='unprofiled'):
        bdd._collect([bdd.root])
    InventoryProfile(bdd,np.array([-1,-1]))
    with pytest.raises(ValueError,match='unprofiled'):
        bdd._collect([bdd.root])


@pytest.mark.parametrize('observations',[(),((6,),),((True,),),((0,)*4097,)])
def test_invalid_records_rejected(observations):
    with pytest.raises(ValueError,match='records'):
        RollingInventoryBDD(observations)
