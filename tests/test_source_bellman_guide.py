"""Independent full-dictionary future enumeration and unchanged history laws."""
import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from tests.test_source_state_lattice import full_reference, source
from voynich.guided_source_particles import IidSuffixGuide
from voynich.native_source_state import NativeStateLattice, build_source_state_native
from voynich.source_bellman_guide import NativeBellmanGuide, actions, backup, build_bellman
from voynich.source_state_lattice import State


def tail_for(cipher, p):
    guide = IidSuffixGuide(cipher, p[0], glyphs=2, rho=.25, cache_entries=64, max_tables=100000)
    def tail(s):
        offsets = np.array([s.offsets], dtype=np.int32)
        return float(guide(np.array([s.key], dtype=np.int32), offsets, offsets == np.array(list(map(len,cipher))))[0])
    return tail


def full_future(state, cipher, p, goto):
    """Enumerate full remaining dictionaries and plaintext strings as rationals.

    No deterministic action scheduler, backups, state merging, tail or local
    candidate generator is used here. Free unused rows integrate via full keys.
    """
    pool = [(g,) for g in range(2)]+list(itertools.product(range(2), repeat=2))
    free = [r for r,k in enumerate(state.key) if k<0]
    rows = [[F(float(v)).limit_denominator(100) for v in row] for row in p]
    assert all(sum(row)==1 for row in rows)
    total = F(0)
    for codes in itertools.product(range(6), repeat=len(free)):
        key = list(state.key)
        for r,code in zip(free,codes,strict=True):
            key[r] = code
        weight = F(1,6)**len(free)
        for record, observed in enumerate(cipher):
            remainder = tuple(observed[state.offsets[record]:])
            if not remainder:
                continue
            subtotal = F(0)
            for n in range(1,len(remainder)+1):
                for text in itertools.product(range(p.shape[1]),repeat=n):
                    if tuple(g for r in text for g in pool[key[r]]) != remainder:
                        continue
                    probability,context = F(1,4)*F(3,4)**n,state.contexts[record]
                    for r in text:
                        probability *= rows[context][r]
                        context = int(goto[context,r])
                    subtotal += probability
            weight *= subtotal
        total += weight
    return total


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    return build_source_state_native(tmp_path_factory.mktemp("bellman-base")),build_bellman(tmp_path_factory.mktemp("bellman-new"))


@pytest.mark.parametrize("schedule",["balanced","sequential"])
def test_all36_future_full_depth_equals_independent_rational_sum(schedule,builds):
    p,goto=source()
    observed=[(g,) for g in range(2)]+list(itertools.product(range(2),repeat=2))
    engine=NativeBellmanGuide(p,goto,builds[1])
    for cipher in itertools.product(observed,repeat=2):
        root=State((0,0),(0,0),(-1,-1))
        states={root}
        states.update(child for child,_ in actions(root,cipher,p,goto,glyphs=2,rho=.25,schedule=schedule))
        tail=tail_for(cipher,p)
        for s in states:
            expected=float(full_future(s,cipher,p,goto))
            python=backup(s,sum(map(len,cipher)),cipher,p,goto,tail,glyphs=2,rho=.25,schedule=schedule)
            native=engine.value(cipher,s,sum(map(len,cipher)),schedule=schedule)
            assert math.exp(python)==pytest.approx(expected,abs=2e-14)
            assert math.exp(native)==pytest.approx(expected,abs=2e-14)
            for depth in (0,1,2):
                target=backup(s,depth,cipher,p,goto,tail,glyphs=2,rho=.25,schedule=schedule)
                assert engine.value(cipher,s,depth,schedule=schedule)==pytest.approx(target,abs=1e-12)


@pytest.mark.parametrize("schedule",["balanced","sequential"])
def test_all36_backed_search_without_pruning_retains_original_rational_law(schedule,builds):
    p,goto=source()
    engine=NativeBellmanGuide(p,goto,builds[1])
    observed=[(g,) for g in range(2)]+list(itertools.product(range(2),repeat=2))
    for cipher in itertools.product(observed,repeat=2):
        reference,_=full_reference(cipher)
        result=engine.search(cipher,glyphs=2,rho=.25,guidance="iid",width=100000,schedule=schedule)
        assert result["complete_search"] and result["lookahead_calls"]>0
        assert math.exp(result["found_log_mass"])==pytest.approx(float(sum(reference.values())),abs=1e-13)
        assert {tuple(t["used_key"]) for t in result["terminals"]}==reference.keys()
        for t in result["terminals"]:
            assert math.exp(t["log_mass"])==pytest.approx(float(reference[tuple(t["used_key"])]),abs=1e-13)


def test_depth_zero_entire_search_matches_untouched_engine_and_counter_resets(builds):
    baseline,engine=NativeStateLattice(*source(),builds[0]),NativeBellmanGuide(*source(),builds[1])
    cipher=((0,1,0),(1,0,1))
    cfg=dict(glyphs=2,rho=.25,guidance="iid",width=3,guide_prewidth=12)
    for cutoff in (0,1,16):
        r=engine.search(cipher,lookahead_depth=0,lookahead_layers=cutoff,**cfg)
        for k,v in baseline.search(cipher,**cfg).items():
            assert r[k]==v
        assert r["lookahead_calls"]==r["lookahead_actions"]==0
    r=engine.search(cipher,lookahead_depth=1,lookahead_layers=1,**cfg)
    assert r["lookahead_calls"]==1 and r["lookahead_actions"]<=4
    again=engine.search(cipher,lookahead_depth=0,**cfg)
    assert again["lookahead_calls"]==again["lookahead_actions"]==0


def test_floor_does_not_invent_completed_search_support_and_absorbing_one(builds):
    engine=NativeBellmanGuide(*source(zero=True),builds[1])
    cipher=((0,1,0),)
    result=engine.search(cipher,glyphs=2,rho=.25,guidance="iid",width=100000)
    assert result["complete_search"] and not result["terminals"] and result["found_log_mass"]==-math.inf
    s=State((3,),(0,),(0,0))
    assert engine.value(cipher,s,8)==0
    s=State((0,),(0,),(0,0))
    assert engine.value(cipher,s,2)==-math.inf
    assert backup(s,2,cipher,*source(zero=True),tail_for(cipher,source(zero=True)[0]),glyphs=2,rho=.25)==-math.inf


def test_iid_guide_is_neither_future_upper_nor_lower_bound():
    p,goto=source()
    comparisons=set()
    for cipher in (((0,),(0,)),((0,),(1,)),((0,0),(0,1)),((0,1),(1,0))):
        root=State((0,0),(0,0),(-1,-1))
        states=[root]+[s for s,_ in actions(root,cipher,p,goto,glyphs=2,rho=.25)]
        tail=tail_for(cipher,p)
        for s in states:
            truth=float(full_future(s,cipher,p,goto))
            approx=math.exp(tail(s))
            if abs(truth-approx)>1e-12:
                comparisons.add(truth>approx)
    assert comparisons=={True,False}


def test_invalid_configs_and_generated_body_pin(builds):
    engine=NativeBellmanGuide(*source(),builds[1])
    for kw in ({"lookahead_depth":True},{"lookahead_depth":3},{"lookahead_layers":-1}):
        with pytest.raises(ValueError):
            engine.search(((0,),),**kw)
    with pytest.raises(ValueError):
        engine.value(((0,),),State((0,),(0,),(-1,-1)),9)
    with pytest.raises(ValueError):
        NativeBellmanGuide(*source(),{**builds[1],"generated_cpp_sha256":"changed"})
