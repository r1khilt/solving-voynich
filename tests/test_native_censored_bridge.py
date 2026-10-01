"""Censoring verified against independent source strings and original closure."""

import copy
import itertools
import math
import shutil

import numpy as np
import pytest

from scripts.check_prefix_bridge001 import CONTEXTS, enumerated
from tests.test_dictionary_smc import KEYS
from tests.test_native_suffix_marginal import source
from voynich.dictionary_smc import FixedKeyBridge, bridge_schedule, run_dictionary_smc
from voynich.native_censored_bridge import CensoredNativeBridge
from voynich.native_suffix_marginal import NativeMarginal, build_native


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not (shutil.which("clang++") or shutil.which("c++")):
        pytest.skip("Optional local C++ compiler unavailable")
    build = build_native(tmp_path_factory.mktemp("censored-native")/"build")
    model = source()
    # A known three-state rational fixture, not an empirical source fit.
    model.probabilities = np.array([[float(p) for p in row] for row in CONTEXTS],dtype=np.float64)
    model.transitions = np.tile(np.array([1,2],dtype=np.uint32),(3,1))
    model.probabilities.setflags(write=False)
    model.transitions.setflags(write=False)
    return NativeMarginal(model,build)


def bridge(native,cipher=((0,1),(1,0)),**kwargs):
    return CensoredNativeBridge(cipher,native,**{"glyphs":2,"rho":.25,**kwargs})


def test_every_binary_key_context_and_short_cut_matches_rational_source_strings(native):
    for ctx in range(3):
        for n in range(1,4):
            for y in itertools.product(range(2),repeat=n):
                b = bridge(native,(y,),contexts=(ctx,))
                for cuts,closed in bridge_schedule((n,)):
                    expected = [float(enumerated(y[:cuts[0]],key,ctx,CONTEXTS,closed=closed)) for key in KEYS]
                    actual = b.log_values(KEYS,cuts,closed=closed)
                    np.testing.assert_allclose(np.exp(actual),expected,rtol=0,atol=2e-15)


def test_two_record_offsets_current_contexts_and_paid_eos(native):
    cipher = ((0,1,0),(0,0))
    for offsets in itertools.product(range(4),range(3)):
        b = bridge(native,cipher,offsets=offsets,contexts=(1,2))
        reference = FixedKeyBridge(cipher,native.probabilities,native.transitions,glyphs=2,rho=.25,
            offsets=offsets,contexts=(1,2))
        for cuts,closed in bridge_schedule(b.lengths):
            np.testing.assert_allclose(b.log_values(KEYS,cuts,closed=closed),
                reference.log_values(KEYS,cuts,closed=closed),rtol=0,atol=2e-14)
    assert b.native_calls==0 and np.all(b.log_values(KEYS,(0,0),closed=True)==0)


def test_mid_unit_witness_removes_only_native_eos(native):
    b = bridge(native,((0,0),))
    key = np.array([[2,2]],dtype=np.int32)
    assert math.exp(b.log_values(key,(1,))[0])==pytest.approx(.75)
    assert math.exp(b.log_values(key,(2,),closed=True)[0])==pytest.approx(3/16)
    assert np.isneginf(bridge(native,((0,),)).log_values(key,(1,),closed=True)[0])


def test_native_closed_graph_and_original_scorer_identical_each_context(native):
    for context in range(3):
        conditional = copy.copy(native)
        conditional.root = context
        for key in KEYS:
            units = tuple("".join("xy"[g] for g in bridge(native).units[k]) for k in key)
            old = conditional.score(units,"xyx",.25)
            b = bridge(native,((0,1,0),),contexts=(context,))
            actual = b.log_values(np.array([key]),(3,),closed=True)[0]
            assert actual==old.log_likelihood
            assert b.nodes==old.reachable_nodes and b.edges==old.edges


def test_native_smc_trajectory_agrees_with_contextual_python_law(native):
    # Equality of the actual algorithm's results is stronger than checking just
    # one returned likelihood. Tolerances apply to log sums, RNG/keys stay fixed.
    ref = FixedKeyBridge(((0,1),(1,0)),native.probabilities,native.transitions,glyphs=2,rho=.25)
    for kernel in ("row","joint"):
        a = run_dictionary_smc(bridge(native),(-1,-1),particles=32,seed=81911,kernel=kernel)
        b = run_dictionary_smc(ref,(-1,-1),particles=32,seed=81911,kernel=kernel)
        assert a["status"]==b["status"]=="complete_particles"
        np.testing.assert_array_equal(a["keys"],b["keys"])
        np.testing.assert_allclose(a["key_log_likelihoods"],b["key_log_likelihoods"],rtol=0,atol=2e-14)
        assert a["log_evidence_estimate"]==pytest.approx(b["log_evidence_estimate"],abs=2e-14)
        assert [r["key_bank_sha256"] for r in a["trace"]]==[r["key_bank_sha256"] for r in b["trace"]]
        # Separate bridge work counters intentionally count different operations.
        ref = FixedKeyBridge(((0,1),(1,0)),native.probabilities,native.transitions,glyphs=2,rho=.25)


def test_source_arrays_pinned_without_copy_or_mutation(native):
    b = bridge(native)
    assert b.p is native.probabilities and b.goto is native.transitions
    assert not b.p.flags.writeable and not b.goto.flags.writeable
    assert b.source_pinned_bytes==native.probabilities.nbytes+native.transitions.nbytes
    altered = copy.copy(native)
    pinned = bridge(altered)
    expected = pinned.log_values(KEYS,(2,2),closed=True)
    altered.probabilities = np.array([[1.,0.]])
    altered.transitions = np.zeros((1,2),dtype=np.uint32)
    altered.function = None
    np.testing.assert_array_equal(pinned.log_values(KEYS,(2,2),closed=True),expected)


def test_work_state_and_edge_caps_raise_never_partial_likelihood(native):
    keys = np.array([[0,1]])
    with pytest.raises(RuntimeError,match="table cap"):
        bridge(native,max_tables=1).log_values(KEYS,(1,0))
    with pytest.raises(RuntimeError,match="node cap"):
        bridge(native,max_nodes_per_record=1).log_values(keys,(2,2),closed=True)
    with pytest.raises(RuntimeError,match="edge cap"):
        bridge(native,max_edges_per_record=1).log_values(keys,(2,2),closed=True)
    with pytest.raises(RuntimeError,match="edge cap"):
        bridge(native,max_edges=1).log_values(keys,(2,2),closed=True)
    with pytest.raises(MemoryError,match="envelope"):
        bridge(native,max_work_bytes=8192).log_values(keys,(1,0))


def test_invalid_contexts_keys_cuts_and_zero_source_rejected_before_access(native):
    for kwargs in ({"contexts":(3,0)},{"offsets":(3,0)},{"rho":False},{"max_tables":True}):
        with pytest.raises(ValueError):
            bridge(native,**kwargs)
    b = bridge(native)
    for keys,cuts,closed in ((np.array([[6,0]]),(1,0),False),(KEYS,(3,0),False),
                             (KEYS,(1,0),True),(np.array([[True,False]]),(1,0),False)):
        with pytest.raises(ValueError):
            b.log_values(keys,cuts,closed=closed)
    altered = copy.copy(native)
    altered.probabilities = np.tile([1.,0.],(3,1))
    altered.probabilities.setflags(write=False)
    altered.probability_pointer = altered.probabilities.ctypes.data_as(type(native.probability_pointer))
    with pytest.raises(ValueError,match="strictly positive"):
        bridge(altered)
    altered.probabilities = native.probabilities.copy()
    altered.probabilities.setflags(write=False)
    with pytest.raises(ValueError,match="pointers"):
        bridge(altered)
