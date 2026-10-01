"""Actual native cap boundaries, including equal per-record/global constraints."""

import numpy as np
import pytest

from tests import test_censored_context001 as fixture_support
from voynich.native_censored_bridge import CensoredNativeBridge
from voynich.native_suffix_marginal import NativeMarginal
from voynich.strict_censored_bridge import CensoredWorkBudgetExceeded,StrictCensoredNativeBridge


environment = fixture_support.environment


@pytest.mark.parametrize('record_limit,total_limit,global_failure',[(100,1,True),(1,100,False),(1,1,True)])
def test_per_record_and_global_native_limits_have_different_failure_types(environment,record_limit,total_limit,global_failure):
    model,build = environment
    native = NativeMarginal(model,build)
    bridge = StrictCensoredNativeBridge(((0,0),),native,max_edges=total_limit,
        max_edges_per_record=record_limit)
    keys = np.zeros((1,23),dtype=np.int32)
    if global_failure:
        with pytest.raises(CensoredWorkBudgetExceeded,match='cumulative edge budget'):
            bridge.log_values(keys,(2,),closed=True)
        assert bridge.edges>=total_limit
    else:
        with pytest.raises(RuntimeError,match='Exact lattice edge cap') as error:
            bridge.log_values(keys,(2,),closed=True)
        assert not isinstance(error.value,CensoredWorkBudgetExceeded) and bridge.edges<total_limit


def test_adapter_keeps_uncapped_scores_and_counters_identical(environment):
    model,build = environment
    native = NativeMarginal(model,build)
    old,new = [cls(((0,1),(1,0)),native) for cls in (CensoredNativeBridge,StrictCensoredNativeBridge)]
    keys = np.random.default_rng(91411).integers(42,size=(16,23),dtype=np.int32)
    for cuts,closed in (((1,0),False),((2,2),False),((2,2),True)):
        np.testing.assert_array_equal(old.log_values(keys,cuts,closed=closed),new.log_values(keys,cuts,closed=closed))
    assert (old.tables,old.edges,old.nodes,old.native_calls)==(new.tables,new.edges,new.nodes,new.native_calls)
