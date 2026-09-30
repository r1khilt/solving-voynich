import itertools
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_suffix_reader002 import infer, probability, reading
from scripts.run_suffix_reader002 import source_interval, window_offsets
from voynich.calibrated_suffix_source import DepthSource
from voynich.sparse_suffix_source import collect_counts, decode


def test_source_calibration_and_selection_do_not_join_body_or_role_boundaries():
    payload={'text':'abcdefghijklm','body_boundaries':[4,9,13]}
    assert source_interval(payload,0,7)==['abcd','efg']
    assert source_interval(payload,7,13)==['hi','jklm']
    with pytest.raises(ValueError):
        source_interval(payload,5,4)


def test_new_fixed_allocation_is_disjoint_and_body_contained():
    offsets=[o for pair in window_offsets() for o in pair]
    assert len(offsets)==32 and min(offsets)==20000 and max(offsets)+224==28160
    assert all(18435<=o and o+224<=35839 for o in offsets)
    assert all(a+224<=b for a,b in zip(offsets,offsets[1:]))
    assert max(offsets)+224<40000 and min(offsets)>14784


@pytest.mark.parametrize('masses',[(4.,),(.5,2.,8.),(1.,4.,16.,64.,.5,2.,32.,8.)])
def test_reference_covers_every_small_observation_and_nonuniform_probability(masses):
    counts=collect_counts(('aabbaabbabaababbab','baaab'),'ab',len(masses))
    model=DepthSource('ab',masses,counts)
    raw=model.to_dict()
    for history in ('','aaaabbaa','bbabaaaabbb','aabbaabbabaababbab'):
        for c in 'ab':
            assert probability(raw,history,c)==pytest.approx(model.probabilities[model.state(history),'ab'.index(c)],abs=3e-16)
    for units in (('x','y'),('x','xy'),('xy','xy'),('xx','y')):
        for n in range(5):
            for letters in itertools.product('xy',repeat=n):
                observed=''.join(letters)
                result=decode(model,units,observed,.25)
                total,best,nodes=infer(raw,units,observed,.25)
                assert result.reachable_nodes==nodes
                assert total==pytest.approx(result.log_likelihood,abs=2e-14)
                assert best==pytest.approx(result.joint_log_probability,abs=2e-14)
                assert reading(raw,units,observed,result.plaintext,.25)==pytest.approx(best,abs=2e-14)


def test_reading_reencoding_and_cap_failures_are_not_silently_accepted():
    raw=DepthSource('ab',(1.,2.),collect_counts(('abba',),'ab',2)).to_dict()
    with pytest.raises(ValueError,match='re-encode'):
        reading(raw,('x','y'),'xx','ab',.5)
    with pytest.raises(RuntimeError,match='cap'):
        infer(raw,('x','y'),'xyxy',.5,max_nodes=1)
    assert reading(raw,('x','y'),'z',None,.5)==-math.inf
