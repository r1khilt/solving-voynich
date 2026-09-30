from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_suffix_reader001 import decompose


def test_mixture_and_conditioned_responsibility_match_hand_fraction_example():
    raw = {'alphabet':list('ab'), 'order':2, 'tau':2.,
           'counts':{'':{'a':4,'b':4}, 'a':{'a':3,'b':1}, 'aa':{'a':2}}}
    weights, posterior, p = decompose(raw, 'baa', 'a')
    assert weights == pytest.approx([1/6, 1/3, 1/2], abs=1e-15)
    assert posterior == pytest.approx([.1, .3, .6], abs=1e-15)
    assert p == pytest.approx(5/6, abs=1e-15)
    weights, posterior, p = decompose(raw, 'aa', 'b')
    assert weights == pytest.approx([1/6, 1/3, 1/2], abs=1e-15)
    assert posterior == pytest.approx([.5, .5, 0], abs=1e-15)
    assert p == pytest.approx(1/6, abs=1e-15)


def test_unseen_history_and_empty_history_reduce_to_root_without_invented_mass():
    raw = {'alphabet':list('ab'), 'order':2, 'tau':2.,
           'counts':{'':{'a':4,'b':4}, 'a':{'a':3,'b':1}, 'aa':{'a':2}}}
    for history in ('', 'abb'):
        weights, posterior, p = decompose(raw, history, 'a')
        assert weights == posterior == [1.,0.,0.]
        assert p == .5
