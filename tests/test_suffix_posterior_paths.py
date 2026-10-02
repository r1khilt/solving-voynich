"""Literal full-source-string sums independently check conditional path laws."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

from voynich.suffix_posterior_paths import SuffixPosteriorPaths


class Source:
    alphabet = ('a','b')

    def __init__(self,contextual=False,zero=False):
        self.rows = ((F(2,3),F(1,3)),(F(1,4),F(3,4)),(F(3,4),F(1,4))) if contextual else ((F(2,3),F(1,3)),)*3
        if zero:
            self.rows = ((F(1),F(0)),)*3
        self.probabilities = np.array([[float(x) for x in row] for row in self.rows],dtype=np.float64)
        self.transitions = np.array([[1,2]]*3,dtype=np.uint32)
        self.probabilities.setflags(write=False)
        self.transitions.setflags(write=False)

    def state(self,_):
        return 0

    def probability(self,text):
        result,state = F(1,3)*F(2,3)**len(text),0
        for letter in text:
            result *= self.rows[state]['ab'.index(letter)]
            state = 1+'ab'.index(letter)
        return result


def law(source,units,record):
    result = {}
    for n in range(len(record)+1):
        for letters in itertools.product('ab',repeat=n):
            text = ''.join(letters)
            if ''.join(units['ab'.index(c)] for c in text)==record and source.probability(text):
                result[text] = source.probability(text)
    return result


def graph_law(lattice):
    result,agenda = {},[(0,'',1.)]
    while agenda:
        node,text,mass = agenda.pop()
        if not lattice.outgoing[node]:
            result[text] = mass
        else:
            agenda.extend((nxt,text+lattice.alphabet[a],mass*p) for a,nxt,p in lattice.choices(node))
    return result


@pytest.mark.parametrize('contextual',[False,True])
def test_all_unit_keys_and_short_records_match_complete_rational_path_distributions(contextual):
    source = Source(contextual)
    for units in itertools.product(('0','1','00','01','10','11'),repeat=2):
        for record in (''.join(c) for n in range(4) for c in itertools.product('01',repeat=n)):
            expected = law(source,units,record)
            lattice = SuffixPosteriorPaths(source,units,record,1/3)
            assert lattice.probabilities is source.probabilities and lattice.transitions is source.transitions
            if not expected:
                assert lattice.log_likelihood==-math.inf and lattice.sample_paths(draws=7,seed=123)==()
                continue
            total = sum(expected.values())
            assert lattice.log_likelihood==pytest.approx(math.log(float(total)),abs=1e-13)
            assert graph_law(lattice)==pytest.approx({k:float(v/total) for k,v in expected.items()},abs=1e-13)
            sampled = lattice.sample_paths(draws=7,seed=123)
            assert sampled==lattice.sample_paths(draws=7,seed=123)
            for path in sampled:
                assert path.text in expected
                assert path.joint_log_probability==pytest.approx(math.log(float(expected[path.text])),abs=1e-13)
                assert path.conditional_log_probability==pytest.approx(math.log(float(expected[path.text]/total)),abs=1e-13)
                assert ''.join(units['ab'.index(c)] for c in path.text)==record


def test_empty_zero_probability_dead_branches_and_zero_draws():
    source = Source(zero=True)
    empty = SuffixPosteriorPaths(source,('0','1'),'',1/3)
    assert empty.sample_paths(draws=2,seed=0)[0].text==''
    impossible = SuffixPosteriorPaths(source,('0','1'),'1',1/3)
    assert impossible.log_likelihood==-math.inf and impossible.sample_paths(draws=2,seed=0)==()
    overlapping = SuffixPosteriorPaths(Source(),('0','01'),'01',1/3)
    # Matching a first glyph is insufficient: its continuation can be impossible.
    assert all(p.text=='b' for p in overlapping.sample_paths(draws=32,seed=7))
    assert overlapping.sample_paths(draws=0,seed=1)==()
    with pytest.raises(ValueError):
        overlapping.choices(1)  # Dead continuation for the single-unit branch.


@pytest.mark.parametrize('settings,error',[
    ({'max_nodes':1},RuntimeError),({'max_edges':1},RuntimeError),({'max_work_bytes':1},MemoryError),
])
def test_caps_raise_instead_of_returning_partial_scores(settings,error):
    with pytest.raises(error,match='cap'):
        SuffixPosteriorPaths(Source(),('0','0'),'00',1/3,**settings)


@pytest.mark.parametrize('units,record,rho',[
    (('','0'),'0',1/3),(('0',),'0',1/3),(('0','1'),'0',True),
    (('0','1'),'0',0.),(('0','1'),[0],1/3),
])
def test_rejects_invalid_channel_settings(units,record,rho):
    with pytest.raises(ValueError):
        SuffixPosteriorPaths(Source(),units,record,rho)


@pytest.mark.parametrize('draws,seed',[(-1,0),(4097,0),(True,0),(1,-1),(1,True)])
def test_rejects_unbounded_draw_settings(draws,seed):
    lattice = SuffixPosteriorPaths(Source(),('0','1'),'0',1/3)
    with pytest.raises(ValueError):
        lattice.sample_paths(draws=draws,seed=seed)


def test_refuses_mutable_invalid_source_rows_and_bad_reachable_transition():
    source = Source()
    source.probabilities.setflags(write=True)
    with pytest.raises(ValueError,match='immutable'):
        SuffixPosteriorPaths(source,('0','1'),'0',1/3)
    source.probabilities[0] = [.5,.7]
    source.probabilities.setflags(write=False)
    with pytest.raises(ValueError,match='normalized'):
        SuffixPosteriorPaths(source,('0','1'),'0',1/3)
    source = Source()
    source.transitions.setflags(write=True)
    source.transitions[0,0] = 100
    source.transitions.setflags(write=False)
    with pytest.raises(ValueError,match='outside'):
        SuffixPosteriorPaths(source,('0','1'),'0',1/3)


def test_draw_output_byte_budget_fails_before_sampling():
    lattice = SuffixPosteriorPaths(Source(),('0','1'),'0',1/3,max_sample_bytes=1)
    with pytest.raises(MemoryError,match='output cap'):
        lattice.sample_paths(draws=1,seed=0)
    assert lattice.sample_paths(draws=0,seed=0)==()
    with pytest.raises(ValueError):
        SuffixPosteriorPaths(Source(),('0','1'),'0',1/3,max_sample_bytes=True)
