import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest
import torch

from voynich.calibrated_suffix_source import DepthSource, Features, calibrate, extract_features, loss_gradient
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def test_uniform_depth_vector_reproduces_existing_source_exactly():
    counts = collect_counts(('abbabaabbabbaa', 'bbab', 'a'), 'abc', 8)
    old = SuffixSource('abc', 8, 256., counts)
    new = DepthSource('abc', [256.] * 8, counts)
    assert np.array_equal(old.probabilities, new.probabilities)
    assert new.to_dict() == DepthSource.from_dict(new.to_dict()).to_dict()


@pytest.mark.parametrize('order', [1, 3, 8, 12])
def test_analytic_gradients_match_autograd_and_central_differences(order):
    counts = collect_counts(('abbaababbaabbaabbaba', 'baaba'), 'abc', order)
    source = SuffixSource('abc', order, 4., counts)
    features = extract_features(source, ('abacbbabbc', 'aaaab', 'c'))
    theta = np.linspace(-.2, 2.3, order)
    loss, gradient = loss_gradient(theta, features)
    variable = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    prediction = torch.tensor(features.root.copy(), dtype=torch.float64)
    for d in range(order):
        mass = variable[d].exp()
        prediction = (torch.tensor(features.next_counts[d].copy()) + mass * prediction) / (
            torch.tensor(features.context_counts[d].copy()) + mass)
    expected = -prediction.log().mean()
    expected.backward()
    assert loss == pytest.approx(expected.item(), abs=2e-15)
    assert gradient == pytest.approx(variable.grad.numpy(), abs=2e-15)
    for d in range(order):
        step = np.eye(order)[d] * 1e-5
        finite = (loss_gradient(theta + step, features)[0] - loss_gradient(theta - step, features)[0]) / 2e-5
        assert gradient[d] == pytest.approx(finite, abs=4e-10)
    new = DepthSource('abc', np.exp(theta), counts)
    assert loss / math.log(2) == pytest.approx(new.bits(('abacbbabbc','aaaab','c')) / 16, abs=2e-15)


def test_feature_extraction_resets_histories_and_does_not_count_calibration_text():
    source = SuffixSource('ab', 3, 4., collect_counts(('aaaabbbb',), 'ab', 3))
    before = source.to_dict()
    f = extract_features(source, ('a','b'))
    assert np.all(f.context_counts == 0) and np.all(f.next_counts == 0)
    assert source.to_dict() == before
    assert np.all(loss_gradient(np.zeros(3), f)[1] == 0)


def test_projected_calibration_is_bounded_deterministic_and_improves_toy_loss():
    source = SuffixSource('ab', 3, 4., collect_counts(('aaaaaaaaabbbaaaaab',), 'ab', 3))
    f = extract_features(source, ('ababababababab', 'baab'))
    first = calibrate(f, 4., steps=80)
    assert first == calibrate(f, 4., steps=80)
    assert first[-1]['loss_nats_per_character'] < first[0]['loss_nats_per_character']
    assert all(math.log(.25) <= x <= math.log(4096.) for r in first for x in r['log_masses'])


def test_nonuniform_source_matches_rational_full_plaintext_enumeration():
    counts = collect_counts(('aabbaab', 'baba'), 'ab', 3)
    source = DepthSource('ab', (.5, 2., 8.), counts)
    def p(history, char):
        if not history:
            return F(2 * counts[''].get(char,0) + 1, 2 * sum(counts[''].values()) + 2)
        row = counts.get(history, {})
        mass = F(source.masses[len(history)-1])
        return (row.get(char,0) + mass * p(history[1:],char)) / (sum(row.values()) + mass)
    for units in (('x','xy'), ('x','x'), ('xx','y')):
        groups = {}
        for size in range(5):
            for chars in itertools.product('ab', repeat=size):
                text = ''.join(chars)
                observed = ''.join(units['ab'.index(c)] for c in text)
                if len(observed)>4:
                    continue
                mass = F(1,4)
                for i,char in enumerate(text):
                    mass *= F(3,4) * p(text[max(0,i-3):i],char)
                groups.setdefault(observed,{})[text]=mass
        for observed, paths in groups.items():
            actual=decode(source,units,observed,.25)
            assert actual.log_likelihood == pytest.approx(math.log(float(sum(paths.values()))),abs=2e-14)
            assert actual.joint_log_probability == pytest.approx(math.log(float(max(paths.values()))),abs=2e-14)
            assert paths[actual.plaintext] == max(paths.values())


def test_invalid_shapes_parameters_and_unknown_source_schema_are_rejected():
    with pytest.raises(ValueError):
        Features(np.ones(2),np.zeros((3,2)),np.zeros((2,2)))
    for mass in (0,True,float('inf'),float('nan')):
        with pytest.raises(ValueError):
            DepthSource('a',[mass],{'':{'a':1}})
    with pytest.raises(ValueError):
        DepthSource.from_dict({'schema_version':1,'kind':'unknown','alphabet':['a'],'masses':[],'counts':{'':{'a':1}}})
    f=Features(np.array([.5]),np.zeros((2,1)),np.zeros((2,1)))
    with pytest.raises(ValueError):
        loss_gradient([0.],f)
    with pytest.raises(ValueError):
        calibrate(f,4.,steps=-1)
