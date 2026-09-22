from types import SimpleNamespace

import numpy as np
import pytest

from voynich.workspace.neuron_campaign import (
    ALTERNATE, contribution_ranking, random_neuron_edit, select_tasks, selected_increment,
)
from voynich.workspace.tasks import build_tasks


def test_neuron_ranking_invariant_to_function_preserving_rescaling():
    changes = np.array([[1., 2., 3.], [2., 1., 3.], [1., 3., 2.], [3., 1., 2.]])
    norms = np.array([2., 3., 4.])
    scale = np.array([100., -.01, 3.])
    order, score, _ = contribution_ranking(changes, norms, list(ALTERNATE))
    scaled_order, scaled_score, _ = contribution_ranking(changes*scale, norms/np.abs(scale), list(ALTERNATE))
    np.testing.assert_array_equal(order, scaled_order)
    np.testing.assert_allclose(score, scaled_score, rtol=1e-12)
    assert np.argmax(np.abs(changes[0])) != np.argmax(np.abs((changes*scale)[0]))


def test_common_relation_ranking_rejects_single_query_spike():
    changes = np.array([[100., 1.], [0., 1.], [0., 1.], [0., 1.]])
    order, score, _ = contribution_ranking(changes, [1., 1.], list(ALTERNATE))
    assert order[0] == 1
    assert score[0] == 0
    with pytest.raises(ValueError, match='four relations'):
        contribution_ranking(changes[:3], [1., 1.], list(ALTERNATE)[:3])


def test_neuron_selection_only_uses_one_paraphrase_and_one_pair_direction():
    training, evaluation = select_tasks(build_tasks(split='development'))
    assert len(training) == 24 and len(evaluation) == 72
    assert all(r['paraphrase_id'] == 0 and r['id'].endswith('/d0') for r in training)
    assert all(r['family'] == 'surface_copy' or r['paraphrase_id'] == 1 for r in evaluation)


def test_random_neuron_control_matches_residual_norm_not_activation_norm():
    weights = np.random.default_rng(2).normal(size=(4, 12)).astype(np.float32)
    mx = SimpleNamespace(array=np.asarray, float32=np.float32, eval=lambda value: None)
    model = SimpleNamespace(mx=mx, layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=weights)))])
    reference = np.array([1., 2., 3., 4.], dtype=np.float32)
    delta, eta, units = random_neuron_edit(model, 0, 3, [0, 1, 2], reference, 7)
    assert not set(units) & {0, 1, 2}
    np.testing.assert_allclose(weights @ eta, delta, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(delta), np.linalg.norm(reference), rtol=1e-6)
    assert np.count_nonzero(eta) == 3


def test_cross_query_delta_distinguishes_shared_from_query_specific_encoding():
    def shared(country, query):
        return np.array([country, query, 1.])
    target = shared(1., 10.) + selected_increment(shared(4., 20.)-shared(1., 20.), [0])
    np.testing.assert_array_equal(target, shared(4., 10.))
    # A query-dependent scale cannot be transferred unchanged across queries.
    def gated(country, query):
        return np.array([country*query, query, 1.])
    wrong = gated(1., 10.) + selected_increment(gated(4., 20.)-gated(1., 20.), [0])
    assert not np.array_equal(wrong, gated(4., 10.))
    with pytest.raises(ValueError, match='Invalid neuron'):
        selected_increment([1., 2.], [0, 0])
