"""Unit tests for EXP-0020 edge-emitting channel (no holdout scores)."""

import numpy as np
import torch

from voynich.edge_emitting import (
    ALPHABET,
    EdgeEmittingChannel,
    equivalent_pair,
    keyed_cycle_null,
    keyed_delayed_parity,
    oracle_next_marginal,
    sample_edge_process,
    sample_family,
)


def test_edge_row_normalization_cycle_and_parity():
    perm = np.arange(ALPHABET)
    for builder in (keyed_cycle_null, keyed_delayed_parity):
        edge, prior = builder(perm)
        np.testing.assert_allclose(edge.sum(axis=(0, 2)), 1.0, atol=1e-9)
        assert np.all(edge >= 0)
        np.testing.assert_allclose(prior.sum(), 1.0, atol=1e-9)


def test_equivalent_generators_agree_on_next_marginals():
    edge_min, prior_min, edge_red, prior_red = equivalent_pair()
    rng = np.random.default_rng(0)
    data = sample_edge_process(edge_min, prior_min, 20, 40, rng)
    for row in data:
        prefix = row[:32]
        p0 = oracle_next_marginal(edge_min, prior_min, prefix)
        p1 = oracle_next_marginal(edge_red, prior_red, prefix)
        np.testing.assert_allclose(p0, p1, atol=1e-8)


def test_model_probability_validity_and_closure():
    model = EdgeEmittingChannel(4)
    with torch.no_grad():
        model.initial_logits.normal_(0, 0.5)
        model.edge_logits.normal_(0, 0.5)
    assert model.probability_validity()["ok"]
    assert model.update_closure_error(n_beliefs=16, seed=3) < 1e-5


def test_score_continuation_matches_sequential_product():
    model = EdgeEmittingChannel(2)
    with torch.no_grad():
        model.initial_logits.zero_()
        model.edge_logits.zero_()  # uniform
    belief = model.initial_belief(1)
    symbols = torch.tensor([[0, 1, 2]], dtype=torch.long)
    joint = float(model.score_continuation(belief, symbols)[0].detach())
    # uniform over 4 symbols each step under K>=1 with flat logits: each step 1/4
    assert abs(joint - 3 * np.log(0.25)) < 1e-5


def test_sample_families_shapes():
    perm = np.arange(ALPHABET)
    rng = np.random.default_rng(1)
    for family in ("iid", "cycle_null", "delayed_parity", "copy_lag", "equivalent"):
        data = sample_family(family, perm, 8, 40, rng)
        assert data.shape == (8, 40)
        assert data.min() >= 0 and data.max() < ALPHABET
