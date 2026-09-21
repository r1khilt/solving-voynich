"""Independent probability/renaming checks for fresh episodic controls."""

import copy
import itertools
import json

import numpy as np
import pytest

from voynich.episodic_data import (
    FAMILIES,
    canonical_raw_probs,
    canonical_targets,
    canonicalize,
    equivalent_state_split,
    make_task,
    oracle_belief,
    oracle_joint,
    oracle_next,
    rename_task,
    sample_task,
    sample_tasks,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_tasks_valid_stationary_reproducible_and_parameter_fresh(family):
    task = make_task(family, 121501)
    assert json.loads(json.dumps(task, allow_nan=False)) == task
    assert task == make_task(family, 121501)
    other = make_task(family, 121502)
    assert task["task_id"] != other["task_id"]
    parameters = {k: v for k, v in task["params"].items() if k != "emission_permutation"}
    other_parameters = {k: v for k, v in other["params"].items() if k != "emission_permutation"}
    assert parameters != other_parameters  # Novel tasks are more than renamed old tasks.
    edge, prior = np.array(task["edge"]), np.array(task["prior"])
    assert edge.min() >= 0 and prior.min() >= 0
    np.testing.assert_allclose(edge.sum(axis=(0, 2)), 1, atol=1e-12)
    np.testing.assert_allclose(prior.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(prior @ edge.sum(axis=0), prior, atol=1e-12)
    first = sample_task(task, 10, 12, 121503)
    np.testing.assert_array_equal(first, sample_task(task, 10, 12, 121503))
    assert first.dtype == np.int64 and first.shape == (10, 12)
    assert first.min() >= 0 and first.max() < 4


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_joint_enumeration_marginalization_and_sampling(family):
    task = make_task(family, 122001)
    edge = np.array(task["edge"])
    prefix = sample_task(task, 1, 5, 122002)[0]
    belief = oracle_belief(task, prefix)
    expected = []
    for continuation in itertools.product(range(4), repeat=3):
        weights = belief.copy()
        for token in continuation:
            weights = weights @ edge[token]
        expected.append(weights.sum())
    joint = oracle_joint(task, prefix, 3)
    np.testing.assert_allclose(joint, expected, rtol=1e-12, atol=1e-15)
    assert np.isfinite(joint).all() and (joint >= 0).all()
    np.testing.assert_allclose(joint.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(joint.reshape(4, 4, 4).sum(axis=(1, 2)), oracle_next(task, prefix))
    np.testing.assert_array_equal(oracle_joint(task, prefix, 0), [1.0])
    for first in range(4):
        conditional = joint.reshape(4, 16)[first] / oracle_next(task, prefix)[first]
        np.testing.assert_allclose(conditional, oracle_joint(task, np.r_[prefix, first], 2))
    draws = sample_task(task, 20_000, 2, 122003)
    histogram = np.bincount(draws[:, 0] * 4 + draws[:, 1], minlength=16) / len(draws)
    np.testing.assert_allclose(histogram, oracle_joint(task, [], 2), rtol=0, atol=0.012)


def test_filter_agrees_with_exhaustive_hidden_path_enumeration():
    task = make_task("pair_parity", 123001)
    edge, prior = np.array(task["edge"]), np.array(task["prior"])
    prefix = np.array([0, 2, 3])
    posterior = np.zeros(len(prior))
    for path in itertools.product(range(len(prior)), repeat=len(prefix) + 1):
        weight = prior[path[0]]
        for t, token in enumerate(prefix):
            weight *= edge[token, path[t], path[t + 1]]
        posterior[path[-1]] += weight
    np.testing.assert_allclose(oracle_belief(task, prefix), posterior / posterior.sum(), atol=1e-12)
    np.testing.assert_array_equal(oracle_belief(task, []), prior)


def test_pair_parity_has_same_one_step_and_different_joint_ready_states():
    task = make_task("pair_parity", 123501)
    counterfactuals = []
    for state in (0, 1):
        altered = copy.deepcopy(task)
        altered["prior"] = np.eye(6)[state].tolist()
        counterfactuals.append(altered)
    np.testing.assert_allclose(oracle_next(counterfactuals[0], []), oracle_next(counterfactuals[1], []))
    distance = np.abs(oracle_joint(counterfactuals[0], [], 2) - oracle_joint(counterfactuals[1], [], 2)).sum()
    assert distance > 1.5


@pytest.mark.parametrize("family", FAMILIES)
def test_split_state_is_observationally_equivalent_for_all_short_prefixes(family):
    task = make_task(family, 124001)
    split = equivalent_state_split(task, 0, 0.31)
    assert split["task_id"] != task["task_id"]
    assert len(split["prior"]) == len(task["prior"]) + 1
    for length in range(3):
        for prefix in itertools.product(range(4), repeat=length):
            np.testing.assert_allclose(oracle_joint(task, prefix, 3), oracle_joint(split, prefix, 3),
                                       rtol=1e-12, atol=1e-15)
            split_belief = oracle_belief(split, prefix)
            aggregate = split_belief[:-1].copy()
            aggregate[0] += split_belief[-1]
            np.testing.assert_allclose(oracle_belief(task, prefix), aggregate, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
def test_task_renaming_preserves_hidden_filter_and_relabels_joints(family):
    task = make_task(family, 124501)
    permutation = np.array([2, 0, 3, 1])
    renamed = rename_task(task, permutation)
    prefix = sample_task(task, 1, 9, 124502)[0]
    np.testing.assert_allclose(oracle_belief(task, prefix), oracle_belief(renamed, permutation[prefix]))
    old = oracle_joint(task, prefix, 2).reshape(4, 4)
    new = oracle_joint(renamed, permutation[prefix], 2).reshape(4, 4)
    np.testing.assert_allclose(new[np.ix_(permutation, permutation)], old)
    restored = rename_task(renamed, np.argsort(permutation))
    assert restored == task


def test_vectorized_sampler_matches_shared_task_sampler_and_mixed_oracles():
    for family in FAMILIES:
        task = make_task(family, 125001)
        np.testing.assert_array_equal(sample_tasks([task] * 17, 15, 125002),
                                      sample_task(task, 17, 15, 125002))
    tasks = [make_task(family, 125101 + i) for i, family in enumerate(FAMILIES)]
    draws = sample_tasks(tasks * 4000, 1, 125201).reshape(4000, len(tasks))
    for column, task in enumerate(tasks):
        histogram = np.bincount(draws[:, column], minlength=4) / len(draws)
        np.testing.assert_allclose(histogram, oracle_next(task, []), rtol=0, atol=0.035)
    assert sample_tasks([], 5, 125202).shape == (0, 5)


def test_canonical_first_occurrence_inverse_and_new_targets():
    raw = np.array([[2, 2, 0, 2, 1, 3], [1, 3, 3, 2, 1, 0]])
    canonical, counts, inverse = canonicalize(raw, 4)
    np.testing.assert_array_equal(canonical, [[0, 0, 1, 0, 2, 3], [0, 1, 1, 2, 0, 3]])
    np.testing.assert_array_equal(counts, [[1, 1, 2, 2, 3, 4], [1, 2, 2, 3, 3, 4]])
    np.testing.assert_array_equal(inverse[0], [[2, -1, -1, -1], [2, -1, -1, -1],
                                             [2, 0, -1, -1], [2, 0, -1, -1],
                                             [2, 0, 1, -1], [2, 0, 1, 3]])
    restored = np.take_along_axis(inverse, canonical[..., None], -1)[..., 0]
    np.testing.assert_array_equal(restored, raw)
    np.testing.assert_array_equal(canonical_targets(raw, 4), [[0, 4, 0, 4, 4], [4, 1, 4, 0, 4]])


def test_canonical_suffix_invariance_even_before_unseen_symbol_appears():
    rng = np.random.default_rng(126001)
    raw = rng.integers(4, size=(5, 12))
    full = canonicalize(raw, 4)
    for t in range(12):
        changed = raw.copy()
        changed[:, t + 1:] = rng.integers(4, size=changed[:, t + 1:].shape)
        altered = canonicalize(changed, 4)
        shortened = canonicalize(raw[:, :t + 1], 4)
        for original, modified, truncated in zip(full, altered, shortened):
            np.testing.assert_array_equal(original[:, :t + 1], modified[:, :t + 1])
            np.testing.assert_array_equal(original[:, :t + 1], truncated)


def test_canonical_renaming_and_raw_prediction_equivariance_all_permutations():
    raw = np.array([[2, 2, 0, 1, 3], [1, 3, 3, 1, 3]])
    canonical, counts, inverse = canonicalize(raw, 4)
    logits = np.random.default_rng(126501).normal(size=(*raw.shape, 5))
    old = canonical_raw_probs(logits, counts, inverse)
    for permutation in itertools.permutations(range(4)):
        permutation = np.array(permutation)
        cnew, nnew, inew = canonicalize(permutation[raw], 4)
        np.testing.assert_array_equal(cnew, canonical)
        np.testing.assert_array_equal(nnew, counts)
        expected_inverse = np.where(inverse >= 0, permutation[np.maximum(inverse, 0)], -1)
        np.testing.assert_array_equal(inew, expected_inverse)
        new = canonical_raw_probs(logits, nnew, inew)
        np.testing.assert_allclose(new[..., permutation], old)


def test_unseen_probability_and_invalid_event_masking():
    logits = np.array([[0., 500., 500., 500., 0.], [500., 500., 500., 500., 0.],
                       [0., 0., 0., 0., 500.]])
    counts = np.array([1, 0, 4])
    inverse = np.array([[2, -1, -1, -1], [-1, -1, -1, -1], [2, 0, 3, 1]])
    actual = canonical_raw_probs(logits, counts, inverse)
    np.testing.assert_allclose(actual, [[1 / 6, 1 / 6, 1 / 2, 1 / 6], [1 / 4] * 4, [1 / 4] * 4])
    np.testing.assert_allclose(actual.sum(axis=-1), 1)
    np.testing.assert_allclose(canonical_raw_probs(np.zeros(5), np.array(0), np.full(4, -1)), [0.25] * 4)


def test_empty_inputs_and_invalid_probabilities_fail_explicitly():
    canonical, counts, inverse = canonicalize(np.empty((3, 0), dtype=np.int64))
    assert canonical.shape == counts.shape == (3, 0) and inverse.shape == (3, 0, 4)
    assert canonical_targets(np.empty((3, 0), dtype=np.int64)).shape == (3, 0)
    task = make_task("iid", 127001)
    assert sample_task(task, 0, 2, 127002).shape == (0, 2)
    assert sample_task(task, 2, 0, 127002).shape == (2, 0)
    for raw in ([[1.2]], [[4]], [[-1]]):
        with pytest.raises(ValueError):
            canonicalize(raw)
    with pytest.raises(ValueError):
        make_task("cycle", 127001, alphabet=3)
    with pytest.raises(ValueError):
        rename_task(task, [0, 0, 1, 2])
    with pytest.raises(ValueError):
        oracle_joint(task, [], 11)
    with pytest.raises(ValueError):
        equivalent_state_split(task, weight=0)
    bad = copy.deepcopy(task)
    bad["edge"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        oracle_next(bad, [])
    impossible = copy.deepcopy(task)
    impossible["edge"] = [[[1]], [[0]], [[0]], [[0]]]
    with pytest.raises(ValueError, match="zero probability"):
        oracle_belief(impossible, [1])
    with pytest.raises(ValueError, match="unique"):
        canonical_raw_probs(np.zeros(5), np.array(2), np.array([1, 1, -1, -1]))
