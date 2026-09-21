"""Causal numerical controls and independent temporal-failure calibration."""

import itertools
import json

import numpy as np
import pytest
import torch
from torch import nn

from voynich.episodic_causal import (
    Pool, audit, evaluate_basis, evaluate_closure, fisher_score_sketch,
    fit_linear_updates, frozen_model, future_jacobian_bases,
    interchange, joint_distribution, joint_log_distribution, kl_bits,
    orthonormalize, output_basis, select_pairs, validate_splits,
)


class ShiftToy(nn.Module):
    """Known finite register with delayed information absent from current output."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer('unchanged_buffer', torch.tensor(7.))

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, 1, 3, device=device or self.scale.device)

    def readout(self, state):
        value = state[:, 0, 0] * self.scale
        return torch.stack([value, -value], -1)

    def step(self, symbols, state):
        updated = torch.stack([state[:, 0, 1], state[:, 0, 2], 2*symbols.to(state.dtype)-1], -1)[:, None]
        return self.readout(updated), updated

    def encode(self, prefix):
        state = self.initial_state(len(prefix), prefix.device)
        for symbol in prefix.T:
            _, state = self.step(symbol, state)
        return state


def brute_joint(model, state, horizon, alphabet):
    values = []
    for continuation in itertools.product(range(alphabet), repeat=horizon):
        current, probability = state, state.new_ones(len(state))
        for symbol in continuation:
            probability = probability * model.readout(current).softmax(-1)[:, symbol]
            _, current = model.step(torch.full((len(state),), symbol), current)
        values.append(probability)
    return torch.stack(values, -1)


@pytest.mark.parametrize('horizon', [1, 2, 3, 4])
def test_joint_enumeration_matches_prefix_only_brute_force_and_normalizes(horizon):
    model = ShiftToy()
    state = torch.tensor([[[.2, -.8, .5]], [[1., .3, -.1]]], requires_grad=True)
    actual = joint_distribution(model, state, horizon, 2)
    torch.testing.assert_close(actual, brute_joint(model, state, horizon, 2))
    torch.testing.assert_close(actual.sum(-1), torch.ones(len(state)), atol=5e-7, rtol=0)
    actual[:, 0].sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0


def test_equal_immediate_predictions_can_hide_distinct_joint_futures():
    model = ShiftToy()
    states = torch.tensor([[[.2, -.8, .5]], [[.2, .8, .5]]])
    one = joint_distribution(model, states, 1, 2)
    two = joint_distribution(model, states, 2, 2)
    torch.testing.assert_close(one[0], one[1], atol=0, rtol=0)
    assert torch.abs(two[0]-two[1]).sum() > .5
    basis = torch.tensor([[0.], [1.], [0.]])
    changed, _ = interchange(states[:1], states[1:], basis)
    torch.testing.assert_close(joint_distribution(model, changed, 2, 2), two[1:], atol=0, rtol=0)


def test_raw_output_contract_and_enumeration_budget_are_checked():
    model, state = ShiftToy(), torch.zeros(2, 1, 3)
    with pytest.raises(ValueError, match='output dimension'):
        joint_distribution(model, state, 2, 4)
    with pytest.raises(ValueError, match='enumeration'):
        joint_distribution(model, state, 12, 2)
    with pytest.raises(ValueError, match='batched tensor'):
        joint_distribution(model, {'state': state}, 2, 2)


def test_interchange_identity_and_equal_rank_norm_control():
    rng = torch.Generator().manual_seed(59)
    base, donor = [torch.randn(12, 2, 3, generator=rng) for _ in range(2)]
    q, random = [orthonormalize(torch.randn(6, 2, generator=rng)) for _ in range(2)]
    identity, zero = interchange(base, base, q)
    torch.testing.assert_close(identity, base, atol=0, rtol=0)
    assert torch.count_nonzero(zero) == 0
    changed, delta = interchange(base, donor, q)
    torch.testing.assert_close(changed.flatten(1) @ q, donor.flatten(1) @ q, atol=8e-7, rtol=1e-5)
    _, random_delta = interchange(base, donor, random, delta.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(random_delta.norm(dim=-1), delta.norm(dim=-1), atol=5e-7, rtol=1e-5)


def test_backbone_restored_after_exception_including_mixed_modes_and_flags():
    model = ShiftToy()
    model.scale.requires_grad_(False)
    model.train()
    initial = model.scale.detach().clone()
    with pytest.raises(RuntimeError, match='intentional'):
        with frozen_model(model) as status:
            assert not model.training and not model.scale.requires_grad
            with torch.no_grad():
                model.scale.add_(4)
                model.unchanged_buffer.add_(2)
            raise RuntimeError('intentional')
    assert model.training and not model.scale.requires_grad
    torch.testing.assert_close(model.scale, initial, atol=0, rtol=0)
    assert model.unchanged_buffer == 7
    assert not status['parameters_and_buffers_unchanged']
    assert status['parameters_and_buffers_restored']


def test_split_leakage_key_disjointness_and_symbol_ids_are_checked():
    pools = [np.asarray([[i, 0], [i, 1]]) for i in range(3)]
    valid = validate_splits(pools, (['a', 'a'], ['b', 'b'], ['c', 'c']), 4)
    assert len(valid) == 3
    with pytest.raises(ValueError, match='prefix overlap'):
        validate_splits((pools[0], pools[0], pools[2]), (None, None, None), 4)
    with pytest.raises(ValueError, match='groups overlap'):
        validate_splits(pools, (['a', 'a'], ['a', 'a'], ['c', 'c']), 4)
    with pytest.raises(ValueError, match='all three'):
        validate_splits(pools, (['a', 'a'], None, None), 4)
    with pytest.raises(ValueError, match='integer'):
        validate_splits([p.astype(float) for p in pools], (None, None, None), 4)
    with pytest.raises(ValueError, match='Duplicate prefix'):
        validate_splits((np.array([[0, 0], [0, 0]]), pools[1], pools[2]), (None, None, None), 4)


def make_pool(model, prefixes, groups):
    states = model.encode(torch.as_tensor(prefixes))
    joint = joint_log_distribution(model, states, 3, 2)
    pairs = select_pairs(prefixes, groups, joint, 3, 2)
    return Pool(prefixes, groups, states, joint, *pairs)


def test_pairs_stay_within_keys_and_match_current_output_but_separate_future():
    model = ShiftToy()
    prefixes = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1],
                         [0, 1, 0], [0, 0, 0]])
    groups = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    pool = make_pool(model, prefixes, groups)
    assert pool.matched.all()
    assert np.all(groups == groups[pool.donors])
    assert np.all(pool.donors != pool.recipients)
    assert np.all(prefixes[:, -1] == prefixes[pool.donors, -1])
    immediate = model.readout(pool.states).softmax(-1)
    torch.testing.assert_close(immediate, immediate[pool.donors], atol=0, rtol=0)
    assert float(kl_bits(pool.log_joint[pool.donors], pool.log_joint).detach().mean()) > 0


def test_temporal_battery_detects_noncommuting_coordinate_and_full_donor_is_exact():
    model = ShiftToy()
    prefixes = np.array(list(itertools.product(range(2), repeat=3)))
    pool = make_pool(model, prefixes, np.array(['one']*len(prefixes)))
    basis = torch.tensor([[0.], [1.], [0.]])
    kwargs = dict(horizon=3, alphabet=2, batch_size=3)
    learned = evaluate_basis(model, pool, basis, **kwargs)
    full = evaluate_basis(model, pool, None, mode='full_donor', **kwargs)
    assert learned['all_pairs']['recipient_immediate_tv']['mean'] < 1e-7
    assert learned['all_pairs']['commuting_symmetric_kl_bits']['mean'] > .1
    assert abs(full['all_pairs']['donor_kl_h3_bits']['mean']) < 1e-7
    assert full['all_pairs']['commuting_symmetric_kl_bits']['mean'] == 0
    witness = learned['largest_probability_counterexample']
    assert len(witness['continuation']) == 3
    assert 0 <= witness['recipient_index'] < len(prefixes)


def test_independently_fit_affine_closure_can_succeed_or_fail():
    model = ShiftToy().requires_grad_(False)
    rng = torch.Generator().manual_seed(20)
    train, test = [torch.randn(80, 1, 3, generator=rng) for _ in range(2)]
    placeholder = np.zeros((len(train), 3), dtype=np.int64)
    args = (placeholder, np.array(['key']*len(train)))
    blank = np.arange(len(train))
    pools = [Pool(*args, states, joint_log_distribution(model, states, 3, 2), blank, blank, blank,
                  np.ones(len(train), dtype=bool)) for states in (train, test)]
    full_basis, partial_basis = torch.eye(3), torch.tensor([[0.], [1.], [0.]])
    results = []
    for basis in (full_basis, partial_basis):
        updates = fit_linear_updates(model, train, basis, 2)
        results.append(evaluate_closure(model, *pools, basis, updates, horizon=2, alphabet=2, batch_size=20))
    assert results[0]['coordinate_mse'] < 1e-7
    assert results[1]['coordinate_mse'] > .1
    assert results[0]['affine_updated_fixed_complement_joint_kl_bits'] < 1e-6


def test_output_control_uses_fit_readout_jacobian_and_discloses_rank():
    model = ShiftToy().requires_grad_(False)
    state = torch.ones(10, 1, 3)
    basis, report = output_basis(model, state, 2)
    assert report['effective_jacobian_rank'] == 1
    assert report['null_completion_dimensions'] == 1
    torch.testing.assert_close(basis.T @ basis, torch.eye(2))
    expected = torch.tensor([1., 0., 0.])
    torch.testing.assert_close((expected @ basis) @ basis.T, expected)


def test_future_fisher_sketch_is_finite_deterministic_and_has_correct_causal_horizon():
    model = ShiftToy().double().requires_grad_(False)
    states = torch.tensor([[[.2, -.4, .7]], [[.4, -.8, .5]]], dtype=torch.float64)
    short = fisher_score_sketch(model, states, horizon=2, alphabet=2, seed=442)
    long = fisher_score_sketch(model, states, horizon=3, alphabet=2, seed=442)
    assert short.shape == long.shape == (16, 3)
    assert torch.isfinite(long).all()
    # The third register enters the prediction only after TWO consumed symbols.
    assert torch.count_nonzero(short[:, 2]) == 0
    assert long[:, 2].square().sum() > .01
    torch.testing.assert_close(long, fisher_score_sketch(model, states, horizon=3, alphabet=2, seed=442),
                               atol=0, rtol=0)
    generator = torch.Generator().manual_seed(442)
    sign = (2*torch.randint(2, (8, 8), generator=generator)[0]-1).to(states)
    fixed_weight = joint_distribution(model, states[:1], 3, 2).sqrt()[0]
    numeric = []
    for coordinate in range(3):
        plus, minus = states[:1].clone(), states[:1].clone()
        plus[0, 0, coordinate] += 1e-5
        minus[0, 0, coordinate] -= 1e-5
        a = (sign*fixed_weight*joint_log_distribution(model, plus, 3, 2)[0]).sum()
        b = (sign*fixed_weight*joint_log_distribution(model, minus, 3, 2)[0]).sum()
        numeric.append((a-b)/2e-5)
    torch.testing.assert_close(long[0]*4, torch.stack(numeric), atol=1e-9, rtol=1e-6)


def test_future_and_delayed_bases_are_orthogonal_with_bounded_fit_only_sketch():
    model = ShiftToy().requires_grad_(False)
    states = torch.linspace(-1, 1, 36).reshape(12, 1, 3)
    snapshot = states.clone()
    output = torch.tensor([[1.], [0.], [0.]])
    bases, report = future_jacobian_bases(model, states, 1, output, alphabet=2, seed=443)
    assert report['fit_state_count'] == 8 and report['sketches_per_state'] == 8
    assert report['sqrt_probability_weights_detached']
    assert report['delayed_jacobian']['output_span_overlap'] < 1e-10
    for name, basis in bases.items():
        torch.testing.assert_close(basis.T @ basis, torch.eye(1), atol=2e-6, rtol=0)
        assert 0 <= report[name]['split_half_subspace_overlap'] <= 1.000001
    torch.testing.assert_close(states, snapshot, atol=0, rtol=0)
    with pytest.raises(ValueError, match='bounded'):
        fisher_score_sketch(model, states, max_states=9)


def test_small_audit_has_honest_controls_artifacts_and_preserves_model_and_rng():
    model, untrained = ShiftToy(), ShiftToy()
    model.train()
    model.scale.grad = torch.tensor(9.)
    torch_rng = torch.random.get_rng_state().clone()
    prefixes = np.array(list(itertools.product(range(2), repeat=6)))
    pools = (prefixes[:16], prefixes[16:32], prefixes[32:48])
    report, artifacts = audit(model, *pools, alphabet=2, rank=1, steps=2, horizon=2, batch_size=8,
                              train_groups=['fit']*16, validation_groups=['dev']*16,
                              test_groups=['final']*16, untrained_model=untrained)
    assert not report['causal_selectivity_claim']
    assert report['untrained_control']['status'] == 'completed'
    assert report['key_disjointness_verified']
    assert report['model_restoration']['parameters_and_buffers_unchanged']
    assert report['numerical_controls']['identity_joint_max_abs'] == 0
    assert set(report['confirmation']) >= {'learned', 'shuffled_targets', 'output_jacobian',
                                          'pca', 'random_norm_0', 'unchanged', 'full_donor',
                                          'future_jacobian', 'delayed_jacobian'}
    metric = report['confirmation']['learned']['all_pairs']['donor_kl_h2_bits']
    assert set(metric['per_key_means']) == {'final'}
    assert metric['per_key_counts'] == {'final': 16}
    assert artifacts['learned_basis'].shape == (3, 1)
    assert artifacts['learned_affine_updates'].shape == (2, 2, 1)
    assert all(value.device.type == 'cpu' for value in artifacts.values())
    assert model.training and model.scale.requires_grad and model.scale.grad == 9
    assert torch.equal(torch_rng, torch.random.get_rng_state())
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('kind', ['gru', 'signed_delta'])
def test_audit_runs_against_actual_raw_recurrent_interfaces(kind):
    from voynich.episodic_models import ModelSpec, build_model

    model = build_model(ModelSpec(kind=kind, alphabet=4, width=8, layers=2))
    rng = np.random.default_rng(172)
    pools = [rng.integers(4, size=(8, 12)) for _ in range(3)]
    report, artifacts = audit(model, *pools, rank=2, steps=1, horizon=2, batch_size=4)
    assert report['state_dimension'] == 16
    assert not report['key_disjointness_verified']
    assert report['untrained_control']['status'] == 'not_run'
    assert report['model_restoration']['parameters_and_buffers_unchanged']
    assert report['numerical_controls']['maximum_joint_normalization_error'] < 1e-5
    assert artifacts['learned_basis'].shape == (16, 2)
    json.dumps(report, allow_nan=False)
