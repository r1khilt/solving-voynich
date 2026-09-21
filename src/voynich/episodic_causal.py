"""Visible-only predictive-state interventions for EXP-0012.

This battery explains a frozen raw-symbol recurrent predictor. Donor-distribution
fitting is not a latent-variable counterfactual and cannot identify a generator.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np
import torch


def orthonormalize(value):
    """Small CPU QR remains differentiable and does not require MPS QR support."""
    q, r = torch.linalg.qr(value.cpu(), mode='reduced')
    signs = torch.where(r.diag() < 0, -1., 1.).to(q)
    return (q * signs).to(value.device)


def joint_log_distribution(model, state, horizon, alphabet=4):
    """Enumerate normalized AR log probabilities in lexicographic string order.

    ``readout(state)`` predicts the next symbol; ``step(a,state)`` consumes a.
    No suffix is fed before its probability is recorded. The complete state
    must be a batched tensor; canonical symbol-map side state is unsupported.
    """
    if not isinstance(state, torch.Tensor) or state.ndim < 2:
        raise ValueError('Complete recurrent state must be a batched tensor')
    if not 1 <= horizon <= 6 or alphabet < 2 or alphabet ** horizon > 65536:
        raise ValueError('Exact enumeration requires 1<=horizon<=6 and at most 65536 strings')
    batch, shape = state.shape[0], state.shape[1:]
    current = state
    log_weight = state.new_zeros(batch)
    for depth in range(horizon):
        logits = model.readout(current)
        if logits.shape != (current.shape[0], alphabet):
            raise ValueError('Raw model output dimension must equal alphabet; map side state cannot be omitted')
        log_weight = (log_weight[:, None] + logits.log_softmax(-1)).reshape(-1)
        if depth + 1 < horizon:
            symbols = torch.arange(alphabet, device=state.device).repeat(current.shape[0])
            parents = current[:, None].expand(-1, alphabet, *shape).reshape(-1, *shape)
            _, current = model.step(symbols, parents)
    return log_weight.reshape(batch, alphabet ** horizon)


def joint_distribution(model, state, horizon, alphabet=4):
    return joint_log_distribution(model, state, horizon, alphabet).exp()


def kl_bits(target_logp, predicted_logp):
    """Per-example KL, retaining tiny roundoff instead of clipping evidence."""
    terms = torch.where(torch.isfinite(target_logp),
                        target_logp.exp() * (target_logp - predicted_logp), 0.)
    return terms.sum(-1) / math.log(2)


def prefix_log_marginal(logp, length, horizon, alphabet):
    return logp.reshape(logp.shape[0], alphabet ** length, alphabet ** (horizon-length)).logsumexp(-1)


def interchange(recipient, donor, basis, displacement_norm=None):
    """Exchange Q coordinates; optional norm matching is a perturbation control."""
    if recipient.shape != donor.shape or basis.shape[0] != recipient[0].numel():
        raise ValueError('State and basis dimensions disagree')
    flat = recipient.flatten(1)
    delta = ((donor.flatten(1) - flat) @ basis) @ basis.T
    if displacement_norm is not None:
        norms = delta.norm(dim=-1, keepdim=True)
        # A zero projection has no direction; record its mismatch rather than
        # silently inventing a direction outside the specified control span.
        delta = delta * displacement_norm / norms.clamp_min(1e-12)
    return (flat + delta).reshape_as(recipient), delta


@contextmanager
def frozen_model(model):
    """Restore weights, buffers, individual modes and gradient flags on failure too."""
    snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    status = {}
    model.eval().requires_grad_(False)
    try:
        yield status
    finally:
        current = model.state_dict()
        unchanged = all(torch.equal(value, current[name]) for name, value in snapshot.items())
        status['parameters_and_buffers_unchanged'] = unchanged
        model.load_state_dict(snapshot)
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        status['parameters_and_buffers_restored'] = all(
            torch.equal(value, model.state_dict()[name]) for name, value in snapshot.items())


@dataclass
class Pool:
    prefixes: np.ndarray
    groups: np.ndarray
    states: torch.Tensor
    log_joint: torch.Tensor
    recipients: np.ndarray
    donors: np.ndarray
    wrong_donors: np.ndarray
    matched: np.ndarray


def _digest(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def validate_splits(prefixes, groups, alphabet):
    """Forbid repeated contexts or named keys across independent analysis pools."""
    all_seen, seen_groups = set(), set()
    supplied = [g is not None for g in groups]
    if any(supplied) and not all(supplied):
        raise ValueError('Supply groups for all three pools, or explicitly use single-key pools')
    result = []
    for split, (values, labels) in enumerate(zip(prefixes, groups, strict=True)):
        raw = np.asarray(values)
        if raw.ndim != 2 or min(raw.shape) < 1 or len(raw) < 2:
            raise ValueError('Each pool needs at least two nonempty equal-length prefixes')
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Prefixes must contain integer symbol IDs')
        raw = raw.astype(np.int64)
        if raw.min() < 0 or raw.max() >= alphabet:
            raise ValueError('Prefix symbols outside raw alphabet')
        unique = {tuple(row) for row in raw.tolist()}
        if len(unique) != len(raw):
            raise ValueError('Duplicate prefix contexts within an analysis split')
        if all_seen.intersection(unique):
            raise ValueError('Exact prefix overlap across independent analysis splits')
        all_seen.update(unique)
        label = np.full(len(raw), f'asserted-single-key-{split}', dtype=object) if labels is None else np.asarray(labels)
        if label.shape != (len(raw),):
            raise ValueError('Groups must have one label per prefix')
        label = label.astype(str)
        keys = set(label.tolist())
        if all(supplied) and seen_groups.intersection(keys):
            raise ValueError('Task/key groups overlap across analysis splits')
        seen_groups.update(keys)
        if any(np.count_nonzero(label == key) < 2 for key in keys):
            raise ValueError('Each task/key needs at least two contexts for within-key interchange')
        result.append((raw, label))
    return result


def select_pairs(prefixes, groups, log_joint, horizon, alphabet, immediate_tv=.05):
    """Visible-only same-key pairs; matching uses no hidden states or oracle.

    Among equal-final-symbol donors with immediate TV<=threshold choose the
    largest later conditional divergence. Otherwise choose the closest
    immediate prediction and mark the row unmatched. Wrong donors minimize
    joint divergence from the recipient within the same key.
    """
    joint = log_joint.detach().double().cpu().numpy()
    immediate_log = prefix_log_marginal(log_joint.detach().double(), 1, horizon, alphabet).cpu().numpy()
    immediate = np.exp(immediate_log)
    donors, wrong, matched = [], [], []
    for i in range(len(prefixes)):
        candidates = np.flatnonzero((groups == groups[i]) & (np.arange(len(groups)) != i))
        tv = .5 * np.abs(immediate[candidates] - immediate[i]).sum(-1)
        joint_kl = (np.exp(joint[candidates]) * (joint[candidates] - joint[i])).sum(-1) / math.log(2)
        one_kl = (immediate[candidates] * (immediate_log[candidates] - immediate_log[i])).sum(-1) / math.log(2)
        eligible = (prefixes[candidates, -1] == prefixes[i, -1]) & (tv <= immediate_tv)
        if eligible.any():
            choice = np.flatnonzero(eligible)[np.argmax((joint_kl-one_kl)[eligible])]
        else:
            choice = int(np.argmin(tv + (prefixes[candidates, -1] != prefixes[i, -1])))
        donors.append(candidates[choice])
        wrong.append(candidates[np.argmin(joint_kl)])
        matched.append(bool(eligible[choice]))
    return np.arange(len(prefixes)), np.asarray(donors), np.asarray(wrong), np.asarray(matched)


def _encode_pool(model, values, groups, horizon, alphabet, batch_size, device, immediate_tv):
    states, log_joint = [], []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            state = model.encode(torch.as_tensor(values[start:start+batch_size], device=device))
            states.append(state)
            log_joint.append(joint_log_distribution(model, state, horizon, alphabet))
    states, log_joint = torch.cat(states), torch.cat(log_joint)
    if not torch.isfinite(states).all() or not torch.isfinite(log_joint).all():
        raise ValueError('Nonfinite recurrent state or joint prediction')
    pairs = select_pairs(values, groups, log_joint, horizon, alphabet, immediate_tv)
    return Pool(values, groups, states, log_joint, *pairs)


def _random_basis(dimension, rank, seed, reference):
    generator = torch.Generator().manual_seed(seed)
    return orthonormalize(torch.randn(dimension, rank, generator=generator).to(reference))


def fit_basis(model, train, validation, *, rank, steps, seed, horizon, alphabet,
              batch_size=32, learning_rate=.025, shuffled=False):
    """Optimize only a QR basis; validation selects among fixed checkpoint steps."""
    raw = torch.nn.Parameter(_random_basis(train.states[0].numel(), rank, seed, train.states))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    rng = np.random.default_rng(seed)
    # Shuffle donor-distribution assignments within key, preserving key support.
    target_order = train.donors.copy()
    validation_target_order = validation.donors.copy()
    if shuffled:
        for pool, order in ((train, target_order), (validation, validation_target_order)):
            for key in np.unique(pool.groups):
                indexes = np.flatnonzero(pool.groups == key)
                order[indexes] = order[rng.permutation(indexes)]

    @torch.no_grad()
    def score(q):
        losses = []
        for start in range(0, len(validation.recipients), batch_size):
            chosen = validation.recipients[start:start+batch_size]
            changed, _ = interchange(validation.states[chosen], validation.states[validation.donors[chosen]], q)
            prediction = joint_log_distribution(model, changed, horizon, alphabet)
            losses.append(kl_bits(validation.log_joint[validation_target_order[chosen]], prediction))
        return float(torch.cat(losses).mean())

    best, best_step = orthonormalize(raw).detach().clone(), 0
    best_loss = score(best)
    history = [{'step': 0, 'validation_kl_bits': best_loss}]
    for step in range(1, steps+1):
        ids = rng.integers(len(train.recipients), size=min(batch_size, len(train.recipients)))
        q = orthonormalize(raw)
        changed, _ = interchange(train.states[ids], train.states[train.donors[ids]], q)
        prediction = joint_log_distribution(model, changed, horizon, alphabet)
        loss = kl_bits(train.log_joint[target_order[ids]], prediction).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all() or not torch.isfinite(loss):
            raise ValueError('Nonfinite basis optimization')
        optimizer.step()
        if step % max(1, steps//5) == 0 or step == steps:
            candidate = orthonormalize(raw).detach()
            measured = score(candidate)
            history.append({'step': step, 'validation_kl_bits': measured})
            if measured < best_loss:
                best_loss, best, best_step = measured, candidate.clone(), step
    return best, {'selected_step': best_step, 'steps': steps, 'shuffled_targets': shuffled,
                  'validation_history': history, 'target_assignment_changed_fraction':
                  float(np.mean(target_order != train.donors)),
                  'validation_target_assignment_changed_fraction':
                  float(np.mean(validation_target_order != validation.donors))}


def output_basis(model, states, rank):
    """Equal-rank span of centered readout Jacobians on fit states only.

    LayerNorm makes the effective readout state dependent; raw head weights
    alone would be an inaccurate control. Rank-deficient spans are completed
    by SVD's null directions and their effective rank is disclosed.
    """
    rows = []
    for state in states[:min(16, len(states))]:
        current = state.detach().clone().requires_grad_(True)
        logits = model.readout(current[None]).squeeze(0)
        contrasts = logits - logits.mean()
        for index in range(len(contrasts)):
            grad, = torch.autograd.grad(contrasts[index], current, retain_graph=True)
            rows.append(grad.flatten().detach().cpu())
    matrix = torch.stack(rows)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
    effective = int((singular > tolerance).sum())
    return vh[:rank].T.to(states), {'effective_jacobian_rank': effective,
                                   'null_completion_dimensions': max(0, rank-effective),
                                   'fit_state_count': min(16, len(states))}


def _pca_basis(states, rank):
    flat = states.flatten(1).cpu()
    _, _, vh = torch.linalg.svd(flat-flat.mean(0), full_matrices=True)
    return vh[:rank].T.to(states)


def fisher_score_sketch(model, states, *, horizon=3, alphabet=4, seed=12099,
                        max_states=8, sketches=8):
    """Fit-only Rademacher sketches of sqrt(p) times the log-probability Jacobian.

    Weights sqrt(p) are detached. The expected sketch outer product is the
    Fisher matrix of the full joint future law with respect to the current
    recurrent state. Differentiating sqrt(p) too would define another operator.
    """
    if not 1 <= max_states <= 8 or not 1 <= sketches <= 8:
        raise ValueError('Fisher sketch is bounded to eight states and eight signs per state')
    generator = torch.Generator().manual_seed(seed)
    rows = []
    indexes = np.linspace(0, len(states)-1, min(max_states, len(states)), dtype=np.int64)
    for index in indexes:
        state = states[int(index)]
        current = state.detach().clone().requires_grad_(True)
        logp = joint_log_distribution(model, current[None], horizon, alphabet).squeeze(0)
        weight = (.5*logp).exp().detach()
        signs = (2*torch.randint(2, (sketches, len(logp)), generator=generator)-1).to(logp)
        for index, sign in enumerate(signs):
            gradient, = torch.autograd.grad((sign*weight*logp).sum(), current,
                                             retain_graph=index+1 < sketches)
            rows.append(gradient.flatten().detach().cpu())
    matrix = torch.stack(rows) / math.sqrt(len(rows))
    if not torch.isfinite(matrix).all():
        raise ValueError('Nonfinite future Fisher sensitivity sketch')
    return matrix


def _sensitivity_basis(matrix, rank, complement=None):
    """SVD within an optional explicit complement, including null completion."""
    reduced = matrix if complement is None else matrix @ complement
    _, singular, vh = torch.linalg.svd(reduced, full_matrices=True)
    basis = vh[:rank].T
    if complement is not None:
        basis = complement @ basis
    tolerance = max(reduced.shape) * torch.finfo(reduced.dtype).eps * singular.max()
    effective = int((singular > tolerance).sum())
    captured = float(singular[:rank].square().sum() / singular.square().sum().clamp_min(1e-30))
    middle = len(reduced)//2
    halves = [torch.linalg.svd(part, full_matrices=True).Vh[:rank].T
              for part in (reduced[:middle], reduced[middle:])]
    stability = float((halves[0].T @ halves[1]).square().sum()/rank)
    return basis, {'effective_sensitivity_rank': effective,
                   'null_completion_dimensions': max(0, rank-effective),
                   'fraction_sketch_energy_captured': captured,
                   'split_half_subspace_overlap': stability,
                   'leading_singular_values': singular[:min(8, len(singular))].tolist()}


def future_jacobian_bases(model, states, rank, output_control_basis, *, alphabet=4, seed=12099):
    """Two bounded sensitivity candidates; every input state must come from fit."""
    matrix = fisher_score_sketch(model, states, alphabet=alphabet, seed=seed)
    future, future_info = _sensitivity_basis(matrix, rank)
    info = {'horizon': 3, 'fit_state_count': min(8, len(states)), 'sketches_per_state': 8,
            'fit_state_indexes': np.linspace(0, len(states)-1, min(8, len(states)), dtype=np.int64).tolist(),
            'seed': seed, 'sqrt_probability_weights_detached': True,
            'future_jacobian': future_info}
    # An explicit complementary coordinate system avoids roundoff singular
    # vectors pointing back into the supposedly excluded immediate span.
    output = output_control_basis.detach().cpu()
    complement = torch.linalg.svd(output.T, full_matrices=True).Vh[output.shape[1]:].T
    if complement.shape[1] < rank:
        info['delayed_jacobian'] = {'status': 'not_run', 'reason': 'Insufficient complement dimension at equal rank.'}
        return {'future_jacobian': future.to(states)}, info
    delayed, delayed_info = _sensitivity_basis(matrix, rank, complement)
    delayed_info.update({'status': 'completed', 'excluded_output_span_dimension': output.shape[1],
                         'output_span_overlap': float((output.T @ delayed).square().sum()/rank),
                         'limitation': 'Excludes only the rank-matched sampled output-control span; not all immediate effects.'})
    info['delayed_jacobian'] = delayed_info
    return {'future_jacobian': future.to(states), 'delayed_jacobian': delayed.to(states)}, info


def fit_linear_updates(model, states, basis, alphabet=4, ridge=1e-3):
    """Independent affine coordinate predictors; no commuting identity is imposed."""
    with torch.no_grad():
        z = (states.flatten(1) @ basis).cpu().double()
        design = torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], -1)
        penalty = torch.eye(design.shape[-1], dtype=z.dtype) * ridge
        penalty[-1, -1] = 0
        coefficients = []
        for symbol in range(alphabet):
            _, updated = model.step(torch.full((len(states),), symbol, device=states.device), states)
            target = (updated.flatten(1) @ basis).cpu().double()
            coefficients.append(torch.linalg.solve(design.T @ design + penalty, design.T @ target))
    return torch.stack(coefficients).to(states)


def _summary(values, groups, mask=None):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if mask is not None:
        values, groups = values[mask], groups[mask]
    if not len(values):
        return {'mean': None, 'equal_key_mean': None, 'rows': 0, 'keys': 0,
                'per_key_means': {}, 'per_key_counts': {}}
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite causal metric')
    key_means = {str(key): float(values[groups == key].mean()) for key in np.unique(groups)}
    means = list(key_means.values())
    return {'mean': float(values.mean()), 'equal_key_mean': float(np.mean(means)),
            'rows': len(values), 'keys': len(means),
            'key_min': min(means), 'key_max': max(means), 'per_key_means': key_means,
            'per_key_counts': {str(key): int(np.count_nonzero(groups == key)) for key in np.unique(groups)}}


@torch.no_grad()
def evaluate_basis(model, pool, basis, *, horizon, alphabet, batch_size,
                   mode='basis', norm_reference=None, updates=True):
    """One-shot patches and finite all-symbol temporal intervention tests."""
    metrics = {f'donor_kl_h{h}_bits': [] for h in range(1, horizon+1)}
    metrics.update({name: [] for name in ('donor_conditional_after_first_bits', 'recipient_immediate_tv',
                    'displacement_norm', 'norm_matching_absolute_error', 'update_donor_kl_bits',
                    'commuting_symmetric_kl_bits', 'wrong_donor_preservation_kl_bits')})
    witness = None
    for start in range(0, len(pool.recipients), batch_size):
        ids = pool.recipients[start:start+batch_size]
        recipient, donor = pool.states[ids], pool.states[pool.donors[ids]]
        reference_norm = None
        if norm_reference is not None:
            _, ref_delta = interchange(recipient, donor, norm_reference)
            reference_norm = ref_delta.norm(dim=-1, keepdim=True)
        if mode == 'unchanged':
            changed = recipient
        elif mode == 'full_donor':
            changed = donor.clone()
        else:
            changed, _ = interchange(recipient, donor, basis, reference_norm)
        predicted = joint_log_distribution(model, changed, horizon, alphabet)
        wanted = pool.log_joint[pool.donors[ids]]
        for h in range(1, horizon+1):
            loss = kl_bits(prefix_log_marginal(wanted, h, horizon, alphabet),
                           prefix_log_marginal(predicted, h, horizon, alphabet))
            metrics[f'donor_kl_h{h}_bits'].append(loss.cpu())
        metrics['donor_conditional_after_first_bits'].append(
            metrics[f'donor_kl_h{horizon}_bits'][-1] - metrics['donor_kl_h1_bits'][-1])
        immediate = prefix_log_marginal(predicted, 1, horizon, alphabet).exp()
        original = prefix_log_marginal(pool.log_joint[ids], 1, horizon, alphabet).exp()
        metrics['recipient_immediate_tv'].append((.5*(immediate-original).abs().sum(-1)).cpu())
        norm = (changed-recipient).flatten(1).norm(dim=-1)
        metrics['displacement_norm'].append(norm.cpu())
        metrics['norm_matching_absolute_error'].append(
            torch.zeros_like(norm).cpu() if reference_norm is None else (norm-reference_norm[:, 0]).abs().cpu())
        # Keep the largest concrete behavior discrepancy, not an attractive
        # latent name: the prefix pair and exact future string can be replayed.
        differences = (wanted.exp()-predicted.exp()).abs()
        maximum, flat_index = differences.flatten().max(0)
        if witness is None or float(maximum) > witness['absolute_probability_gap']:
            row, token = divmod(int(flat_index), alphabet ** horizon)
            digits = []
            for power in reversed(range(horizon)):
                digit, token = divmod(token, alphabet ** power)
                digits.append(digit)
            col = int(flat_index) % (alphabet ** horizon)
            witness = {'recipient_index': int(ids[row]), 'donor_index': int(pool.donors[ids[row]]),
                       'continuation': digits, 'donor_probability': float(wanted[row, col].exp()),
                       'patched_probability': float(predicted[row, col].exp()),
                       'absolute_probability_gap': float(maximum)}
        if updates:
            after_kl, commuting = [], []
            for symbol in range(alphabet):
                symbols = torch.full((len(ids),), symbol, device=recipient.device)
                _, changed_next = model.step(symbols, changed)
                _, recipient_next = model.step(symbols, recipient)
                _, donor_next = model.step(symbols, donor)
                if mode == 'unchanged':
                    reintervened = recipient_next
                elif mode == 'full_donor':
                    reintervened = donor_next
                else:
                    next_reference_norm = None
                    if norm_reference is not None:
                        _, next_delta = interchange(recipient_next, donor_next, norm_reference)
                        next_reference_norm = next_delta.norm(dim=-1, keepdim=True)
                    reintervened, _ = interchange(recipient_next, donor_next, basis, next_reference_norm)
                next_horizon = max(1, horizon-1)
                after = joint_log_distribution(model, changed_next, next_horizon, alphabet)
                actual = joint_log_distribution(model, donor_next, next_horizon, alphabet)
                commuting_target = joint_log_distribution(model, reintervened, next_horizon, alphabet)
                after_kl.append(kl_bits(actual, after))
                commuting.append(.5*(kl_bits(after, commuting_target)+kl_bits(commuting_target, after)))
            weights = prefix_log_marginal(wanted, 1, horizon, alphabet).exp()
            metrics['update_donor_kl_bits'].append((torch.stack(after_kl, -1)*weights).sum(-1).cpu())
            metrics['commuting_symmetric_kl_bits'].append((torch.stack(commuting, -1)*weights).sum(-1).cpu())
            wrong = pool.states[pool.wrong_donors[ids]]
            if mode == 'unchanged':
                preserved = recipient
            elif mode == 'full_donor':
                preserved = wrong
            else:
                wrong_norm = None
                if norm_reference is not None:
                    _, wrong_delta = interchange(recipient, wrong, norm_reference)
                    wrong_norm = wrong_delta.norm(dim=-1, keepdim=True)
                preserved, _ = interchange(recipient, wrong, basis, wrong_norm)
            wrong_log = joint_log_distribution(model, preserved, horizon, alphabet)
            metrics['wrong_donor_preservation_kl_bits'].append(kl_bits(pool.log_joint[ids], wrong_log).cpu())
    values = {name: torch.cat(rows).numpy() for name, rows in metrics.items() if rows}
    return {'all_pairs': {name: _summary(rows, pool.groups) for name, rows in values.items()},
            'immediate_matched_pairs': {name: _summary(rows, pool.groups, pool.matched)
                                        for name, rows in values.items()},
            'largest_probability_counterexample': witness}


@torch.no_grad()
def evaluate_closure(model, fit, test, basis, coefficients, *, horizon, alphabet, batch_size):
    squared_error, squared_scale, behavior, current_sufficiency = [], [], [], []
    mean_state = fit.states.flatten(1).mean(0)
    mean_coordinates = mean_state @ basis
    for start in range(0, len(test.states), batch_size):
        states = test.states[start:start+batch_size]
        z = states.flatten(1) @ basis
        design = torch.cat([z, torch.ones(len(z), 1, device=z.device, dtype=z.dtype)], -1)
        current_rebuilt = (mean_state + (z-mean_coordinates) @ basis.T).reshape_as(states)
        current_sufficiency.append(kl_bits(joint_log_distribution(model, states, horizon, alphabet),
                                           joint_log_distribution(model, current_rebuilt, horizon, alphabet)).cpu())
        for symbol in range(alphabet):
            _, updated = model.step(torch.full((len(states),), symbol, device=states.device), states)
            actual = updated.flatten(1) @ basis
            predicted = design @ coefficients[symbol]
            squared_error.append((actual-predicted).square().mean(-1).cpu())
            squared_scale.append((actual-mean_coordinates).square().mean(-1).cpu())
            rebuilt = (mean_state + (predicted-mean_coordinates) @ basis.T).reshape_as(states)
            behavior.append(kl_bits(joint_log_distribution(model, updated, horizon, alphabet),
                                    joint_log_distribution(model, rebuilt, horizon, alphabet)).cpu())
    mse = float(torch.cat(squared_error).mean())
    scale = float(torch.cat(squared_scale).mean())
    return {'coordinate_mse': mse, 'coordinate_mse_over_fit_mean_baseline': mse/max(scale, 1e-12),
            'current_state_fixed_complement_joint_kl_bits': float(torch.cat(current_sufficiency).mean()),
            'affine_updated_fixed_complement_joint_kl_bits': float(torch.cat(behavior).mean()),
            'symbols_weighted_uniformly': True,
            'interpretation': 'Descriptive held-out affine closure; neither imposed algebraically nor a causal success rule.'}


def _audit_one(model, splits, *, alphabet, rank, steps, seed, horizon, batch_size, immediate_tv):
    device = next(model.parameters()).device
    with frozen_model(model) as restoration:
        pools = [_encode_pool(model, values, groups, horizon, alphabet, batch_size, device, immediate_tv)
                 for values, groups in splits]
        train, validation, test = pools
        dimension = train.states[0].numel()
        if not 1 <= rank < dimension:
            raise ValueError('A selective candidate rank must be positive and smaller than full state dimension')
        learned, fit_report = fit_basis(model, train, validation, rank=rank, steps=steps, seed=seed,
                                        horizon=horizon, alphabet=alphabet, batch_size=batch_size)
        shuffled, shuffle_report = fit_basis(model, train, validation, rank=rank, steps=steps, seed=seed,
                                             horizon=horizon, alphabet=alphabet, batch_size=batch_size,
                                             shuffled=True)
        output, output_info = output_basis(model, train.states, rank)
        bases = {'learned': learned, 'shuffled_targets': shuffled, 'pca': _pca_basis(train.states, rank),
                 'output_jacobian': output,
                 'random_norm_0': _random_basis(dimension, rank, seed+101, train.states),
                 'random_norm_1': _random_basis(dimension, rank, seed+102, train.states),
                 'random_norm_2': _random_basis(dimension, rank, seed+103, train.states)}
        sensitivity_bases, sensitivity_info = future_jacobian_bases(
            model, train.states, rank, output, alphabet=alphabet, seed=seed+201)
        bases.update(sensitivity_bases)
        scores, closure, artifacts = {}, {}, {}
        for mode in ('unchanged', 'full_donor'):
            scores[mode] = evaluate_basis(model, test, None, horizon=horizon, alphabet=alphabet,
                                          batch_size=batch_size, mode=mode)
        for name, basis in bases.items():
            reference = learned if name.startswith('random_norm') else None
            scores[name] = evaluate_basis(model, test, basis, horizon=horizon, alphabet=alphabet,
                                          batch_size=batch_size, norm_reference=reference)
            coefficients = fit_linear_updates(model, train.states, basis, alphabet)
            closure[name] = evaluate_closure(model, train, test, basis, coefficients,
                                             horizon=max(1, horizon-1), alphabet=alphabet, batch_size=batch_size)
            artifacts[f'{name}_basis'] = basis.cpu()
            artifacts[f'{name}_affine_updates'] = coefficients.cpu()
        artifacts['fit_mean_state'] = train.states.flatten(1).mean(0).cpu()
        numerical = {}
        with torch.no_grad():
            state = test.states[:min(8, len(test.states))]
            identity, _ = interchange(state, state, learned)
            numerical['identity_state_max_abs'] = float((identity-state).abs().max())
            numerical['identity_joint_max_abs'] = float((joint_distribution(model, identity, horizon, alphabet)-
                                                         joint_distribution(model, state, horizon, alphabet)).abs().max())
            numerical['basis_orthogonality_max_abs'] = float((learned.T@learned-
                                                             torch.eye(rank, device=device)).abs().max())
            numerical['maximum_joint_normalization_error'] = max(
                float((pool.log_joint.exp().sum(-1)-1).abs().max()) for pool in pools)
            numerical['state_encoding_replay_max_abs'] = float((model.encode(torch.as_tensor(
                test.prefixes[:len(state)], device=device))-state).abs().max())
        pair_reports = {}
        for name, pool in zip(('fit', 'validation', 'confirmation'), pools, strict=True):
            target_one = prefix_log_marginal(pool.log_joint[pool.donors], 1, horizon, alphabet).exp()
            base_one = prefix_log_marginal(pool.log_joint, 1, horizon, alphabet).exp()
            pair_reports[name] = {'prefix_sha256': _digest(pool.prefixes), 'rows': len(pool.prefixes),
                                  'group_sha256': hashlib.sha256(json.dumps(pool.groups.tolist()).encode()).hexdigest(),
                                  'unique_prefixes': len({tuple(row) for row in pool.prefixes.tolist()}),
                                  'keys': len(np.unique(pool.groups)), 'matched_rows': int(pool.matched.sum()),
                                  'mean_donor_recipient_immediate_tv': float(.5*(target_one-base_one).abs().sum(-1).mean()),
                                  'within_key_pairs_verified': bool(np.all(pool.groups == pool.groups[pool.donors])),
                                  'pair_indices_sha256': _digest(np.stack([pool.recipients, pool.donors,
                                                                          pool.wrong_donors], -1)),
                                  'wrong_donor_equals_target_fraction': float(np.mean(pool.wrong_donors == pool.donors))}
            artifacts[f'{name}_pair_indices'] = torch.from_numpy(np.stack([pool.recipients, pool.donors,
                                                                           pool.wrong_donors], -1))
            artifacts[f'{name}_immediate_matched'] = torch.from_numpy(pool.matched)
        report = {'fit': fit_report, 'shuffled_fit': shuffle_report, 'output_control': output_info,
                  'future_sensitivity': sensitivity_info,
                  'state_dimension': dimension, 'pools': pair_reports, 'confirmation': scores,
                  'held_out_affine_closure': closure, 'numerical_controls': numerical}
    report['model_restoration'] = restoration
    return report, artifacts


def audit(model, train_prefixes, validation_prefixes, test_prefixes, *, alphabet=4, rank=4, steps=150,
          seed=12091, train_groups=None, validation_groups=None, test_groups=None, horizon=3,
          batch_size=32, immediate_tv=.05, untrained_model=None):
    """Return JSON-safe report and CPU tensor artifacts; never fit a backbone.

    Group IDs must identify the actual generator/key and be disjoint across
    splits. If omitted the caller asserts one key per pool; key transfer is
    then unverified. Confirmation never chooses a basis, rank, or threshold.
    Supply a freshly initialized same-architecture model for the untrained
    control. No automatic scientific pass/fail or hidden-label claim is made.
    """
    if steps < 0 or batch_size < 1 or not 0 <= immediate_tv <= 1:
        raise ValueError('Invalid bounded fit settings')
    if not 2 <= horizon <= 4:
        raise ValueError('Audit requires a future joint horizon from 2 through 4')
    if untrained_model is not None:
        if untrained_model is model:
            raise ValueError('Untrained control must be a separate model')
        def architecture(module):
            return [(name, tuple(value.shape)) for name, value in module.state_dict().items()]
        if type(untrained_model) is not type(model) or architecture(untrained_model) != architecture(model):
            raise ValueError('Untrained control must have the same architecture')
    start = time.monotonic()
    splits = validate_splits((train_prefixes, validation_prefixes, test_prefixes),
                            (train_groups, validation_groups, test_groups), alphabet)
    kwargs = dict(alphabet=alphabet, rank=rank, steps=steps, seed=seed, horizon=horizon,
                  batch_size=batch_size, immediate_tv=immediate_tv)
    report, artifacts = _audit_one(model, splits, **kwargs)
    report.update({'schema_version': 1, 'seed': seed, 'rank': rank, 'horizon': horizon,
                   'alphabet': alphabet, 'immediate_matching_tv_threshold': immediate_tv,
                   'key_disjointness_verified': train_groups is not None,
                   'scope': 'Visible-only low-rank donor predictive-state transplantation; no latent counterfactual identified.',
                   'causal_selectivity_claim': False,
                   'limitations': ['Teacher distributions are targets, not generator truth.',
                                   'Immediate-output preservation is not an independent semantic nuisance test.',
                                   'Affine closure is descriptive and may fail for valid nonlinear state updates.',
                                   'Interchanged hybrid states may be off the natural state manifold.',
                                   'Prefix/key bootstrap intervals and seed stability require campaign-level aggregation.']})
    if untrained_model is None:
        report['untrained_control'] = {'status': 'not_run', 'reason': 'No independently initialized model supplied.'}
    else:
        control, control_artifacts = _audit_one(untrained_model, splits, **kwargs)
        report['untrained_control'] = {'status': 'completed', 'report': control}
        artifacts.update({f'untrained_{key}': value for key, value in control_artifacts.items()})
    report['elapsed_seconds'] = time.monotonic()-start
    return report, artifacts
