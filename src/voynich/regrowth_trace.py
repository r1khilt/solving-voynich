"""Fixed-length chain receipts, retaining candidate failures and reference laws."""
import math

import numpy as np


def log_target(sampler, path):
    numerator, denominator = sampler.target_integers(path)
    return math.log(numerator)-math.log(denominator)


def path_receipt(path):
    return {'actions': list(path.actions), 'key': list(path.state.key), 'offsets': list(path.state.offsets),
            'texts': [list(t) for t in path.state.texts], 'counts': list(path.counts),
            'source_terms': [list(t) for t in path.source_terms], 'complete': path.complete,
            'grid_bits': path.grid_bits, 'reference_policy_log_probability': path.policy_log_probability(),
            'not_chain_state_sampling_density': True}


def chain(sampler, initial_actions, *, seed, steps, progress=lambda: None):
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 1:
        raise ValueError('Registered nonnegative seed and finite positive steps required')
    rng = np.random.Generator(np.random.PCG64(seed))
    state = sampler.path(forced_actions=tuple(initial_actions))
    if not state.complete:
        raise ValueError('Complete literal initialization required')
    initial = path_receipt(state)
    initial_target = log_target(sampler, state)
    proposals, accepted, inventory_changes, key_changes = [], 0, 0, 0
    length_changes, failures, blocks, cuts = 0, 0, 0, []
    for index in range(steps):
        before = state
        state, candidate, info = sampler.step(before, rng, progress=progress)
        changed_inventory = (set(before.state.key)-{-1}) != (set(candidate.state.key)-{-1})
        changed_key = before.state.key != candidate.state.key
        changed_length = len(before.actions) != len(candidate.actions)
        accepted += info['accepted']
        inventory_changes += info['accepted'] and changed_inventory
        key_changes += info['accepted'] and changed_key
        length_changes += info['accepted'] and changed_length
        failures += info['failed']
        blocks += info['acceptance_blocks']
        cuts.append(info['cut'])
        proposals.append({'index': index, **info, 'candidate': path_receipt(candidate),
            'candidate_joint_log_potential': log_target(sampler, candidate) if candidate.complete else None,
            'retained_joint_log_potential': log_target(sampler, state),
            'candidate_changed_inventory': changed_inventory, 'candidate_changed_key': changed_key,
            'candidate_changed_length': changed_length})
    summary = {'attempts': steps, 'complete_proposals': steps-failures, 'failed_proposals': failures,
        'accepted': accepted, 'accepted_inventory_changes': inventory_changes, 'accepted_key_changes': key_changes,
        'accepted_length_changes': length_changes, 'root_cuts': cuts.count(0), 'acceptance_raw64_blocks': blocks,
        'initial_joint_log_potential': initial_target, 'final_joint_log_potential': log_target(sampler, state),
        'joint_log_potential_change': log_target(sampler, state)-initial_target}
    return {'seed': seed, 'steps': steps, 'initial': initial, 'proposals': proposals, 'final': path_receipt(state),
            'final_rng_state': rng.bit_generator.state, 'summary': summary}
