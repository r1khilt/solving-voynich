"""Neural binding-input interventions, preserving the original symbolic task.

Counterfactual feature packets need not describe a valid dictionary state;
legal masks, observed records, preceding actions and target traces stay fixed.
This is input dependence, not a certified latent mechanism or decipherment.
"""

import torch

from voynich.joint_key_proposal import unit_pool


def binding_packet(packed, *, rows, glyphs, mode):
    if (type(rows) is not int or not 1 <= rows <= 64 or type(glyphs) is not int
            or not 1 <= glyphs <= 64 or mode not in ('sham', 'erase', 'rotate')
            or not isinstance(packed, tuple) or len(packed) != 8):
        raise ValueError('Declared bounded packet and sham/erase/rotate intervention required')
    keys, active = packed[1], packed[-2]
    stride = len(unit_pool(glyphs))+1
    if (keys.dtype != torch.long or keys.ndim != 3 or keys.shape[-1] != rows
            or active.dtype != torch.bool or active.shape != keys.shape[:2]):
        raise ValueError('Long binding indices and matching Boolean active-query mask required')
    base = torch.arange(rows, device=keys.device)*stride
    units = keys-base
    if ((units[active] < 0).any() or (units[active] >= stride).any()):
        raise ValueError('Active slots must retain their declared row identities')
    result = keys.clone()
    if mode != 'sham':
        for b, t in active.nonzero().tolist():
            used = (units[b, t] > 0).nonzero().flatten()
            if mode == 'erase':
                result[b, t, used] = base[used]
            elif len(used) > 1:
                result[b, t, used] = base[used]+torch.roll(units[b, t, used], 1)
    return (packed[0], result, *packed[2:])


def action_groups(trace):
    snapshots, actions, _ = trace
    return tuple('first_binding' if state.key[action//2] < 0 else 'reuse'
                 for state, action in zip(snapshots, actions, strict=True))


def first_deviation(environment, trace, prediction, truth_key):
    """Gold-key diagnosis ONLY; never a feature or inference-time repair."""
    actual, expected = tuple(prediction['actions']), tuple(trace[1])
    common = 0
    while common < min(len(actual), len(expected)) and actual[common] == expected[common]:
        common += 1
    if common == len(expected):
        if len(actual) != len(expected):
            raise ValueError('A valid complete truth path cannot have extra actions')
        return {'exact_actions': True, 'shared_prefix_actions': common}
    if common == len(actual):
        raise ValueError('A truth-compatible prefix cannot genuinely dead-end early')
    state = environment.initial
    for action in actual[:common]:
        state = environment.advance(state, action)
    chosen, gold = actual[common], expected[common]
    next_state = environment.advance(state, chosen)
    row = chosen//2
    return {'exact_actions': False, 'shared_prefix_actions': common,
            'teacher_group': 'first_binding' if state.key[gold//2] < 0 else 'reuse',
            'chosen_group': 'first_binding' if state.key[row] < 0 else 'reuse',
            'wrong_source_row': row != gold//2, 'wrong_unit_length': chosen%2 != gold%2,
            'introduced_wrong_dictionary_binding': state.key[row] < 0 and next_state.key[row] != truth_key[row]}
