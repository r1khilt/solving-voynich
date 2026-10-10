"""Preparatory PCG64 dispatcher for qualified relaxed-channel components.

No original-source experiment, production recovery campaign or neural policy is
launched by this module. Zero-target candidates consume one raw64 rejection block.
"""
from fractions import Fraction as F

import numpy as np

from voynich import constraint_reading as law
from voynich.exact_radical import RadicalRatio, exact_radical_bernoulli
from voynich.reading_label_landscape import target_fingerprint


def _rng(rng):
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full-state PCG64 generator required')


def _bindings(env, state, record, first, last, rows, rng):
    outside = set(a for r, text in enumerate(state.texts) for i, a in enumerate(text)
                  if r != record or i < first or i >= last)
    return {a: int(rng.integers(len(env.pool))) for a in sorted(set(rows)-outside)}


def propose(env, state, rng):
    _rng(rng)
    law.validate(env, state)
    kernel = ('rebind', 'relabel', 'boundary')[int(rng.integers(3))]
    info = {'kernel': kernel}
    if kernel == 'rebind':
        row, unit = int(rng.integers(env.rows)), int(rng.integers(len(env.pool)))
        move = law.rebind(env, state, row, unit)
        info.update({'row': row, 'unit': unit})
    elif kernel == 'relabel':
        anchor = int(rng.integers(sum(map(len, state.texts))))
        record, index = 0, anchor
        while index >= len(state.texts[record]):
            index -= len(state.texts[record])
            record += 1
        row = int(rng.integers(env.rows))
        bindings = _bindings(env, state, record, index, index+1, (row,), rng)
        move = law.relabel(env, state, record, index, row, bindings)
        info.update({'anchor': anchor, 'record': record, 'index': index, 'row': row,
                     'bindings': [list(x) for x in sorted(bindings.items())]})
    else:
        count = sum(len(r)-1 for r in env.records)
        rank = int(rng.integers(count)) if count else 0
        if not count:
            move = law.boundary(env, state, rank, (), {})
            info.update({'rank': rank, 'record': None, 'start': None, 'rows': [], 'bindings': []})
        else:
            record, start = 0, rank
            while start >= len(env.records[record])-1:
                start -= len(env.records[record])-1
                record += 1
            offsets, offset = {0: 0}, 0
            for j, width in enumerate(state.widths[record]):
                offset += width
                offsets[offset] = j+1
            if start not in offsets or start+2 not in offsets:
                rows, bindings = (), {}
            else:
                first, last = offsets[start], offsets[start+2]
                size = 2 if last-first == 1 else 1
                rows = tuple(int(rng.integers(env.rows)) for _ in range(size))
                bindings = _bindings(env, state, record, first, last, rows, rng)
            move = law.boundary(env, state, rank, rows, bindings)
            info.update({'rank': rank, 'record': record, 'start': start, 'rows': list(rows),
                         'bindings': [list(x) for x in sorted(bindings.items())]})
    return move, info


def decide(ratio, rng, maximum_blocks=16):
    _rng(rng)
    if (not isinstance(ratio, F) or ratio < 0 or type(maximum_blocks) is not int
            or not 1 <= maximum_blocks <= 16):
        raise ValueError('Nonnegative exact ratio and bounded acceptance blocks required')
    if ratio == 0:
        rng.bit_generator.random_raw()
        return False, 1
    return exact_radical_bernoulli(RadicalRatio(ratio.numerator, ratio.denominator, 1), rng,
                                  maximum_blocks=maximum_blocks)


def fingerprint(value):
    return target_fingerprint(value.numerator, value.denominator) if value else None


def step(source, env, state, epsilon, rng, *, progress=lambda: None):
    """Equal three-kernel mixture; actual component reverse correction stays explicit."""
    _rng(rng)
    progress()
    old = law.target(source, env, state, epsilon)
    if old <= 0:
        raise ValueError('Current state must have positive target in its constraint layer')
    move, info = propose(env, state, rng)
    progress()
    new = law.target(source, env, move.state, epsilon)
    ratio = new/old*move.reverse/move.forward
    accepted, blocks = decide(ratio, rng)
    retained = move.state if accepted else state
    info.update({'kind': move.kind, 'forward': str(move.forward), 'reverse': str(move.reverse),
        'old_violations': law.violations(env, state), 'candidate_violations': law.violations(env, move.state),
        'old_target_sha256': fingerprint(old), 'candidate_target_sha256': fingerprint(new),
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)],
        'accepted': accepted, 'acceptance_blocks': blocks, 'changed': retained != state,
        'source_length_delta': sum(map(len, move.state.texts))-sum(map(len, state.texts)),
        'visited_delta': sum(k >= 0 for k in move.state.key)-sum(k >= 0 for k in state.key)})
    progress()
    return retained, move.state, ratio, info


def exchange(env, left, right, epsilon_left, epsilon_right, rng):
    ratio = law.swap_ratio(env, left, right, epsilon_left, epsilon_right)
    accepted, blocks = decide(ratio, rng)
    retained = (right, left) if accepted else (left, right)
    return retained, ratio, {'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)],
        'accepted': accepted, 'acceptance_blocks': blocks, 'changed': accepted and left != right}


def replica_sweep(source, env, states, epsilons, rng, index, *, progress=lambda: None):
    """Target-invariant local composition and alternating disjoint adjacent exchanges."""
    if (type(states) is not tuple or type(epsilons) is not tuple or not 2 <= len(epsilons) <= 8
            or len(states) != len(epsilons) or type(index) is not int or not 0 <= index <= 4096
            or any(not isinstance(e, F) or not 0 <= e <= 1 for e in epsilons)
            or epsilons[0] != 0 or any(a >= b for a, b in zip(epsilons, epsilons[1:]))):
        raise ValueError('Bounded increasing hard-to-soft constraint ladder and sweep index required')
    _rng(rng)
    retained, locals_, swaps = list(states), [], []
    for slot, epsilon in enumerate(epsilons):
        retained[slot], candidate, ratio, info = step(source, env, retained[slot], epsilon, rng, progress=progress)
        locals_.append((candidate, ratio, info))
    for slot in range(index % 2, len(states)-1, 2):
        progress()
        pair, ratio, info = exchange(env, retained[slot], retained[slot+1], epsilons[slot], epsilons[slot+1], rng)
        retained[slot], retained[slot+1] = pair
        swaps.append((slot, ratio, info))
    return tuple(retained), tuple(locals_), tuple(swaps)
