"""PCG64 shape-only local MH and exact collapsed replica exchanges."""
from fractions import Fraction as F

from voynich import constraint_shape as law
from voynich.constraint_sampling import decide, fingerprint, _rng


def propose(env, state, rng):
    _rng(rng)
    law.validate(env, state)
    kernel = ('relabel', 'boundary')[int(rng.integers(2))]
    info = {'kernel': kernel}
    if kernel == 'relabel':
        anchor = int(rng.integers(sum(map(len, state.texts))))
        record, index = 0, anchor
        while index >= len(state.texts[record]):
            index -= len(state.texts[record])
            record += 1
        row = int(rng.integers(env.rows))
        move = law.relabel(env, state, record, index, row)
        info.update({'anchor': anchor, 'record': record, 'index': index, 'row': row})
    else:
        count = sum(len(record)-1 for record in env.records)
        rank = int(rng.integers(count)) if count else 0
        if count:
            record, start = 0, rank
            while start >= len(env.records[record])-1:
                start -= len(env.records[record])-1
                record += 1
            offsets, offset = {0: 0}, 0
            for index, width in enumerate(state.widths[record]):
                offset += width
                offsets[offset] = index+1
            if start not in offsets or start+2 not in offsets:
                rows = ()
            else:
                size = 2 if offsets[start+2]-offsets[start] == 1 else 1
                rows = tuple(int(rng.integers(env.rows)) for _ in range(size))
        else:
            record, start, rows = None, None, ()
        move = law.boundary(env, state, rank, rows)
        info.update({'rank': rank, 'record': record, 'start': start, 'rows': list(rows)})
    return move, info


def step(source, env, state, epsilon, rng, *, progress=lambda: None):
    _rng(rng)
    progress()
    old = law.target(source, env, state, epsilon)
    if not old:
        raise ValueError('Current shape needs positive target in its layer')
    move, info = propose(env, state, rng)
    progress()
    new = law.target(source, env, move.state, epsilon)
    ratio = new/old*move.reverse/move.forward
    accepted, blocks = decide(ratio, rng)
    retained = move.state if accepted else state
    info.update({'kind': move.kind, 'forward': str(move.forward), 'reverse': str(move.reverse),
        'old_target_sha256': fingerprint(old), 'candidate_target_sha256': fingerprint(new),
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)], 'accepted': accepted,
        'acceptance_blocks': blocks, 'changed': retained != state,
        'source_length_delta': sum(map(len, move.state.texts))-sum(map(len, state.texts)),
        'visited_delta': len({a for t in move.state.texts for a in t})-len({a for t in state.texts for a in t})})
    progress()
    return retained, move.state, ratio, info


def exchange(env, left, right, epsilon_left, epsilon_right, rng):
    ratio = law.swap_ratio(env, left, right, epsilon_left, epsilon_right)
    accepted, blocks = decide(ratio, rng)
    pair = (right, left) if accepted else (left, right)
    return pair, ratio, {'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)],
        'accepted': accepted, 'acceptance_blocks': blocks, 'changed': accepted and left != right}


def replica_sweep(source, env, states, epsilons, rng, index, *, progress=lambda: None):
    if (type(states) is not tuple or type(epsilons) is not tuple or not 2 <= len(epsilons) <= 8
            or len(states) != len(epsilons) or type(index) is not int or not 0 <= index < 4096
            or any(not isinstance(e, F) or not 0 <= e <= 1 for e in epsilons)
            or epsilons[0] != 0 or any(a >= b for a, b in zip(epsilons, epsilons[1:]))):
        raise ValueError('Bounded increasing hard-to-soft constraint ladder and index required')
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
