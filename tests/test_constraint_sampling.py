"""Artificial independent selector/construction/full-target/PCG replay of soft moves."""
from fractions import Fraction as F
import math

import numpy as np
import pytest

from scripts.check_constraint_reading001 import independent_distance
from tests.test_constraint_reading import source
from tests.test_exact_radical import quantile_decision
from voynich import constraint_reading as law
from voynich import constraint_sampling as actual
from voynich.exact_radical import RadicalRatio
from voynich.reading_label_landscape import target_fingerprint
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def full_mass(expert, env, state, epsilon):
    # Independent one-shot integer products, including arbitrary source root context.
    numerators, exponent = [], 0
    for text in state.texts:
        context = expert.state('')
        for a in text:
            numerator, denominator = float(expert.probabilities[context, a]).as_integer_ratio()
            assert numerator > 0 and denominator & (denominator-1) == 0
            numerators.append(numerator)
            exponent += denominator.bit_length()-1
            context = int(expert.transitions[context, a])
    n = sum(map(len, state.texts))
    used = len(set(a for text in state.texts for a in text))
    mass = F(math.prod(numerators)*224**n, 225**(n+len(state.texts))*(1 << exponent)*len(env.pool)**used)
    return mass*epsilon**independent_distance(env, state)


def alternative_step(expert, env, state, epsilon, rng):
    kernel = ('rebind', 'relabel', 'boundary')[int(rng.integers(3))]
    metadata = {'kernel': kernel}
    old = full_mass(expert, env, state, epsilon)
    assert old > 0
    if kernel == 'rebind':
        row, unit = int(rng.integers(env.rows)), int(rng.integers(len(env.pool)))
        key = tuple(unit if a == row and k >= 0 else k for a, k in enumerate(state.key))
        candidate = law.ConstraintReading(key, state.widths, state.texts)
        forward = reverse = F(1, env.rows*len(env.pool))
        kind = 'rebind'
        metadata.update({'row': row, 'unit': unit})
    else:
        valid = True
        if kernel == 'relabel':
            anchors = [(r, j) for r, text in enumerate(state.texts) for j in range(len(text))]
            rank = int(rng.integers(len(anchors)))
            record, first = anchors[rank]
            last = first+1
            rows = (int(rng.integers(env.rows)),)
            widths = (state.widths[record][first],)
            forward_factor = reverse_factor = len(anchors)*env.rows
            metadata.update({'anchor': rank, 'record': record, 'index': first, 'row': rows[0]})
        else:
            anchors = [(r, i) for r, obs in enumerate(env.records) for i in range(len(obs)-1)]
            rank = int(rng.integers(len(anchors))) if anchors else 0
            if not anchors:
                valid, record, start = False, None, None
            else:
                record, start = anchors[rank]
                spans, cursor = [], 0
                for j, width in enumerate(state.widths[record]):
                    spans.append((cursor, cursor+width, j))
                    cursor += width
                inside = [s for s in spans if s[0] >= start and s[1] <= start+2]
                valid = bool(inside) and inside[0][0] == start and inside[-1][1] == start+2
            if valid:
                first, last = inside[0][2], inside[-1][2]+1
                widths = (1, 1) if len(inside) == 1 else (2,)
                rows = tuple(int(rng.integers(env.rows)) for _ in widths)
                forward_factor, reverse_factor = len(anchors)*env.rows**len(rows), len(anchors)*env.rows**len(inside)
            else:
                rows = ()
            metadata.update({'rank': rank, 'record': record, 'start': start, 'rows': list(rows)})
        if not valid:
            candidate = state
            forward = reverse = F(1, len(anchors) or 1)
            metadata['bindings'] = []
            kind = 'boundary_identity'
        else:
            outside = {a for r, text in enumerate(state.texts) for j, a in enumerate(text)
                       if r != record or j not in range(first, last)}
            old_fresh = set(state.texts[record][first:last])-outside
            new_fresh = set(rows)-outside
            bindings = [(a, int(rng.integers(len(env.pool)))) for a in sorted(new_fresh)]
            key = [state.key[a] if a in outside else -1 for a in range(env.rows)]
            for a, unit in bindings:
                key[a] = unit
            texts = tuple(t[:first]+rows+t[last:] if r == record else t for r, t in enumerate(state.texts))
            partitions = tuple(w[:first]+widths+w[last:] if r == record else w for r, w in enumerate(state.widths))
            candidate = law.ConstraintReading(tuple(key), partitions, texts)
            forward, reverse = F(1, forward_factor*len(env.pool)**len(new_fresh)), F(
                1, reverse_factor*len(env.pool)**len(old_fresh))
            metadata['bindings'] = [list(pair) for pair in bindings]
            kind = kernel
    new = full_mass(expert, env, candidate, epsilon)
    ratio = new/old*reverse/forward
    if ratio == 0:
        rng.bit_generator.random_raw()
        accepted, blocks = False, 1
    else:
        accepted, blocks = quantile_decision(RadicalRatio(ratio.numerator, ratio.denominator, 1), rng, 16)
    retained = candidate if accepted else state
    def fingerprint(value):
        return target_fingerprint(value.numerator, value.denominator) if value else None
    metadata.update({'kind': kind, 'forward': str(forward), 'reverse': str(reverse),
        'old_violations': independent_distance(env, state), 'candidate_violations': independent_distance(env, candidate),
        'old_target_sha256': fingerprint(old), 'candidate_target_sha256': fingerprint(new),
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)], 'accepted': accepted,
        'acceptance_blocks': blocks, 'changed': retained != state,
        'source_length_delta': sum(map(len, candidate.texts))-sum(map(len, state.texts)),
        'visited_delta': len(set(a for t in candidate.texts for a in t))-len(set(a for t in state.texts for a in t))})
    return retained, candidate, ratio, metadata


def alternative_exchange(expert, env, left, right, ex, ey, rng):
    old = full_mass(expert, env, left, ex)*full_mass(expert, env, right, ey)
    new = full_mass(expert, env, right, ex)*full_mass(expert, env, left, ey)
    ratio = new/old
    if not ratio:
        rng.bit_generator.random_raw()
        accepted, blocks = False, 1
    else:
        accepted, blocks = quantile_decision(RadicalRatio(ratio.numerator, ratio.denominator, 1), rng, 16)
    return (right, left) if accepted else (left, right), ratio, {
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)], 'accepted': accepted,
        'acceptance_blocks': blocks, 'changed': accepted and left != right}


@pytest.mark.parametrize('records', [((0, 1, 0), (1, 0)), ((0,), (1,))])
def test_every_actual_dispatch_draw_target_zero_rejection_and_replica_rng(records):
    expert = source()
    env = ReadingEnvironment(records, rows=3, glyphs=2)
    hard = ReadingState((0, 1, -1), tuple(map(len, records)), records)
    initial = law.from_hard(env, hard)
    epsilons = (F(0), F(1, 8), F(1, 2), F(1))
    for seed in (96521, 96529):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        observed = expected = (initial,)*len(epsilons)
        counters = {'zero_local': 0, 'zero_exchange': 0, 'changed': 0}
        kernels = set()
        for index in range(128):
            observed, locals_, swaps = actual.replica_sweep(expert, env, observed, epsilons, a, index)
            paths, alternative_locals, alternative_swaps = list(expected), [], []
            for slot, epsilon in enumerate(epsilons):
                paths[slot], candidate, ratio, info = alternative_step(expert, env, paths[slot], epsilon, b)
                alternative_locals.append((candidate, ratio, info))
                kernels.add(info['kernel'])
                counters['zero_local'] += ratio == 0
                counters['changed'] += info['changed']
            for slot in range(index % 2, len(epsilons)-1, 2):
                pair, ratio, info = alternative_exchange(expert, env, paths[slot], paths[slot+1],
                                                       epsilons[slot], epsilons[slot+1], b)
                paths[slot], paths[slot+1] = pair
                alternative_swaps.append((slot, ratio, info))
                counters['zero_exchange'] += ratio == 0
            expected = tuple(paths)
            assert observed == expected and locals_ == tuple(alternative_locals) and swaps == tuple(alternative_swaps)
            assert a.bit_generator.state == b.bit_generator.state
            assert independent_distance(env, observed[0]) == 0
            for slot, state in enumerate(observed):
                assert full_mass(expert, env, state, epsilons[slot]) > 0
        assert kernels == {'rebind', 'relabel', 'boundary'} and all(counters.values())


def test_decision_zero_raw_block_forced_boundary_abort_and_argument_guards():
    a, b = np.random.default_rng(96537), np.random.default_rng(96537)
    assert actual.decide(F(0), a) == (False, 1)
    b.bit_generator.random_raw()
    assert a.bit_generator.state == b.bit_generator.state
    raw = int(np.random.default_rng(96541).bit_generator.random_raw())
    boundary = F(2*raw+1, 2**65)
    with pytest.raises(RuntimeError):
        actual.decide(boundary, np.random.default_rng(96541), maximum_blocks=1)
    for ratio in (.5, F(-1)):
        with pytest.raises(ValueError):
            actual.decide(ratio, a)
    with pytest.raises(ValueError):
        actual.decide(F(1), np.random.Generator(np.random.Philox(1)))
    with pytest.raises(ValueError):
        actual.decide(F(1), a, maximum_blocks=True)


def test_original_23row_42unit_artificial_source_and_nonzero_root():
    from tests.test_window_reading_admission import fixture
    expert, sampler, hard = fixture()
    expert.state = lambda _: 5
    env = sampler.env
    initial = law.from_hard(env, hard.state)
    epsilons = (F(0), F(1, 8), F(1, 2), F(1))
    a, b = np.random.default_rng(96543), np.random.default_rng(96543)
    observed = expected = (initial,)*len(epsilons)
    for index in range(64):
        observed, locals_, swaps = actual.replica_sweep(expert, env, observed, epsilons, a, index)
        paths, checks, exchanges = list(expected), [], []
        for slot, epsilon in enumerate(epsilons):
            paths[slot], candidate, ratio, info = alternative_step(expert, env, paths[slot], epsilon, b)
            checks.append((candidate, ratio, info))
        for slot in range(index % 2, len(epsilons)-1, 2):
            pair, ratio, info = alternative_exchange(expert, env, paths[slot], paths[slot+1],
                                                   epsilons[slot], epsilons[slot+1], b)
            paths[slot], paths[slot+1] = pair
            exchanges.append((slot, ratio, info))
        expected = tuple(paths)
        assert observed == expected and locals_ == tuple(checks) and swaps == tuple(exchanges)
        assert a.bit_generator.state == b.bit_generator.state
        for slot, state in enumerate(observed):
            assert law.target(expert, env, state, epsilons[slot]) == full_mass(expert, env, state, epsilons[slot])
