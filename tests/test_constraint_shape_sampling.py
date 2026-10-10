"""Independent integer marginal, construction, quantile and PCG replay."""
from fractions import Fraction as F
import math

import numpy as np
import pytest

from tests.test_constraint_shape import source
from tests.test_exact_radical import quantile_decision
from tests.test_window_reading_admission import fixture
from voynich import constraint_shape as law
from voynich import constraint_shape_sampling as actual
from voynich.exact_radical import RadicalRatio
from voynich.reading_label_landscape import target_fingerprint
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def independent_mass(expert, env, state, epsilon):
    """One-shot integer products, common row denominators, and separate edit cases."""
    spans = {}
    for obs, widths, labels in zip(env.records, state.widths, state.texts, strict=True):
        pos = 0
        for width, row in zip(widths, labels, strict=True):
            spans.setdefault(row, []).append(obs[pos:pos+width])
            pos += width
    def edit(a, b):
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b, strict=True))
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        return 1 if shorter[0] in longer else 2
    p, q = epsilon.numerator, epsilon.denominator
    numerators, exponent, row_numerators, powers = [], 0, [], 0
    for observations in spans.values():
        costs = [sum(edit(unit, obs) for obs in observations) for unit in env.pool]
        degree = max(costs)
        row_numerators.append(sum(p**cost*q**(degree-cost) for cost in costs))
        powers += degree
    for text in state.texts:
        context = expert.state('')
        for row in text:
            numerator, denominator = float(expert.probabilities[context, row]).as_integer_ratio()
            assert numerator > 0 and denominator & (denominator-1) == 0
            numerators.append(numerator)
            exponent += denominator.bit_length()-1
            context = int(expert.transitions[context, row])
    n = sum(map(len, state.texts))
    return F(math.prod(numerators)*math.prod(row_numerators)*224**n,
             (1 << exponent)*225**(n+len(state.texts))*len(env.pool)**len(spans)*q**powers)


def decision(ratio, rng):
    if not ratio:
        rng.bit_generator.random_raw()
        return False, 1
    return quantile_decision(RadicalRatio(ratio.numerator, ratio.denominator, 1), rng, 16)


def fingerprint(mass):
    return target_fingerprint(mass.numerator, mass.denominator) if mass else None


def alternative_step(expert, env, state, epsilon, rng):
    kernel = ('relabel', 'boundary')[int(rng.integers(2))]
    info = {'kernel': kernel}
    old = independent_mass(expert, env, state, epsilon)
    assert old > 0
    if kernel == 'relabel':
        anchors = [(r, i) for r, text in enumerate(state.texts) for i in range(len(text))]
        anchor = int(rng.integers(len(anchors)))
        record, first = anchors[anchor]
        last = first+1
        rows = (int(rng.integers(env.rows)),)
        widths = (state.widths[record][first],)
        forward = reverse = F(1, len(anchors)*env.rows)
        info.update({'anchor': anchor, 'record': record, 'index': first, 'row': rows[0]})
        valid = True
    else:
        anchors = [(r, i) for r, obs in enumerate(env.records) for i in range(len(obs)-1)]
        anchor = int(rng.integers(len(anchors))) if anchors else 0
        valid, record, start = False, None, None
        if anchors:
            record, start = anchors[anchor]
            spans, offset = [], 0
            for i, width in enumerate(state.widths[record]):
                spans.append((offset, offset+width, i))
                offset += width
            inside = [span for span in spans if start <= span[0] and span[1] <= start+2]
            valid = bool(inside) and inside[0][0] == start and inside[-1][1] == start+2
        if valid:
            first, last = inside[0][2], inside[-1][2]+1
            widths = (1, 1) if len(inside) == 1 else (2,)
            rows = tuple(int(rng.integers(env.rows)) for _ in widths)
            forward, reverse = F(1, len(anchors)*env.rows**len(rows)), F(1, len(anchors)*env.rows**len(inside))
        else:
            rows = ()
            forward = reverse = F(1, len(anchors) or 1)
        info.update({'rank': anchor, 'record': record, 'start': start, 'rows': list(rows)})
    if valid:
        texts = tuple(t[:first]+rows+t[last:] if r == record else t for r, t in enumerate(state.texts))
        spans = tuple(w[:first]+widths+w[last:] if r == record else w for r, w in enumerate(state.widths))
        candidate = law.ReadingShape(spans, texts)
    else:
        candidate = state
    new = independent_mass(expert, env, candidate, epsilon)
    ratio = new/old*reverse/forward
    accepted, blocks = decision(ratio, rng)
    retained = candidate if accepted else state
    info.update({'kind': kernel if valid else 'boundary_identity', 'forward': str(forward), 'reverse': str(reverse),
        'old_target_sha256': fingerprint(old), 'candidate_target_sha256': fingerprint(new),
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)], 'accepted': accepted,
        'acceptance_blocks': blocks, 'changed': retained != state,
        'source_length_delta': sum(map(len, candidate.texts))-sum(map(len, state.texts)),
        'visited_delta': len({a for t in candidate.texts for a in t})-len({a for t in state.texts for a in t})})
    return retained, candidate, ratio, info


def alternative_exchange(expert, env, left, right, ex, ey, rng):
    old = independent_mass(expert, env, left, ex)*independent_mass(expert, env, right, ey)
    new = independent_mass(expert, env, right, ex)*independent_mass(expert, env, left, ey)
    ratio = new/old
    accepted, blocks = decision(ratio, rng)
    return (right, left) if accepted else (left, right), ratio, {
        'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)], 'accepted': accepted,
        'acceptance_blocks': blocks, 'changed': accepted and left != right}


def replay_case(expert, env, initial, seed, sweeps):
    epsilons = (F(0), F(1, 8), F(1, 2), F(1))
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    observed = expected = (initial,)*4
    coverage = {'zero_local': 0, 'zero_exchange': 0, 'changed': 0}
    kernels = set()
    for index in range(sweeps):
        observed, locals_, swaps = actual.replica_sweep(expert, env, observed, epsilons, a, index)
        paths, checks, exchanges = list(expected), [], []
        for slot, epsilon in enumerate(epsilons):
            paths[slot], candidate, ratio, info = alternative_step(expert, env, paths[slot], epsilon, b)
            checks.append((candidate, ratio, info))
            kernels.add(info['kernel'])
            coverage['zero_local'] += ratio == 0
            coverage['changed'] += info['changed']
        for slot in range(index % 2, 3, 2):
            pair, ratio, info = alternative_exchange(expert, env, paths[slot], paths[slot+1],
                                                    epsilons[slot], epsilons[slot+1], b)
            paths[slot], paths[slot+1] = pair
            exchanges.append((slot, ratio, info))
            coverage['zero_exchange'] += ratio == 0
        expected = tuple(paths)
        assert observed == expected and locals_ == tuple(checks) and swaps == tuple(exchanges)
        assert a.bit_generator.state == b.bit_generator.state
        law.to_hard(env, observed[0])
        for state, epsilon in zip(observed, epsilons, strict=True):
            assert law.target(expert, env, state, epsilon) == independent_mass(expert, env, state, epsilon) > 0
    assert kernels == {'relabel', 'boundary'}
    return coverage


@pytest.mark.parametrize('records', (((0, 1, 0), (1, 0)), ((0,), (1,))))
def test_all_shape_dispatch_construction_targets_quantiles_and_pcg(records):
    env, expert = ReadingEnvironment(records, rows=3, glyphs=2), source(3, True, root=2)
    hard = ReadingState((0, 1, -1), tuple(map(len, records)), records)
    for seed in (96571, 96579):
        assert all(replay_case(expert, env, law.from_hard(env, hard), seed, 128).values())


def test_original_23row_42unit_artificial_shape_and_nonzero_source_root():
    expert, sampler, base = fixture()
    expert.state = lambda _: 5
    replay_case(expert, sampler.env, law.from_hard(sampler.env, base.state), 96583, 64)


def test_shape_sampler_bad_generator_ladder_and_sweep_index():
    env, expert = ReadingEnvironment(((0,), (1,)), rows=3, glyphs=2), source(3, True)
    state = law.ReadingShape(((1,), (1,)), ((0,), (1,)))
    with pytest.raises(ValueError):
        actual.step(expert, env, state, F(0), np.random.Generator(np.random.Philox(1)))
    for ladder, index in (((F(0), F(0)), 0), ((F(0), F(1)), 4096), ((F(0), F(1)), True)):
        with pytest.raises(ValueError):
            actual.replica_sweep(expert, env, (state, state), ladder, np.random.default_rng(1), index)
