"""Complete visited-reading replicas: collapsed target**(1/k), cold k=1.

The source and reference action policy stay unchanged. Only target factors are
tempered. Deterministic pair/label points replay policy counts after acceptance.
"""
import math
from fractions import Fraction as F

import numpy as np

from voynich.exact_radical import exact_radical_bernoulli, swap_ratio, tempered_ratio
from voynich.reading_label_transport import LabelPoint, reference_reading, score_label_transport
from voynich.reading_pair_rewrite import _guard, eligible_triplets, reading_actions, rewrite_pair
from voynich.reading_regrowth import cut_probability


def score_complete_state(sampler, state, *, progress=lambda: None):
    """Action-ordered source terms, without quantized reference-policy allocation."""
    actions = reading_actions(sampler.env, state)
    offsets, contexts, terms = [0]*len(state.texts), [sampler.root]*len(state.texts), []
    for action in actions:
        progress()
        unfinished = [i for i, r in enumerate(sampler.env.records) if offsets[i] < len(r)]
        selected = unfinished[0]
        for i in unfinished[1:]:
            if offsets[i]*len(sampler.env.records[selected]) < offsets[selected]*len(sampler.env.records[i]):
                selected = i
        row, length = action//2, action%2+1
        pairs, exponents, _ = sampler._row(contexts[selected])
        terms.append((pairs[row][0], exponents[row]))
        contexts[selected] = int(sampler.t[contexts[selected], row])
        offsets[selected] += length
    stop, continuation = sampler.config.stop, 1-sampler.config.stop
    n, records = len(actions), len(state.texts)
    numerator = math.prod(p for p, _ in terms)*continuation.numerator**n*stop.numerator**records
    denominator = (1 << sum(e for _, e in terms))*continuation.denominator**n*stop.denominator**records
    denominator *= len(sampler.env.pool)**sum(k >= 0 for k in state.key)
    return LabelPoint(actions, state, tuple(terms), numerator, denominator, sampler.token)


def point_target(point):
    return F(point.numerator, point.denominator)


def path_target(sampler, path):
    return F(*sampler.target_integers(path))


def suffix_correction(sampler, old, candidate, cut):
    """Marginal reverse cut times OLD suffix q / forward cut times NEW suffix q."""
    if (old.law_token is not sampler.token or candidate.law_token is not sampler.token
            or not old.complete or not candidate.complete or type(cut) is not int
            or not 0 <= cut < min(len(old.actions), len(candidate.actions))
            or old.actions[:cut] != candidate.actions[:cut]):
        raise ValueError('Supported complete paths with identical pre-cut prefix required')
    old_cut = cut_probability(len(old.actions), cut, sampler.config.root_mass)
    new_cut = cut_probability(len(candidate.actions), cut, sampler.config.root_mass)
    if not old_cut or not new_cut:
        raise ValueError('Positive marginal cuts required')
    result = F(math.prod(old.counts[cut:]), math.prod(candidate.counts[cut:]))*new_cut/old_cut
    shift = sampler.config.grid_bits*(len(candidate.actions)-len(old.actions))
    return result*(1 << shift) if shift >= 0 else result/F(1 << -shift)


def _validate(sampler, path, rng, degree):
    _guard(sampler, path)
    # Construction enforces the degree domain before a proposal consumes RNG.
    tempered_ratio(F(1), F(1), degree)
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Full64-bit PCG64 generator required')


def local_step(sampler, path, rng, degree, *, kernel, progress=lambda: None):
    _validate(sampler, path, rng, degree)
    old_target = path_target(sampler, path)
    info = {'kernel': kernel, 'degree': degree, 'failed': False, 'accepted': False, 'acceptance_blocks': 0}
    if kernel == 'regrowth':
        root = sampler.config.root_mass
        is_root = int(rng.integers(root.denominator)) < root.numerator
        cut = 0 if is_root else int(rng.integers(len(path.actions)))
        candidate = sampler.path(prefix=path.actions[:cut], rng=rng, progress=progress)
        info['cut'] = cut
        if not candidate.complete:
            info['failed'] = True
            return path, candidate, None, info
        correction = suffix_correction(sampler, path, candidate, cut)
        target = path_target(sampler, candidate)
    elif kernel == 'pair':
        eligible = eligible_triplets(sampler.env, path.state)
        info['old_degree'] = len(eligible)
        if not eligible:
            info.update({'branch': 'identity', 'new_degree': 0, 'triplet': None})
            return path, path, None, info
        triplet = eligible[int(rng.integers(len(eligible)))]
        change = rewrite_pair(sampler.env, path.state, *triplet)
        reverse = eligible_triplets(sampler.env, change.state)
        assert triplet in reverse and change.branch != 'identity'
        candidate = score_complete_state(sampler, change.state, progress=progress)
        correction, target = F(len(eligible), len(reverse)), point_target(candidate)
        info.update({'triplet': list(triplet), 'new_degree': len(reverse), 'branch': change.branch,
                     'replacements': change.replacements})
    elif kernel == 'label':
        used = [row for row, k in enumerate(path.state.key) if k >= 0]
        a = used[int(rng.integers(len(used)))]
        b = int(rng.integers(sampler.env.rows-1))
        b += b >= a
        a, b = sorted((a, b))
        candidate = score_label_transport(sampler, path, a, b, progress=progress)
        correction, target = F(1), point_target(candidate)
        info['pair'] = [a, b]
    else:
        raise ValueError('Registered regrowth/pair/label kernel required')
    ratio = tempered_ratio(target/old_target, correction, degree)
    accepted, blocks = exact_radical_bernoulli(ratio, rng,
                                            maximum_blocks=sampler.config.maximum_acceptance_blocks)
    info.update({'accepted': accepted, 'acceptance_blocks': blocks})
    retained = reference_reading(sampler, candidate, progress=progress) if accepted and isinstance(
        candidate, LabelPoint) else candidate if accepted else path
    return retained, candidate, ratio, info


def mixed_step(sampler, path, rng, degree, *, progress=lambda: None):
    _validate(sampler, path, rng, degree)
    # Fixed mixture: half regrowth, quarter global pair, quarter label.
    kernel = ('regrowth', 'regrowth', 'pair', 'label')[int(rng.integers(4))]
    return local_step(sampler, path, rng, degree, kernel=kernel, progress=progress)


def exchange(sampler, cold, warm, cold_degree, warm_degree, rng):
    _validate(sampler, cold, rng, cold_degree)
    _validate(sampler, warm, rng, warm_degree)
    ratio = swap_ratio(path_target(sampler, warm)/path_target(sampler, cold), cold_degree, warm_degree)
    accepted, blocks = exact_radical_bernoulli(ratio, rng,
                                            maximum_blocks=sampler.config.maximum_acceptance_blocks)
    return (warm, cold) if accepted else (cold, warm), ratio, {'accepted': accepted, 'acceptance_blocks': blocks}


def replica_sweep(sampler, paths, degrees, rng, sweep, *, progress=lambda: None):
    if (type(paths) is not tuple or type(degrees) is not tuple or len(paths) != len(degrees)
            or not degrees or degrees[0] != 1 or tuple(sorted(set(degrees))) != degrees
            or type(sweep) is not int or sweep < 0):
        raise ValueError('Fixed increasing cold-first replica ladder and sweep index required')
    for path, degree in zip(paths, degrees, strict=True):
        _validate(sampler, path, rng, degree)
    updated, locals_, swaps = list(paths), [], []
    for i, degree in enumerate(degrees):
        updated[i], candidate, ratio, info = mixed_step(sampler, updated[i], rng, degree, progress=progress)
        locals_.append((candidate, ratio, info))
    for i in range(sweep % 2, len(degrees)-1, 2):
        (updated[i], updated[i+1]), ratio, info = exchange(
            sampler, updated[i], updated[i+1], degrees[i], degrees[i+1], rng)
        swaps.append((i, ratio, info))
    return tuple(updated), locals_, swaps
