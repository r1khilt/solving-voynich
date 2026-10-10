"""Full-family cold repair traces; paired observed-window opportunities, no Gold."""
from fractions import Fraction as F
import math

import numpy as np

from voynich.exact_radical import RadicalRatio, exact_radical_bernoulli
from voynich.reading_label_transport import reference_reading
from voynich.reading_window import observed_window, window_choices
from voynich.regrowth_trace import path_receipt
from voynich.tempered_reading import score_complete_state
from voynich.window_conditionals import rational_categorical, window_ratio


def summarize(rows):
    return {'local_attempts': len(rows), 'valid_windows': sum(r['valid_endpoints'] for r in rows),
        'candidate_targets': sum(len(r['candidates']) for r in rows),
        'accepted': sum(r['accepted'] for r in rows), 'changed': sum(r['changed'] for r in rows),
        'raw64_blocks': sum(r['raw64_blocks'] for r in rows)}


def repair_trace(sampler, initial_actions, *, seed, sweeps, arm, progress=lambda: None):
    if (type(seed) is not int or seed < 0 or type(sweeps) is not int or not 1 <= sweeps <= 4096
            or arm not in ('uniform', 'conditional')):
        raise ValueError('Bounded fixed allocation and declared cold repair arm required')
    path = sampler.path(forced_actions=tuple(initial_actions), progress=progress)
    initial = path_receipt(path)
    rows = []
    count = sum(2*len(r)-1 for r in sampler.env.records)
    for index in range(sweeps):
        # Per-step stream keeps observed-window ranks paired across diverging arms.
        rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
        before = rng.bit_generator.state
        rank = int(rng.integers(count))
        window = observed_window(sampler.env, rank)
        family = window_choices(sampler.env, path.state, *window)
        row = {'index': index, 'rank': rank, 'window': list(window), 'rng_before': before,
            'valid_endpoints': family is not None, 'candidates': [], 'selected_index': None,
            'raw64_blocks': 0, 'accepted': False, 'changed': False}
        if family is not None:
            values = []
            for replacement in family.replacements:
                progress()
                value = window_ratio(sampler, path, *window, replacement)
                values.append(value)
                row['candidates'].append({'replacement': list(replacement),
                    'ratio_hex': [hex(value.ratio.numerator), hex(value.ratio.denominator)]})
            if arm == 'conditional':
                chosen, blocks = rational_categorical(tuple(v.ratio for v in values), rng,
                    maximum_blocks=sampler.config.maximum_acceptance_blocks)
                accepted = True
            else:
                chosen = int(rng.integers(len(values)))
                ratio = values[chosen].ratio
                accepted, blocks = exact_radical_bernoulli(RadicalRatio(ratio.numerator, ratio.denominator, 1),
                    rng, maximum_blocks=sampler.config.maximum_acceptance_blocks)
            candidate = values[chosen]
            if accepted and candidate.state != path.state:
                point = score_complete_state(sampler, candidate.state, progress=progress)
                n, d = sampler.target_integers(path)
                assert F(point.numerator*d, point.denominator*n) == candidate.ratio
                path = reference_reading(sampler, point, progress=progress)
                row['changed'] = True
            row.update({'selected_index': chosen, 'raw64_blocks': blocks, 'accepted': accepted})
        row.update({'rng_after': rng.bit_generator.state, 'retained': [path_receipt(path)]})
        rows.append(row)
        progress()
    n, d = sampler.target_integers(path)
    return {'seed': seed, 'arm_kernel': arm, 'degrees': [1], 'sweeps': sweeps, 'initial': initial,
        'trace': rows, 'final': [path_receipt(path)], 'final_log_targets': [math.log(n)-math.log(d)],
        'summary': summarize(rows), 'no_gold_or_optimization': True}
