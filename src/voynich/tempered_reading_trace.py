"""Finite replica traces; candidates, failed draws and RNG remain inspectable."""
import math

import numpy as np

from voynich.reading_label_landscape import target_fingerprint
from voynich.reading_label_transport import LabelPoint
from voynich.regrowth_trace import path_receipt
from voynich.tempered_reading import path_target, replica_sweep


def ratio_receipt(ratio):
    return None if ratio is None else {'radicand_sha256': target_fingerprint(ratio.numerator, ratio.denominator),
                                      'degree': ratio.degree}


def candidate_receipt(candidate):
    if not isinstance(candidate, LabelPoint):
        return {'kind': 'reference_path', 'reading': path_receipt(candidate)}
    return {'kind': 'deterministic_point', 'reading': {'actions': list(candidate.actions),
        'key': list(candidate.state.key), 'offsets': list(candidate.state.offsets),
        'texts': list(map(list, candidate.state.texts)), 'complete': True},
        'source_terms': list(map(list, candidate.source_terms)),
        'target_sha256': target_fingerprint(candidate.numerator, candidate.denominator)}


def summarize(sweeps, degrees):
    locals_ = [entry for sweep in sweeps for entry in sweep['local']]
    swaps = [entry for sweep in sweeps for entry in sweep['swaps']]
    return {'sweeps': len(sweeps), 'local_attempts': len(locals_), 'exchange_attempts': len(swaps),
        'by_degree': {str(k): {name: sum(bool(entry['info'][name]) for entry in locals_
            if entry['info']['degree'] == k) for name in ('accepted', 'failed')} for k in degrees},
        'by_kernel': {kernel: {'attempts': sum(e['info']['kernel'] == kernel for e in locals_),
            'accepted': sum(e['info']['kernel'] == kernel and e['info']['accepted'] for e in locals_),
            'failed': sum(e['info']['kernel'] == kernel and e['info']['failed'] for e in locals_)}
            for kernel in ('regrowth', 'pair', 'label')},
        'accepted_exchanges': sum(e['info']['accepted'] for e in swaps),
        'acceptance_raw64_blocks': sum(e['info']['acceptance_blocks'] for e in locals_+swaps)}


def replica_trace(sampler, initial_actions, *, seed, sweeps, degrees, progress=lambda: None):
    if type(seed) is not int or seed < 0 or type(sweeps) is not int or sweeps < 1:
        raise ValueError('Fixed nonnegative seed and positive finite sweep allocation required')
    initial = sampler.path(forced_actions=tuple(initial_actions), progress=progress)
    paths = tuple(initial for _ in degrees)
    rng = np.random.Generator(np.random.PCG64(seed))
    trace = []
    for index in range(sweeps):
        rng_before = rng.bit_generator.state
        paths, locals_, swaps = replica_sweep(sampler, paths, degrees, rng, index, progress=progress)
        trace.append({'index': index, 'rng_before': rng_before, 'rng_after': rng.bit_generator.state,
            'local': [{'candidate': candidate_receipt(candidate), 'ratio': ratio_receipt(ratio), 'info': info}
                      for candidate, ratio, info in locals_],
            'swaps': [{'slot': slot, 'ratio': ratio_receipt(ratio), 'info': info} for slot, ratio, info in swaps],
            'retained': [path_receipt(path) for path in paths]})
        progress()
    targets = [path_target(sampler, path) for path in paths]
    return {'seed': seed, 'degrees': list(degrees), 'sweeps': sweeps, 'initial': path_receipt(initial),
        'trace': trace, 'final': [path_receipt(p) for p in paths], 'final_rng': rng.bit_generator.state,
        'final_log_targets': [math.log(t.numerator)-math.log(t.denominator) for t in targets],
        'summary': summarize(trace, degrees), 'no_gold_or_optimization': True}
