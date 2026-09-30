"""Explicit post-cap-failure numeric-storage revision; old experiment stays failed."""
import argparse
import bisect
import json
import math
import signal
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import torch

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import (
    PATHS as MODEL_PATHS, ROOT, TAUS, artifact, limit_resources, load_inputs,
)
from voynich.compact_suffix_source import CompactSuffixSource, fit_compact
from voynich.recurrent_latin_source import ALPHABET, chunks

EXPERIMENT = 'LATIN-SOURCE-COMPACT-001'
OUT = ROOT / 'results' / EXPERIMENT
BULK = ROOT / 'outputs' / EXPERIMENT
PATHS = list(MODEL_PATHS) + [
    'src/voynich/compact_suffix_source.py', 'scripts/run_latin_source_compact001.py',
    'tests/test_compact_suffix_source.py', 'docs/experiments/LATIN-SOURCE-COMPACT-001.md',
    'results/LATIN-SOURCE-MODEL-001/statistical-large-failure.json',
    'results/LATIN-SOURCE-MODEL-001/statistical-small.json',
]


def context_string(code, depth):
    letters = []
    for _ in range(depth):
        code, value = divmod(int(code), len(ALPHABET))
        letters.append(ALPHABET[value])
    return ''.join(reversed(letters))


def fit(dataset, freeze):
    training, validation, identity = load_inputs(dataset)
    save_new(OUT / f'{dataset}-inputs.json', identity)
    model = fit_compact(training, ALPHABET)
    BULK.mkdir(parents=True, exist_ok=True)
    path = BULK / f'{dataset}.npz'
    model.save(path)
    if path.stat().st_size > 512 * 1024**2:
        raise RuntimeError('Compressed archive exceeds fixed 512MiB cap')
    scores = model.score(validation, TAUS)
    for row in scores:
        row['selection_bpc'] = row['selection_bits'] / row['characters']
    winner = min(scores, key=lambda r: (r['selection_bpc'], r['tau']))
    result = {'experiment': EXPERIMENT, 'status': 'completed_source_only', 'dataset': dataset,
              'freeze': freeze, 'inputs': artifact(OUT / f'{dataset}-inputs.json'),
              'counts': artifact(path), 'scores': scores, 'selected': winner, 'order': 12, 'minimum': 4,
              'contexts_by_depth': [len(level['contexts']) for level in model.levels],
              'nonzero_successors_by_depth': [len(level['joints']) for level in model.levels],
              'array_bytes': sum(a.nbytes for level in model.levels for a in level.values()),
              'older_large_baseline_status': 'fixed_cap_failure_retained'}
    return result


def audit(dataset, freeze):
    result = json.loads((OUT / f'{dataset}.json').read_text())
    for spec in (result['inputs'], result['counts']):
        if artifact(ROOT / spec['path']) != spec:
            raise ValueError('Saved input/count hash mismatch')
    training, validation, identity = load_inputs(dataset)
    if json.loads((ROOT / result['inputs']['path']).read_text()) != identity:
        raise ValueError('Input identity changed')
    model = CompactSuffixSource.load(ROOT / result['counts']['path'])
    root = model.levels[0]
    actual_root = {ALPHABET[int(c)]: int(n) for c, n in zip(root['joints'], root['frequencies'])}
    if actual_root != dict(Counter(''.join(training))):
        raise ValueError('Independent root counts failed')
    count_checks = 0
    for depth, level in enumerate(model.levels[1:], 1):
        if np.any(level['totals'] < 4):
            raise ValueError('Context cutoff changed')
        # Four evenly-spaced contexts at every represented depth, fixed in advance.
        for index in np.linspace(0, len(level['contexts']) - 1, min(4, len(level['contexts'])), dtype=int):
            code = int(level['contexts'][index])
            text = context_string(code, depth)
            expected = Counter()
            for record in training:
                offset = 0
                while True:
                    found = record.find(text, offset)
                    if found < 0:
                        break
                    if found + depth < len(record):
                        expected[record[found + depth]] += 1
                    offset = found + 1
            start = bisect.bisect_left(level['joints'], np.uint64(code * len(ALPHABET)))
            end = bisect.bisect_left(level['joints'], np.uint64((code + 1) * len(ALPHABET)))
            actual = {ALPHABET[int(v) % len(ALPHABET)]: int(n)
                      for v, n in zip(level['joints'][start:end], level['frequencies'][start:end])}
            if actual != dict(expected):
                raise ValueError('Independent overlapping substring count mismatch')
            count_checks += 1
    old_small_delta = None
    if dataset == 'small':
        old = json.loads((ROOT / 'results/LATIN-SOURCE-MODEL-001/statistical-small.json').read_text())
        old_counts = load_archive(old['counts'])['counts']
        rebuilt = {}
        for depth, level in enumerate(model.levels):
            for code, count in zip(level['joints'], level['frequencies']):
                text = context_string(int(code), depth + 1)
                rebuilt.setdefault(text[:-1], {})[text[-1]] = int(count)
        if rebuilt != old_counts:
            raise ValueError('Changed estimator rather than changed storage')
        old_small_delta = max(abs(a['selection_bits'] - b['selection_bits'])
                              for a, b in zip(result['scores'], old['scores']))
        if old_small_delta > 1e-6:
            raise ValueError('Old small-corpus selection scores changed')
    # Separate scalar bisect + recursive raw-history evaluation, no vector score code.
    def get(keys, values, code):
        index = bisect.bisect_left(keys, np.uint64(code))
        return int(values[index]) if index < len(keys) and int(keys[index]) == code else 0
    tau = result['selected']['tau']
    @lru_cache(maxsize=200000)
    def probability(history, char):
        if not history:
            return (actual_root.get(char, 0) + .5) / (sum(actual_root.values()) + .5 * len(ALPHABET))
        code = 0
        for c in history:
            code = code * len(ALPHABET) + ALPHABET.index(c)
        level = model.levels[len(history)]
        total = get(level['contexts'], level['totals'], code)
        if not total:
            return probability(history[1:], char)
        count = get(level['joints'], level['frequencies'], code * len(ALPHABET) + ALPHABET.index(char))
        return (count + tau * probability(history[1:], char)) / (total + tau)
    terms = [-math.log2(probability(text[max(0, i - 12):i], char))
             for _, text in chunks(validation) for i, char in enumerate(text)]
    delta = abs(math.fsum(terms) - result['selected']['selection_bits'])
    if delta > 1e-6 or len(terms) != 359276:
        raise ValueError('Separate scalar probability replay mismatch')
    if min(result['scores'], key=lambda r: (r['selection_bpc'], r['tau'])) != result['selected']:
        raise ValueError('Wrong selected mass')
    return {'experiment': EXPERIMENT, 'status': 'PASS', 'dataset': dataset, 'audit_freeze': freeze,
            'result': artifact(OUT / f'{dataset}.json'), 'root_and_context_checks': count_checks + 1,
            'all_small_counts_equal_old_source': dataset == 'small',
            'small_four_score_maximum_difference_bits': old_small_delta,
            'scalar_selection_replay_delta_bits': delta, 'replayed_characters': len(terms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'audit'))
    parser.add_argument('--dataset', choices=('small', 'large'), required=True)
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    name = f'{args.dataset}-{args.stage}'
    save_new(OUT / f'{name}-started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(1800, 1800)
    torch.set_num_threads(1)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = fit(args.dataset, args.freeze) if args.stage == 'fit' else audit(args.dataset, args.freeze)
        result['resources'] = resource_report(wall, cpu)
        target = f'{args.dataset}.json' if args.stage == 'fit' else f'{args.dataset}-audit.json'
        save_new(OUT / target, result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
