"""Exact lookup representation benchmark on exposed fitting records only."""
import argparse
import cProfile
import gc
import json
import pstats
import signal
import time

import numpy as np

from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_key_bank_fit001 import PATHS as BANK_PATHS
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.local_key_bank import one_move_bank
from voynich.sparse_suffix_source import decode

EXP = 'DENSE-SOURCE-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
PATHS = sorted(set(BANK_PATHS + ['src/voynich/dense_suffix_adapter.py',
    'tests/test_dense_suffix_adapter.py', 'scripts/benchmark_dense_source001.py',
    'docs/experiments/DENSE-SOURCE-001.md']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / 'started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(420, 420)
    wall, cpu = time.monotonic(), time.process_time()
    selected_path = ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'
    selected = json.loads(selected_path.read_text())
    audit = json.loads(selected_path.with_name('large-audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['result'] != artifact(selected_path)
            or artifact(ROOT / selected['counts']['path']) != selected['counts']):
        raise ValueError('Source audit/identity mismatch')
    compact = CompactSuffixSource.load(ROOT / selected['counts']['path'])
    began = time.monotonic()
    dense = DenseSuffixAdapter(compact, selected['selected']['tau'])
    construction_seconds = time.monotonic() - began
    lazy = CompactSuffixAdapter(compact, selected['selected']['tau'])
    sampled, delta, transitions = 0, 0., 0
    for depth, level in enumerate(compact.levels):
        indices = sorted({i * (len(level['contexts']) - 1) // 31 for i in range(32)}) if len(level['contexts']) else []
        for index in indices:
            state = int(lazy.offsets[depth]) + index
            delta = max(delta, float(np.max(np.abs(dense.row(state) - lazy.row(state)))))
            for letter in range(len(dense.alphabet)):
                if dense.step(state, letter) != lazy.step(state, letter):
                    raise ValueError('Source transition mismatch')
                transitions += 1
            sampled += 1
    if delta > 1e-15:
        raise ValueError('Source probability mismatch')
    del lazy
    gc.collect()
    manifest = json.loads((ROOT / 'data/manifests/blind_channel_confirm001.json').read_text())
    results = {}
    BULK.mkdir(parents=True, exist_ok=True)
    for name in ('B-key1', 'B-key1-shuffle'):
        spec = manifest['cases'][name]['artifacts']['fit']
        observed = checked_artifact(spec)
        parent_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
        parent = json.loads(parent_path.read_text())
        if parent['fit_sha256'] != spec['sha256']:
            raise ValueError('Parent fitting identity changed')
        keys = one_move_bank(parent['units'], observed['context']['glyph_alphabet'])
        indices = [i * (len(keys) - 1) // 7 for i in range(8)]
        timings, readings = {}, {}
        for backend in (('lazy', 'dense') if name == 'B-key1' else ('dense', 'lazy')):
            source = dense if backend == 'dense' else CompactSuffixAdapter(compact, selected['selected']['tau'])
            began, clock = time.monotonic(), time.process_time()
            rows = []
            for index in indices:
                for record in observed['records']:
                    rows.append(decode(source, keys[index], record, observed['context']['stop_probability'], max_nodes=500_000).to_dict())
            timings[backend] = {'wall_seconds': time.monotonic() - began, 'cpu_seconds': time.process_time() - clock,
                                'nodes': sum(r['reachable_nodes'] for r in rows), 'edges': sum(r['edges'] for r in rows)}
            readings[backend] = rows
            if backend == 'lazy':
                del source
                gc.collect()
        if readings['lazy'] != readings['dense']:
            raise ValueError('Probability, reading or graph counts differ across source representations')
        archive = save_new(BULK / f'{name}.json.gz', readings, compressed=True)
        result = {'fit': spec, 'parent': artifact(parent_path), 'indices': indices, 'timings': timings,
                  'identical_full_readings': True, 'readings': archive,
                  'speedup_observed_not_universal': timings['lazy']['wall_seconds'] / timings['dense']['wall_seconds']}
        results[name] = result
        save_new(OUT / f'{name}.json', result)
        if resource_report(wall, cpu)['peak_rss_bytes'] > 2 * 1024**3:
            raise MemoryError('Sampled 2GiB benchmark RSS cap')
        print(json.dumps({'case': name, 'timings': timings}), flush=True)
    # Separate profile: excluded from both timing comparisons; no warm-cache advantage.
    observed = checked_artifact(manifest['cases']['B-key1-shuffle']['artifacts']['fit'])
    units = json.loads((ROOT / 'results/BLIND-CHANNEL-CONFIRM-001/B-key1-shuffle_freeze.json').read_text())['units']
    source = CompactSuffixAdapter(compact, selected['selected']['tau'])
    profiler = cProfile.Profile()
    profiler.enable()
    decode(source, units, observed['records'][0], 1 / 225, max_nodes=500_000)
    profiler.disable()
    stats = pstats.Stats(profiler).stats
    top = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:20]
    profile = [{'function': str(key), 'primitive_calls': value[0], 'total_calls': value[1],
                'self_seconds': value[2], 'cumulative_seconds': value[3]} for key, value in top]
    save_new(OUT / 'result.json', {'freeze': args.freeze, 'source': selected['counts'],
        'construction_seconds': construction_seconds, 'array_bytes': dense.array_bytes,
        'states': len(dense.probabilities), 'sampled_rows': sampled, 'sampled_transitions': transitions,
        'maximum_sampled_probability_delta': delta, 'cases': results, 'lazy_parent_profile': profile,
        'resources': resource_report(wall, cpu), 'status': 'representation_benchmark_only',
        'no_new_key_selection_or_transfer_reading': True})
    signal.alarm(0)


if __name__ == '__main__':
    main()
