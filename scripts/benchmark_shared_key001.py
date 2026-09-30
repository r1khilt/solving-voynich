"""Frozen source-generated workload; resource measurements, not recovery evidence."""
import argparse
import gc
import itertools
import json
import random
import signal
import time
from dataclasses import asdict

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_compact001 import PATHS as SOURCE_PATHS
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.fresh_reader_panel import encode_known
from voynich.local_key_bank import one_move_bank
from voynich.shared_key_mixture import decode_shared_keys
from voynich.sparse_suffix_source import decode

EXP = 'SHARED-KEY-BENCH-001'
OUT = ROOT / 'results' / EXP
PATHS = sorted(set(SOURCE_PATHS + [
    'results/LATIN-SOURCE-COMPACT-001/large.json',
    'results/LATIN-SOURCE-COMPACT-001/large-audit.json',
    'src/voynich/compact_suffix_adapter.py', 'src/voynich/shared_key_mixture.py',
    'src/voynich/local_key_bank.py', 'src/voynich/fresh_reader_panel.py',
    'tests/test_shared_key_mixture.py', 'tests/test_local_key_bank.py',
    'scripts/benchmark_shared_key001.py', 'docs/experiments/SHARED-KEY-BENCH-001.md',
]))


def timeout(_signum, _frame):
    raise TimeoutError('Fixed per-operation wall cap')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / 'started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(600, 600)
    signal.signal(signal.SIGALRM, timeout)
    wall, cpu = time.monotonic(), time.process_time()
    selected = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    audit = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large-audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['result'] != artifact(ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json')
            or artifact(ROOT / selected['counts']['path']) != selected['counts']):
        raise ValueError('Source identity/audit mismatch')
    source = CompactSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])
    rng = random.Random(552003)
    pairs = [''.join(x) for x in itertools.product('ABCDEF', repeat=2)]
    rng.shuffle(pairs)
    parent = list('ABCDEF') + pairs[:17]
    rng.shuffle(parent)
    bank = one_move_bank(parent, 'ABCDEF')
    texts = []
    for _ in range(6):
        context, text = source.state(''), ''
        for _ in range(224):
            letter = rng.choices(range(len(source.alphabet)), weights=source.row(context), k=1)[0]
            text += source.alphabet[letter]
            context = source.step(context, letter)
        texts.append(text)
    records = [encode_known(t, source.alphabet, parent) for t in texts]
    shuffled = []
    for row in records:
        letters = list(row)
        rng.shuffle(letters)
        shuffled.append(''.join(letters))
    save_new(OUT / 'inputs.json', {'parent': parent, 'records': records, 'shuffled': shuffled,
                                 'seed': 552003, 'bank_size': len(bank), 'source': selected['counts']})
    indices = [i * (len(bank) - 1) // 31 for i in range(32)]
    results = {}
    for name, rows in [('source_generated', records), ('glyph_shuffle', shuffled)]:
        result = {'bank_size': len(bank), 'sample_indices': indices, 'source_generated_not_empirical': True}
        started, clock = time.monotonic(), time.process_time()
        count, nodes, edges = 0, 0, 0
        signal.alarm(120)
        try:
            for i in indices:
                for record in rows[:4]:
                    value = decode(source, bank[i], record, 1 / 225, max_nodes=500_000)
                    count += 1
                    nodes += value.reachable_nodes
                    edges += value.edges
            result['fit_sample_status'] = 'complete'
        except (RuntimeError, TimeoutError) as error:
            result['fit_sample_status'] = type(error).__name__ + ': ' + str(error)
        finally:
            signal.alarm(0)
        result['fit_sample'] = {'records': count, 'nodes': nodes, 'edges': edges,
                                'wall_seconds': time.monotonic() - started,
                                'cpu_seconds': time.process_time() - clock}
        if count == 128:
            result['linear_full_bank_fit_seconds'] = result['fit_sample']['wall_seconds'] * len(bank) / 32
        gc.collect()
        started, clock = time.monotonic(), time.process_time()
        signal.alarm(120)
        try:
            value = decode_shared_keys(source, bank, rows[4:], 1 / 225, max_nodes=500_000)
            result['mixture_status'] = 'complete'
            result['mixture'] = asdict(value)
        except (RuntimeError, TimeoutError) as error:
            result['mixture_status'] = type(error).__name__ + ': ' + str(error)
        finally:
            signal.alarm(0)
        result['mixture_wall_seconds'] = time.monotonic() - started
        result['mixture_cpu_seconds'] = time.process_time() - clock
        result['resources'] = resource_report(wall, cpu)
        if result['resources']['peak_rss_bytes'] > 4 * 1024**3:
            raise MemoryError('Post-operation 4GiB RSS bound failed')
        save_new(OUT / f'{name}.json', result)
        results[name] = result
        print(json.dumps({'case': name, 'fit_status': result['fit_sample_status'],
                          'mixture_status': result['mixture_status']}), flush=True)
        gc.collect()
    save_new(OUT / 'result.json', {'freeze': args.freeze, 'results': results,
                                 'resources': resource_report(wall, cpu), 'paid_cost': 0,
                                 'status': 'resource_probe_only',
                                 'fit_scores_not_used_to_weight_the_uniform_mixture': True})


if __name__ == '__main__':
    main()
