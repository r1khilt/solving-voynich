"""Fit-only complete local key banks on exposed development cases.

Never opens transfer or answer artifacts. Partial runs cannot produce a
complete-bank freeze and are retained without restart.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.audit_blind_channel_dev004 import literal_model_bits
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from scripts.run_neural_reader001 import PATHS as SOURCE_PATHS
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.local_key_bank import one_move_bank
from voynich.shared_key_mixture import _logsum
from voynich.sparse_suffix_source import decode

EXP = 'KEY-BANK-FIT-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
MANIFEST = 'data/manifests/blind_channel_confirm001.json'
NAMES = tuple(f'B-key{i}' + suffix for i in range(1, 9) for suffix in ('', '-shuffle'))
PATHS = sorted(set(SOURCE_PATHS + [MANIFEST,
    'scripts/run_key_bank_fit001.py', 'scripts/audit_blind_channel_dev004.py',
    'src/voynich/local_key_bank.py', 'src/voynich/shared_key_mixture.py',
    'tests/test_local_key_bank.py', 'tests/test_key_bank_fit001.py',
    'docs/experiments/KEY-BANK-FIT-001.md',
] + [f'results/BLIND-CHANNEL-CONFIRM-001/{n}_freeze.json' for n in NAMES]))


def build_bank(source, observed, parent, *, progress=None, max_nodes=500_000):
    context = observed['context']
    if tuple(context['source_alphabet']) != tuple(source.alphabet):
        raise ValueError('Source alphabet mismatch')
    bank = one_move_bank(parent, context['glyph_alphabet'], context['max_emission_length'])
    rows = []
    for index, units in enumerate(bank):
        values = [decode(source, units, record, context['stop_probability'], max_nodes=max_nodes)
                  for record in observed['records']]
        supported = all(v.plaintext is not None for v in values)
        log_likelihood = math.fsum(v.log_likelihood for v in values) if supported else None
        code = literal_model_bits(units, context)
        row = {'index': index, 'units': units, 'model_bits': code,
               'fit_log_likelihood': log_likelihood,
               'log_weight_unnormalized': None if not supported else log_likelihood - code * math.log(2),
               'record_log_likelihoods': [v.log_likelihood if v.plaintext is not None else None for v in values],
               'record_nodes': [v.reachable_nodes for v in values]}
        rows.append(row)
        if progress is not None:
            progress(row, len(bank))
    finite = [r for r in rows if r['log_weight_unnormalized'] is not None]
    if not finite:
        raise ValueError('Entire declared bank has zero fit evidence')
    normalizer = _logsum(r['log_weight_unnormalized'] for r in finite)
    best = max(finite, key=lambda r: r['log_weight_unnormalized'])
    for row in rows:
        row['log_weight'] = (None if row['log_weight_unnormalized'] is None
                             else row['log_weight_unnormalized'] - normalizer)
    return {'bank': rows, 'bank_size': len(rows), 'finite_keys': len(finite),
            'best_index': best['index'], 'best_units': best['units'],
            'fit_log_normalizer_restricted_not_full_family_evidence': normalizer,
            'maximum_weight': math.exp(best['log_weight']),
            'all_positive_weights_retained_in_log_domain': True}


def fit(name, freeze):
    if name not in NAMES:
        raise ValueError('Case outside fixed panel')
    require_frozen(freeze, PATHS)
    save_new(OUT / f'{name}-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(650, 600)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest = json.loads((ROOT / MANIFEST).read_text())
        spec = manifest['cases'][name]['artifacts']['fit']
        observed = checked_artifact(spec)
        warm_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
        warm = json.loads(warm_path.read_text())
        if warm['fit_sha256'] != spec['sha256'] or len(observed['records']) != 4:
            raise ValueError('Original fit identity/dimensions changed')
        source_path = ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'
        selected = json.loads(source_path.read_text())
        audit = json.loads(source_path.with_name('large-audit.json').read_text())
        if (audit['status'] != 'PASS' or audit['result'] != artifact(source_path)
                or artifact(ROOT / selected['counts']['path']) != selected['counts']):
            raise ValueError('Source identity/audit mismatch')
        source = CompactSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])
        BULK.mkdir(parents=True, exist_ok=True)
        with gzip.open(BULK / f'{name}-progress.jsonl.gz', 'xb') as stream:
            def progress(row, total):
                stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
                if row['index'] % 64 == 0:
                    stream.flush()
                    print(json.dumps({'case': name, 'completed': row['index'] + 1, 'total': total}), flush=True)
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024**3:
                        raise MemoryError('Sampled per-process 4GiB RSS cap')
            result = build_bank(source, observed, warm['units'], progress=progress)
        archive = save_new(BULK / f'{name}-bank.json.gz', result, compressed=True)
        compact = {k: v for k, v in result.items() if k != 'bank'}
        compact.update(experiment=EXP, case=name, freeze=freeze, fit=spec,
                       parent=artifact(warm_path), source=artifact(source_path), bank=archive,
                       resources=resource_report(wall, cpu), status='complete_fit_only_bank')
        save_new(OUT / f'{name}.json', compact)
        print(json.dumps(compact), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error),
                                             'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT / 'campaign-started.json', {'freeze': freeze, 'start_unix': time.time(), 'workers': 2})
    start = time.monotonic()
    BULK.mkdir(parents=True, exist_ok=True)

    def launch(name):
        with (BULK / f'{name}.log').open('x') as stream:
            try:
                value = subprocess.run([sys.executable, __file__, 'fit', '--case', name, '--freeze', freeze],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=660)
                return {'case': name, 'returncode': value.returncode}
            except subprocess.TimeoutExpired:
                return {'case': name, 'returncode': None, 'status': 'outer_case_timeout'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(launch, NAMES))
    payload = {'freeze': freeze, 'results': results, 'wall_seconds': time.monotonic() - start,
               'status': 'complete' if all(r['returncode'] == 0 for r in results) else 'incomplete_failures_retained',
               'paid_spend_usd': 0}
    save_new(OUT / 'campaign.json', payload)
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('fit', 'campaign'))
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'campaign':
        campaign(args.freeze)
    else:
        fit(args.case, args.freeze)
