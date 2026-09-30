"""Four fixed reading decisions from frozen fit-only banks; never opens gold."""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.benchmark_dense_source001 import PATHS as DENSE_PATHS
from scripts.benchmark_shared_key002 import serializable
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_fit001 import MANIFEST, NAMES
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.fresh_reader_panel import encode_known
from voynich.kbest_suffix import bounded_mixture
from voynich.sparse_suffix_source import decode

EXP = 'KEY-BANK-READ-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
FIT = ROOT / 'results/KEY-BANK-FIT-001'
PATHS = sorted(set(DENSE_PATHS + [
    'scripts/run_key_bank_read001.py', 'scripts/evaluate_key_bank_read001.py',
    'scripts/benchmark_shared_key002.py', 'scripts/audit_key_bank_fit001.py',
    'src/voynich/kbest_suffix.py', 'tests/test_kbest_suffix.py', 'tests/test_key_bank_read001.py',
    'docs/experiments/KEY-BANK-READ-001.md', 'results/DENSE-SOURCE-001/result.json',
    'results/DENSE-SOURCE-001/audit.json', 'results/KEY-BANK-FIT-001/campaign.json',
]))


def admission(freeze):
    campaign = json.loads((FIT / 'campaign.json').read_text())
    original_manifest = json.loads((ROOT / MANIFEST).read_text())
    source_result = artifact(ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json')
    statuses = {row['case']: row['returncode'] for row in campaign['results']}
    if (len(campaign['results']) != len(NAMES) or set(statuses) != set(NAMES)
            or campaign['status'] not in ('complete', 'incomplete_failures_retained')):
        raise ValueError('Original fitting campaign is not completely accounted')
    paths = list(PATHS)
    admitted = {}
    for name in NAMES:
        if statuses[name] == 0:
            result_path, audit_path = FIT / f'{name}.json', FIT / f'{name}-audit.json'
            result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
            if (result['status'] != 'complete_fit_only_bank' or audit['status'] != 'PASS'
                    or audit['result'] != artifact(result_path)
                    or audit['candidates_accounted'] != result['bank_size']
                    or result['freeze'] != campaign['freeze'] or result['case'] != name
                    or result['fit'] != original_manifest['cases'][name]['artifacts']['fit']
                    or result['parent'] != artifact(ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json')
                    or result['source'] != source_result):
                raise ValueError('Available bank lacks matching complete audit')
            paths += [str(p.relative_to(ROOT)) for p in (result_path, audit_path)]
            admitted[name] = result
        else:
            admitted[name] = None
    require_frozen(freeze, paths)
    benchmark_path = ROOT / 'results/DENSE-SOURCE-001/result.json'
    benchmark = json.loads(benchmark_path.read_text())
    audit = json.loads((ROOT / 'results/DENSE-SOURCE-001/audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['result'] != artifact(benchmark_path)
            or any(not row['identical_full_readings'] for row in benchmark['cases'].values())):
        raise ValueError('Dense source equivalence benchmark unavailable')
    return admitted, campaign


def path_score(source, texts, rho):
    values = []
    for text in texts:
        values += [math.log(rho), len(text) * math.log1p(-rho)]
        state = source.state('')
        for char in text:
            letter = source.alphabet.index(char)
            values.append(math.log(source.probabilities[state, letter]))
            state = source.step(state, letter)
    return math.fsum(values)


def read_case(source, records, parent, bank, rho, *, save_points=None, progress=None):
    points = {'original': [decode(source, parent, row, rho, max_nodes=500_000).to_dict() for row in records]}
    for observed, value in zip(records, points['original']):
        if value['plaintext'] is not None and encode_known(value['plaintext'], source.alphabet, parent) != observed:
            raise ValueError('Original point reading fails encoding')
    if bank is None:
        points['fit_selected'] = None
        if save_points is not None:
            save_points(points)
        return {'points': points, 'status': 'fit_bank_unavailable', 'joint': None, 'mixture': None}
    chosen = bank['bank'][bank['best_index']]['units']
    points['fit_selected'] = [decode(source, chosen, row, rho, max_nodes=500_000).to_dict() for row in records]
    for observed, value in zip(records, points['fit_selected']):
        if value['plaintext'] is not None and encode_known(value['plaintext'], source.alphabet, chosen) != observed:
            raise ValueError('Selected point reading fails encoding')
    if save_points is not None:
        save_points(points)
    active = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None]
    keys = [bank['bank'][i]['units'] for i in active]
    weights = [bank['bank'][i]['log_weight'] for i in active]
    checks = {'keys': 0, 'tuples': 0, 'forward_records': 0, 'maximum_delta': 0.}

    def check(index, values):
        for texts, score in values.readings:
            if [encode_known(t, source.alphabet, keys[index]) for t in texts] != list(records):
                raise ValueError('Per-key tuple fails global encoding')
            checks['maximum_delta'] = max(checks['maximum_delta'], abs(path_score(source, texts, rho) - score))
        if index % 32 == 0 or index == len(keys) - 1:
            rows = [decode(source, keys[index], r, rho, max_nodes=500_000) for r in records]
            evidence = math.fsum(r.log_likelihood for r in rows)
            if math.isfinite(evidence) != math.isfinite(values.log_likelihood):
                raise ValueError('Independent forward support mismatch')
            if math.isfinite(evidence):
                checks['maximum_delta'] = max(checks['maximum_delta'], abs(evidence - values.log_likelihood))
            checks['forward_records'] += len(records)
        checks['keys'] += 1
        checks['tuples'] += len(values.readings)
        if checks['maximum_delta'] > 1e-7:
            raise ValueError('Independent likelihood/path mismatch')
        if progress is not None:
            progress(index, active[index], values)

    mixture = bounded_mixture(source, keys, records, rho, k=8, log_weights=weights, progress=check,
                              max_nodes=500_000, max_edges=2_000_000, max_expanded=50_000)
    candidates = [(weight + values.readings[0][1], -i, values.readings[0][0])
                  for i, (weight, values) in enumerate(zip(weights, mixture['per_key'])) if values.readings]
    joint = None
    if candidates:
        score, minus_index, texts = max(candidates)
        joint = {'plaintexts': texts, 'key_index': active[-minus_index], 'log_probability': score}
        if mixture['joint_log_probability'] + 1e-7 < score:
            raise ValueError('Marginal candidate cannot be worse than the joint candidate score')
    if mixture['plaintexts'] is not None:
        expected = [i for i, key in enumerate(keys)
                    if [encode_known(t, source.alphabet, key) for t in mixture['plaintexts']] == list(records)]
        if tuple(expected) != mixture['compatible_key_indices']:
            raise ValueError('Independent global winner support mismatch')
    return {'points': points, 'status': 'complete', 'joint': joint, 'mixture': mixture,
            'active_bank_indices': active, 'checks': checks}


def predict(name, freeze):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    admitted, _ = admission(freeze)
    save_new(OUT / f'{name}-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(650, 600)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest = json.loads((ROOT / MANIFEST).read_text())
        spec = manifest['cases'][name]['artifacts']['transfer']
        observed = checked_artifact(spec)
        if (len(observed['records']) != 2 or observed['context']['stop_probability'] != 1 / 225):
            raise ValueError('Original transfer dimensions changed')
        selected_path = ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'
        selected = json.loads(selected_path.read_text())
        if artifact(ROOT / selected['counts']['path']) != selected['counts']:
            raise ValueError('Source archive identity changed')
        source = DenseSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])
        if tuple(observed['context']['source_alphabet']) != source.alphabet:
            raise ValueError('Transfer alphabet changed')
        parent_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
        parent = json.loads(parent_path.read_text())['units']
        bank = None if admitted[name] is None else load_archive(admitted[name]['bank'])
        BULK.mkdir(parents=True, exist_ok=True)
        points_archive = None

        def keep_points(value):
            nonlocal points_archive
            points_archive = save_new(BULK / f'{name}-points.json.gz', value, compressed=True)
            save_new(OUT / f'{name}-points.json', {'readings': points_archive, 'transfer': spec,
                                                'fit_bank_available': bank is not None})
        with (BULK / f'{name}-progress.jsonl').open('x') as stream:
            def progress(index, original, values):
                stream.write(json.dumps({'active_index': index, 'bank_index': original,
                                         'nodes': values.nodes, 'edges': values.edges, 'expanded': values.expanded}) + '\n')
                if index % 64 == 0:
                    stream.flush()
                    print(json.dumps({'case': name, 'keys': index + 1}), flush=True)
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4 * 1024**3:
                        raise MemoryError('Sampled 4GiB RSS cap')
            result = read_case(source, observed['records'], parent, bank, 1 / 225,
                               save_points=keep_points, progress=progress)
        result.pop('points')
        archive = save_new(BULK / f'{name}.json.gz', serializable(result), compressed=True)
        save_new(OUT / f'{name}.json', {'freeze': freeze, 'status': result['status'], 'case': name,
            'transfer': spec, 'bank': None if bank is None else admitted[name]['bank'],
            'readings': archive, 'points': points_archive, 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error),
                                             'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    admitted, original = admission(freeze)
    save_new(OUT / 'campaign-started.json', {'freeze': freeze, 'start_unix': time.time(), 'workers': 2,
        'fit_campaign': artifact(FIT / 'campaign.json'), 'fit_available': [n for n in NAMES if admitted[n] is not None],
        'fit_status': original['status']})
    start = time.monotonic()
    BULK.mkdir(parents=True, exist_ok=True)
    def launch(name):
        with (BULK / f'{name}.log').open('x') as stream:
            try:
                value = subprocess.run([sys.executable, __file__, 'predict', '--case', name, '--freeze', freeze],
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
    parser.add_argument('mode', choices=('predict', 'campaign'))
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'campaign':
        campaign(args.freeze)
    else:
        predict(args.case, args.freeze)
