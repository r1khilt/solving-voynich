"""Frozen expanded-bank evidence for all cases; four-arm MAP for positives only."""
import argparse
import json
import math
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.audit_key_bank_read001 import close, logsum
from scripts.benchmark_shared_key002 import serializable
from scripts.run_blind_channel_dev001 import baseline_log_likelihood, checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_expand001 import PATHS as EXPAND_PATHS, admission as native_admission
from scripts.run_key_bank_fit001 import MANIFEST, NAMES
from scripts.run_key_bank_read001 import PATHS as READ_PATHS, read_case
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.native_suffix_marginal import NativeMarginal

EXP = 'KEY-BANK-READ-002'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
FIT = ROOT / 'results/KEY-BANK-EXPAND-001'
PATHS = sorted(set(EXPAND_PATHS + READ_PATHS + [
    'scripts/run_key_bank_read002.py', 'scripts/evaluate_key_bank_read002.py',
    'scripts/audit_key_bank_read002.py', 'scripts/audit_key_bank_read001.py',
    'scripts/audit_key_bank_expand001.py', 'tests/test_key_bank_read002.py',
    'docs/experiments/KEY-BANK-READ-002.md',
    'results/KEY-BANK-EXPAND-001/campaign.json', 'results/KEY-BANK-EXPAND-001/accounting.json',
    'results/KEY-BANK-READ-001/evaluation.json',
] + [f'results/KEY-BANK-EXPAND-001/{n}{suffix}.json' for n in NAMES for suffix in ('', '-audit')]
    + [f'results/KEY-BANK-READ-001/{n}-points.json' for n in NAMES if not n.endswith('-shuffle')]))


def admission(freeze):
    require_frozen(freeze, PATHS)
    benchmark = native_admission(freeze)
    campaign = json.loads((FIT / 'campaign.json').read_text())
    if (campaign['status'] != 'complete' or [r['case'] for r in campaign['results']] != list(NAMES)
            or any(r['returncode'] != 0 for r in campaign['results'])):
        raise ValueError('Expanded fitting is not complete for every registered case')
    manifest = json.loads((ROOT / MANIFEST).read_text())
    admitted = {}
    for name in NAMES:
        path = FIT / f'{name}.json'
        result = json.loads(path.read_text())
        audit = json.loads((FIT / f'{name}-audit.json').read_text())
        if (result['status'] != 'complete_fit_only_expanded_bank' or result['case'] != name
                or result['freeze'] != campaign['freeze'] or audit['status'] != 'PASS'
                or audit['result'] != artifact(path) or audit['candidates_accounted'] != result['bank_size']
                or audit['auditor'] != artifact(ROOT / 'scripts/audit_key_bank_expand001.py')
                or result['fit'] != manifest['cases'][name]['artifacts']['fit']
                or result['parent'] != artifact(ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json')
                or result['source'] != artifact(ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json')):
            raise ValueError('Expanded bank audit/input binding mismatch')
        admitted[name] = result
    return admitted, benchmark


def evidence_case(scorer, records, bank, rho, *, progress=None):
    """One shared key across records; preserve arbitrarily small positive weights."""
    active = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None]
    weights = [bank['bank'][i]['log_weight'] for i in active]
    if not weights or any(not math.isfinite(w) for w in weights):
        raise ValueError('Invalid fitting weights')
    close(logsum(weights), 0.)
    if len({tuple(row['units']) for row in bank['bank']}) != len(bank['bank']):
        raise ValueError('Duplicate bank keys')
    rows = []
    for index in active:
        values = [scorer.score(bank['bank'][index]['units'], text, rho,
                              max_nodes=500_000, max_edges=2_000_000) for text in records]
        logs = [v.log_likelihood for v in values]
        if any(v != -math.inf and not math.isfinite(v) for v in logs):
            raise ValueError('Invalid transfer evidence')
        row = {'bank_index': index, 'record_log_likelihoods': logs,
               'log_likelihood': math.fsum(logs), 'nodes': [v.reachable_nodes for v in values],
               'edges': [v.edges for v in values]}
        rows.append(row)
        if progress is not None:
            progress(row)
    value = logsum(bank['bank'][r['bank_index']]['log_weight'] + r['log_likelihood'] for r in rows)
    return {'active_bank_indices': active, 'per_key': rows, 'log_likelihood': value,
            'supported_keys': sum(math.isfinite(r['log_likelihood']) for r in rows),
            'records': len(records), 'all_positive_weights_retained': True}


def load_source():
    selected = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    if artifact(ROOT / selected['counts']['path']) != selected['counts']:
        raise ValueError('Source archive changed')
    return DenseSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])


def predict(name, freeze):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    admitted, benchmark = admission(freeze)
    save_new(OUT / f'{name}-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(1200, 1100)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        spec = json.loads((ROOT / MANIFEST).read_text())['cases'][name]['artifacts']['transfer']
        observed = checked_artifact(spec)
        source = load_source()
        if (len(observed['records']) != 2 or observed['context']['stop_probability'] != 1/225
                or tuple(observed['context']['source_alphabet']) != source.alphabet):
            raise ValueError('Original transfer context changed')
        records, rho = observed['records'], observed['context']['stop_probability']
        parent_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
        parent = json.loads(parent_path.read_text())
        bank = load_archive(admitted[name]['bank'])
        scorer = NativeMarginal(source, benchmark['build'])
        BULK.mkdir(parents=True, exist_ok=True)
        with (BULK / f'{name}-evidence-progress.jsonl').open('x') as stream:
            def progress(row):
                stream.write(json.dumps(serializable(row), allow_nan=False) + '\n')
                if len_index[0] % 64 == 0:
                    stream.flush()
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4 * 1024**3:
                        raise MemoryError('Sampled 4GiB worker cap')
                len_index[0] += 1
            len_index = [0]
            evidence = evidence_case(scorer, records, bank, rho, progress=progress)
        evidence['original_log_likelihood'] = math.fsum(scorer.score(parent['units'], r, rho).log_likelihood for r in records)
        evidence['fit_selected_log_likelihood'] = next(r['log_likelihood'] for r in evidence['per_key']
                                                       if r['bank_index'] == bank['best_index'])
        evidence['iid_log_likelihood'] = baseline_log_likelihood(parent['baseline'], records)
        evidence['gain_bits_vs_iid'] = (evidence['log_likelihood'] - evidence['iid_log_likelihood']) / math.log(2)
        evidence_archive = save_new(BULK / f'{name}-evidence.json.gz', serializable(evidence), compressed=True)
        saved = {'case': name, 'freeze': freeze, 'transfer': spec, 'bank': admitted[name]['bank'],
                 'parent': artifact(parent_path), 'evidence': evidence_archive,
                 'summary': serializable({k: v for k, v in evidence.items() if k not in ('per_key', 'active_bank_indices')}),
                 'resources': resource_report(wall, cpu)}
        save_new(OUT / f'{name}-evidence.json', saved)
        print(json.dumps({'case': name, 'stage': 'evidence_complete', 'gain_bits': evidence['gain_bits_vs_iid']}), flush=True)
        if name.endswith('-shuffle'):
            save_new(OUT / f'{name}.json', {**saved, 'status': 'complete_evidence_only_by_design', 'readings': None})
            return
        points_artifact = None
        def keep_points(value):
            nonlocal points_artifact
            points_artifact = save_new(BULK / f'{name}-points.json.gz', value, compressed=True)
            save_new(OUT / f'{name}-points.json', {'readings': points_artifact, 'transfer': spec, 'freeze': freeze})
        with (BULK / f'{name}-progress.jsonl').open('x') as stream:
            def read_progress(index, bank_index, values):
                # Independent native evidence for every key, including zero support.
                expected = evidence['per_key'][index]
                if expected['bank_index'] != bank_index:
                    raise ValueError('Native and k-best key inventory mismatch')
                close(values.log_likelihood, expected['log_likelihood'])
                stream.write(json.dumps({'active_index': index, 'bank_index': bank_index,
                                         'nodes': values.nodes, 'edges': values.edges, 'expanded': values.expanded}) + '\n')
                if index % 64 == 0:
                    stream.flush()
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4 * 1024**3:
                        raise MemoryError('Sampled 4GiB worker cap')
            reading = read_case(source, records, parent['units'], bank, rho, save_points=keep_points, progress=read_progress)
        close(reading['mixture']['log_likelihood'], evidence['log_likelihood'])
        readings = save_new(BULK / f'{name}-readings.json.gz', serializable(reading), compressed=True)
        save_new(OUT / f'{name}.json', {**saved, 'status': 'complete', 'readings': readings, 'points': points_artifact,
                                       'checks': reading['checks'], 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error),
                                             'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    admission(freeze)
    save_new(OUT / 'campaign-started.json', {'freeze': freeze, 'start_unix': time.time(), 'workers': 2})
    start = time.monotonic()
    BULK.mkdir(parents=True, exist_ok=True)
    def launch(name):
        with (BULK / f'{name}.log').open('x') as stream:
            try:
                run = subprocess.run([sys.executable, '-u', __file__, 'predict', '--case', name, '--freeze', freeze],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=1210)
                return {'case': name, 'returncode': run.returncode}
            except subprocess.TimeoutExpired:
                return {'case': name, 'returncode': None, 'status': 'outer_timeout'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(launch, NAMES))
    value = {'freeze': freeze, 'results': rows, 'wall_seconds': time.monotonic() - start,
             'status': 'complete' if all(r['returncode'] == 0 for r in rows) else 'incomplete_failures_retained',
             'paid_spend_usd': 0}
    save_new(OUT / 'campaign.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('campaign', 'predict'))
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'campaign':
        campaign(args.freeze)
    else:
        predict(args.case, args.freeze)
