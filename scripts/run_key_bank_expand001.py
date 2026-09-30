"""Four complete fit-only neighborhood rounds, retaining their entire key union."""
import argparse
import gzip
import json
import math
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.audit_blind_channel_dev004 import literal_model_bits
from scripts.benchmark_native_suffix001 import PATHS as NATIVE_PATHS
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_key_bank_fit001 import MANIFEST, NAMES
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.expanded_key_bank import expand_key_bank
from voynich.native_suffix_marginal import NativeMarginal

EXP = 'KEY-BANK-EXPAND-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
BENCHMARK = ROOT / 'results/NATIVE-SUFFIX-001/result.json'
PATHS = sorted(set(NATIVE_PATHS + ['src/voynich/expanded_key_bank.py', 'tests/test_expanded_key_bank.py',
    'scripts/run_key_bank_expand001.py', 'tests/test_key_bank_expand001.py',
    'docs/experiments/KEY-BANK-EXPAND-001.md', 'results/NATIVE-SUFFIX-001/result.json',
    'results/NATIVE-SUFFIX-001/audit.json']))


def score_fit(scorer, observed, units):
    context = observed['context']
    if tuple(context['source_alphabet']) != tuple(scorer.alphabet):
        raise ValueError('Source alphabet mismatch')
    values = [scorer.score(units, record, context['stop_probability']) for record in observed['records']]
    if not values or any(not math.isfinite(v.log_likelihood) and v.log_likelihood != -math.inf for v in values):
        raise ValueError('Invalid fitting likelihood')
    logs = [None if v.log_likelihood == -math.inf else v.log_likelihood for v in values]
    total = None if None in logs else math.fsum(logs)
    code = literal_model_bits(units, context)
    return {'record_log_likelihoods': logs, 'record_nodes': [v.reachable_nodes for v in values],
            'record_edges': [v.edges for v in values], 'fit_log_likelihood': total, 'model_bits': code,
            'log_weight_unnormalized': None if total is None else total - code * math.log(2)}


def admission(freeze):
    require_frozen(freeze, PATHS)
    benchmark = json.loads(BENCHMARK.read_text())
    audit = json.loads(BENCHMARK.with_name('audit.json').read_text())
    if (benchmark['engineering_gate'] != 'PASS' or not all(benchmark['engineering_clauses'].values())
            or audit['status'] != 'PASS' or audit['result'] != artifact(BENCHMARK)
            or artifact(ROOT / benchmark['library']['path']) != benchmark['library']):
        raise ValueError('Native scorer lacks matching passing benchmark/audit')
    return benchmark


def fit(name, freeze):
    if name not in NAMES:
        raise ValueError('Case outside registered panel')
    benchmark = admission(freeze)
    save_new(OUT / f'{name}-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(650, 600)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest = json.loads((ROOT / MANIFEST).read_text())
        spec = manifest['cases'][name]['artifacts']['fit']
        observed = checked_artifact(spec)
        parent_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
        parent = json.loads(parent_path.read_text())
        if (parent['fit_sha256'] != spec['sha256'] or len(observed['records']) != 4
                or observed['context']['stop_probability'] != 1/225
                or observed['context']['max_emission_length'] != 2):
            raise ValueError('Original fitting identity/context changed')
        selected_path = ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'
        selected = json.loads(selected_path.read_text())
        if selected['counts'] != benchmark['source'] or artifact(ROOT / selected['counts']['path']) != selected['counts']:
            raise ValueError('Source count archive changed')
        source = DenseSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])
        scorer = NativeMarginal(source, benchmark['build'])
        BULK.mkdir(parents=True, exist_ok=True)
        with gzip.open(BULK / f'{name}-progress.jsonl.gz', 'xb') as stream:
            def progress(row):
                stream.write((json.dumps({'kind': 'candidate', 'value': row}, allow_nan=False) + '\n').encode())
                if row['index'] % 64 == 0:
                    stream.flush()
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4 * 1024**3:
                        raise MemoryError('Sampled 4GiB per-worker cap')
            def round_progress(row):
                stream.write((json.dumps({'kind': 'round', 'value': row}, allow_nan=False) + '\n').encode())
                stream.flush()
                print(json.dumps({'case': name, 'round': row['round'], 'keys': row['cumulative_unique_keys'],
                                  'gain_nats': row['gain_nats'], 'accepted': row['accepted']}), flush=True)
            result = expand_key_bank(lambda units: score_fit(scorer, observed, units), parent['units'],
                observed['context']['glyph_alphabet'], max_rounds=4, tolerance=1e-6,
                progress=progress, round_progress=round_progress)
        archive = save_new(BULK / f'{name}-bank.json.gz', result, compressed=True)
        compact = {k: v for k, v in result.items() if k not in ('bank', 'rounds')}
        compact['rounds'] = [{**{k: v for k, v in row.items() if k != 'member_indices'},
                              'member_count': len(row['member_indices'])} for row in result['rounds']]
        compact.update(experiment=EXP, case=name, freeze=freeze, fit=spec, parent=artifact(parent_path),
            source=artifact(selected_path), native_benchmark=artifact(BENCHMARK), bank=archive,
            progress=artifact(BULK / f'{name}-progress.jsonl.gz'), resources=resource_report(wall, cpu),
            status='complete_fit_only_expanded_bank')
        save_new(OUT / f'{name}.json', compact)
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
                run = subprocess.run([sys.executable, '-u', __file__, 'fit', '--case', name, '--freeze', freeze],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=660)
                return {'case': name, 'returncode': run.returncode}
            except subprocess.TimeoutExpired:
                return {'case': name, 'returncode': None, 'status': 'outer_timeout'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(launch, NAMES))
    result = {'freeze': freeze, 'results': rows, 'wall_seconds': time.monotonic() - start,
              'status': 'complete' if all(r['returncode'] == 0 for r in rows) else 'incomplete_failures_retained',
              'paid_spend_usd': 0}
    save_new(OUT / 'campaign.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('campaign', 'fit'))
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'campaign':
        campaign(args.freeze)
    else:
        fit(args.case, args.freeze)
