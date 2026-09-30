"""Fresh transfer evidence and fixed reading decisions; no answer access."""
import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.audit_key_bank_read001 import close
from scripts.benchmark_shared_key002 import serializable
from scripts.confirm002_common import BULK, NAMES, OUT, ROOT, WORKERS, fit_status, load_source
from scripts.run_blind_channel_confirm001 import inference_rows, sources
from scripts.run_blind_channel_dev001 import baseline_log_likelihood, checked_artifact
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_read001 import read_case
from scripts.run_key_bank_read002 import evidence_case
from scripts.run_latin_source_model001 import limit_resources
from voynich.native_suffix_marginal import NativeMarginal


def predict(name, freeze):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    panel, benchmark, admitted, _ = fit_status(freeze)
    save_new(OUT / f'{name}-prediction-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(1200, 1100)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        spec = panel['cases'][name]['transfer']
        if admitted[name] is None:
            save_new(OUT / f'{name}.json', {'case': name, 'freeze': freeze, 'status': 'fit_unavailable',
                'transfer': spec, 'readings': None, 'resources': resource_report(wall, cpu)})
            return
        observed = checked_artifact(spec)
        source = load_source()
        if (len(observed['records']) != 2 or observed['context']['stop_probability'] != 1/225
                or tuple(observed['context']['source_alphabet']) != source.alphabet):
            raise ValueError('Original transfer context changed')
        records, rho = observed['records'], observed['context']['stop_probability']
        parent = checked_artifact(admitted[name]['parent'])
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
                 'parent': admitted[name]['parent'], 'evidence': evidence_archive,
                 'summary': serializable({k: v for k, v in evidence.items() if k not in ('per_key', 'active_bank_indices')}),
                 'resources': resource_report(wall, cpu)}
        save_new(OUT / f'{name}-evidence.json', saved)
        print(json.dumps({'case': name, 'stage': 'evidence_complete', 'gain_bits': evidence['gain_bits_vs_iid']}), flush=True)
        if not panel['cases'][name]['positive']:
            save_new(OUT / f'{name}.json', {**saved, 'status': 'complete_evidence_only_by_design', 'readings': None})
            return
        _, source3, models = sources()
        frequency, delta = inference_rows(source3, models['order3'], parent['frequency_units'], records, rho)
        frequency_spec = save_new(BULK / f'{name}-frequency.json.gz', frequency, compressed=True)
        save_new(OUT / f'{name}-frequency.json', {'readings': frequency_spec, 'freeze': freeze,
            'transfer': spec, 'parent': admitted[name]['parent'], 'independent_maximum_delta': delta})
        del source3, models
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
                                       'frequency_order3': frequency_spec, 'checks': reading['checks'], 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error),
                                             'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    fit_status(freeze)
    save_new(OUT / 'campaign-started.json', {'freeze': freeze, 'start_unix': time.time(), 'workers': WORKERS})
    start, cpu = time.monotonic(), time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    BULK.mkdir(parents=True, exist_ok=True)
    def launch(name):
        with (BULK / f'{name}.log').open('x') as stream:
            try:
                run = subprocess.run([sys.executable, '-u', __file__, 'predict', '--case', name, '--freeze', freeze],
                                     cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                                VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH='.:src'), stdout=stream, stderr=subprocess.STDOUT, timeout=1210)
                return {'case': name, 'returncode': run.returncode}
            except subprocess.TimeoutExpired:
                return {'case': name, 'returncode': None, 'status': 'outer_timeout'}
            except OSError as error:
                return {'case': name, 'returncode': None, 'status': 'launch_failure', 'error': str(error)}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(launch, NAMES))
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    value = {'freeze': freeze, 'results': rows, 'wall_seconds': time.monotonic() - start,
             'status': 'complete' if all(r['returncode'] == 0 for r in rows) else 'incomplete_failures_retained',
             'parent_cpu_seconds': time.process_time()-cpu,
             'all_children_cpu_seconds_including_failed': (children_after.ru_utime+children_after.ru_stime
                                                         -children_before.ru_utime-children_before.ru_stime),
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
