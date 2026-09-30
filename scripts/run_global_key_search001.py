"""All-case exposed development: strong-objective replica search, fit inputs only."""
import argparse
import gc
import gzip
import itertools
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from scripts.benchmark_global_search_systems001 import PATHS as SYSTEM_PATHS, score_fit as base_score
from scripts.confirm002_common import FIT_PANEL, NAMES, PATHS as ORIGINAL_PATHS, fit_inputs, observed
from scripts.diagnose_blind_channel_confirm002a import endpoints
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_read002 import load_source
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.local_key_bank import one_move_bank
from voynich.native_suffix_marginal import NativeMarginal
from voynich.tempered_unit_search import TemperedConfig, search_tempered

EXP = 'GLOBAL-KEY-SEARCH-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
OLD = ROOT/'results/BLIND-CHANNEL-CONFIRM-002'
WORKERS = 8
MAX_NODES, MAX_EDGES = 10_000_000, 40_000_000
PATHS = sorted(set(SYSTEM_PATHS+ORIGINAL_PATHS+[
    'scripts/run_global_key_search001.py', 'scripts/audit_global_key_search001.py',
    'tests/test_global_key_search001.py', 'docs/experiments/GLOBAL-KEY-SEARCH-001.md',
    'scripts/diagnose_blind_channel_confirm002a.py', 'src/voynich/local_key_bank.py',
    'results/GLOBAL-SEARCH-SYSTEMS-001/result.json', 'results/GLOBAL-SEARCH-SYSTEMS-001/audit.json',
    FIT_PANEL, 'results/BLIND-CHANNEL-CONFIRM-002/fit-campaign.json',
    *[f'results/BLIND-CHANNEL-CONFIRM-002/{name}{suffix}.json'
      for name in NAMES for suffix in ('-fit', '-parent', '-fit-audit')],
]))


def config_for(name):
    return TemperedConfig(seed=95101+509*NAMES.index(name), iterations=24_000,
                          max_scored_keys=150_000, max_seconds=1200.)


class BoundedScorer:
    """Enlarge graph work caps, never reinterpret a resource failure as zero."""
    def __init__(self, native):
        self.native = native

    def score(self, key, record, rho):
        return self.native.score(key, record, rho, max_nodes=MAX_NODES, max_edges=MAX_EDGES)


def warm_starts(candidates, selected, score):
    """Cold replica keeps old selected key; hotter replicas get distinct endpoints."""
    selected = tuple(selected)
    unique = list(dict.fromkeys([selected, *(tuple(key) for key in candidates)]))
    values = [(key, score(key)) for key in unique]
    if any(value['log_weight_unnormalized'] is None for _, value in values):
        raise ValueError('Existing supported fitting endpoint lost support')
    ordered = sorted(enumerate(values), key=lambda row: (-row[1][1]['log_weight_unnormalized'], row[0]))
    hot = [key for _, (key, _) in ordered if key != selected]
    starts = [selected]+hot[:7]
    starts += [selected]*(8-len(starts))
    return starts, [{'units': key, **value} for key, value in values]


def reader_union(old, visited, neighborhood, *, keep=64):
    """Preserve whole old bank; select new keys by fit only, not visit counts."""
    previous = {tuple(row['units']): row for row in old['bank']}
    for row in visited:
        if tuple(row['units']) in previous:
            prior = previous[tuple(row['units'])]
            a, b = prior['log_weight_unnormalized'], row['log_weight_unnormalized']
            if ((a is None) != (b is None) or prior['model_bits'] != row['model_bits']
                    or (a is not None and abs(a-b) > 1e-7)):
                raise ValueError('Same dictionary has inconsistent frozen fit score')
    eligible = [row for row in visited if row['log_weight_unnormalized'] is not None]
    top = sorted(enumerate(eligible), key=lambda pair: (-pair[1]['log_weight_unnormalized'], pair[0]))[:keep]
    bank, index = [], {}
    for row in [*old['bank'], *(r for _, r in top), *neighborhood]:
        key = tuple(row['units'])
        if key in index:
            previous = bank[index[key]]
            a, b = previous['log_weight_unnormalized'], row['log_weight_unnormalized']
            if ((a is None) != (b is None) or previous['model_bits'] != row['model_bits']
                    or (a is not None and abs(a-b) > 1e-7)):
                raise ValueError('Same dictionary has inconsistent frozen fit score')
            continue
        index[key] = len(bank)
        bank.append({k: v for k, v in row.items() if k not in ('index', 'log_weight')})
    finite = [row['log_weight_unnormalized'] for row in bank if row['log_weight_unnormalized'] is not None]
    if not finite:
        raise ValueError('No supported reader-bank dictionary')
    high = max(finite)
    log_total = high+math.log(math.fsum(math.exp(v-high) for v in finite))
    for i, row in enumerate(bank):
        row['index'] = i
        value = row['log_weight_unnormalized']
        row['log_weight'] = None if value is None else value-log_total
    best = max((i for i, row in enumerate(bank) if row['log_weight'] is not None),
               key=lambda i: bank[i]['log_weight_unnormalized'])
    return {'bank': bank, 'bank_size': len(bank), 'finite_keys': len(finite), 'best_index': best,
            'best_units': bank[best]['units'], 'retained_new_top_count': len(top),
            'old_bank_size': len(old['bank']), 'neighborhood_size': len(neighborhood),
            'whole_old_bank_preserved': True, 'visit_counts_used_as_weights': False,
            'global_optimality_claimed': False}


def admitted(freeze):
    require_frozen(freeze, PATHS)
    result = checked_artifact(artifact(ROOT/'results/GLOBAL-SEARCH-SYSTEMS-001/result.json'))
    audit = checked_artifact(artifact(ROOT/'results/GLOBAL-SEARCH-SYSTEMS-001/audit.json'))
    if result['engineering_gate'] != 'PASS' or audit['status'] != 'PASS' or audit['result'] != artifact(
            ROOT/'results/GLOBAL-SEARCH-SYSTEMS-001/result.json'):
        raise ValueError('Artificial systems qualification missing')
    return fit_inputs(freeze)


def fit(name, freeze):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    panel, benchmark = admitted(freeze)
    save_new(OUT/f'{name}-started.json', {'freeze': freeze, 'start_unix': time.time(), 'config': asdict(config_for(name))})
    limit_resources(1750, 1700)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        spec = panel['cases'][name]['fit']
        data = observed(spec, 4)
        parent_path, old_path = OLD/f'{name}-parent.json', OLD/f'{name}-fit.json'
        parent, old = checked_artifact(artifact(parent_path)), checked_artifact(artifact(old_path))
        audit = checked_artifact(artifact(OLD/f'{name}-fit-audit.json'))
        if (parent['fit'] != spec or old['fit'] != spec or audit['status'] != 'PASS'
                or audit['result'] != artifact(old_path)):
            raise ValueError('Old fitting input/audit identity mismatch')
        trace = load_archive(parent['stage1_trace'])
        if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
            raise MemoryError('Stage-trace4GiB fit guard')
        terminal = endpoints(trace['trace'])
        if len(terminal) != parent['stage1_accounting']['scored_initializations'] or len(terminal) > 16:
            raise ValueError('Old accepted restart inventory differs')
        candidates = [row['units'] for row in terminal.values() if row['score'] is not None]
        del trace, terminal
        gc.collect()
        source = load_source()
        if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
            raise MemoryError('Source-construction4GiB fit guard')
        scorer = BoundedScorer(NativeMarginal(source, benchmark['build']))
        pool = tuple(''.join(cs) for n in (1, 2) for cs in itertools.product(data['context']['glyph_alphabet'], repeat=n))
        cache, work = {}, []
        BULK.mkdir(parents=True, exist_ok=True)
        with gzip.open(BULK/f'{name}-progress.jsonl.gz', 'xb') as stream:
            def progress(kind, row):
                stream.write((json.dumps({'kind': kind, 'value': row}, allow_nan=False)+'\n').encode())
            def exact(key):
                key = tuple(key)
                if key not in cache:
                    row = {'units': key, **base_score(scorer, data, key)}
                    cache[key] = row
                    work.append(row)
                    progress('score', row)
                    if len(work) % 16 == 0 and resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
                        raise MemoryError('Sampled4GiB fit guard')
                    if len(work) % 256 == 0:
                        stream.flush()
                        if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
                            raise MemoryError('Sampled4GiB fit guard')
                        print(json.dumps({'case': name, 'scores': len(work), 'seconds': time.monotonic()-wall}), flush=True)
                return cache[key]
            starts, initial = warm_starts(candidates, old['best_units'], exact)
            save_new(OUT/f'{name}-warm.json', {'parent': artifact(parent_path), 'old_fit': artifact(old_path),
                                             'starts': starts, 'scores': initial, 'trace': parent['stage1_trace']})
            search = search_tempered(lambda key: exact(key)['log_weight_unnormalized'], starts, pool,
                                     config=config_for(name), on_event=lambda row: progress('event', row))
            selected = tuple(search['best_units'])
            neighborhood_keys = one_move_bank(selected, data['context']['glyph_alphabet'])
            neighborhood_start = time.monotonic()
            neighborhood = []
            for key in neighborhood_keys:
                if time.monotonic()-neighborhood_start >= 300.:
                    raise TimeoutError('Registered300second complete-neighborhood cap')
                neighborhood.append(exact(key))
        old_bank = load_archive(old['bank'])
        visited = [cache[tuple(row['units'])] for row in search['bank']]
        reader = reader_union(old_bank, visited, neighborhood)
        complete_work = save_new(BULK/f'{name}-search.json.gz', {'search': search, 'scores': work,
            'warm_starts': starts, 'warm': initial, 'neighborhood_units': neighborhood_keys}, compressed=True)
        reader_spec = save_new(BULK/f'{name}-reader-bank.json.gz', reader, compressed=True)
        if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
            raise MemoryError('Final4GiB fit guard')
        save_new(OUT/f'{name}.json', {'status': 'complete_fit_only', 'case': name, 'freeze': freeze,
            'fit': spec, 'old_fit': artifact(old_path), 'old_bank': old['bank'], 'parent': artifact(parent_path),
            'warm': artifact(OUT/f'{name}-warm.json'), 'search': complete_work, 'reader_bank': reader_spec,
            'progress': artifact(BULK/f'{name}-progress.jsonl.gz'), 'config': asdict(config_for(name)),
            'graph_caps': {'nodes': MAX_NODES, 'edges': MAX_EDGES},
            'reader_summary': {k: v for k, v in reader.items() if k != 'bank'},
            'search_summary': {k: v for k, v in search.items() if k not in ('bank', 'events')},
            'unique_full_fit_scores_including_warm_and_neighborhood': len(work),
            'fit_objective_gain_nats': reader['bank'][reader['best_index']]['log_weight_unnormalized']-
                old_bank['bank'][old_bank['best_index']]['log_weight_unnormalized'],
            'gain_vs_best_warm_nats': reader['bank'][reader['best_index']]['log_weight_unnormalized']-
                max(row['log_weight_unnormalized'] for row in initial),
            'search_gain_vs_best_warm_nats': search['best_score']-max(row['log_weight_unnormalized'] for row in initial),
            'neighborhood_seconds': time.monotonic()-neighborhood_start,
            'resources': resource_report(wall, cpu), 'transfer_or_answer_files_opened': False})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/f'{name}-failure.json', {'type': type(error).__name__, 'error': str(error),
                                           'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    admitted(freeze)
    save_new(OUT/'campaign-started.json', {'freeze': freeze, 'cases': NAMES, 'workers': WORKERS, 'start_unix': time.time()})
    BULK.mkdir(parents=True, exist_ok=True)
    wall = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    def launch(name):
        row = {'case': name, 'fit_returncode': None, 'audit_returncode': None}
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH='.:src')
        for stage, command, timeout in (
                ('fit', [__file__, 'fit', '--case', name, '--freeze', freeze], 1760),
                ('audit', ['scripts/audit_global_key_search001.py', '--case', name], 920)):
            if stage == 'audit' and row['fit_returncode'] != 0:
                break
            with (BULK/f'{name}-{stage}.log').open('x') as stream:
                try:
                    job = subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                    row[stage+'_returncode'] = job.returncode
                except subprocess.TimeoutExpired:
                    row[stage+'_status'] = 'outer_timeout'
                except OSError as error:
                    row[stage+'_status'] = 'launch_failure'
                    row[stage+'_error'] = str(error)
        save_new(OUT/f'{name}-process.json', row)
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(launch, NAMES))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/'campaign.json', {'freeze': freeze, 'results': rows,
        'status': 'complete' if all(r['fit_returncode'] == 0 and r['audit_returncode'] == 0 for r in rows) else 'failures_retained',
        'wall_seconds': time.monotonic()-wall,
        'all_children_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime, 'paid_spend_usd': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('campaign', 'fit'))
    parser.add_argument('--case')
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    campaign(args.freeze) if args.mode == 'campaign' else fit(args.case, args.freeze)
