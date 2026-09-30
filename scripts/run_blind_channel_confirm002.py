"""From-scratch fit-only three-stage search; all fresh cases share the same budget."""
import argparse
import gc
import gzip
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from scripts import audit_blind_channel_dev001 as scalar
from scripts import audit_blind_channel_dev004 as independent
from scripts import summarize_blind_channel_dev003_search as search_audit
from scripts.confirm002_common import (BULK, EXP, NAMES, OUT, PATHS, ROOT, WORKERS,
                                      fit_inputs, load_source, observed)
from scripts.run_blind_channel_confirm001 import frequency_units, sources
from scripts.run_blind_channel_dev001 import digest, fit_glyph_baseline
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_blind_channel_dev005 import trace_accounting
from scripts.run_key_bank_expand001 import score_fit
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.expanded_key_bank import expand_key_bank
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_refine import refine_units
from voynich.native_suffix_marginal import NativeMarginal
from voynich.unit_channel_search import UnitSearchConfig, search_unit_channel


def initial_search(source1, source3, models, data, seed, *, save_stage1, save_stage2):
    """Identical original two-stage objectives/settings; no previous empirical key."""
    context = CodingContext(**data['context'])
    baseline = fit_glyph_baseline(data['records'], context.glyph_alphabet)
    if baseline != scalar.baseline_from_fit(data['records'], context.glyph_alphabet):
        raise ValueError('Independent iid fitting mismatch')
    config = UnitSearchConfig(seed=seed, restarts=16, max_sweeps=80, max_seconds=300., batch_size=256, max_units=256)
    first = search_unit_channel(source1, data['records'], context, config=config).to_dict()
    if first['units'] is None:
        raise ValueError('No supported from-scratch initialization')
    frequency = frequency_units(source1, data['records'], context)
    if list(frequency) != first['trace'][0]['units']:
        raise ValueError('Frequency initializer differs')
    replay1 = math.fsum(independent.infer_record(models['order1'], tuple(first['units']), r, context.stop_probability)['log_likelihood']
                        for r in data['records'])
    if abs(replay1-first['score']['log_likelihood']) > 1e-7:
        raise ValueError('Initial selected likelihood mismatch')
    first_artifact, first_summary, accounting1 = save_stage1(first)
    initial, first_score = tuple(first['units']), first['score']
    del first
    gc.collect()
    second = refine_units(source3, data['records'], context, initial, max_sweeps=20, max_seconds=300., tolerance=1e-8)
    accounting2 = trace_accounting(second, models['order3'], context)
    replay2 = math.fsum(independent.infer_record(models['order3'], tuple(second['units']), r, context.stop_probability)['log_likelihood']
                        for r in data['records'])
    if abs(replay2-second['score']['log_likelihood']) > 1e-7:
        raise ValueError('Refined selected likelihood mismatch')
    second_artifact = save_stage2(second)
    return {'units': second['units'], 'score': second['score'], 'baseline': baseline,
            'frequency_units': list(frequency), 'stage1_units': list(initial), 'stage1_score': first_score,
            'stage1_trace': first_artifact, 'stage1_summary': first_summary, 'stage2_trace': second_artifact,
            'stage1_accounting': accounting1, 'stage2_accounting': accounting2, 'stage2_stop': second['stop_reason'],
            'independent_final_score_deltas': [abs(replay1-first_score['log_likelihood']), abs(replay2-second['score']['log_likelihood'])]}


def fit(name, freeze):
    if name not in NAMES:
        raise ValueError('Unregistered opaque case')
    panel, benchmark = fit_inputs(freeze)
    save_new(OUT/f'{name}-fit-started.json', {'data_freeze': freeze, 'start_unix': time.time()})
    limit_resources(1500, 1440)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        case = panel['cases'][name]
        data = observed(case['fit'], 4)  # Sole per-case data access during fitting.
        source1, source3, models = sources()
        source1_sha = digest(json.dumps(models['order1'], sort_keys=True).encode())
        def stage1_save(first):
            full = save_new(BULK/f'{name}-stage1.json.gz', first, compressed=True)
            compact = {k: v for k, v in first.items() if k != 'trace'}
            compact.update(experiment=EXP, case_id=name, source_freeze=panel['source_freeze'],
                source_sha256=source1_sha,
                input_sha256=case['fit']['sha256'], full_output=full,
                cpu_seconds_before_final_write=time.process_time()-cpu,
                peak_rss_bytes=resource_report(wall, cpu)['peak_rss_bytes'], environment=resource_report(wall, cpu),
                source_hashes={p: digest((ROOT/p).read_bytes()) for p in PATHS})
            accounting = search_audit.validate_case(compact, first, data['context'])
            summary = save_new(BULK/f'{name}-stage1-summary.json.gz', compact, compressed=True)
            if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
                raise MemoryError('Sampled4GiB fitting limit')
            return full, summary, accounting
        parent = initial_search(source1, source3, models, data, case['search_seed'], save_stage1=stage1_save,
                                save_stage2=lambda v: save_new(BULK/f'{name}-stage2.json.gz', v, compressed=True))
        parent.update(case=name, fit=case['fit'], source_freeze=panel['source_freeze'], data_freeze=freeze)
        parent_artifact = save_new(OUT/f'{name}-parent.json', parent)
        print(json.dumps({'case': name, 'stage': 'initial_search_complete', 'seconds': time.monotonic()-wall}), flush=True)
        del source1, source3, models
        gc.collect()
        source = load_source()
        scorer = NativeMarginal(source, benchmark['build'])
        BULK.mkdir(parents=True, exist_ok=True)
        with gzip.open(BULK/f'{name}-bank-progress.jsonl.gz', 'xb') as stream:
            def progress(row):
                stream.write((json.dumps({'kind': 'candidate', 'value': row}, allow_nan=False)+'\n').encode())
                if row['index'] % 64 == 0:
                    stream.flush()
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
                        raise MemoryError('Sampled4GiB fitting limit')
            def round_progress(row):
                stream.write((json.dumps({'kind': 'round', 'value': row}, allow_nan=False)+'\n').encode())
                stream.flush()
                print(json.dumps({'case': name, 'round': row['round'], 'keys': row['cumulative_unique_keys']}), flush=True)
            bank = expand_key_bank(lambda units: score_fit(scorer, data, units), parent['units'],
                                   data['context']['glyph_alphabet'], max_rounds=4, tolerance=1e-6,
                                   progress=progress, round_progress=round_progress)
        packed = save_new(BULK/f'{name}-bank.json.gz', bank, compressed=True)
        summary = {k: v for k, v in bank.items() if k not in ('bank', 'rounds')}
        summary['rounds'] = [{**{k: v for k, v in r.items() if k != 'member_indices'}, 'member_count': len(r['member_indices'])} for r in bank['rounds']]
        save_new(OUT/f'{name}-fit.json', {**summary, 'status': 'complete_fit_only', 'experiment': EXP, 'case': name,
            'source_freeze': panel['source_freeze'], 'data_freeze': freeze, 'fit': case['fit'], 'parent': parent_artifact,
            'bank': packed, 'progress': artifact(BULK/f'{name}-bank-progress.jsonl.gz'),
            'native_benchmark': artifact(ROOT/'results/NATIVE-SUFFIX-001/result.json'),
            'source': artifact(ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json'), 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/f'{name}-fit-failure.json', {'type': type(error).__name__, 'error': str(error), 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def campaign(freeze):
    panel, _ = fit_inputs(freeze)
    save_new(OUT/'fit-campaign-started.json', {'data_freeze': freeze, 'source_freeze': panel['source_freeze'],
        'workers': WORKERS, 'start_unix': time.time(), 'cases': NAMES})
    BULK.mkdir(parents=True, exist_ok=True)
    wall, cpu = time.monotonic(), time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    def launch(name):
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH='.:src')
        result = {'case': name, 'fit_returncode': None, 'audit_returncode': None}
        for stage, command, timeout in (
                ('fit', [__file__, 'fit', '--case', name, '--freeze', freeze], 1510),
                ('audit', ['scripts/audit_blind_channel_confirm002.py', 'fit', '--case', name], 660)):
            if stage == 'audit' and result['fit_returncode'] != 0:
                break
            with (BULK/f'{name}-{stage}.log').open('x') as stream:
                try:
                    job = subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                    result[stage+'_returncode'] = job.returncode
                except subprocess.TimeoutExpired:
                    result[stage+'_status'] = 'outer_timeout'
                except OSError as error:
                    result[stage+'_status'] = 'launch_failure'
                    result[stage+'_error'] = str(error)
        save_new(OUT/f'{name}-fit-process.json', result)
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(launch, NAMES))
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/'fit-campaign.json', {'source_freeze': panel['source_freeze'], 'data_freeze': freeze, 'results': results,
        'wall_seconds': time.monotonic()-wall, 'status': 'complete' if all(r['fit_returncode'] == 0 and r['audit_returncode'] == 0 for r in results) else 'failures_retained',
        'parent_cpu_seconds': time.process_time()-cpu,
        'all_children_cpu_seconds_including_failed': (children_after.ru_utime+children_after.ru_stime
                                                    -children_before.ru_utime-children_before.ru_stime),
        'paid_spend_usd': 0})


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
