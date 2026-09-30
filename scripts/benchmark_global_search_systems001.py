"""Frozen artificial resource/equivalence benchmark; no cipher panel access."""
import argparse
import gzip
import itertools
import json
import math
import random
import signal
import time
from dataclasses import asdict

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.audit_blind_channel_dev004 import literal_model_bits
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.native_partial_bound import NativePartialBound
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.native_suffix_marginal import NativeMarginal, marginal_python
from voynich.partial_unit_bound import relaxed_record
from voynich.recurrent_latin_source import ALPHABET
from voynich.tempered_unit_search import TemperedConfig, search_tempered

EXP = 'GLOBAL-SEARCH-SYSTEMS-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
NEW_PATHS = [
    'src/voynich/tempered_unit_search.py', 'src/voynich/partial_unit_bound.py',
    'src/voynich/native_partial_bound.py', 'tests/test_tempered_unit_search.py',
    'tests/test_partial_unit_bound.py', 'tests/test_native_partial_bound.py',
    'scripts/benchmark_global_search_systems001.py', 'tests/test_global_search_systems001.py',
    'scripts/audit_global_search_systems001.py',
    'docs/experiments/GLOBAL-SEARCH-SYSTEMS-001.md',
    'docs/research/global-dictionary-search-after-confirm002.md',
    'results/LATIN-SOURCE-COMPACT-001/large.json',
    'results/LATIN-SOURCE-COMPACT-001/large-audit.json',
]
PATHS = sorted(set(NEW_PATHS+[
    'src/voynich/native/suffix_marginal.cpp', 'src/voynich/native_suffix_marginal.py',
    'src/voynich/compact_suffix_source.py', 'src/voynich/compact_suffix_adapter.py',
    'src/voynich/dense_suffix_adapter.py', 'src/voynich/recurrent_latin_source.py',
    'scripts/run_blind_channel_dev001.py', 'scripts/run_blind_channel_dev004.py',
    'scripts/run_latin_source_model001.py', 'scripts/audit_blind_channel_dev004.py',
    'results/NATIVE-SUFFIX-001/result.json', 'results/NATIVE-SUFFIX-001/audit.json',
]))
CONFIG = TemperedConfig(seed=94739, iterations=256, max_scored_keys=2048, max_seconds=120.)


def admission(freeze):
    """Read engineering/source provenance only, never old empirical cipher inputs."""
    require_frozen(freeze, PATHS)
    path = ROOT/'results/NATIVE-SUFFIX-001/result.json'
    benchmark = json.loads(path.read_text())
    audit = json.loads(path.with_name('audit.json').read_text())
    selected_path = ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json'
    selected = json.loads(selected_path.read_text())
    source_audit = json.loads(selected_path.with_name('large-audit.json').read_text())
    if (benchmark['engineering_gate'] != 'PASS' or not all(benchmark['engineering_clauses'].values())
            or audit['status'] != 'PASS' or audit['result'] != artifact(path)
            or artifact(ROOT/benchmark['library']['path']) != benchmark['library']
            or selected['counts'] != benchmark['source']
            or artifact(ROOT/selected['counts']['path']) != selected['counts']
            or source_audit['status'] != 'PASS' or source_audit['result'] != artifact(selected_path)):
        raise ValueError('Native/source qualification or identities differ')
    return benchmark


def load_source():
    selected = json.loads((ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    return DenseSuffixAdapter(CompactSuffixSource.load(ROOT/selected['counts']['path']), selected['selected']['tau'])


def score_fit(scorer, data, key):
    values = [scorer.score(key, r, data['context']['stop_probability']) for r in data['records']]
    logs = [None if v.log_likelihood == -math.inf else v.log_likelihood for v in values]
    total = None if None in logs else math.fsum(logs)
    bits = literal_model_bits(key, data['context'])
    return {'record_log_likelihoods': logs, 'record_nodes': [v.reachable_nodes for v in values],
            'record_edges': [v.edges for v in values], 'fit_log_likelihood': total,
            'model_bits': bits, 'log_weight_unnormalized': None if total is None else total-bits*math.log(2)}


def workloads():
    rng = random.Random(94731)
    pool = tuple(''.join(cs) for n in (1, 2) for cs in itertools.product('ABCDEF', repeat=n))
    key = list('ABCDEF')+rng.sample(list(pool[6:]), 17)
    rng.shuffle(key)
    variants = [tuple(key)]
    for _ in range(3):
        changed = list(key)
        rng.shuffle(changed)
        variants.append(tuple(changed))
    plain = [''.join(rng.choice(ALPHABET) for _ in range(224)) for _ in range(4)]
    mapping = dict(zip(ALPHABET, key, strict=True))
    observed = [''.join(mapping[c] for c in record) for record in plain]
    small = [''.join(mapping[c] for c in record[:32]) for record in plain]
    context = {'source_alphabet': list(ALPHABET), 'glyph_alphabet': list('ABCDEF'),
               'max_emission_length': 2, 'max_states': 2, 'max_alternatives': 3,
               'source_count': 1, 'stop_probability': 1/225}
    return {'seed': 94731, 'pool': pool, 'variants': variants, 'plain': plain,
            'full_records': observed, 'short_records': small, 'context': context}


def timed(call):
    wall, cpu = time.monotonic(), time.process_time()
    try:
        return {'status': 'complete', 'value': call(), 'wall_seconds': time.monotonic()-wall,
                'cpu_seconds': time.process_time()-cpu}
    except RuntimeError as error:
        if 'cap; no pruning' not in str(error):
            raise
        return {'status': 'graph_cap', 'error': str(error), 'wall_seconds': time.monotonic()-wall,
                'cpu_seconds': time.process_time()-cpu}


def score_value(row):
    return {'log_likelihood': None if row.log_likelihood == -math.inf else row.log_likelihood,
            'nodes': row.reachable_nodes, 'edges': row.edges}


def main(freeze):
    require_frozen(freeze, PATHS)
    benchmark = admission(freeze)
    save_new(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'config': asdict(CONFIG)})
    limit_resources(360, 340)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        inputs = workloads()
        input_spec = save_new(OUT/'artificial-inputs.json', inputs)
        source = load_source()
        selected = json.loads((ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
        if selected['counts'] != benchmark['source']:
            raise ValueError('Source counts differ')
        scorer = NativeMarginal(source, benchmark['build'])
        bound = NativePartialBound(source, benchmark['build'])
        timings, equivalence, relaxations = [], [], []
        maximum = 0.
        for index, key in enumerate(inputs['variants']):
            for label, records in (('short', inputs['short_records']), ('full', inputs['full_records'])):
                data = {'records': records, 'context': inputs['context']}
                row = timed(lambda: score_fit(scorer, data, key))
                row.update(variant=index, length=label)
                timings.append(row)
                if row['status'] != 'complete':
                    raise RuntimeError('Artificial complete-key workload cannot be scored within graph caps')
            for record_index, record in enumerate(inputs['short_records']):
                native = score_value(scorer.score(key, record, 1/225))
                reference = score_value(marginal_python(source, key, record, 1/225))
                if ((native['log_likelihood'] is None) != (reference['log_likelihood'] is None)
                        or native['nodes'] != reference['nodes'] or native['edges'] != reference['edges']):
                    raise ValueError('Artificial complete graph/support mismatch')
                delta = 0. if native['log_likelihood'] is None else abs(native['log_likelihood']-reference['log_likelihood'])
                maximum = max(maximum, delta)
                equivalence.append({'variant': index, 'record': record_index, 'native': native, 'reference': reference, 'delta': delta})
        save_new(OUT/'complete-scores.json', {'timings': timings, 'equivalence': equivalence})
        for assigned in (0, 11, 20, 23):
            partial = tuple(u if i < assigned else None for i, u in enumerate(inputs['variants'][0]))
            record, pool = inputs['full_records'][0], inputs['pool']
            args = {'max_nodes': 20_000, 'max_edges': 80_000}
            native = timed(lambda: bound.record(partial, pool, record, 1/225, **args))
            python = timed(lambda: relaxed_record(source, partial, pool, record, 1/225, **args))
            if native['status'] != python['status']:
                raise ValueError('Relaxed cap/support outcome mismatch')
            if native['status'] == 'complete':
                a, b = native['value'], python['value']
                if (a['nodes'] != b['nodes'] or a['edges'] != b['edges']
                        or (a['log_likelihood_upper_bound'] is None) != (b['log_likelihood_upper_bound'] is None)):
                    raise ValueError('Relaxed graph/support mismatch')
                delta = 0. if a['log_likelihood_upper_bound'] is None else abs(a['log_likelihood_upper_bound']-b['log_likelihood_upper_bound'])
                maximum = max(maximum, delta)
                if assigned == 23 and abs(a['log_likelihood_upper_bound']-timings[1]['value']['record_log_likelihoods'][0]) > 1e-7:
                    raise ValueError('Complete partial bound is not exact')
            # Python insertion order and native unordered traversal can hit
            # different cap types first; both must stop, never return a bound.
            row = {'assigned': assigned, 'native': native, 'python': python}
            relaxations.append(row)
            save_new(OUT/f'partial-{assigned:02d}.json', row)
            print(json.dumps({'assigned': assigned, 'status': native['status'], 'native_seconds': native['wall_seconds']}), flush=True)
        if maximum > 1e-7:
            raise ValueError('Likelihood tolerance failed')
        data = {'records': inputs['full_records'], 'context': inputs['context']}
        work = []
        BULK.mkdir(parents=True, exist_ok=True)
        with gzip.open(BULK/'progress.jsonl.gz', 'xb') as stream:
            def progress(kind, row):
                stream.write((json.dumps({'kind': kind, 'value': row}, allow_nan=False)+'\n').encode())
            def exact(key):
                value = score_fit(scorer, data, key)
                work.append({'units': key, **value})
                progress('score', work[-1])
                if len(work) % 64 == 0:
                    stream.flush()
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
                        raise MemoryError('Sampled4GiB system cap')
                    print(json.dumps({'scored_keys': len(work), 'seconds': time.monotonic()-wall}), flush=True)
                return value['log_weight_unnormalized']
            search = search_tempered(exact, [inputs['variants'][1]], inputs['pool'], config=CONFIG,
                                     on_event=lambda row: progress('event', row))
        if search['scored_keys'] != len(work) or search['scored_keys'] > CONFIG.max_scored_keys:
            raise ValueError('Search work accounting mismatch')
        archive = save_new(BULK/'search.json.gz', {'search': search, 'scores': work}, compressed=True)
        clauses = {'exact_equivalence': maximum <= 1e-7, 'all_artificial_workloads': len(timings) == 8,
                   'resource_limits': resource_report(wall, cpu)['peak_rss_bytes'] <= 4*1024**3,
                   'score_accounting': len(work) == search['scored_keys'],
                   'bounded_search_completed': search['proposals'] > 0}
        save_new(OUT/'result.json', {'experiment': EXP, 'freeze': freeze, 'inputs': input_spec,
            'native_benchmark': artifact(ROOT/'results/NATIVE-SUFFIX-001/result.json'),
            'source': artifact(ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json'),
            'source_counts': benchmark['source'], 'config': asdict(CONFIG), 'search': archive,
            'progress': artifact(BULK/'progress.jsonl.gz'),
            'search_summary': {k: v for k, v in search.items() if k not in ('bank', 'events')},
            'maximum_likelihood_delta': maximum, 'complete_scores': artifact(OUT/'complete-scores.json'),
            'partial_results': [artifact(OUT/f'partial-{n:02d}.json') for n in (0, 11, 20, 23)],
            'engineering_clauses': clauses, 'engineering_gate': 'PASS' if all(clauses.values()) else 'FAIL',
            'no_empirical_cipher_panel_opened': True, 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/'failure.json', {'type': type(error).__name__, 'error': str(error), 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    main(parser.parse_args().freeze)
