"""Three-way unchanged-probability scoring benchmark on all old fit cases."""
import argparse
import gc
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark_dense_source001 import PATHS as DENSE_PATHS
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_key_bank_fit001 import MANIFEST, NAMES
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.local_key_bank import one_move_bank
from voynich.native_suffix_marginal import NativeMarginal, build_native, marginal_python
from voynich.sparse_suffix_source import decode

EXP = 'NATIVE-SUFFIX-001'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
PATHS = sorted(set(DENSE_PATHS + ['src/voynich/native/suffix_marginal.cpp',
    'src/voynich/native_suffix_marginal.py', 'tests/test_native_suffix_marginal.py',
    'tests/native_suffix_sanitizer.cpp', 'results/NATIVE-SUFFIX-001/sanitizer.json',
    'scripts/benchmark_native_suffix001.py', 'docs/experiments/NATIVE-SUFFIX-001.md',
    'results/DENSE-SOURCE-001/result.json', 'results/DENSE-SOURCE-001/audit.json']))
BACKENDS = ('full_python', 'marginal_python', 'native')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / 'started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(900, 900)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        selected_path = ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'
        selected = checked_artifact(artifact(selected_path))
        source_audit = json.loads(selected_path.with_name('large-audit.json').read_text())
        dense_audit = json.loads((ROOT / 'results/DENSE-SOURCE-001/audit.json').read_text())
        if (source_audit['status'] != 'PASS' or source_audit['result'] != artifact(selected_path)
                or artifact(ROOT / selected['counts']['path']) != selected['counts']
                or dense_audit['status'] != 'PASS'
                or dense_audit['result'] != artifact(ROOT / 'results/DENSE-SOURCE-001/result.json')):
            raise ValueError('Source/equivalence identities changed')
        compact = CompactSuffixSource.load(ROOT / selected['counts']['path'])
        began = time.monotonic()
        source = DenseSuffixAdapter(compact, selected['selected']['tau'])
        construction_seconds = time.monotonic() - began
        began = time.monotonic()
        build = build_native(BULK / 'build')
        native = NativeMarginal(source, build)
        compilation_seconds = time.monotonic() - began
        manifest = json.loads((ROOT / MANIFEST).read_text())
        cases = {}
        maximum_delta = 0.
        for case_index, name in enumerate(NAMES):
            observed = checked_artifact(manifest['cases'][name]['artifacts']['fit'])
            parent_path = ROOT / f'results/BLIND-CHANNEL-CONFIRM-001/{name}_freeze.json'
            parent = json.loads(parent_path.read_text())
            if (parent['fit_sha256'] != manifest['cases'][name]['artifacts']['fit']['sha256']
                    or len(observed['records']) != 4 or observed['context']['stop_probability'] != 1/225
                    or tuple(observed['context']['source_alphabet']) != source.alphabet):
                raise ValueError('Fitting identity/context mismatch')
            bank = one_move_bank(parent['units'], observed['context']['glyph_alphabet'])
            indices = [i * (len(bank) - 1) // 7 for i in range(8)]
            timings, readings = {}, {}
            order = BACKENDS[case_index % 3:] + BACKENDS[:case_index % 3]
            for backend in order:
                began, clock = time.monotonic(), time.process_time()
                rows = []
                for index in indices:
                    for record in observed['records']:
                        if backend == 'full_python':
                            value = decode(source, bank[index], record, 1/225, max_nodes=500_000)
                            row = {'log_likelihood': value.log_likelihood,
                                   'reachable_nodes': value.reachable_nodes, 'edges': value.edges}
                        elif backend == 'marginal_python':
                            row = asdict(marginal_python(source, bank[index], record, 1/225))
                        else:
                            row = asdict(native.score(bank[index], record, 1/225))
                        if row['edges'] > 2_000_000:
                            raise RuntimeError('Common exact lattice edge cap exceeded')
                        if row['log_likelihood'] == -math.inf:
                            row['log_likelihood'] = None
                        rows.append(row)
                timings[backend] = {'wall_seconds': time.monotonic() - began,
                                    'cpu_seconds': time.process_time() - clock}
                readings[backend] = rows
                gc.collect()
                if resource_report(wall, cpu)['peak_rss_bytes'] > 2 * 1024**3:
                    raise MemoryError('Sampled 2GiB benchmark RSS cap')
            delta = 0.
            for original, python, compiled in zip(*(readings[b] for b in BACKENDS), strict=True):
                for row in (python, compiled):
                    if (row['reachable_nodes'] != original['reachable_nodes'] or row['edges'] != original['edges']
                            or (row['log_likelihood'] is None) != (original['log_likelihood'] is None)):
                        raise ValueError('Graph/support mismatch')
                    if row['log_likelihood'] is not None:
                        delta = max(delta, abs(row['log_likelihood'] - original['log_likelihood']))
            if delta > 1e-7:
                raise ValueError('Likelihood equivalence tolerance exceeded')
            maximum_delta = max(maximum_delta, delta)
            archive = save_new(BULK / f'{name}.json.gz', readings, compressed=True)
            result = {'fit': manifest['cases'][name]['artifacts']['fit'], 'parent': artifact(parent_path),
                      'indices': indices, 'full_bank_size': len(bank), 'backend_order': order,
                      'timings': timings, 'maximum_log_likelihood_delta': delta,
                      'identical_support_nodes_edges': True, 'readings': archive}
            cases[name] = result
            save_new(OUT / f'{name}.json', result)
            print(json.dumps({'case': name, 'delta': delta, 'timings': timings}), flush=True)
        totals = {backend: math.fsum(case['timings'][backend]['wall_seconds'] for case in cases.values())
                  for backend in BACKENDS}
        clauses = {'all_sixteen_cases': len(cases) == 16, 'equivalent_likelihood_graph_support': maximum_delta <= 1e-7,
                   'threefold_vs_full_python': totals['full_python'] >= 3 * totals['native'],
                   'twofold_vs_marginal_python': totals['marginal_python'] >= 2 * totals['native']}
        save_new(OUT / 'result.json', {'freeze': args.freeze, 'source': selected['counts'],
            'construction_seconds': construction_seconds, 'compilation_seconds': compilation_seconds,
            'build': build, 'library': artifact(Path(build['library_path'])),
            'cases': cases, 'totals_wall_seconds': totals,
            'group_wall_seconds': {group: {backend: math.fsum(case['timings'][backend]['wall_seconds']
                for name, case in cases.items() if name.endswith('-shuffle') == (group == 'shuffle'))
                for backend in BACKENDS} for group in ('positive', 'shuffle')}, 'unique_workloads': 512,
            'scorings': 1536, 'maximum_log_likelihood_delta': maximum_delta,
            'engineering_clauses': clauses, 'engineering_gate': 'PASS' if all(clauses.values()) else 'FAIL',
            'no_new_keys_fitted_or_transfer_read': True, 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error),
                                      'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
