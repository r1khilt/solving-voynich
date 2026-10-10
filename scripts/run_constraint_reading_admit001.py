"""ONE bounded dispatcher/RNG/original-source admission, never recovery scoring."""
import argparse
from fractions import Fraction as F
import json
import signal
import time

import numpy as np
import torch

from scripts import run_window_reading_admit001 as prior
from scripts import check_constraint_reading001 as finite
from tests.test_constraint_reading import source as tiny_source
from voynich import constraint_reading as law
from voynich import constraint_sampling as sampling
from voynich.reading_pair_rewrite import reading_actions
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import path_receipt
from voynich.source_action_proposal import ReadingEnvironment, ReadingState

EXP = 'CONSTRAINT-READING-ADMIT-001'
ROOT, training = prior.ROOT, prior.training
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
EPSILONS = (F(0), F(1, 8), F(1, 2), F(1))
TINY_RECORDS = (((0, 1, 0), (1, 0)), ((0,), (1,)))
TINY_SEEDS, TINY_SWEEPS, SOURCE_SWEEPS, SOURCE_SEED = (96521, 96529), 128, 4, 96561
WALL, CPU, AUDIT_WALL, AUDIT_CPU = 3600, 3300, 5400, 5100
HOST, PRIVATE = 2*1024**3, 256*1024**2
PATHS = tuple(sorted(set([*finite.PATHS,
    'results/CONSTRAINT-READING-THEORY-001/result.json', 'results/CONSTRAINT-READING-THEORY-001/audit.json',
    'src/voynich/constraint_sampling.py', 'tests/test_constraint_sampling.py',
    'scripts/run_constraint_reading_admit001.py', 'scripts/audit_constraint_reading_admit001.py',
    'tests/test_constraint_reading_admission.py', 'docs/experiments/CONSTRAINT-READING-ADMIT-001.md'])))


def require_prior():
    old = prior.require_prior()
    frames = [c for c in old['cells'] if c['phase'] in prior.PHASES]
    assert [(c['case'], c['phase']) for c in frames] == [(i, p) for i in range(prior.CASES) for p in prior.PHASES]
    for out, expected, audited in ((prior.OUT, 'PASS_original_source_conditional_tables_and_cost',
        'PASS_original_source_all_conditional_targets_and_rng'), (ROOT/'results'/finite.EXP,
        'PASS_exact_constraint_component_target_exchange_and_support_laws',
        'PASS_constraint_receipt_hash_and_arithmetic_closure')):
        result, audit = (json.loads((out/n).read_text()) for n in ('result.json', 'audit.json'))
        assert result['status'] == expected and audit['status'] == audited
        assert audit['result'] == training.artifact(out/'result.json')
        assert not (out/'failure.json').exists() and not (out/'audit-failure.json').exists()
    return {**old, 'cells': frames}


def snapshot(env, state, target_hash):
    return {'key': list(state.key), 'widths': list(map(list, state.widths)), 'texts': list(map(list, state.texts)),
            'violations': law.violations(env, state), 'target_sha256': target_hash}


def summary(rows):
    local = [entry for row in rows for entry in row['local']]
    swaps = [entry for row in rows for entry in row['swaps']]
    return {'local_attempts': len(local), 'exchange_attempts': len(swaps),
        'accepted_local_changes': sum(e['info']['changed'] for e in local),
        'zero_target_local_rejections': sum(e['info']['candidate_target_sha256'] is None for e in local),
        'zero_target_exchange_rejections': sum(e['info']['ratio_hex'][0] == '0x0' for e in swaps),
        'accepted_exchange_changes': sum(e['info']['changed'] for e in swaps),
        'warm_inconsistent_retained_states': sum(s['violations'] > 0 for r in rows for s in r['retained'][1:]),
        'raw64_acceptance_blocks': sum(e['info']['acceptance_blocks'] for e in local+swaps),
        'by_kernel_attempts': {k: sum(e['info']['kernel'] == k for e in local)
                               for k in ('rebind', 'relabel', 'boundary')}}


def trace(source, sampler, base, *, seed, sweeps, progress=lambda: None):
    initial = law.from_hard(sampler.env, base.state)
    original = F(*sampler.target_integers(base))
    assert law.target(source, sampler.env, initial, F(0)) == original
    states = (initial,)*len(EPSILONS)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(sweeps):
        before = rng.bit_generator.state
        states, locals_, swaps = sampling.replica_sweep(source, sampler.env, states, EPSILONS, rng, index,
                                                      progress=progress)
        retained = [snapshot(sampler.env, state, sampling.fingerprint(law.target(source, sampler.env, state, epsilon)))
                    for state, epsilon in zip(states, EPSILONS, strict=True)]
        cold = law.to_hard(sampler.env, states[0])
        reference = sampler.path(forced_actions=reading_actions(sampler.env, cold), progress=progress)
        assert reference.complete and reference.state == cold
        assert F(*sampler.target_integers(reference)) == law.target(source, sampler.env, states[0], F(0))
        rows.append({'index': index, 'rng_before': before, 'rng_after': rng.bit_generator.state,
            'local': [{'candidate': snapshot(sampler.env, candidate, info['candidate_target_sha256']), 'info': info}
                      for candidate, _, info in locals_],
            'swaps': [{'slot': slot, 'info': info} for slot, _, info in swaps],
            'retained': retained, 'cold_reference': path_receipt(reference)})
        progress()
    return {'seed': seed, 'sweeps': sweeps, 'epsilons': [str(e) for e in EPSILONS],
        'initial': snapshot(sampler.env, initial, sampling.fingerprint(original)),
        'initial_reference': path_receipt(base), 'trace': rows, 'summary': summary(rows),
        'no_gold_selection_training_or_recovery_metrics': True}


def tiny_cases():
    for seed in TINY_SEEDS:
        for records in TINY_RECORDS:
            source = tiny_source()
            env = ReadingEnvironment(records, rows=3, glyphs=2)
            hard = ReadingState((0, 1, -1), tuple(map(len, records)), records)
            sampler = SourceRegrowth(source, env)
            base = sampler.path(forced_actions=reading_actions(env, hard))
            yield source, sampler, base, seed


def gates(cells):
    source = [c for c in cells if c['group'] == 'source']
    attempts = sum(c['summary']['local_attempts'] for c in source)
    return {'some_original_source_local_zero_rejections': sum(
        c['summary']['zero_target_local_rejections'] for c in source) > 0,
        'some_original_source_exchange_zero_rejections': sum(c['summary']['zero_target_exchange_rejections'] for c in source) > 0,
        'some_original_source_warm_inconsistent_states': sum(c['summary']['warm_inconsistent_retained_states'] for c in source) > 0,
        'all_kernels_attempted_on_original_source': all(sum(c['summary']['by_kernel_attempts'][k] for c in source) > 0
            for k in ('rebind', 'relabel', 'boundary')),
        'mean_source_cell_cpu_per_local_under_two_seconds': sum(c['cpu_seconds'] for c in source)/attempts <= 2}


def admission(freeze):
    training.old.require_frozen(freeze, PATHS)
    old = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'tiny_cells': len(TINY_SEEDS)*len(TINY_RECORDS), 'source_cells': len(old['cells']),
        'tiny_sweeps': TINY_SWEEPS, 'source_sweeps': SOURCE_SWEEPS,
        'no_gold_selection_training_or_recovery_metrics': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    cells, private = [], 0
    try:
        def guard():
            r = training.resource_report(wall, cpu)
            if r['wall_seconds'] > WALL or r['cpu_seconds'] > CPU or r['peak_rss_bytes'] > HOST or private > PRIVATE:
                raise MemoryError('Registered soft source/RNG admission cap; no retry or extension')
            return r
        def save_cell(group, source, sampler, base, seed, sweeps, metadata):
            nonlocal private
            began, began_cpu = time.monotonic(), time.process_time()
            value = trace(source, sampler, base, seed=seed, sweeps=sweeps, progress=guard)
            spec = training.save_new(BULK/f'{group}-{len(cells)}.json.gz', {'group': group,
                'metadata': metadata, 'records': list(map(list, sampler.env.records)), **value}, compressed=True)
            private += spec['bytes']
            cells.append({'group': group, 'metadata': metadata, 'archive': spec, 'summary': value['summary'],
                'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu})
            guard()
            print(json.dumps({'completed_cells': len(cells)}), flush=True)
        for index, (source, sampler, base, seed) in enumerate(tiny_cases()):
            save_cell('tiny', source, sampler, base, seed, TINY_SWEEPS, {'tiny_index': index})
        source, selected = prior.prior.load_source()
        arrays = prior.prior.array_identity(source)
        assert arrays == old['source_arrays']
        for index, cell in enumerate(old['cells']):
            archive = prior.prior.load_archive(cell['archive'])
            env = ReadingEnvironment(tuple(map(tuple, archive['records'])), rows=23, glyphs=6)
            sampler = SourceRegrowth(source, env)
            base = sampler.path(forced_actions=tuple(archive['base']['actions']), progress=guard)
            assert prior.prior.prior.reading_receipt(base) == archive['base']
            save_cell('source', source, sampler, base, SOURCE_SEED+index, SOURCE_SWEEPS,
                {'source_index': index, 'case': cell['case'], 'kind': cell['kind'], 'phase': cell['phase'],
                 'source_archive': cell['archive']})
        outcome = gates(cells)
        assert all(outcome.values())
        assert prior.prior.array_identity(source) == arrays
        training.old.require_frozen(freeze, PATHS)
        training.save_new(OUT/'result.json', {'status': 'PASS_constraint_dispatch_original_source_cost_and_cold_conformance',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells,
            'gates': outcome, 'source_arrays': arrays, 'source_counts': selected['counts'],
            'private_archive_bytes': private, 'resources': guard(),
            'mean_source_cell_cpu_per_local_seconds': sum(c['cpu_seconds'] for c in cells if c['group'] == 'source')/sum(
                c['summary']['local_attempts'] for c in cells if c['group'] == 'source'),
            'no_gold_selection_training_or_recovery_metrics': True})
        return outcome
    except Exception as error:
        training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'private_archive_bytes': private, 'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(admission(parser.parse_args().freeze))
