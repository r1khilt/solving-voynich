"""ONE bounded original-source conditional-family correctness/cost admission."""
import argparse
from fractions import Fraction as F
import json
import signal
import time

import numpy as np
import torch

from scripts import check_window_reading001 as finite
from scripts import run_reading_pair_admit001 as prior
from voynich.reading_label_transport import reference_reading
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment
from voynich.tempered_reading import score_complete_state
from voynich.regrowth_trace import path_receipt
from voynich.reading_window import observed_window, window_choices
from voynich.window_conditionals import rational_categorical, window_ratio

EXP = 'WINDOW-READING-ADMIT-001'
ROOT, training = prior.ROOT, prior.training
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
CASES, PHASES, WINDOWS, SEED = 97, prior.PHASES, 4, 96391
WALL, CPU, HOST, PRIVATE = 3600, 3300, 2*1024**3, 256*1024**2
PATHS = tuple(sorted(set([*prior.PATHS, *finite.PATHS,
    'results/READING-PAIR-REWRITE-ADMIT-001/result.json', 'results/READING-PAIR-REWRITE-ADMIT-001/audit.json',
    'results/WINDOW-READING-THEORY-001/result.json', 'results/WINDOW-READING-THEORY-001/audit.json',
    'scripts/run_window_reading_admit001.py', 'scripts/audit_window_reading_admit001.py',
    'tests/test_window_reading_admission.py', 'docs/experiments/WINDOW-READING-ADMIT-001.md'])))


def require_prior():
    previous = prior.require_prior()
    for out, status, audited in ((ROOT/'results'/finite.EXP, 'PASS_exact_window_flux_conditional_support_and_rng',
        'PASS_window_receipt_hash_and_arithmetic_closure'), (prior.OUT,
        'PASS_original_source_structural_literal_ratio_and_cost',
        'PASS_original_source_map_eligibility_target_and_ratio_replay')):
        result, audit = (json.loads((out/name).read_text()) for name in ('result.json', 'audit.json'))
        assert result['status'] == status and audit['status'] == audited
        assert audit['result'] == training.artifact(out/'result.json')
        assert not (out/'failure.json').exists() and not (out/'audit-failure.json').exists()
    return previous


def indices(count):
    if type(count) is not int or count <= 0:
        raise ValueError('Positive observed-window count required')
    return tuple(sorted({j*count//WINDOWS for j in range(WINDOWS)}))


def table(sampler, base, rank, seed, *, progress=lambda: None):
    """One fixed window in the unchanged base, not a chain extension."""
    began, began_cpu = time.monotonic(), time.process_time()
    window = observed_window(sampler.env, rank)
    family = window_choices(sampler.env, base.state, *window)
    result = {'rank': rank, 'window': list(window), 'seed': seed, 'valid_endpoints': family is not None,
        'candidates': [], 'selected_index': None, 'raw64_blocks': 0, 'retained': path_receipt(base),
        'rng_before': None, 'rng_after': None, 'changed': False}
    if family is not None:
        values = []
        for replacement in family.replacements:
            progress()
            value = window_ratio(sampler, base, *window, replacement)
            values.append(value)
            result['candidates'].append({'replacement': list(replacement),
                'ratio_hex': [hex(value.ratio.numerator), hex(value.ratio.denominator)],
                'suffix_steps': value.suffix_steps, 'omitted_suffix_steps': value.omitted_suffix_steps,
                'contexts_matched': value.contexts_matched, 'source_length_delta': value.source_length_delta,
                'visited_delta': value.visited_delta})
        rng = np.random.default_rng(seed)
        result['rng_before'] = rng.bit_generator.state
        selected, blocks = rational_categorical(tuple(v.ratio for v in values), rng,
            maximum_blocks=sampler.config.maximum_acceptance_blocks)
        result['rng_after'] = rng.bit_generator.state
        point = score_complete_state(sampler, values[selected].state, progress=progress)
        old_n, old_d = sampler.target_integers(base)
        assert F(point.numerator*old_d, point.denominator*old_n) == values[selected].ratio
        retained = reference_reading(sampler, point, progress=progress)
        result.update({'selected_index': selected, 'raw64_blocks': blocks,
            'retained': path_receipt(retained), 'changed': retained.state != base.state})
    result.update({'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu})
    return result


def admission(freeze):
    training.old.require_frozen(freeze, PATHS)
    previous = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'allocated_frames': CASES*len(PHASES), 'maximum_windows': CASES*len(PHASES)*WINDOWS,
        'seed_base': SEED, 'no_chain_gold_selection_or_neural_calls': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    cells, private = [], 0
    try:
        def guard():
            r = training.resource_report(wall, cpu)
            if (r['wall_seconds'] > WALL or r['cpu_seconds'] > CPU or r['peak_rss_bytes'] > HOST
                    or private > PRIVATE):
                raise MemoryError('Registered conditional admission cap; no retry or extension')
            return r

        source, _ = prior.load_source()
        arrays = prior.array_identity(source)
        assert arrays == previous['source_arrays']
        frames = [c for c in previous['cells'] if c['phase'] in PHASES]
        assert [(c['case'], c['phase']) for c in frames] == [(i, p) for i in range(CASES) for p in PHASES]
        for frame_index, frame in enumerate(frames):
            original = prior.load_archive(frame['archive'])
            env = ReadingEnvironment(original['records'], rows=23, glyphs=6)
            sampler = SourceRegrowth(source, env)
            base = sampler.path(forced_actions=tuple(original['base']['actions']), progress=guard)
            assert prior.prior.reading_receipt(base) == original['base']
            ranks = indices(sum(2*len(r)-1 for r in env.records))
            tables = [table(sampler, base, rank, SEED+frame_index*WINDOWS+j, progress=guard)
                      for j, rank in enumerate(ranks)]
            value = {'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'],
                'input_archive': frame['archive'], 'records': original['records'],
                'base': path_receipt(base), 'tables': tables, 'no_chain_or_gold_selection': True}
            spec = training.save_new(BULK/f"{frame['case']}-{frame['phase']}.json.gz", value, compressed=True)
            private += spec['bytes']
            cells.append({'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'], 'archive': spec,
                'windows': len(tables), 'valid_windows': sum(t['valid_endpoints'] for t in tables),
                'candidates': sum(len(t['candidates']) for t in tables),
                'valid_two_glyph_windows': sum(t['valid_endpoints'] and t['window'][2] == 2 for t in tables),
                'length_changing_candidates': sum(c['source_length_delta'] != 0 for t in tables for c in t['candidates']),
                'changed_draws': sum(t['changed'] for t in tables),
                'window_cpu_seconds': sum(t['cpu_seconds'] for t in tables)})
            guard()
            print(json.dumps({'completed_frames': len(cells), 'allocated_frames': len(frames)}), flush=True)
        valid, candidates = sum(c['valid_windows'] for c in cells), sum(c['candidates'] for c in cells)
        assert sum(c['windows'] for c in cells) <= CASES*len(PHASES)*WINDOWS
        assert candidates <= CASES*len(PHASES)*WINDOWS*552
        mean = sum(c['window_cpu_seconds'] for c in cells)/max(1, valid)
        gates = {'some_valid_windows': valid > 0,
            'some_valid_two_glyph_windows': sum(c['valid_two_glyph_windows'] for c in cells) > 0,
            'some_source_length_changing_candidates': sum(c['length_changing_candidates'] for c in cells) > 0,
            'mean_window_cpu_per_valid_table_under_two_seconds': mean <= 2}
        assert all(gates.values())
        training.old.require_frozen(freeze, PATHS)
        assert prior.array_identity(source) == arrays
        training.save_new(OUT/'result.json', {'status': 'PASS_original_source_conditional_tables_and_cost',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells,
            'valid_windows': valid, 'candidates': candidates, 'gates': gates,
            'mean_window_cpu_per_valid_table_seconds': mean, 'source_arrays': arrays,
            'private_archive_bytes': private, 'resources': guard(),
            'no_gold_metrics_neural_training_or_optimizer': True, 'no_recovery_mixing_or_historical_claim': True,
            'cost_includes_invalid_window_work_but_excludes_frame_initialization_io': True})
        return gates
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
