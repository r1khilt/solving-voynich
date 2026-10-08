"""Bounded original-source structural cost admission; no chain or Gold selection."""
import argparse
import json
import math
import signal
import time

import torch

from scripts import check_reading_pair_rewrite001 as finite
from scripts import run_reading_label_landscape001 as prior
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.reading_label_landscape import target_fingerprint
from voynich.reading_pair_rewrite import (eligible_triplets, pair_rewrite_candidate,
    rewrite_pair, valid_pair_rewrite_ratio)
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment

EXP = 'READING-PAIR-REWRITE-ADMIT-001'
ROOT, training = prior.ROOT, prior.training
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
PHASES, POINTS, CASES = ('initial', '92821-regrowth'), 8, 97
WALL, CPU, HOST, PRIVATE = 3600, 3300, 2*1024**3, 128*1024**2
PATHS = sorted(set([*prior.PATHS, *finite.PATHS,
    'results/READING-LABEL-LANDSCAPE-001/result.json',
    'results/READING-LABEL-LANDSCAPE-001/sealed-cells.json',
    'results/READING-LABEL-LANDSCAPE-001/audit.json',
    'results/READING-PAIR-REWRITE-THEORY-001/result.json',
    'results/READING-PAIR-REWRITE-THEORY-001/audit.json',
    'scripts/run_reading_pair_admit001.py', 'scripts/audit_reading_pair_admit001.py',
    'tests/test_reading_pair_admission.py', 'docs/experiments/READING-PAIR-REWRITE-ADMIT-001.md']))


def selected_indices(degree):
    if type(degree) is not int or degree < 0:
        raise ValueError('Nonnegative eligibility count required')
    return tuple(sorted({j*degree//POINTS for j in range(POINTS)})) if degree else ()


def require_prior():
    for out, status in ((ROOT/'results'/finite.EXP, 'PASS_exact_global_pair_involution_and_flux'),
                       (prior.OUT, 'COMPLETE_registered_all_pairs_development_diagnostic')):
        result = json.loads((out/'result.json').read_text())
        audit = json.loads((out/'audit.json').read_text())
        assert result['status'] == status and audit['result'] == training.artifact(out/'result.json')
        assert audit['status'] == ('PASS_receipt_hash_and_arithmetic_closure' if out.name == finite.EXP else
                                  'PASS_independent_all_pairs_target_literal_and_orbit_replay')
        assert not (out/'failure.json').exists() and not (out/'audit-failure.json').exists()
    return json.loads((prior.OUT/'result.json').read_text())


def target_receipt(sampler, path):
    n, d = sampler.target_integers(path)
    return {'target_fingerprint': target_fingerprint(n, d), 'log_target': math.log(n)-math.log(d),
            'reference_logq': path.policy_log_probability(), 'length': len(path.actions),
            'visited': sum(k >= 0 for k in path.state.key)}


def admission(freeze):
    training.old.require_frozen(freeze, PATHS)
    previous = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'allocated_frames': CASES*len(PHASES), 'maximum_points': CASES*len(PHASES)*POINTS,
        'no_chain_or_gold_selection': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    BULK.mkdir(parents=True, exist_ok=True)
    cells, private_bytes = [], 0
    try:
        def guard():
            r = training.resource_report(wall, cpu)
            if (r['wall_seconds'] > WALL or r['cpu_seconds'] > CPU or r['peak_rss_bytes'] > HOST
                    or private_bytes > PRIVATE):
                raise MemoryError('Registered structural admission limit; no retry or extension')
            return r

        source, _ = load_source()
        arrays = array_identity(source)
        assert arrays == previous['source_arrays']
        frames = [c for c in previous['cells'] if c['phase'] in PHASES]
        assert [(c['case'], c['phase']) for c in frames] == [(i, p) for i in range(CASES) for p in PHASES]
        for frame in frames:
            old = load_archive(frame['archive'])
            env = ReadingEnvironment(old['records'], rows=23, glyphs=6)
            sampler = SourceRegrowth(source, env)
            path = sampler.path(forced_actions=tuple(old['base']['actions']), progress=guard)
            assert prior.reading_receipt(path) == old['base']
            eligible = eligible_triplets(env, path.state)
            base_reference = prior.recovery.admitted.point_check(source, sampler, path)
            points = []
            for index in selected_indices(len(eligible)):
                began, began_cpu = time.monotonic(), time.process_time()
                triplet = eligible[index]
                candidate, change = pair_rewrite_candidate(sampler, path, *triplet, progress=guard)
                reverse = eligible_triplets(env, candidate.state)
                assert triplet in reverse and rewrite_pair(env, candidate.state, *triplet).state == path.state
                reference = prior.recovery.admitted.point_check(source, sampler, candidate)
                n, d = valid_pair_rewrite_ratio(sampler, path, candidate, len(eligible), len(reverse))
                points.append({'eligible_index': index, 'triplet': list(triplet), 'branch': change.branch,
                    'replacements': change.replacements, 'new_degree': len(reverse),
                    'reading': prior.reading_receipt(candidate), 'target': target_receipt(sampler, candidate),
                    'valid_ratio_fingerprint': target_fingerprint(n, d),
                    'valid_log_ratio': math.log(n)-math.log(d), 'source_reference': reference,
                    'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu})
            value = {'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'],
                'input_archive': frame['archive'], 'records': old['records'], 'base': old['base'],
                'base_target': target_receipt(sampler, path), 'base_source_reference': base_reference,
                'old_degree': len(eligible), 'points': points, 'no_chain_or_gold_selection': True}
            spec = training.save_new(BULK/f"{frame['case']}-{frame['phase']}.json.gz", value, compressed=True)
            private_bytes += spec['bytes']
            cells.append({'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'],
                'archive': spec, 'old_degree': len(eligible), 'points': len(points),
                'candidate_cpu_seconds': sum(p['cpu_seconds'] for p in points),
                'branches': {b: sum(p['branch'] == b for p in points) for b in ('merge', 'split')}})
            print(json.dumps({'completed_frames': len(cells), 'allocated_frames': CASES*len(PHASES)}), flush=True)
        total_points = sum(c['points'] for c in cells)
        mean_cpu = sum(c['candidate_cpu_seconds'] for c in cells)/max(1, total_points)
        gates = {'some_nonidentity_candidates': total_points > 0,
                 'mean_candidate_cpu_under_two_seconds': mean_cpu <= 2}
        assert all(gates.values())
        training.old.require_frozen(freeze, PATHS)
        assert array_identity(source) == arrays
        training.save_new(OUT/'result.json', {'status': 'PASS_original_source_structural_literal_ratio_and_cost',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells,
            'total_points': total_points, 'mean_candidate_cpu_seconds': mean_cpu, 'gates': gates,
            'source_arrays': arrays, 'resources': guard(), 'private_archive_bytes': private_bytes,
            'paid_spend_usd': 0, 'no_chain_gold_metrics_neural_calls_or_optimizer': True,
            'recovery_mixing_speedup_or_historical_claim': False})
        return gates
    except Exception as error:
        training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'private_archive_bytes': private_bytes, 'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(admission(parser.parse_args().freeze))
