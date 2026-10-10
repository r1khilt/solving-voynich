"""ONE bounded replica transport/cost admission on original archived readings."""
import argparse
import json
import signal
import time

import torch

from scripts import check_tempered_reading001 as theory
from scripts import run_reading_pair_admit001 as prior
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment
from voynich.tempered_reading_trace import replica_trace

EXP = 'TEMPERED-READING-ADMIT-001'
ROOT, training = prior.ROOT, prior.training
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
CASES, PHASES, SWEEPS, SEED = 97, prior.PHASES, 8, 96101
DEGREES = theory.DEGREES
WALL, CPU, HOST, PRIVATE = 3600, 3300, 2*1024**3, 128*1024**2
PATHS = tuple(sorted(set([*prior.PATHS, *theory.PATHS,
    'results/READING-PAIR-REWRITE-ADMIT-001/result.json', 'results/READING-PAIR-REWRITE-ADMIT-001/audit.json',
    'results/TEMPERED-READING-THEORY-001/result.json', 'results/TEMPERED-READING-THEORY-001/audit.json',
    'src/voynich/tempered_reading_trace.py', 'src/voynich/regrowth_trace.py',
    'scripts/run_tempered_reading_admit001.py', 'scripts/audit_tempered_reading_admit001.py',
    'tests/test_tempered_reading_admission.py', 'docs/experiments/TEMPERED-READING-ADMIT-001.md'])))


def require_prior():
    previous = prior.require_prior()
    for out, status, audit_status in ((prior.OUT, 'PASS_original_source_structural_literal_ratio_and_cost',
            'PASS_original_source_map_eligibility_target_and_ratio_replay'), (ROOT/'results'/theory.EXP,
            'PASS_exact_power_flux_radical_and_product_laws', 'PASS_receipt_hash_and_arithmetic_closure')):
        result, audit = (json.loads((out/n).read_text()) for n in ('result.json', 'audit.json'))
        assert result['status'] == status and audit['status'] == audit_status
        assert audit['result'] == training.artifact(out/'result.json')
        assert not (out/'failure.json').exists() and not (out/'audit-failure.json').exists()
    return previous


def admission(freeze):
    training.old.require_frozen(freeze, PATHS)
    previous = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'allocated_frames': CASES*len(PHASES), 'allocated_local_attempts': CASES*len(PHASES)*SWEEPS*len(DEGREES),
        'degrees': list(DEGREES), 'sweeps': SWEEPS, 'seed_base': SEED, 'no_gold_metrics': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    cells, private = [], 0
    try:
        def guard():
            resources = training.resource_report(wall, cpu)
            if (resources['wall_seconds'] > WALL or resources['cpu_seconds'] > CPU
                    or resources['peak_rss_bytes'] > HOST or private > PRIVATE):
                raise MemoryError('Registered replica admission cap; no retry or extension')
            return resources

        source, _ = prior.load_source()
        arrays = prior.array_identity(source)
        assert arrays == previous['source_arrays']
        frames = [c for c in previous['cells'] if c['phase'] in PHASES]
        assert [(c['case'], c['phase']) for c in frames] == [(i, p) for i in range(CASES) for p in PHASES]
        for index, frame in enumerate(frames):
            original = prior.load_archive(frame['archive'])
            env = ReadingEnvironment(original['records'], rows=23, glyphs=6)
            sampler = SourceRegrowth(source, env)
            began, began_cpu = time.monotonic(), time.process_time()
            trace = replica_trace(sampler, original['base']['actions'], seed=SEED+index, sweeps=SWEEPS,
                                  degrees=DEGREES, progress=guard)
            assert prior.prior.reading_receipt(sampler.path(forced_actions=tuple(trace['initial']['actions']))) == original['base']
            value = {'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'],
                'input_archive': frame['archive'], 'records': original['records'], **trace}
            spec = training.save_new(BULK/f"{frame['case']}-{frame['phase']}.json.gz", value, compressed=True)
            private += spec['bytes']
            cells.append({'case': frame['case'], 'kind': frame['kind'], 'phase': frame['phase'],
                'seed': SEED+index, 'archive': spec, 'summary': trace['summary'],
                'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu})
            print(json.dumps({'completed_frames': len(cells), 'allocated_frames': CASES*len(PHASES)}), flush=True)
        locals_ = sum(c['summary']['local_attempts'] for c in cells)
        mean = sum(c['cpu_seconds'] for c in cells)/locals_
        gates = {'complete_fixed_allocation': locals_ == CASES*len(PHASES)*SWEEPS*len(DEGREES),
                 'mean_cpu_per_local_attempt_under_two_seconds': mean <= 2}
        assert all(gates.values())
        training.old.require_frozen(freeze, PATHS)
        assert prior.array_identity(source) == arrays
        training.save_new(OUT/'result.json', {'status': 'PASS_original_source_replica_cost_and_recorded_transport',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells,
            'source_arrays': arrays, 'gates': gates, 'mean_cpu_per_local_attempt_seconds': mean,
            'private_archive_bytes': private, 'resources': guard(), 'no_gold_metrics_neural_training_or_optimizer': True,
            'no_recovery_mixing_or_historical_claim': True})
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
