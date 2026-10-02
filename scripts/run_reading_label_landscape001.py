"""One complete all-pairs development diagnostic; no chain, fitting or Gold selection."""
import argparse
import json
import signal
import time

import torch

from scripts import check_reading_label_transport001 as theory
from scripts import run_reading_regrowth_recovery001 as recovery
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.reading_endpoint_metrics import reading_endpoint_metrics
from voynich.reading_label_landscape import complete_label_landscape
from voynich.reading_label_orbit import label_orbit_obstructions
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment
from voynich.source_action_training import action_episode

training = recovery.admitted.prior.training
ROOT = recovery.ROOT
EXP = 'READING-LABEL-LANDSCAPE-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
PHASES = ('initial', *(f'{seed}-{arm}' for seed in recovery.SEEDS for arm in recovery.ARMS))
WALL, CPU, HOST, PRIVATE = 7200, 6900, 2*1024**3, 256*1024**2
PATHS = sorted(set([*recovery.PATHS, *theory.PATHS,
    'results/READING-REGROWTH-RECOVERY-001/result.json',
    'results/READING-REGROWTH-RECOVERY-001/sealed-cells.json',
    'results/READING-REGROWTH-RECOVERY-001/audit.json',
    'results/READING-LABEL-TRANSPORT-THEORY-001/result.json',
    'results/READING-LABEL-TRANSPORT-THEORY-001/audit.json',
    'src/voynich/reading_label_orbit.py', 'src/voynich/reading_label_landscape.py',
    'tests/test_reading_label_orbit.py', 'tests/test_reading_label_landscape.py',
    'tests/test_reading_label_landscape_driver.py',
    'scripts/run_reading_label_landscape001.py', 'scripts/audit_reading_label_landscape001.py',
    'docs/experiments/READING-LABEL-LANDSCAPE-001.md',
    'docs/research/reading-label-landscape-2026-10-02.md']))


def require_prior():
    for out, status, audit_status in ((recovery.OUT, 'COMPLETE_registered_recovery_campaign',
            'PASS_full_registered_recovery_replay'), (ROOT/'results'/theory.EXP,
            'PASS_exact_finite_label_and_fixed_mixture', 'PASS_receipt_hash_and_arithmetic_closure')):
        result, audit = (json.loads((out/name).read_text()) for name in ('result.json', 'audit.json'))
        assert result['status'] == status and audit['status'] == audit_status
        assert audit['result'] == training.artifact(out/'result.json')
        assert not (out/'audit-failure.json').exists() and not (out/'failure.json').exists()
    return json.loads((recovery.OUT/'result.json').read_text())


def endpoint_frames(prior, initials, case):
    initial = load_archive(initials[case]['archive'])
    yield 'initial', initial['prediction']['actions'], initials[case]['archive']
    for seed in recovery.SEEDS:
        for arm in recovery.ARMS:
            cell = next(c for c in prior['cells'] if c['seed'] == seed and c['case'] == case and c['arm'] == arm)
            archive = load_archive(cell['archive'])
            yield f'{seed}-{arm}', archive['final']['actions'], cell['archive']
            del archive


def reading_receipt(point):
    return {'actions': list(point.actions), 'key': list(point.state.key),
        'offsets': list(point.state.offsets), 'texts': list(map(list, point.state.texts)), 'complete': True,
        'point_not_policy_draw_or_chain_density': True}


def measure(freeze):
    training.old.require_frozen(freeze, PATHS)
    prior = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'allocated_states': recovery.CASES*len(PHASES), 'allocated_pairs': recovery.CASES*len(PHASES)*253})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    cells, private_bytes = [], 0
    try:
        def guard():
            value = training.resource_report(wall, cpu)
            if (value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST
                    or private_bytes > PRIVATE):
                raise MemoryError('Registered all-pairs resource cap; no retry')
            return value
        _, episodes, episode_sampler, allocated, initials = recovery.admitted.load_inputs()
        assert len(allocated) == recovery.CASES and len(episodes) == recovery.POSITIVE
        source, _ = load_source()
        arrays = array_identity(source)
        for case, (kind, paired, records) in enumerate(allocated):
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            for phase, actions, input_archive in endpoint_frames(prior, initials, case):
                sampler = SourceRegrowth(source, env)
                path = sampler.path(forced_actions=tuple(actions), progress=guard)
                assert path.complete
                pairs, best, summary = complete_label_landscape(sampler, path, progress=guard)
                assert len(pairs) == 253
                value = {'case': case, 'kind': kind, 'paired_case': paired, 'phase': phase,
                    'records': list(map(list, records)), 'input_archive': input_archive,
                    'base': reading_receipt(path), 'best_by_target': reading_receipt(best or path),
                    'pairs': pairs, 'summary': summary, 'gold_metrics_not_yet_computed': True}
                archive = training.save_new(BULK/f'{case}-{phase}.json.gz', value, compressed=True)
                private_bytes += archive['bytes']
                cells.append({'case': case, 'kind': kind, 'phase': phase, 'archive': archive, 'summary': summary})
                del value, pairs, best, path, sampler
                print(json.dumps({'completed_states': len(cells), 'allocated_states': recovery.CASES*len(PHASES),
                                  'private_bytes': private_bytes, **guard()}), flush=True)
        assert [(c['case'], c['phase']) for c in cells] == [(i, p) for i in range(recovery.CASES) for p in PHASES]
        sealed = training.save_new(OUT/'sealed-cells.json', {'freeze': freeze, 'cells': cells,
            'all_pairs_complete_before_gold_metrics': True, 'resources': guard()})
        known, metrics = {}, {}
        for case, episode in enumerate(episodes):
            env, trace = action_episode(episode_sampler, episode)
            known[case] = env, trace[2]
        for phase in PHASES:
            positive = [load_archive(c['archive']) for c in cells if c['phase'] == phase and c['kind'] == 'positive']
            assert len(positive) == recovery.POSITIVE
            metrics[phase] = {'base': reading_endpoint_metrics([p['base'] for p in positive], episodes, episode_sampler),
                'best_by_target': reading_endpoint_metrics([p['best_by_target'] for p in positive], episodes, episode_sampler),
                'orbit_obstructions': [label_orbit_obstructions(known[p['case']][0],
                    recovery.admitted.literal_replay(known[p['case']][0], p['base']), known[p['case']][1])
                    for p in positive]}
            del positive
        training.old.require_frozen(freeze, PATHS)
        assert array_identity(source) == arrays
        result = {'status': 'COMPLETE_registered_all_pairs_development_diagnostic', 'freeze': freeze,
            'inputs': [training.artifact(ROOT/p) for p in PATHS], 'sealed_cells': sealed, 'cells': cells,
            'known_answer_metrics': metrics, 'source_arrays': arrays, 'private_archive_bytes': private_bytes,
            'resources': guard(), 'paid_spend_usd': 0, 'total_pairs': sum(c['summary']['pairs'] for c in cells),
            'no_sampling_training_or_empirical_recovery_gate': True, 'not_historical_or_posterior_claim': True}
        training.save_new(OUT/'result.json', result)
        return {'states': len(cells), 'pairs': result['total_pairs']}
    except Exception as error:
        training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'private_archive_bytes': private_bytes, 'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(measure(parser.parse_args().freeze))
