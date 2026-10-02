"""One longer, two-seed joint reading recovery campaign; fixed approximate cost match."""
import argparse
import json
import signal
import time

import torch

from scripts import run_reading_regrowth_admit001 as admitted
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.reading_endpoint_metrics import competence_gate, reading_endpoint_metrics
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import chain
from voynich.source_action_proposal import ReadingEnvironment

EXP = 'READING-REGROWTH-RECOVERY-001'
ROOT = admitted.prior.training.ROOT
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
SEEDS, CASES, POSITIVE = (92821, 93821), 97, 64
ARMS = admitted.ARMS
MOVES = {'independence': 1280, 'regrowth': 512}
WALL, CPU, HOST, PRIVATE = 21600, 21000, 2*1024**3, 4*1024**3
PATHS = sorted(set([*admitted.PATHS, 'src/voynich/reading_endpoint_metrics.py',
    'tests/test_reading_endpoint_metrics.py', 'tests/test_reading_regrowth_recovery.py',
    'scripts/run_reading_regrowth_recovery001.py', 'scripts/audit_reading_regrowth_recovery001.py',
    'docs/experiments/READING-REGROWTH-RECOVERY-001.md',
    'results/READING-REGROWTH-ADMIT-001/result.json', 'results/READING-REGROWTH-ADMIT-001/audit.json',
    'results/READING-REGROWTH-ADMIT-001/publication-started.json',
    'results/READING-REGROWTH-ADMIT-001/publication-summary.json']))


def expected_cells():
    return [(seed, i, arm) for seed in SEEDS for i in range(CASES) for arm in ARMS]


def gate_summary(metrics, initial):
    flags = {}
    for seed in SEEDS:
        root, local = metrics[str(seed)]['independence'], metrics[str(seed)]['regrowth']
        flags[str(seed)] = {'regrowth_competence': bool(competence_gate(local)),
            'regrowth_better_edits_than_root_and_initial': local['edits'] < min(root['edits'], initial['edits']),
            'regrowth_better_used_matches_than_root_and_initial': local['used_matches'] > max(
                root['used_matches'], initial['used_matches'])}
    return {'per_seed': flags, 'all_recovery_gates_pass': all(all(r.values()) for r in flags.values()),
            'not_historical_or_posterior_convergence_gate': True}


def summaries(cells):
    assert [(c['seed'], c['case'], c['arm']) for c in cells] == expected_cells()
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        for arm in ARMS:
            rows = [c for c in cells if c['seed'] == seed and c['arm'] == arm]
            keys = ('attempts', 'complete_proposals', 'failed_proposals', 'accepted', 'accepted_inventory_changes',
                    'accepted_key_changes', 'accepted_length_changes', 'root_cuts', 'acceptance_raw64_blocks')
            totals = {k: sum(c['summary'][k] for c in rows) for k in keys}
            totals.update({'cell_wall_seconds': sum(c['wall_seconds'] for c in rows),
                'cell_cpu_seconds': sum(c['cpu_seconds'] for c in rows),
                'positive_joint_log_potential_change_sum': sum(c['summary']['joint_log_potential_change']
                    for c in rows if c['kind'] == 'positive'),
                'null_kinds': {kind: {k: sum(c['summary'][k] for c in rows if c['kind'] == kind) for k in keys}
                    for kind in dict.fromkeys(c['kind'] for c in rows if c['kind'] != 'positive')}})
            result[str(seed)][arm] = totals
    return result


def recover(freeze):
    training = admitted.prior.training
    training.old.require_frozen(freeze, PATHS)
    prior = json.loads((admitted.OUT/'result.json').read_text())
    closure = json.loads((admitted.OUT/'audit.json').read_text())
    assert prior['status'] == 'PASS_full_source_regrowth_density_literal_and_cost_admission'
    assert closure['status'] == 'PASS_registered_trace_rng_literal_density_and_source_replay'
    assert closure['result'] == training.artifact(admitted.OUT/'result.json')
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'seeds': SEEDS, 'moves': MOVES, 'allocated_attempts': CASES*len(SEEDS)*sum(MOVES.values())})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    BULK.mkdir(parents=True, exist_ok=True)
    cells, private_bytes, endpoints, initial_endpoints = [], 0, {}, []
    try:
        def guard():
            value = training.resource_report(wall, cpu)
            if (value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST
                    or private_bytes > PRIVATE):
                raise MemoryError('Registered recovery resource bound; no retry or extension')
            return value
        manifest, episodes, episode_sampler, allocated, initials = admitted.load_inputs()
        assert len(allocated) == CASES and len(episodes) == POSITIVE
        source, selected = load_source()
        arrays = array_identity(source)
        for seed in SEEDS:
            endpoints[str(seed)] = {arm: [] for arm in ARMS}
            for i, (kind, paired, records) in enumerate(allocated):
                env = ReadingEnvironment(records, rows=23, glyphs=6)
                initial = load_archive(initials[i]['archive'])
                assert initial['records'] == [list(r) for r in records]
                admitted.initial_receipt(env, initial['prediction'])
                for arm in ARMS:
                    began, began_cpu = time.monotonic(), time.process_time()
                    sampler = SourceRegrowth(source, env, admitted.configuration(arm))
                    value = chain(sampler, initial['prediction']['actions'], seed=seed+i,
                                  steps=MOVES[arm], progress=guard)
                    value.update({'seed_base': seed, 'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm,
                        'records': [list(r) for r in records], 'initial_archive': initials[i]['archive']})
                    final = sampler.path(forced_actions=tuple(value['final']['actions']), progress=guard)
                    admitted.literal_replay(env, value['final'])
                    reference = admitted.point_check(source, sampler, final)
                    spec = training.save_new(BULK/f'{seed}-{i}-{arm}.json.gz', value, compressed=True)
                    private_bytes += spec['bytes']
                    cell = {'seed': seed, 'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm,
                        'archive': spec, 'summary': value['summary'], 'final_source_reference': reference,
                        'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu}
                    cells.append(cell)
                    if i < POSITIVE:
                        endpoints[str(seed)][arm].append(value['final'])
                        if seed == SEEDS[0] and arm == ARMS[0]:
                            initial_endpoints.append(value['initial'])
                    del value
                    print(json.dumps({'completed_cells': len(cells), 'allocated_cells': len(expected_cells()),
                        'seed': seed, 'case': i, 'arm': arm, 'summary': cell['summary'], **guard()}), flush=True)
        assert [(c['seed'], c['case'], c['arm']) for c in cells] == expected_cells()
        # All endpoints are sealed BEFORE opening known-answer metrics or choosing any reading.
        sealed = training.save_new(OUT/'sealed-cells.json', {'freeze': freeze, 'cells': cells,
            'no_gold_metrics_yet': True, 'private_archive_bytes': private_bytes, 'resources': guard()})
        initial_metrics = reading_endpoint_metrics(initial_endpoints, episodes, episode_sampler)
        metrics = {seed: {arm: reading_endpoint_metrics(receipts, episodes, episode_sampler)
                         for arm, receipts in arms.items()} for seed, arms in endpoints.items()}
        training.old.require_frozen(freeze, PATHS)
        assert array_identity(source) == arrays
        result = {'status': 'COMPLETE_registered_recovery_campaign', 'freeze': freeze, 'sealed_cells': sealed,
            'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells, 'summary': summaries(cells),
            'initial_metrics': initial_metrics, 'endpoint_metrics': metrics, 'gates': gate_summary(metrics, initial_metrics),
            'source_arrays': arrays, 'source_counts': selected['counts'], 'source_inputs': selected['inputs'],
            'private_archive_bytes': private_bytes, 'resources': guard(), 'paid_spend_usd': 0,
            'fixed_attempts_approximate_cost_match_not_exact_equal_time': True,
            'chain_state_posterior_or_historical_claim': False, 'original_training_unchanged': True}
        training.save_new(OUT/'result.json', result)
        return result['gates']
    except Exception as error:
        training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'private_archive_bytes': private_bytes, 'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(recover(parser.parse_args().freeze))
