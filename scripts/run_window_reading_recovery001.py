"""ONE fresh-key full-family cold repair comparison; seal before all Gold metrics."""
import argparse
import json
import math
import signal
import time

import numpy as np
import torch

from scripts import run_tempered_reading_admit001 as admitted
from scripts import run_window_reading_admit001 as window_admitted
from scripts import run_tempered_reading_recovery001 as previous
from voynich.joint_key_training import EpisodeSampler, dictionary_code
from voynich.reading_endpoint_metrics import competence_gate, reading_endpoint_metrics
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import path_receipt
from voynich.source_action_cipher_diagnosis import source_greedy
from voynich.source_action_proposal import ReadingEnvironment
from voynich.source_action_training import action_episode, control_records
from voynich.reading_label_orbit import label_orbit_obstructions
from voynich.tempered_cold_selection import select_cold
from voynich.window_recovery_trace import repair_trace

EXP = 'WINDOW-READING-RECOVERY-001'
ROOT, training = admitted.ROOT, admitted.training
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
POSITIVE, CONTROL_COUNT, CASES = 16, 8, 33
GENERATION_SEED, CONTROL_SEED, SEEDS = 96401, 96411, (96421, 96429)
ARMS = ('uniform', 'conditional')
DEGREES = {a: (1,) for a in ARMS}
SWEEPS = {a: 256 for a in ARMS}
WALL, CPU, AUDIT_WALL, AUDIT_CPU = 7200, 7000, 10800, 10400
HOST, PRIVATE = 2*1024**3, 4*1024**3
PATHS = tuple(sorted(set([*window_admitted.PATHS, *previous.PATHS,
    'results/WINDOW-READING-ADMIT-001/result.json', 'results/WINDOW-READING-ADMIT-001/audit.json',
    'results/TEMPERED-READING-RECOVERY-001/result.json', 'results/TEMPERED-READING-RECOVERY-001/audit.json',
    'src/voynich/window_recovery_trace.py', 'scripts/run_window_reading_recovery001.py',
    'scripts/audit_window_reading_recovery001.py', 'tests/test_window_reading_recovery.py',
    'docs/experiments/WINDOW-READING-RECOVERY-001.md'])))


def require_prior():
    window_admitted.require_prior()
    previous.require_prior()
    for out, status, audit_status in ((window_admitted.OUT,
        'PASS_original_source_conditional_tables_and_cost', 'PASS_original_source_all_conditional_targets_and_rng'),
        (previous.OUT, 'COMPLETE_registered_fresh_key_recovery_comparison',
         'PASS_full_fresh_recovery_rng_selection_and_metrics_replay')):
        result, audit = (json.loads((out/n).read_text()) for n in ('result.json', 'audit.json'))
        assert result['status'] == status and audit['status'] == audit_status
        assert audit['result'] == training.artifact(out/'result.json')
        assert not (out/'failure.json').exists() and not (out/'audit-failure.json').exists()
    return json.loads((window_admitted.OUT/'result.json').read_text())


def allocate():
    manifest, _, texts, prior_episodes = admitted.prior.training.load_prepared()
    raw = set(manifest['old_forbidden_raw']) | {dictionary_code(e[2]['raw_indices']) for e in prior_episodes}
    canonical = set(manifest['old_forbidden_canonical']) | {dictionary_code(e[1]) for e in prior_episodes}
    prior_result = json.loads((previous.OUT/'result.json').read_text())
    prior_allocation = admitted.prior.load_archive(prior_result['allocation'])
    raw.update(dictionary_code(e['raw_indices']) for e in prior_allocation['metadata'])
    canonical.update(dictionary_code(e['canonical_key']) for e in prior_allocation['metadata'])
    sampler = EpisodeSampler(texts, forbidden_raw=raw, forbidden_canonical=canonical)
    rng = np.random.default_rng(GENERATION_SEED)
    episodes = []
    for _ in range(POSITIVE):
        episode = sampler.sample(rng)
        sampler.forbidden_raw.add(dictionary_code(episode[2]['raw_indices']))
        sampler.forbidden_canonical.add(dictionary_code(episode[1]))
        episodes.append(episode)
    assert not raw.intersection(dictionary_code(e[2]['raw_indices']) for e in episodes)
    assert not canonical.intersection(dictionary_code(e[1]) for e in episodes)
    controls = control_records(episodes, count=CONTROL_COUNT, seed=CONTROL_SEED)
    cases = [('positive', i, e[0]) for i, e in enumerate(episodes)]
    cases.extend((c['kind'], c['paired_case'], c['records']) for c in controls)
    assert len(cases) == CASES and len(episodes) == POSITIVE
    provenance = {'parent_inputs': training.artifact(admitted.prior.training.OUT/'inputs.json'),
        'source_validation': manifest['validation_source'], 'generation_seed': GENERATION_SEED,
        'control_seed': CONTROL_SEED, 'rejected_keys': sampler.rejected_keys,
        'previous_recovery_allocation': prior_result['allocation'],
        'forbidden_raw_count': len(raw), 'forbidden_canonical_count': len(canonical),
        'metadata': [e[2] for e in episodes], 'controls': controls, 'no_new_author_or_language_holdout': True}
    return sampler, episodes, cases, provenance


def initial_reading(source, records):
    """Cipher-only frozen source greedy; dead initialization gets a declared singleton fallback."""
    env = ReadingEnvironment(records, rows=23, glyphs=6)
    prediction, _ = source_greedy(source, env, 'source_increment')
    fallback = prediction['status'] != 'complete_path'
    if fallback:
        state, actions = env.initial, []
        while (i := env.selected_record(state)) is not None:
            action = 2*env.records[i][state.offsets[i]]
            actions.append(action)
            state = env.advance(state, action)
    else:
        actions = prediction['actions']
    sampler = SourceRegrowth(source, env)
    path = sampler.path(forced_actions=tuple(actions))
    assert path.complete
    return sampler, path, fallback


def expected_cells():
    return [(seed, i, arm) for seed in SEEDS for i in range(CASES) for arm in ARMS]


def local_attempts_per_case():
    values = {SWEEPS[a]*len(DEGREES[a]) for a in ARMS}
    assert len(values) == 1 and min(values) > 0
    return values.pop()


def gates(metrics, initial):
    per_seed = {}
    for seed in SEEDS:
        uniform, conditional = (metrics[str(seed)][a] for a in ARMS)
        per_seed[str(seed)] = {'conditional_competent': bool(competence_gate(conditional)),
            'conditional_better_edits_than_uniform_and_initial': conditional['edits'] < min(
                uniform['edits'], initial['edits']),
            'conditional_better_used_matches_than_uniform_and_initial': conditional['used_matches'] > max(
                uniform['used_matches'], initial['used_matches'])}
    return {'per_seed': per_seed, 'all_recovery_gates_pass': all(all(x.values()) for x in per_seed.values()),
            'uniform_competence_by_seed': {str(s): bool(competence_gate(metrics[str(s)]['uniform'])) for s in SEEDS},
            'not_historical_or_null_semantic_classification': True}


def post_gold_diagnosis(source, sampler, episodes, chosen, *, progress=lambda: None):
    """Read Gold only after output seal; available-score and orbit diagnosis, not selection."""
    result = []
    for i, episode in enumerate(episodes):
        env, truth = action_episode(sampler, episode)
        source_sampler = SourceRegrowth(source, env)
        gold = source_sampler.path(forced_actions=truth[1], progress=progress)
        n, d = source_sampler.target_integers(gold)
        gold_log = math.log(n)-math.log(d)
        by_seed = {}
        for seed in SEEDS:
            by_seed[str(seed)] = {}
            for arm in ARMS:
                receipt = chosen[str(seed)][arm][i]
                selected = source_sampler.path(forced_actions=tuple(receipt['actions']), progress=progress)
                sn, sd = source_sampler.target_integers(selected)
                by_seed[str(seed)][arm] = {'gold_target_strictly_higher': n*sd > sn*d,
                    'gold_minus_selected_log_target': gold_log-(math.log(sn)-math.log(sd)),
                    'label_orbit': label_orbit_obstructions(env, selected.state, truth[2])}
        result.append({'case': i, 'gold_log_target': gold_log, 'by_seed': by_seed})
    return result


def recover(freeze):
    training.old.require_frozen(freeze, PATHS)
    prior = require_prior()
    training.save_new(OUT/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time(),
        'allocated_cells': len(expected_cells()), 'allocated_local_attempts': len(expected_cells())*local_attempts_per_case(),
        'seeds': list(SEEDS), 'degrees': DEGREES, 'sweeps': SWEEPS, 'no_gold_metrics_before_seal': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    cells, private, initial_receipts, chosen = [], 0, [], {}
    try:
        def guard():
            r = training.resource_report(wall, cpu)
            if r['wall_seconds'] > WALL or r['cpu_seconds'] > CPU or r['peak_rss_bytes'] > HOST or private > PRIVATE:
                raise MemoryError('Registered fresh recovery cap; no retry or extension')
            return r

        episode_sampler, episodes, allocated, provenance = allocate()
        input_spec = training.save_new(BULK/'allocation.json.gz', provenance, compressed=True)
        private += input_spec['bytes']
        source, selected_source = admitted.prior.load_source()
        arrays = admitted.prior.array_identity(source)
        assert arrays == prior['source_arrays']
        for seed in SEEDS:
            chosen[str(seed)] = {a: [] for a in ARMS}
            for i, (kind, paired, records) in enumerate(allocated):
                sampler, initial, fallback = initial_reading(source, records)
                if seed == SEEDS[0]:
                    initial_receipts.append(path_receipt(initial))
                for arm in ARMS:
                    began, began_cpu = time.monotonic(), time.process_time()
                    trace = repair_trace(sampler, initial.actions, seed=seed+i, sweeps=SWEEPS[arm],
                                         arm=arm, progress=guard)
                    selection = select_cold(trace)
                    value = {'seed_base': seed, 'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm,
                        'records': list(map(list, records)), 'initial_fallback': fallback,
                        'selection': selection, **trace}
                    spec = training.save_new(BULK/f'{seed}-{i}-{arm}.json.gz', value, compressed=True)
                    private += spec['bytes']
                    cells.append({'seed': seed, 'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm,
                        'archive': spec, 'summary': trace['summary'], 'initial_fallback': fallback,
                        'selected_sweep': selection['selected_sweep'], 'selected_log_target': selection['log_target'],
                        'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-began_cpu})
                    chosen[str(seed)][arm].append(selection['reading'])
                    guard()
                    print(json.dumps({'completed_cells': len(cells), 'allocated_cells': len(expected_cells())}), flush=True)
        assert [(c['seed'], c['case'], c['arm']) for c in cells] == expected_cells()
        seal = training.save_new(OUT/'sealed-cells.json', {'freeze': freeze, 'cells': cells,
            'allocation': input_spec, 'all_selected_cold_outputs_fixed_before_gold_metrics': True})
        # FIRST scoring against known generating texts/key occurs after all outputs are sealed.
        metrics = {str(seed): {arm: reading_endpoint_metrics(chosen[str(seed)][arm][:POSITIVE], episodes,
                    episode_sampler) for arm in ARMS} for seed in SEEDS}
        initial = reading_endpoint_metrics(initial_receipts[:POSITIVE], episodes, episode_sampler)
        diagnosis = post_gold_diagnosis(source, episode_sampler, episodes, chosen, progress=guard)
        training.old.require_frozen(freeze, PATHS)
        assert admitted.prior.array_identity(source) == arrays
        training.save_new(OUT/'result.json', {'status': 'COMPLETE_registered_fresh_window_recovery_comparison',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'cells': cells, 'sealed_cells': seal,
            'metrics': metrics, 'initial_metrics': initial, 'gates': gates(metrics, initial),
            'post_seal_gold_diagnosis': diagnosis,
            'source_arrays': arrays, 'source_counts': selected_source['counts'], 'allocation': input_spec,
            'private_archive_bytes': private, 'resources': guard(), 'no_historical_or_null_semantic_claim': True,
            'match_is_local_attempts_not_wall_time_or_exact_CPU': True,
            'both_arms_evaluate_all_local_family_targets': True, 'paired_observed_window_streams': True,
            'uniform_full_table_cost_is_not_natural_uniform_sampler_cost': True, 'no_neural_fit_or_paid_calls': True})
        return gates(metrics, initial)
    except Exception as error:
        training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'private_archive_bytes': private, 'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(recover(parser.parse_args().freeze))
