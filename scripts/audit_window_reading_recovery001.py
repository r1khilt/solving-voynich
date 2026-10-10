"""ONE fresh window recovery audit; full alternate families, targets, RNG and metrics.

Controller/metric scaffolding copied from the prior qualified audit; old bytes unchanged.
"""
import argparse
import json
import math
import signal
import time
from fractions import Fraction as F

import numpy as np
import torch

from scripts import run_window_reading_recovery001 as run
from scripts.audit_reading_pair_admit001 import independent_point
from scripts.audit_window_reading_admit001 import direct_family
from tests.test_window_conditionals import lattice_quantile
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from voynich.exact_radical import RadicalRatio
from voynich.reading_label_landscape import target_fingerprint
from voynich.regrowth_trace import path_receipt
from voynich.source_action_training import action_episode
from voynich.tempered_cold_selection import select_cold
from tests.test_exact_radical import quantile_decision


def independent_recovery_replay(source, sampler, saved, *, degrees, sweeps, progress=lambda: None):
    """Every family, full target, conditional/uniform choice and retained path independently checked."""
    env = sampler.env
    path = sampler.path(forced_actions=tuple(saved['initial']['actions']), progress=progress)
    assert saved['initial'] == path_receipt(path) and linear_literal(env, saved['initial']) == path.state
    assert tuple(saved['degrees']) == degrees == (1,) and saved['sweeps'] == sweeps
    assert saved['arm_kernel'] == saved['arm'] in run.ARMS
    totals = dict.fromkeys(('local_attempts', 'exchange_attempts', 'root_decisions', 'independent_complete_targets'), 0)
    windows = [(r, i, w) for r, c in enumerate(env.records) for i in range(len(c))
               for w in (1, 2) if i+w <= len(c)]
    assert [r['index'] for r in saved['trace']] == list(range(sweeps))
    for index, row in enumerate(saved['trace']):
        progress()
        rng = np.random.default_rng(np.random.SeedSequence((saved['seed'], index)))
        assert row['rng_before'] == rng.bit_generator.state
        rank = int(rng.integers(len(windows)))
        assert row['rank'] == rank and row['window'] == list(windows[rank])
        family = direct_family(env, path.state, *windows[rank])
        assert row['valid_endpoints'] == bool(family)
        candidates, ratios = [], []
        if family:
            old_mass = F(*independent_point(source, env, path.state, progress))
            assert old_mass == F(*sampler.target_integers(path))
            totals['independent_complete_targets'] += 1
            for replacement, state in family.items():
                mass = F(*independent_point(source, env, state, progress))
                ratio = mass/old_mass
                ratios.append(ratio)
                candidates.append({'replacement': list(replacement),
                    'ratio_hex': [hex(ratio.numerator), hex(ratio.denominator)]})
                totals['independent_complete_targets'] += 1
        assert row['candidates'] == candidates
        chosen, blocks, accepted, changed = None, 0, False, False
        if family:
            if saved['arm'] == 'conditional':
                chosen, blocks = lattice_quantile(tuple(ratios), rng, sampler.config.maximum_acceptance_blocks)
                accepted = True
            else:
                chosen = int(rng.integers(len(family)))
                ratio = ratios[chosen]
                accepted, blocks = quantile_decision(RadicalRatio(ratio.numerator, ratio.denominator, 1),
                    rng, sampler.config.maximum_acceptance_blocks)
                totals['root_decisions'] += 1
            state = list(family.values())[chosen]
            changed = accepted and state != path.state
            if changed:
                assert linear_literal(env, row['retained'][0]) == state
                path = sampler.path(forced_actions=tuple(row['retained'][0]['actions']), progress=progress)
                assert path.state == state
        assert (row['selected_index'], row['raw64_blocks'], row['accepted'], row['changed']) == (
            chosen, blocks, accepted, changed)
        assert row['rng_after'] == rng.bit_generator.state and row['retained'] == [path_receipt(path)]
        totals['local_attempts'] += 1
    n, d = independent_point(source, env, path.state, progress)
    totals['independent_complete_targets'] += 1
    assert saved['final'] == [path_receipt(path)]
    assert saved['final_log_targets'] == [math.log(n)-math.log(d)]
    independent_summary = {'local_attempts': sweeps,
        'valid_windows': sum(r['valid_endpoints'] for r in saved['trace']),
        'candidate_targets': sum(len(r['candidates']) for r in saved['trace']),
        'accepted': sum(r['accepted'] for r in saved['trace']), 'changed': sum(r['changed'] for r in saved['trace']),
        'raw64_blocks': sum(r['raw64_blocks'] for r in saved['trace'])}
    assert saved['summary'] == independent_summary and saved['no_gold_or_optimization'] is True
    return totals


def integer_edits(left, right):
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(previous[j]+1, row[-1]+1, previous[j-1]+(a != b)))
        previous = row
    return previous[-1]


def independent_metrics(receipts, episodes, sampler):
    assert len(receipts) == len(episodes) > 0
    cases = []
    for receipt, episode in zip(receipts, episodes, strict=True):
        env, trace = action_episode(sampler, episode)
        state = linear_literal(env, receipt)
        truth = trace[2]
        used = [i for i, code in enumerate(truth.key) if code >= 0]
        complete = bool(receipt['complete'])
        letters = sum(map(len, truth.texts))
        matches = sum(state.key[i] == episode[1][i] for i in used) if complete else 0
        cases.append({'complete': complete, 'edits': sum(integer_edits(t, p) for t, p in
            zip(truth.texts, state.texts, strict=True)) if complete else letters, 'true_letters': letters,
            'used_matches': matches, 'used_rows': len(used), 'exact_records': sum(t == p for t, p in
            zip(truth.texts, state.texts, strict=True)) if complete else 0,
            'complete_used_key': complete and matches == len(used)})
    score = {k: sum(c[k] for c in cases) for k in ('edits', 'true_letters', 'used_matches', 'used_rows',
        'exact_records', 'complete_used_key', 'complete')}
    return {**score, 'episodes': len(episodes), 'record_attempts': sum(len(e[0]) for e in episodes), 'cases': cases,
        'unused_rows_not_scored': True, 'failure_penalty_is_full_length': True, 'no_state_density_claim': True}


def verify_cold_selection(source, sampler, saved):
    """Compare selected point to all cold targets using independently rescored readings."""
    states = [(saved['initial'], -1), *((r['retained'][0], r['index']) for r in saved['trace'])]
    best, index, best_target, raw = None, None, F(0), None
    # Rescore each distinct cold state once; repeat states still participate in tie/order checks.
    cache = {}
    for receipt, sweep in states:
        state = linear_literal(sampler.env, receipt)
        if state not in cache:
            cache[state] = independent_point(source, sampler.env, state, lambda: None)
        n, d = cache[state]
        if F(n, d) > best_target:
            best, index, best_target, raw = receipt, sweep, F(n, d), (n, d)
    assert saved['selection'] == {'reading': best, 'selected_sweep': index,
        'target_sha256': target_fingerprint(*raw), 'log_target': math.log(raw[0])-math.log(raw[1]),
        'cold_states_only': True, 'earliest_exact_target_maximum': True}
    assert saved['selection'] == select_cold(saved)
    return best


def independent_gates(metrics, initial):
    def competent(m):
        return (min(m[k] for k in ('used_rows', 'true_letters', 'episodes', 'record_attempts')) > 0
            and 10*m['used_matches'] >= 9*m['used_rows'] and 10*m['edits'] <= m['true_letters']
            and 2*m['exact_records'] >= m['record_attempts'] and 2*m['complete_used_key'] >= m['episodes'])
    per_seed = {}
    for seed in run.SEEDS:
        baseline, teacher = metrics[str(seed)]['uniform'], metrics[str(seed)]['conditional']
        per_seed[str(seed)] = {'conditional_competent': competent(teacher),
            'conditional_better_edits_than_uniform_and_initial': teacher['edits'] < baseline['edits']
                and teacher['edits'] < initial['edits'],
            'conditional_better_used_matches_than_uniform_and_initial': teacher['used_matches'] > baseline['used_matches']
                and teacher['used_matches'] > initial['used_matches']}
    return {'per_seed': per_seed, 'all_recovery_gates_pass': all(all(v.values()) for v in per_seed.values()),
        'uniform_competence_by_seed': {str(s): competent(metrics[str(s)]['uniform']) for s in run.SEEDS},
        'not_historical_or_null_semantic_classification': True}


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(run.AUDIT_WALL, run.AUDIT_CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    totals = dict.fromkeys(('local_attempts', 'exchange_attempts', 'root_decisions', 'independent_complete_targets'), 0)
    try:
        def guard():
            r = run.training.resource_report(wall, cpu)
            if r['wall_seconds'] > run.AUDIT_WALL or r['cpu_seconds'] > run.AUDIT_CPU or r['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered fresh recovery audit cap; no retry')
            return r

        result = json.loads((run.OUT/'result.json').read_text())
        assert result['status'] == 'COMPLETE_registered_fresh_window_recovery_comparison' and result['freeze'] == freeze
        assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        assert [(c['seed'], c['case'], c['arm']) for c in result['cells']] == run.expected_cells()
        seal_path = run.ROOT/result['sealed_cells']['path']
        assert run.training.artifact(seal_path) == result['sealed_cells']
        seal = json.loads(seal_path.read_text())
        assert seal == {'freeze': freeze, 'cells': result['cells'], 'allocation': result['allocation'],
                        'all_selected_cold_outputs_fixed_before_gold_metrics': True}
        episode_sampler, episodes, allocated, provenance = run.allocate()
        assert run.admitted.prior.load_archive(result['allocation']) == json.loads(json.dumps(provenance))
        source, selected_source = run.admitted.prior.load_source()
        assert run.admitted.prior.array_identity(source) == result['source_arrays']
        assert selected_source['counts'] == result['source_counts']
        initial_receipts = []
        selected = {str(s): {a: [] for a in run.ARMS} for s in run.SEEDS}
        private = result['allocation']['bytes']
        for index, cell in enumerate(result['cells']):
            seed, i, arm = cell['seed'], cell['case'], cell['arm']
            saved = run.admitted.prior.load_archive(cell['archive'])
            kind, paired, records = allocated[i]
            assert (saved['seed_base'], saved['case'], saved['arm'], saved['kind'], saved['paired_case']) == (
                seed, i, arm, kind, paired)
            assert saved['seed'] == seed+i and saved['records'] == list(map(list, records))
            sampler, initial, fallback = run.initial_reading(source, records)
            assert saved['initial'] == path_receipt(initial) and saved['initial_fallback'] == fallback
            if seed == run.SEEDS[0] and arm == run.ARMS[0]:
                initial_receipts.append(path_receipt(initial))
            checked = independent_recovery_replay(source, sampler, saved, degrees=run.DEGREES[arm],
                sweeps=run.SWEEPS[arm], progress=guard)
            for key in totals:
                totals[key] += checked[key]
            best = verify_cold_selection(source, sampler, saved)
            selected[str(seed)][arm].append(best)
            assert cell['summary'] == saved['summary'] and cell['initial_fallback'] == fallback
            assert cell['selected_sweep'] == saved['selection']['selected_sweep']
            assert cell['selected_log_target'] == saved['selection']['log_target']
            assert cell['wall_seconds'] >= 0 and cell['cpu_seconds'] >= 0
            private += cell['archive']['bytes']
            guard()
            print(json.dumps({'audited_cells': index+1, 'allocated_cells': len(result['cells'])}), flush=True)
        metrics = {str(seed): {arm: independent_metrics(selected[str(seed)][arm][:run.POSITIVE], episodes,
                    episode_sampler) for arm in run.ARMS} for seed in run.SEEDS}
        initial = independent_metrics(initial_receipts[:run.POSITIVE], episodes, episode_sampler)
        assert result['metrics'] == metrics and result['initial_metrics'] == initial
        assert result['gates'] == independent_gates(metrics, initial) == run.gates(metrics, initial)
        # Oracle helpers are shared; known-answer endpoint metrics and search replay above are separate.
        assert result['post_seal_gold_diagnosis'] == run.post_gold_diagnosis(
            source, episode_sampler, episodes, selected, progress=guard)
        assert totals['local_attempts'] == len(run.expected_cells())*run.local_attempts_per_case()
        assert totals['exchange_attempts'] == 0
        assert private == result['private_archive_bytes'] <= run.PRIVATE
        r = result['resources']
        assert r['wall_seconds'] <= run.WALL and r['cpu_seconds'] <= run.CPU and r['peak_rss_bytes'] <= run.HOST
        assert r['paid_spend_usd'] == 0
        for flag in ('no_historical_or_null_semantic_claim', 'match_is_local_attempts_not_wall_time_or_exact_CPU',
                     'no_neural_fit_or_paid_calls', 'both_arms_evaluate_all_local_family_targets',
                     'paired_observed_window_streams', 'uniform_full_table_cost_is_not_natural_uniform_sampler_cost'):
            assert result[flag] is True
        assert run.admitted.prior.array_identity(source) == result['source_arrays']
        run.training.old.require_frozen(freeze, run.PATHS)
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_full_fresh_window_families_rng_selection_and_metrics_replay',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'totals': totals,
            'resources': guard(), 'gates': result['gates'], 'independent_expert_review': False,
            'legacy_suffix_and_initialization_shared': True, 'gold_oracle_diagnosis_helpers_shared': True,
            'new_kernel_replay_and_integer_metrics_separate': True, 'no_chain_extension': True})
        return totals
    except Exception as error:
        run.training.save_new(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'totals': totals, 'resources': run.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
