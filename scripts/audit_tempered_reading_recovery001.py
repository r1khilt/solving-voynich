"""ONE fresh recovery replay; alternate selectors/ratios/roots and independent metrics.

The allocation-generalized replay below is a copied version of the separately
frozen admission auditor. Its original bytes and fixed allocation stay unchanged.
"""
import argparse
import json
import math
import signal
import time
from fractions import Fraction as F

import numpy as np
import torch

from scripts import run_tempered_reading_recovery001 as run
from scripts.audit_reading_pair_admit001 import direct_eligible, independent_point
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from voynich.exact_radical import RadicalRatio
from voynich.reading_label_landscape import target_fingerprint
from voynich.regrowth_trace import path_receipt
from voynich.source_action_proposal import ReadingState
from voynich.source_action_training import action_episode
from voynich.tempered_cold_selection import select_cold
from tests.test_exact_radical import quantile_decision


def independent_recovery_replay(source, sampler, saved, *, degrees, sweeps, progress=lambda: None):
    """Qualified suffix policy shared; NEW selectors/ratios/decisions implemented separately."""
    env, cache = sampler.env, {}

    def target(path):
        if path.state not in cache:
            raw = independent_point(source, env, path.state, progress)
            assert F(*raw) == F(*sampler.target_integers(path))
            cache[path.state] = (F(*raw), raw, direct_eligible(env, path.state))
        return cache[path.state]

    def ratio_receipt(ratio):
        return None if ratio is None else {'radicand_sha256': target_fingerprint(ratio.numerator, ratio.denominator),
                                          'degree': ratio.degree}

    def reconstruct(actions):
        return sampler.path(forced_actions=tuple(actions), progress=progress)

    initial = reconstruct(saved['initial']['actions'])
    assert initial.complete and path_receipt(initial) == saved['initial']
    assert linear_literal(env, saved['initial']) == initial.state
    paths = [initial for _ in saved['degrees']]
    assert tuple(saved['degrees']) == degrees and saved['sweeps'] == sweeps
    rng = np.random.Generator(np.random.PCG64(saved['seed']))
    checked, roots, swaps_checked = 0, 0, 0
    assert [s['index'] for s in saved['trace']] == list(range(sweeps))
    for sweep, row in enumerate(saved['trace']):
        assert row['rng_before'] == rng.bit_generator.state
        assert len(row['local']) == len(degrees)
        for slot, (k, entry) in enumerate(zip(degrees, row['local'], strict=True)):
            old = paths[slot]
            kernel = ('regrowth', 'regrowth', 'pair', 'label')[int(rng.integers(4))]
            info = {'kernel': kernel, 'degree': k, 'failed': False, 'accepted': False, 'acceptance_blocks': 0}
            old_t, _, eligible = target(old)
            ratio, candidate = None, old
            if kernel == 'regrowth':
                root = sampler.config.root_mass
                is_root = int(rng.integers(root.denominator)) < root.numerator
                cut = 0 if is_root else int(rng.integers(len(old.actions)))
                candidate = sampler.path(prefix=old.actions[:cut], rng=rng, progress=progress)
                info['cut'] = cut
                if not candidate.complete:
                    info['failed'] = True
                else:
                    forward_cut = root*(cut == 0)+(1-root)/len(old.actions)
                    reverse_cut = root*(cut == 0)+(1-root)/len(candidate.actions)
                    old_q = math.prod(F(c, 2**sampler.config.grid_bits) for c in old.counts[cut:])
                    new_q = math.prod(F(c, 2**sampler.config.grid_bits) for c in candidate.counts[cut:])
                    h = reverse_cut*old_q/(forward_cut*new_q)
            elif kernel == 'pair':
                info['old_degree'] = len(eligible)
                if not eligible:
                    info.update({'branch': 'identity', 'new_degree': 0, 'triplet': None})
                else:
                    triplet = eligible[int(rng.integers(len(eligible)))]
                    state, branch, replacements = run.admitted.prior.finite.direct_rewrite(env, old.state, triplet)
                    candidate = reconstruct(run.admitted.prior.finite.direct_actions(env, state))
                    assert candidate.state == state
                    reverse = target(candidate)[2]
                    assert triplet in reverse
                    h = F(len(eligible), len(reverse))
                    info.update({'triplet': list(triplet), 'new_degree': len(reverse),
                                 'branch': branch, 'replacements': replacements})
            else:
                used = [a for a, code in enumerate(old.state.key) if code >= 0]
                a = used[int(rng.integers(len(used)))]
                b = int(rng.integers(env.rows-1))
                b += b >= a
                a, b = sorted((a, b))
                key = list(old.state.key)
                key[a], key[b] = key[b], key[a]
                def rename(j):
                    return b if j == a else a if j == b else j
                state = ReadingState(tuple(key), old.state.offsets,
                                     tuple(tuple(rename(j) for j in t) for t in old.state.texts))
                candidate = reconstruct(run.admitted.prior.finite.direct_actions(env, state))
                assert candidate.state == state
                h = F(1)
                info['pair'] = [a, b]
            is_identity = kernel == 'pair' and not eligible
            if not info['failed'] and not is_identity:
                value = target(candidate)[0]/old_t*h**k
                ratio = RadicalRatio(value.numerator, value.denominator, k)
                accepted, blocks = quantile_decision(ratio, rng, sampler.config.maximum_acceptance_blocks)
                info.update({'accepted': accepted, 'acceptance_blocks': blocks})
                roots += 1
                if accepted:
                    paths[slot] = candidate
            assert entry['info'] == info and entry['ratio'] == ratio_receipt(ratio)
            observed = entry['candidate']
            assert linear_literal(env, observed['reading']) == candidate.state
            if observed['kind'] == 'reference_path':
                assert observed['reading'] == path_receipt(candidate)
                assert kernel == 'regrowth' or is_identity
            else:
                assert observed['kind'] == 'deterministic_point' and kernel in ('pair', 'label') and not is_identity
                assert observed['reading'] == {'actions': list(candidate.actions), 'key': list(candidate.state.key),
                    'offsets': list(candidate.state.offsets), 'texts': list(map(list, candidate.state.texts)), 'complete': True}
                assert observed['source_terms'] == list(map(list, candidate.source_terms))
                assert observed['target_sha256'] == target_fingerprint(*target(candidate)[1])
            checked += 1
        slots = list(range(sweep % 2, len(degrees)-1, 2))
        assert [s['slot'] for s in row['swaps']] == slots
        for slot, entry in zip(slots, row['swaps'], strict=True):
            exponent = F(1, degrees[slot])-F(1, degrees[slot+1])
            radicand = (target(paths[slot+1])[0]/target(paths[slot])[0])**exponent.numerator
            ratio = RadicalRatio(radicand.numerator, radicand.denominator, exponent.denominator)
            accepted, blocks = quantile_decision(ratio, rng, sampler.config.maximum_acceptance_blocks)
            assert entry['info'] == {'accepted': accepted, 'acceptance_blocks': blocks}
            assert entry['ratio'] == ratio_receipt(ratio)
            if accepted:
                paths[slot], paths[slot+1] = paths[slot+1], paths[slot]
            roots += 1
            swaps_checked += 1
        assert row['retained'] == [path_receipt(p) for p in paths]
        assert row['rng_after'] == rng.bit_generator.state
        progress()
    assert saved['final'] == [path_receipt(p) for p in paths]
    assert saved['final_rng'] == rng.bit_generator.state
    assert saved['final_log_targets'] == [math.log(target(p)[0].numerator)-math.log(target(p)[0].denominator) for p in paths]
    # All fields counted here have already been independently replayed above.
    from voynich.tempered_reading_trace import summarize
    assert saved['summary'] == summarize(saved['trace'], degrees)
    assert saved['no_gold_or_optimization'] is True
    return {'local_attempts': checked, 'exchange_attempts': swaps_checked, 'root_decisions': roots,
            'independent_complete_targets': len(cache)}


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
        assert result['status'] == 'COMPLETE_registered_fresh_key_recovery_comparison' and result['freeze'] == freeze
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
        assert result['gates'] == run.gates(metrics, initial)
        # Oracle helpers are shared; known-answer endpoint metrics and search replay above are separate.
        assert result['post_seal_gold_diagnosis'] == run.post_gold_diagnosis(
            source, episode_sampler, episodes, selected, progress=guard)
        assert totals['local_attempts'] == len(run.expected_cells())*run.local_attempts_per_case()
        assert totals['exchange_attempts'] == len(run.SEEDS)*run.CASES*(run.SWEEPS['replicas']//2)*3
        assert private == result['private_archive_bytes'] <= run.PRIVATE
        r = result['resources']
        assert r['wall_seconds'] <= run.WALL and r['cpu_seconds'] <= run.CPU and r['peak_rss_bytes'] <= run.HOST
        assert r['paid_spend_usd'] == 0
        for flag in ('no_historical_or_null_semantic_claim', 'match_is_local_attempts_not_wall_time_or_exact_CPU',
                     'no_neural_fit_or_paid_calls'):
            assert result[flag] is True
        assert run.admitted.prior.array_identity(source) == result['source_arrays']
        run.training.old.require_frozen(freeze, run.PATHS)
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_full_fresh_recovery_rng_selection_and_metrics_replay',
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

