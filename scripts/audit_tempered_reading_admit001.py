"""ONE full trace replay with alternate selectors, targets and root decisions."""
import argparse
import json
import math
import signal
import time
from fractions import Fraction as F

import numpy as np
import torch

from scripts import run_tempered_reading_admit001 as run
from scripts.audit_reading_pair_admit001 import direct_eligible, independent_point
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from tests.test_exact_radical import quantile_decision
from voynich.exact_radical import RadicalRatio
from voynich.reading_label_landscape import target_fingerprint
from voynich.regrowth_trace import path_receipt
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def independent_replay(source, sampler, saved, *, progress=lambda: None):
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
    degrees = tuple(saved['degrees'])
    assert degrees == run.DEGREES and saved['sweeps'] == run.SWEEPS
    rng = np.random.Generator(np.random.PCG64(saved['seed']))
    checked, roots, swaps_checked = 0, 0, 0
    assert [s['index'] for s in saved['trace']] == list(range(run.SWEEPS))
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
                    state, branch, replacements = run.prior.finite.direct_rewrite(env, old.state, triplet)
                    candidate = reconstruct(run.prior.finite.direct_actions(env, state))
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
                candidate = reconstruct(run.prior.finite.direct_actions(env, state))
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


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(run.WALL, run.CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    totals = dict.fromkeys(('local_attempts', 'exchange_attempts', 'root_decisions', 'independent_complete_targets'), 0)
    try:
        def guard():
            r = run.training.resource_report(wall, cpu)
            if r['wall_seconds'] > run.WALL or r['cpu_seconds'] > run.CPU or r['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered replica audit cap; no retry')
            return r

        result = json.loads((run.OUT/'result.json').read_text())
        assert result['status'] == 'PASS_original_source_replica_cost_and_recorded_transport' and result['freeze'] == freeze
        assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        assert [(c['case'], c['phase']) for c in result['cells']] == [(i, p) for i in range(run.CASES) for p in run.PHASES]
        source, _ = run.prior.load_source()
        assert run.prior.array_identity(source) == result['source_arrays']
        private = 0
        for index, cell in enumerate(result['cells']):
            saved = run.prior.load_archive(cell['archive'])
            original = run.prior.load_archive(saved['input_archive'])
            assert (saved['case'], saved['kind'], saved['phase'], saved['seed']) == (
                cell['case'], cell['kind'], cell['phase'], run.SEED+index)
            assert saved['records'] == original['records']
            env = ReadingEnvironment(saved['records'], rows=23, glyphs=6)
            sampler = run.SourceRegrowth(source, env)
            initial = sampler.path(forced_actions=tuple(saved['initial']['actions']))
            assert run.prior.prior.reading_receipt(initial) == original['base']
            checked = independent_replay(source, sampler, saved, progress=guard)
            for key in totals:
                totals[key] += checked[key]
            assert cell['summary'] == saved['summary'] and cell['seed'] == saved['seed']
            assert cell['wall_seconds'] >= 0 and cell['cpu_seconds'] >= 0
            private += cell['archive']['bytes']
            print(json.dumps({'audited_frames': index+1, 'allocated_frames': len(result['cells'])}), flush=True)
        assert private == result['private_archive_bytes'] <= run.PRIVATE
        assert totals['local_attempts'] == run.CASES*len(run.PHASES)*run.SWEEPS*len(run.DEGREES)
        assert totals['exchange_attempts'] == run.CASES*len(run.PHASES)*(run.SWEEPS//2)*3
        mean = sum(c['cpu_seconds'] for c in result['cells'])/totals['local_attempts']
        assert mean == result['mean_cpu_per_local_attempt_seconds']
        assert result['gates'] == {'complete_fixed_allocation': True,
                                  'mean_cpu_per_local_attempt_under_two_seconds': mean <= 2}
        assert all(result['gates'].values())
        r = result['resources']
        assert r['wall_seconds'] <= run.WALL and r['cpu_seconds'] <= run.CPU and r['peak_rss_bytes'] <= run.HOST
        assert r['paid_spend_usd'] == 0 and result['no_gold_metrics_neural_training_or_optimizer'] is True
        assert result['no_recovery_mixing_or_historical_claim'] is True
        assert run.prior.array_identity(source) == result['source_arrays']
        run.training.old.require_frozen(freeze, run.PATHS)
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_full_replica_rng_literal_target_and_radical_replay',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'totals': totals,
            'resources': guard(), 'independent_expert_review': False,
            'legacy_suffix_policy_shared_not_independently_rederived': True,
            'new_selectors_and_ratios_independent': True, 'radical_oracle_uses_integer_root_quantiles': True,
            'no_gold_or_chain_extension': True})
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
