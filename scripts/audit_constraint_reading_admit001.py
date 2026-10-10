"""ONE full independent construction, source-target, RNG and cold-reference replay."""
import argparse
from fractions import Fraction as F
import json
import signal
import time

import numpy as np
import torch

from scripts import run_constraint_reading_admit001 as run
from tests.test_constraint_sampling import alternative_step, alternative_exchange, full_mass
from scripts.check_constraint_reading001 import independent_distance
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from voynich import constraint_reading as law
from voynich.reading_label_landscape import target_fingerprint
from voynich.regrowth_trace import path_receipt
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment


def check_snapshot(expert, env, state, epsilon, saved):
    mass = full_mass(expert, env, state, epsilon)
    fingerprint = target_fingerprint(mass.numerator, mass.denominator) if mass else None
    assert saved == {'key': list(state.key), 'widths': list(map(list, state.widths)),
        'texts': list(map(list, state.texts)), 'violations': independent_distance(env, state),
        'target_sha256': fingerprint}


def replay(expert, sampler, base, saved, *, seed, sweeps, progress=lambda: None):
    env = sampler.env
    state = law.from_hard(env, base.state)
    assert F(*sampler.target_integers(base)) == full_mass(expert, env, state, F(0))
    check_snapshot(expert, env, state, F(0), saved['initial'])
    assert saved['initial_reference'] == path_receipt(base)
    assert saved['seed'] == seed and saved['sweeps'] == sweeps
    assert saved['epsilons'] == [str(e) for e in run.EPSILONS]
    states = (state,)*len(run.EPSILONS)
    rng = np.random.default_rng(seed)
    assert [r['index'] for r in saved['trace']] == list(range(sweeps))
    local_count = exchange_count = 0
    for index, row in enumerate(saved['trace']):
        progress()
        assert row['rng_before'] == rng.bit_generator.state
        assert len(row['local']) == len(run.EPSILONS)
        retained = list(states)
        for slot, (epsilon, entry) in enumerate(zip(run.EPSILONS, row['local'], strict=True)):
            retained[slot], candidate, _, info = alternative_step(expert, env, retained[slot], epsilon, rng)
            assert entry['info'] == info
            check_snapshot(expert, env, candidate, epsilon, entry['candidate'])
            local_count += 1
        slots = list(range(index % 2, len(run.EPSILONS)-1, 2))
        assert [r['slot'] for r in row['swaps']] == slots
        for slot, entry in zip(slots, row['swaps'], strict=True):
            pair, _, info = alternative_exchange(expert, env, retained[slot], retained[slot+1],
                                                run.EPSILONS[slot], run.EPSILONS[slot+1], rng)
            assert entry['info'] == info
            retained[slot], retained[slot+1] = pair
            exchange_count += 1
        states = tuple(retained)
        assert row['rng_after'] == rng.bit_generator.state and len(row['retained']) == len(states)
        for state, epsilon, snap in zip(states, run.EPSILONS, row['retained'], strict=True):
            check_snapshot(expert, env, state, epsilon, snap)
            assert full_mass(expert, env, state, epsilon) > 0
        cold = law.to_hard(env, states[0])
        assert independent_distance(env, states[0]) == 0
        assert linear_literal(env, row['cold_reference']) == cold
        reference = sampler.path(forced_actions=tuple(row['cold_reference']['actions']), progress=progress)
        assert reference.state == cold and path_receipt(reference) == row['cold_reference']
        assert F(*sampler.target_integers(reference)) == full_mass(expert, env, states[0], F(0))
    # Summary fields refer only to already replayed entries; aggregation helper is shared.
    assert saved['summary'] == run.summary(saved['trace'])
    assert saved['no_gold_selection_training_or_recovery_metrics'] is True
    return {'local_attempts': local_count, 'exchange_attempts': exchange_count, 'cold_references': sweeps}


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    old = run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(run.AUDIT_WALL, run.AUDIT_CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    totals = dict.fromkeys(('local_attempts', 'exchange_attempts', 'cold_references'), 0)
    try:
        def guard():
            r = run.training.resource_report(wall, cpu)
            if r['wall_seconds'] > run.AUDIT_WALL or r['cpu_seconds'] > run.AUDIT_CPU or r['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered soft source/RNG audit cap; no retry')
            return r
        result = json.loads((run.OUT/'result.json').read_text())
        assert result['status'] == 'PASS_constraint_dispatch_original_source_cost_and_cold_conformance'
        assert result['freeze'] == freeze and result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        expected_tiny = list(run.tiny_cases())
        cells = result['cells']
        assert len(cells) == len(expected_tiny)+len(old['cells'])
        source, selected = run.prior.prior.load_source()
        assert run.prior.prior.array_identity(source) == result['source_arrays'] == old['source_arrays']
        assert selected['counts'] == result['source_counts']
        private = 0
        for index, cell in enumerate(cells):
            saved = run.prior.prior.load_archive(cell['archive'])
            assert saved['group'] == cell['group'] and saved['metadata'] == cell['metadata']
            if index < len(expected_tiny):
                expert, sampler, base, seed = expected_tiny[index]
                sweeps = run.TINY_SWEEPS
                assert cell['group'] == 'tiny' and cell['metadata'] == {'tiny_index': index}
            else:
                j = index-len(expected_tiny)
                old_cell = old['cells'][j]
                assert cell['group'] == 'source' and cell['metadata'] == {'source_index': j,
                    'case': old_cell['case'], 'kind': old_cell['kind'], 'phase': old_cell['phase'],
                    'source_archive': old_cell['archive']}
                original = run.prior.prior.load_archive(old_cell['archive'])
                env = ReadingEnvironment(tuple(map(tuple, original['records'])), rows=23, glyphs=6)
                expert, sampler = source, SourceRegrowth(source, env)
                base = sampler.path(forced_actions=tuple(original['base']['actions']), progress=guard)
                assert run.prior.prior.prior.reading_receipt(base) == original['base']
                seed, sweeps = run.SOURCE_SEED+j, run.SOURCE_SWEEPS
            assert saved['records'] == list(map(list, sampler.env.records))
            checks = replay(expert, sampler, base, saved, seed=seed, sweeps=sweeps, progress=guard)
            for k, count in checks.items():
                totals[k] += count
            assert cell['summary'] == saved['summary'] and min(cell['wall_seconds'], cell['cpu_seconds']) >= 0
            private += cell['archive']['bytes']
            guard()
            print(json.dumps({'audited_cells': index+1, 'allocated_cells': len(cells)}), flush=True)
        assert result['gates'] == run.gates(cells) and all(result['gates'].values())
        assert result['private_archive_bytes'] == private <= run.PRIVATE
        assert result['mean_source_cell_cpu_per_local_seconds'] == sum(
            c['cpu_seconds'] for c in cells if c['group'] == 'source')/sum(
                c['summary']['local_attempts'] for c in cells if c['group'] == 'source')
        r = result['resources']
        assert 0 <= r['wall_seconds'] <= run.WALL and 0 <= r['cpu_seconds'] <= run.CPU
        assert r['peak_rss_bytes'] <= run.HOST and r['paid_spend_usd'] == 0
        assert result['no_gold_selection_training_or_recovery_metrics'] is True
        assert run.prior.prior.array_identity(source) == result['source_arrays']
        run.training.old.require_frozen(freeze, run.PATHS)
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_full_constraint_dispatch_rng_source_and_cold_reference_replay',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'totals': totals,
            'resources': guard(), 'independent_expert_review': False,
            'independent_selectors_construction_full_targets_and_acceptance_quantiles': True,
            'legacy_reference_counts_summary_and_gate_aggregation_shared': True,
            'no_training_gold_selection_or_recovery_metrics': True})
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
