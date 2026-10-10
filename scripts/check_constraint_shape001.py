"""ONE finite qualification of exact shape marginals, proposals and exchanges."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import signal
import time

from scripts import run_constraint_reading_admit001 as prior
from tests.test_constraint_shape import direct_shapes, independent_components, summed_keys, source, EPSILONS
from tests.test_constraint_reading import states as key_states
from voynich import constraint_shape as law
from voynich.source_action_proposal import ReadingEnvironment

EXP = 'CONSTRAINT-SHAPE-THEORY-001'
ROOT = prior.ROOT
training = prior.training
PANELS = tuple((rows, contextual, records) for rows in (3, 4) for contextual in (False, True)
    for records in (((0, 1),), ((0, 0),), ((0,), (1,)), ((0, 1, 0),)))
KERNELS = ('relabel', 'boundary')
WALL, CPU, HOST = 600, 550, 512*1024**2
PATHS = tuple(sorted(set([*prior.PATHS,
    'results/CONSTRAINT-READING-ADMIT-001/result.json', 'results/CONSTRAINT-READING-ADMIT-001/audit.json',
    'docs/research/constraint-return-and-key-marginalization-2026-10-10.md',
    'tests/test_constraint_marginal_theory.py', 'src/voynich/constraint_shape.py',
    'src/voynich/constraint_shape_sampling.py', 'tests/test_constraint_shape.py',
    'tests/test_constraint_shape_sampling.py', 'scripts/check_constraint_shape001.py',
    'scripts/audit_constraint_shape001.py', 'tests/test_constraint_shape_qualification.py',
    'docs/experiments/CONSTRAINT-SHAPE-THEORY-001.md'])))


def finite_panel(rows, contextual, records, *, progress=lambda: None):
    env, expert = ReadingEnvironment(records, rows=rows, glyphs=2), source(rows, contextual)
    ordered = tuple(sorted(direct_shapes(env), key=lambda s: (s.widths, s.texts)))
    weights = summed_keys(env, expert)
    counted = defaultdict(int)
    counted['states'] = len(ordered)
    counted['enumerated_visited_dictionary_assignments'] = len(key_states(env))
    counted['summed_dictionary_terms'] = len(EPSILONS)*counted['enumerated_visited_dictionary_assignments']
    hard = tuple(s for s in ordered if weights[F(0)][s])
    counted['hard_states'] = len(hard)
    for epsilon in EPSILONS:
        for state in ordered:
            progress()
            assert law.target(expert, env, state, epsilon) == weights[epsilon][state]
            counted['independent_marginal_target_checks'] += 1
    for state in hard:
        assert law.from_hard(env, law.to_hard(env, state)) == state
        counted['hard_bijection_checks'] += 1
    proposal = {}
    for kernel in KERNELS:
        proposal[kernel] = {}
        for state in ordered:
            progress()
            q = defaultdict(F)
            for candidate, forward, args in independent_components(env, state, kernel):
                move = getattr(law, kernel)(env, state, *args)
                assert move.state == candidate and move.forward == forward
                q[candidate] += forward
                counted['direct_proposal_component_checks'] += 1
            assert sum(q.values()) == 1
            proposal[kernel][state] = dict(q)
            counted['component_normalizations'] += 1
        for state in ordered:
            for candidate, _, args in independent_components(env, state, kernel):
                if candidate != state:
                    assert getattr(law, kernel)(env, state, *args).reverse == proposal[kernel][candidate][state]
                    counted['reverse_component_checks'] += 1
        for epsilon in EPSILONS:
            incoming = dict.fromkeys(ordered, F(0))
            for state in ordered:
                progress()
                p = weights[epsilon][state]
                if not p:
                    continue
                stay = F(1)
                for candidate, q in proposal[kernel][state].items():
                    if candidate == state:
                        continue
                    other, qr = weights[epsilon][candidate], proposal[kernel][candidate][state]
                    ratio = other*qr/(p*q)
                    moved = q*min(F(1), ratio)
                    incoming[candidate] += p*moved
                    stay -= moved
                    if other:
                        assert p*moved == other*qr*min(F(1), 1/ratio)
                    else:
                        assert moved == 0
                        counted['hard_incompatible_local_rejections'] += 1
                    counted['exact_local_flux_checks'] += 1
                    # Omit reverse component correction on the asymmetric boundary.
                    wrong = p*q*min(F(1), other/p)
                    back = other*qr*min(F(1), p/other) if other else F(0)
                    counted['wrong_proposal_flux_failures'] += wrong != back
                    ms, mt = len({a for t in state.texts for a in t}), len({a for t in candidate.texts for a in t})
                    altered = ratio*F(len(env.pool))**(mt-ms)
                    wrong = p*q*min(F(1), altered)
                    back = other*qr*min(F(1), 1/altered) if altered else F(0)
                    counted['wrong_visited_prior_flux_failures'] += wrong != back
                assert 0 <= stay <= 1
                incoming[state] += p*stay
            assert incoming == weights[epsilon]
            counted['stationarity_equations'] += len(ordered)
    reached, frontier = {ordered[0]}, [ordered[0]]
    while frontier:
        state = frontier.pop()
        for kernel in proposal.values():
            for candidate in kernel[state]:
                if candidate not in reached:
                    reached.add(candidate)
                    frontier.append(candidate)
    assert reached == set(ordered)
    counted['soft_connected_states'] = len(reached)
    for ex, ey in zip(EPSILONS, EPSILONS[1:]):
        for x in ordered:
            if not weights[ex][x]:
                continue
            progress()
            for y in ordered:
                old = weights[ex][x]*weights[ey][y]
                new = weights[ex][y]*weights[ey][x]
                actual = law.swap_ratio(env, x, y, ex, ey)
                assert actual == new/old
                if new:
                    back = law.swap_ratio(env, y, x, ex, ey)
                    assert old*min(F(1), actual) == new*min(F(1), back)
                    # Explicit-key rule accepts any two hard-compatible states with ratio1;
                    # after key marginalization that rule can violate flux, even when valid.
                    counted['wrong_explicit_key_exchange_flux_failures'] += old != new
                else:
                    assert actual == 0
                    counted['hard_incompatible_exchange_rejections'] += 1
                counted['exact_exchange_flux_checks'] += 1
    # Singleton separate records have only symmetric relabel and boundary identity:
    # wrong proposal correction may have no witness in that panel, but must fail overall.
    assert all(counted[k] > 0 for k in ('states', 'hard_states', 'exact_local_flux_checks',
        'exact_exchange_flux_checks', 'wrong_visited_prior_flux_failures',
        'wrong_explicit_key_exchange_flux_failures'))
    return {'rows': rows, 'contextual': contextual, 'records': list(map(list, records)),
        'state_sha256': hashlib.sha256(json.dumps([{'widths': s.widths, 'texts': s.texts} for s in ordered],
                                                sort_keys=True).encode()).hexdigest(), **dict(counted)}


def check(freeze):
    training.old.require_frozen(freeze, PATHS)
    prior.require_prior()
    result, audit = (json.loads((prior.OUT/name).read_text()) for name in ('result.json', 'audit.json'))
    assert result['status'] == 'PASS_constraint_dispatch_original_source_cost_and_cold_conformance'
    assert audit['status'] == 'PASS_full_constraint_dispatch_rng_source_and_cold_reference_replay'
    assert audit['result'] == training.artifact(prior.OUT/'result.json')
    out = ROOT/'results'/EXP
    training.save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        def guard():
            value = training.resource_report(wall, cpu)
            if value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST:
                raise MemoryError('Registered shape finite qualification cap; no retry')
            return value
        panels = []
        for args in PANELS:
            panels.append(finite_panel(*args, progress=guard))
            print(json.dumps({'completed_panels': len(panels), 'allocated_panels': len(PANELS)}), flush=True)
        keys = set(panels[0])-{'rows', 'contextual', 'records', 'state_sha256'}
        totals = {k: sum(p[k] for p in panels) for k in sorted(keys)}
        assert all(v > 0 for v in totals.values())
        training.old.require_frozen(freeze, PATHS)
        training.save_new(out/'result.json', {'status': 'PASS_exact_collapsed_shape_targets_components_and_exchanges',
            'freeze': freeze, 'inputs': [training.artifact(ROOT/p) for p in PATHS], 'panels': panels,
            'totals': totals, 'epsilons': [str(e) for e in EPSILONS], 'kernels': list(KERNELS), 'resources': guard(),
            'software': {name: importlib.metadata.version(name) for name in ('numpy', 'torch')},
            'no_original_source_rng_neural_training_or_recovery': True, 'independent_expert_review': False,
            'independent_direct_shapes_components_and_full_dictionary_sums': True})
        return totals
    except Exception as error:
        training.save_new(out/'failure.json', {'error': repr(error), 'panels': locals().get('panels', []),
            'resources': training.resource_report(wall, cpu), 'no_retry': True})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(check(parser.parse_args().freeze))
