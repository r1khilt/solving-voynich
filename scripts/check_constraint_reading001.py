"""ONE prospective finite qualification of relaxed-channel component and exchange laws."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import signal
import time
from types import SimpleNamespace

import numpy as np

from scripts import run_window_reading_recovery001 as previous
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from tests.test_constraint_reading import states, components
from voynich import constraint_reading as c
from voynich.source_action_proposal import ReadingEnvironment

EXP = 'CONSTRAINT-READING-THEORY-001'
PANELS = tuple((r, context, records) for r in (3, 4) for context in (False, True)
    for records in (((0, 1),), ((0, 0),), ((0,), (1,))))
EPSILONS = (F(0), F(1, 8), F(1, 2), F(1))
KERNELS = ('rebind', 'relabel', 'boundary')
WALL, CPU, HOST = 600, 550, 512*1024**2
PATHS = tuple(sorted(set([*previous.PATHS, 'src/voynich/constraint_reading.py',
    'tests/test_constraint_reading.py', 'tests/test_window_support_obstruction.py',
    'scripts/check_constraint_reading001.py', 'scripts/audit_constraint_reading001.py',
    'tests/test_constraint_reading_qualification.py',
    'docs/research/window-support-and-constraint-relaxation-2026-10-10.md',
    'docs/experiments/CONSTRAINT-READING-THEORY-001.md'])))


def independent_distance(env, s):
    """All strings have length1/2: independent finite edit-distance cases."""
    result = 0
    for observed, widths, text in zip(env.records, s.widths, s.texts, strict=True):
        pos = 0
        for width, row in zip(widths, text, strict=True):
            a, b = env.pool[s.key[row]], observed[pos:pos+width]
            if len(a) == len(b):
                result += sum(x != y for x, y in zip(a, b, strict=True))
            else:
                shorter, longer = (a, b) if len(a) < len(b) else (b, a)
                result += 1 if shorter[0] in longer else 2
            pos += width
    return result


def independent_base(source, env, s):
    value = F(224, 225)**sum(map(len, s.texts))*F(1, 225)**len(s.texts)
    value /= len(env.pool)**len({a for text in s.texts for a in text})
    for text in s.texts:
        context = 0
        for a in text:
            value *= F(*float(source.probabilities[context, a]).as_integer_ratio())
            context = int(source.transitions[context, a])
    return value


def finite_panel(rows, contextual, records, *, progress=lambda: None):
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    probabilities = np.array(([.5, .25, .25] if rows == 3 else [.5, .25, .125, .125]), dtype=np.float64)
    p = np.array([np.roll(probabilities, i) for i in range(rows if contextual else 1)])
    t = np.array([[(i+a+1) % rows if contextual else 0 for a in range(rows)]
                  for i in range(len(p))], dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    ordered = tuple(sorted(states(env), key=lambda s: (s.key, s.widths, s.texts)))
    counted = defaultdict(int)
    counted['states'] = len(ordered)
    distance = {s: independent_distance(env, s) for s in ordered}
    base = {s: independent_base(source, env, s) for s in ordered}
    weights = {e: {s: base[s]*e**distance[s] for s in ordered} for e in EPSILONS}
    hard = tuple(s for s in ordered if distance[s] == 0)
    counted['hard_states'] = len(hard)
    assert hard
    for s in ordered:
        assert c.violations(env, s) == distance[s]
        counted['independent_distance_checks'] += 1
        for e in EPSILONS:
            assert c.target(source, env, s, e) == weights[e][s]
            counted['independent_target_checks'] += 1
        if s in hard:
            assert c.from_hard(env, c.to_hard(env, s)) == s
            counted['hard_bijection_checks'] += 1
    proposal = {}
    for kernel in KERNELS:
        proposal[kernel] = {}
        for s in ordered:
            progress()
            q, moves = defaultdict(F), list(components(env, s, kernel))
            for move in moves:
                assert move.state in base
                q[move.state] += move.forward
            assert sum(q.values()) == 1
            counted['component_normalizations'] += 1
            counted['augmented_components'] += len(moves)
            proposal[kernel][s] = (dict(q), moves)
        for s, (_, moves) in proposal[kernel].items():
            for move in moves:
                if move.state != s:
                    assert move.reverse == proposal[kernel][move.state][0][s]
                    counted['reverse_component_checks'] += 1
    # Entire soft union graph must be connected, independent of target values.
    reached, frontier = {ordered[0]}, [ordered[0]]
    while frontier:
        s = frontier.pop()
        for kernel in proposal.values():
            for other in kernel[s][0]:
                if other not in reached:
                    reached.add(other)
                    frontier.append(other)
    assert reached == set(ordered)
    counted['soft_connected_states'] = len(reached)
    for e, target in weights.items():
        positive = tuple(s for s in ordered if target[s])
        for kernel in proposal.values():
            incoming = dict.fromkeys(positive, F(0))
            for s in positive:
                progress()
                stay = F(1)
                for other, q in kernel[s][0].items():
                    if other == s:
                        continue
                    qr = kernel[other][0][s]
                    moved = q*min(F(1), target[other]*qr/(target[s]*q))
                    flux = target[s]*moved
                    if target[other]:
                        back = qr*min(F(1), target[s]*q/(target[other]*qr))
                        assert flux == target[other]*back
                        incoming[other] += flux
                    else:
                        assert moved == 0
                        counted['hard_inconsistent_rejections'] += 1
                    stay -= moved
                    counted['exact_local_flux_checks'] += 1
                    # Deliberately omit all reverse proposal correction.
                    wrong = target[s]*q*min(F(1), target[other]/target[s])
                    wrong_back = target[other]*qr*min(F(1), target[s]/target[other]) if target[other] else F(0)
                    counted['wrong_uncorrected_proposal_flux_failures'] += wrong != wrong_back
                    # Deliberately omit the visited-row prior in the accept ratio.
                    ms, mt = sum(k >= 0 for k in s.key), sum(k >= 0 for k in other.key)
                    altered_ratio = target[other]*F(len(env.pool))**(mt-ms)/target[s]*qr/q
                    wrong = target[s]*q*min(F(1), altered_ratio)
                    wrong_back = target[other]*qr*min(F(1), 1/altered_ratio) if altered_ratio else F(0)
                    counted['wrong_prior_flux_failures'] += wrong != wrong_back
                assert 0 <= stay <= 1
                incoming[s] += target[s]*stay
            assert incoming == {s: target[s] for s in positive}
            counted['stationarity_equations'] += len(positive)
    # Every positive joint pair on every adjacent constraint layer, no normalizer.
    for ex, ey in zip(EPSILONS, EPSILONS[1:]):
        for x in ordered:
            if not weights[ex][x]:
                continue
            progress()
            for y in ordered:
                old = weights[ex][x]*weights[ey][y]
                new = weights[ex][y]*weights[ey][x]
                actual = c.swap_ratio(env, x, y, ex, ey)
                assert actual == new/old
                if new:
                    backward = c.swap_ratio(env, y, x, ex, ey)
                    assert old*min(F(1), actual) == new*min(F(1), backward)
                else:
                    assert actual == 0
                    counted['hard_inconsistent_exchange_rejections'] += 1
                counted['exact_exchange_flux_checks'] += 1
                counted['wrong_unpenalized_exchange_flux_failures'] += old != new
    assert counted['component_normalizations'] == len(KERNELS)*len(ordered)
    assert all(v > 0 for v in counted.values())
    canonical = [{'key': s.key, 'widths': s.widths, 'texts': s.texts} for s in ordered]
    return {'rows': rows, 'contextual': contextual, 'records': list(map(list, records)),
        'state_sha256': hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
        **dict(counted)}


def boundary_witnesses():
    env = ReadingEnvironment(((0, 1, 0),), rows=3, glyphs=2)
    checks, payload_rejections = 0, 0
    for widths, key, rank in (((2, 1), (env.pool.index((0, 1)), 0, -1), 1),
                             ((1, 2), (0, env.pool.index((1, 0)), -1), 0)):
        for soften in (False, True):
            state = c.ConstraintReading(key, (widths,), ((0, 1),))
            if soften:
                state = c.rebind(env, state, 0, len(env.pool)-1).state
            move = c.boundary(env, state, rank, (), {})
            assert move.state == state and move.forward == move.reverse == F(1, 2)
            assert move.kind == 'boundary_identity'
            checks += 1
            try:
                c.boundary(env, state, rank, (0,), {})
            except ValueError:
                payload_rejections += 1
            else:
                raise AssertionError('Invalid boundary accepted a forged replacement payload')
    return {'invalid_endpoint_identities': checks, 'invalid_endpoint_payload_rejections': payload_rejections}


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        def guard():
            value = resource_report(wall, cpu)
            if value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST:
                raise MemoryError('Registered finite constraint qualification cap; no retry')
            return value
        panels = []
        for panel in PANELS:
            panels.append(finite_panel(*panel, progress=guard))
            print(json.dumps({'completed_panels': len(panels), 'allocated_panels': len(PANELS)}), flush=True)
        keys = set(panels[0])-{'rows', 'contextual', 'records', 'state_sha256'}
        totals = {k: sum(p[k] for p in panels) for k in sorted(keys)}
        assert all(v > 0 for v in totals.values())
        witnesses = boundary_witnesses()
        require_frozen(freeze, PATHS)
        save_new(out/'result.json', {'status': 'PASS_exact_constraint_component_target_exchange_and_support_laws',
            'freeze': freeze, 'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels, 'totals': totals,
            'epsilons': [str(e) for e in EPSILONS], 'kernels': list(KERNELS), 'resources': guard(),
            'boundary_witnesses': witnesses,
            'software': {n: importlib.metadata.version(n) for n in ('numpy', 'torch')},
            'no_original_source_scoring_gold_training_chain_rng_or_recovery': True,
            'independent_expert_review': False, 'preparation_enumerator_and_components_shared': True})
        return totals
    except Exception as error:
        save_new(out/'failure.json', {'error': repr(error), 'no_retry': True, 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(check(parser.parse_args().freeze))
