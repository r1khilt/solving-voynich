"""ONE bounded algebraic qualification of complete-reading replica kernels."""
import argparse
import importlib.metadata
import itertools
import math
import signal
import time
from collections import defaultdict
from decimal import Decimal as D, localcontext
from fractions import Fraction as F

from scripts import check_reading_pair_rewrite001 as pair
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from tests.test_exact_radical import qualification_decisions
from tests.test_reading_label_transport import exact_terminal_law
from voynich.exact_radical import swap_ratio, tempered_ratio
from voynich.reading_label_transport import score_label_transport, reference_reading
from voynich.reading_pair_rewrite import eligible_triplets
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.tempered_reading import score_complete_state, suffix_correction

EXP = 'TEMPERED-READING-THEORY-001'
DEGREES = (1, 2, 4, 16)
PANELS = [(r, c, records) for r in (3, 4) for c in (False, True) for records in
          ((((0, 1), (0, 1)), ((0, 0), (0, 0)), ((0, 0, 0, 1),)) if r == 3 else
           (((0, 1), (0, 1)),))]
PATHS = tuple(sorted(set([*pair.PATHS, 'src/voynich/exact_radical.py', 'src/voynich/tempered_reading.py',
    'src/voynich/reading_label_transport.py', 'tests/test_reading_label_transport.py',
    'tests/test_exact_radical.py', 'tests/test_tempered_reading.py', 'tests/test_tempered_reading_theory.py',
    'scripts/check_tempered_reading001.py', 'scripts/audit_tempered_reading001.py',
    'docs/research/tempered-structural-reading-proposal-2026-10-08.md',
    'docs/research/tempered-reading-implementation-2026-10-10.md',
    'docs/experiments/TEMPERED-READING-THEORY-001.md'])))


def decimal(fraction):
    return D(fraction.numerator)/D(fraction.denominator)


def acceptance(ratio):
    return min(F(1), F(ratio.numerator, ratio.denominator))


def finite_panel(rows, contextual, records):
    source = pair.source_fixture(rows, contextual)
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    config = RegrowthConfig(stop=F(1, 3), grid_bits=8)
    sampler = SourceRegrowth(source, env, config)
    terminal = exact_terminal_law(source, env, config)
    paths, target, q = {}, {}, {}
    for state, actions, counts, probability, complete in terminal:
        path = sampler.path(forced_actions=actions)
        assert (path.state, path.counts, path.complete) == (state, counts, complete)
        if complete:
            paths[state] = path
            target[state] = pair.direct_target(source, env, state, config.stop)
            assert F(*sampler.target_integers(path)) == target[state]
            point = score_complete_state(sampler, state)
            assert reference_reading(sampler, point) == path
            q[state] = probability
    assert set(paths) == set(pair.enumerate_complete(env))
    states = tuple(paths)
    eligible, label_edges, structural_edges, regrowth_edges = {}, {}, {}, {}
    counters = dict.fromkeys(('fast_score_reference_states', 'structural_components', 'label_components',
        'regrowth_complete_components', 'failed_suffix_components', 'exact_power_flux_checks',
        'wrong_powered_selector_flux_failures', 'wrong_powered_suffix_flux_failures',
        'wrong_warm_prior_flux_failures', 'replica_exchange_checks', 'reversed_swap_flux_failures',
        'zero_eligibility_states'), 0)
    for state, path in paths.items():
        counters['fast_score_reference_states'] += 1
        eligible[state] = tuple(t for t in itertools.permutations(range(rows), 3)
                                if pair.direct_rewrite(env, state, t)[1] != 'identity')
        assert eligible_triplets(env, state) == eligible[state]
        counters['zero_eligibility_states'] += not eligible[state]
    # Independently constructed selection laws. Every tuple is one augmented
    # proposal edge, so duplicate endpoints are accumulated, not deduplicated.
    for state, path in paths.items():
        structural = []
        for triplet in eligible[state]:
            destination, _, _ = pair.direct_rewrite(env, state, triplet)
            reverse = eligible[destination]
            assert triplet in reverse
            point = score_complete_state(sampler, destination)
            assert reference_reading(sampler, point) == paths[destination]
            structural.append((destination, F(1, len(eligible[state])), F(1, len(reverse)),
                               F(len(eligible[state]), len(reverse))))
        if not structural:
            structural = [(state, F(1), F(1), F(1))]
        structural_edges[state] = structural
        labels, used = [], sum(k >= 0 for k in state.key)
        for a, b in itertools.combinations(range(rows), 2):
            probability = F(int(state.key[a] >= 0)+int(state.key[b] >= 0), used*(rows-1))
            if not probability:
                continue
            key = list(state.key)
            key[a], key[b] = key[b], key[a]
            def rename(j):
                return b if j == a else a if j == b else j
            destination = ReadingState(tuple(key), state.offsets,
                                       tuple(tuple(rename(j) for j in text) for text in state.texts))
            point = score_label_transport(sampler, path, a, b)
            assert point.state == destination and reference_reading(sampler, point) == paths[destination]
            labels.append((destination, probability, probability, F(1)))
        assert sum(e[1] for e in labels) == sum(e[1] for e in structural) == 1
        label_edges[state] = labels
        regrowth = []
        for cut in range(len(path.actions)):
            cx = F(7, 8*len(path.actions))+(F(1, 8) if cut == 0 else 0)
            prefix = math.prod(F(c, 256) for c in path.counts[:cut])
            successors = [t for t in terminal if t[1][:cut] == path.actions[:cut]]
            assert sum(t[3]/prefix for t in successors) == 1
            for destination, _, _, probability, complete in successors:
                forward = cx*probability/prefix
                if not complete:
                    regrowth.append((state, forward, forward, F(1), True))
                    continue
                other = paths[destination]
                cy = F(7, 8*len(other.actions))+(F(1, 8) if cut == 0 else 0)
                reverse = cy*q[state]/prefix
                h = reverse/forward
                assert suffix_correction(sampler, path, other, cut) == h
                regrowth.append((destination, forward, reverse, h, False))
        assert sum(e[1] for e in regrowth) == 1
        regrowth_edges[state] = regrowth
    maximum = D(0)
    for k in DEGREES:
        weights = {s: (decimal(target[s]).ln()/k).exp() for s in states}
        kernels = []
        for name, proposals in (('structural', structural_edges), ('label', label_edges),
                                 ('regrowth', regrowth_edges)):
            kernel = {}
            for state in states:
                row = defaultdict(D)
                for edge in proposals[state]:
                    destination, forward, reverse, h = edge[:4]
                    if name == 'regrowth' and edge[4]:
                        row[state] += decimal(forward)
                        counters['failed_suffix_components'] += 1
                        continue
                    r = target[destination]/target[state]
                    ratio = tempered_ratio(r, h, k)
                    assert F(ratio.numerator, ratio.denominator) == r*h**k
                    reverse_ratio = tempered_ratio(1/r, 1/h, k)
                    assert (target[state]*forward**k*acceptance(ratio)
                            == target[destination]*reverse**k*acceptance(reverse_ratio))
                    counters['exact_power_flux_checks'] += 1
                    counters[{'structural': 'structural_components', 'label': 'label_components',
                              'regrowth': 'regrowth_complete_components'}[name]] += 1
                    alpha = min(D(1), weights[destination]/weights[state]*decimal(h))
                    row[destination] += decimal(forward)*alpha
                    row[state] += decimal(forward)*(1-alpha)
                    if k > 1:
                        bad = min(F(1), r*h)
                        back = min(F(1), 1/(r*h))
                        wrong = target[state]*forward**k*bad != target[destination]*reverse**k*back
                        if name == 'structural':
                            counters['wrong_powered_selector_flux_failures'] += wrong
                        elif name == 'regrowth':
                            counters['wrong_powered_suffix_flux_failures'] += wrong
                        change = sum(j >= 0 for j in destination.key)-sum(j >= 0 for j in state.key)
                        wrong_prior = r*F(len(env.pool))**(-(k-1)*change)*h**k
                        counters['wrong_warm_prior_flux_failures'] += (target[state]*forward**k*min(F(1), wrong_prior)
                            != target[destination]*reverse**k*min(F(1), 1/wrong_prior))
                maximum = max(maximum, abs(sum(row.values())-1))
                assert all(v >= 0 for v in row.values())
                kernel[state] = row
            for destination in states:
                stationary = sum(weights[s]*kernel[s][destination] for s in states)
                maximum = max(maximum, abs(stationary-weights[destination]))
            kernels.append(kernel)
        # Direct mixture row sums and stationarity; weights are fixed, not adaptive.
        for destination in states:
            incoming = sum(weights[s]*(kernels[0][s][destination]/4+kernels[1][s][destination]/4
                                        +kernels[2][s][destination]/2) for s in states)
            maximum = max(maximum, abs(incoming-weights[destination]))
        for cold, warm in ((1, 2), (2, 4), (4, 16)):
            if k != cold:
                continue
            exponent = F(1, cold)-F(1, warm)
            common_power = math.lcm(cold, warm)
            for s, t in itertools.product(states, repeat=2):
                r = target[t]/target[s]
                ratio, inverse = swap_ratio(r, cold, warm), swap_ratio(1/r, cold, warm)
                assert F(ratio.numerator, ratio.denominator) == r**exponent.numerator
                assert ratio.degree == exponent.denominator and common_power % ratio.degree == 0
                old = target[s]**(common_power//cold)*target[t]**(common_power//warm)
                new = target[t]**(common_power//cold)*target[s]**(common_power//warm)
                scale = common_power//ratio.degree
                assert old*acceptance(ratio)**scale == new*acceptance(inverse)**scale
                counters['replica_exchange_checks'] += 1
                counters['reversed_swap_flux_failures'] += old*acceptance(inverse)**scale != new*acceptance(ratio)**scale
    assert maximum < D('1e-75')
    return {'rows': rows, 'contextual': contextual, 'records': list(map(list, records)), 'states': len(states),
            'degrees': list(DEGREES), 'maximum_decimal_residual': str(maximum),
            'kernel_stationarity_equations': 4*len(DEGREES)*len(states), **counters}


def product_sweep_panel():
    """Direct27-state product enumeration with rational perfect-power targets."""
    degrees, values = (1, 2, 4), (1, 2, 3)
    targets = {i: F(values[i]**4) for i in range(3)}
    states = tuple(itertools.product(range(3), repeat=3))
    weight = {s: math.prod(F(values[i]**(4//k)) for i, k in zip(s, degrees, strict=True)) for s in states}
    distribution = weight.copy()
    for sweep in (0, 1):
        for slot, k in enumerate(degrees):
            updated = defaultdict(F)
            for state, mass in distribution.items():
                for replacement in range(3):
                    alpha = min(F(1), F(values[replacement], values[state[slot]])**(4//k))
                    ratio = tempered_ratio(targets[replacement]/targets[state[slot]], F(1), k)
                    assert acceptance(ratio) == alpha**k
                    new = state[:slot]+(replacement,)+state[slot+1:]
                    updated[new] += mass*alpha/3
                    updated[state] += mass*(1-alpha)/3
            distribution = dict(updated)
        for slot in range(sweep % 2, 2, 2):
            updated = defaultdict(F)
            for state, mass in distribution.items():
                new = list(state)
                new[slot], new[slot+1] = new[slot+1], new[slot]
                new = tuple(new)
                alpha = min(F(1), weight[new]/weight[state])
                ratio = swap_ratio(targets[state[slot+1]]/targets[state[slot]], degrees[slot], degrees[slot+1])
                assert acceptance(ratio) == alpha**ratio.degree
                updated[new] += mass*alpha
                updated[state] += mass*(1-alpha)
            distribution = dict(updated)
        assert distribution == weight
    return {'product_states': 27, 'exact_product_stationarity_equations': 54}


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    limit_resources(1800, 1700)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        with localcontext() as context:
            context.prec = 100
            panels = [finite_panel(*p) for p in PANELS]
        totals = {key: sum(p[key] for p in panels) for key in panels[0] if key not in
                  ('rows', 'contextual', 'records', 'degrees', 'maximum_decimal_residual')}
        assert all(v > 0 for v in totals.values())
        root = qualification_decisions()
        product = product_sweep_panel()
        resources = resource_report(wall, cpu)
        assert resources['wall_seconds'] <= 1800 and resources['cpu_seconds'] <= 1700
        assert resources['peak_rss_bytes'] <= 512*1024**2
        save_new(out/'result.json', {'status': 'PASS_exact_power_flux_radical_and_product_laws', 'freeze': freeze,
            'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels, 'totals': totals,
            'radical_checks': root, 'product_checks': product, 'resources': resources,
            'software': {n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'pytest')},
            'no_original_source_training_gold_or_recovery': True, 'independent_expert_review': False})
        return totals
    except Exception as error:
        save_new(out/'failure.json', {'error': repr(error), 'no_retry': True,
                                    'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(check(parser.parse_args().freeze))
