"""Prospective finite qualification of conditional observed-window repair."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import importlib.metadata
import signal
import time

import numpy as np

from scripts import check_reading_pair_rewrite001 as pair
from scripts import check_tempered_reading001 as prior
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from tests.test_exact_radical import quantile_decision
from tests.test_window_conditionals import lattice_quantile
from voynich.exact_radical import RadicalRatio, tempered_ratio
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.reading_window import observed_window, replace_window, window_choices, window_step
from voynich.source_action_proposal import ReadingEnvironment
from voynich.tempered_reading import point_target, score_complete_state
from voynich.window_conditionals import cold_window_heatbath, rational_categorical, window_ratio

EXP = 'WINDOW-READING-THEORY-001'
DEGREES = (1, 2, 4, 16)
PANELS = tuple((3, c, records) for c in (False, True) for records in
               (((0, 1), (0, 1)), ((0, 0), (0, 0)), ((0, 0, 1),)))+tuple(
               (4, c, ((0, 1), (0, 1))) for c in (False, True))
PATHS = tuple(sorted(set([*prior.PATHS, 'src/voynich/reading_occurrence.py',
    'src/voynich/reading_window.py', 'src/voynich/window_conditionals.py',
    'tests/test_reading_occurrence.py', 'tests/test_reading_window.py', 'tests/test_window_conditionals.py',
    'scripts/check_window_reading001.py', 'scripts/audit_window_reading001.py',
    'tests/test_window_reading_qualification.py', 'docs/research/occurrence-repair-2026-10-10.md',
    'docs/research/window-repair-2026-10-10.md', 'docs/experiments/WINDOW-READING-THEORY-001.md'])))
WALL, CPU, HOST = 600, 550, 512*1024**2
STEPS = 256


def signature(environment, state, window):
    """Alternate interval-indexed conditioning, without release/bind construction."""
    record, start, width = window
    outside, inside, endpoints = [], [], set()
    for r, text in enumerate(state.texts):
        offset = 0
        for row in text:
            unit = environment.pool[state.key[row]]
            end = offset+len(unit)
            assert environment.records[r][offset:end] == unit
            if r == record:
                endpoints.update((offset, end))
            if r != record or end <= start or offset >= start+width:
                outside.append((r, offset, end, row, state.key[row]))
            elif r == record and start <= offset and end <= start+width:
                inside.append(row)
            offset = end
        assert offset == len(environment.records[r])
    if start not in endpoints or start+width not in endpoints:
        return None, None
    return tuple(outside), tuple(inside)


def oracle_groups(environment, states, windows):
    groups, valid_counts = [], dict.fromkeys(states, 0)
    for window in windows:
        grouped = defaultdict(dict)
        for state in states:
            outside, inside = signature(environment, state, window)
            if outside is not None:
                assert inside not in grouped[outside]
                grouped[outside][inside] = state
                valid_counts[state] += 1
        groups.append(dict(grouped))
    return groups, valid_counts


def finite_panel(rows, contextual, records):
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    source = pair.source_fixture(rows, contextual)
    sampler = SourceRegrowth(source, env, RegrowthConfig(stop=F(1, 3), grid_bits=8))
    states = pair.enumerate_complete(env)
    target = {s: pair.direct_target(source, env, s, sampler.config.stop) for s in states}
    windows = [(r, start, width) for r, obs in enumerate(records) for start in range(len(obs))
               for width in (1, 2) if start+width <= len(obs)]
    assert [observed_window(env, i) for i in range(len(windows))] == windows
    groups, valid_counts = oracle_groups(env, states, windows)
    paths = {s: sampler.path(forced_actions=pair.direct_actions(env, s)) for s in states}
    for state in states:
        assert paths[state].complete and paths[state].state == state
        assert F(*sampler.target_integers(paths[state])) == point_target(score_complete_state(sampler, state)) == target[state]
    counters = dict.fromkeys(('states', 'augmented_components', 'valid_components', 'invalid_boundary_components',
        'split_components', 'merge_components', 'visited_count_changes', 'sole_window_row_rebindings',
        'exact_power_flux_checks', 'cold_stationarity_equations', 'wrong_prior_flux_failures',
        'wrong_continuation_flux_failures', 'uncorrected_valid_anchor_flux_failures',
        'seeded_steps', 'independent_root_decisions', 'exact_incremental_target_checks',
        'heatbath_stationarity_equations', 'seeded_heatbath_steps', 'independent_categorical_draws',
        'rescored_suffix_witnesses', 'cancelled_suffix_witnesses'), 0)
    counters['states'] = len(states)
    incoming = defaultdict(F)
    heatbath_incoming = defaultdict(F)
    for state in states:
        normalization = F(0)
        used = sum(k >= 0 for k in state.key)
        length = sum(map(len, state.texts))
        for wi, window in enumerate(windows):
            outside, old_inside = signature(env, state, window)
            family = window_choices(env, state, *window)
            if outside is None:
                assert family is None
                incoming[state] += target[state]/len(windows)
                heatbath_incoming[state] += target[state]/len(windows)
                normalization += F(1, len(windows))
                counters['invalid_boundary_components'] += 1
                counters['augmented_components'] += 1
                counters['exact_power_flux_checks'] += len(DEGREES)
                continue
            alternatives = groups[wi][outside]
            ordered = tuple(sorted(alternatives, key=lambda replacement: (len(replacement), replacement)))
            assert family.replacements == ordered
            q = F(1, len(windows)*len(ordered))
            family_mass = sum(target[s] for s in alternatives.values())
            for replacement, other in alternatives.items():
                assert replace_window(env, state, *window, replacement) == other
                reverse_family = window_choices(env, other, *window)
                assert (reverse_family.replacements, reverse_family.outside_key) == (family.replacements, family.outside_key)
                assert replace_window(env, other, *window, old_inside) == state
                normalization += q
                ratio = target[other]/target[state]
                incremental = window_ratio(sampler, paths[state], *window, replacement)
                assert incremental.state == other and incremental.ratio == ratio
                assert not incremental.omitted_suffix_steps or incremental.contexts_matched
                counters['exact_incremental_target_checks'] += 1
                counters['rescored_suffix_witnesses'] += incremental.suffix_steps > 0
                counters['cancelled_suffix_witnesses'] += incremental.omitted_suffix_steps > 0
                heatbath_incoming[other] += target[state]*target[other]/family_mass/len(windows)
                alpha = min(F(1), ratio)
                incoming[other] += target[state]*q*alpha
                incoming[state] += target[state]*q*(1-alpha)
                counters['augmented_components'] += 1
                counters['valid_components'] += 1
                other_used = sum(k >= 0 for k in other.key)
                other_length = sum(map(len, other.texts))
                counters['split_components'] += other_length == length+1
                counters['merge_components'] += other_length == length-1
                counters['visited_count_changes'] += other_used != used
                counters['sole_window_row_rebindings'] += any(a >= 0 and b >= 0 and a != b
                    for a, b in zip(state.key, other.key, strict=True))
                for degree in DEGREES:
                    root = tempered_ratio(ratio, F(1), degree)
                    assert F(root.numerator, root.denominator) == ratio and root.degree == degree
                    # Raising strictly positive real detailed-balance flux to k
                    # is injective; this rational equality proves the root law.
                    assert target[state]*q**degree*min(F(1), ratio) == target[other]*q**degree*min(F(1), 1/ratio)
                    counters['exact_power_flux_checks'] += 1
                    wrong_prior = ratio*F(len(env.pool))**(other_used-used)
                    counters['wrong_prior_flux_failures'] += target[state]*min(F(1), wrong_prior) != target[other]*min(F(1), 1/wrong_prior)
                    wrong_continuation = ratio/(1-sampler.config.stop)**(other_length-length)
                    counters['wrong_continuation_flux_failures'] += target[state]*min(F(1), wrong_continuation) != target[other]*min(F(1), 1/wrong_continuation)
                    forward = F(1, valid_counts[state]*len(ordered))
                    reverse = F(1, valid_counts[other]*len(ordered))
                    counters['uncorrected_valid_anchor_flux_failures'] += (
                        target[state]*forward**degree*min(F(1), ratio) != target[other]*reverse**degree*min(F(1), 1/ratio))
        assert normalization == 1
    assert dict(incoming) == target
    assert dict(heatbath_incoming) == target
    counters['cold_stationarity_equations'] = len(states)
    counters['heatbath_stationarity_equations'] = len(states)
    # Oracle selectors use interval groups over ALL complete states. Root
    # decisions use integer-root quantiles rather than interval inequalities.
    seed = 96321+rows+int(contextual)
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    state = states[0]
    for step in range(STEPS):
        degree = DEGREES[step % len(DEGREES)]
        retained, candidate, root, info = window_step(sampler, paths[state], a, degree)
        wi = int(b.integers(len(windows)))
        window = windows[wi]
        outside, _ = signature(env, state, window)
        assert (info['rank'], info['record'], info['start'], info['width']) == (wi, *window)
        other, accept, blocks = state, False, 0
        if outside is not None:
            alternatives = groups[wi][outside]
            ordered = tuple(sorted(alternatives, key=lambda replacement: (len(replacement), replacement)))
            replacement = ordered[int(b.integers(len(ordered)))]
            other = alternatives[replacement]
            assert info['replacement'] == replacement and info['family_size'] == len(ordered)
            if other != state:
                ratio = target[other]/target[state]
                assert candidate.state == other and F(root.numerator, root.denominator) == ratio
                accept, blocks = quantile_decision(RadicalRatio(ratio.numerator, ratio.denominator, degree), b,
                    sampler.config.maximum_acceptance_blocks)
                counters['independent_root_decisions'] += 1
        else:
            assert not info['valid_endpoints']
        assert info['accepted'] == accept and info['acceptance_blocks'] == blocks
        assert retained == paths[other if accept else state]
        assert a.bit_generator.state == b.bit_generator.state
        state = other if accept else state
        counters['seeded_steps'] += 1
    a, b = np.random.default_rng(seed+100), np.random.default_rng(seed+100)
    state = states[0]
    for _ in range(STEPS):
        retained, info = cold_window_heatbath(sampler, paths[state], a)
        wi = int(b.integers(len(windows)))
        outside, _ = signature(env, state, windows[wi])
        assert (info['rank'], info['record'], info['start'], info['width']) == (wi, *windows[wi])
        other, blocks = state, 0
        if outside is not None:
            alternatives = groups[wi][outside]
            ordered = tuple(sorted(alternatives, key=lambda replacement: (len(replacement), replacement)))
            weights = tuple(target[alternatives[x]] for x in ordered)
            selected, blocks = lattice_quantile(weights, b, sampler.config.maximum_acceptance_blocks)
            other = alternatives[ordered[selected]]
            assert info['selected'] == ordered[selected] and info['family_size'] == len(ordered)
            counters['independent_categorical_draws'] += 1
        assert retained == paths[other] and info['raw64_blocks'] == blocks
        assert a.bit_generator.state == b.bit_generator.state
        state = other
        counters['seeded_heatbath_steps'] += 1
    return {'rows': rows, 'contextual': contextual, 'records': list(map(list, records)),
        'windows': len(windows), 'degrees': list(DEGREES), **counters}


def categorical_qualification():
    checks = 0
    for weights in ((F(1),), (F(1, 3), F(2, 3)), (F(1, 2**160), F(1), F(23, 19)),
                    (F(3, 7), F(8, 9), F(1, 6), F(5, 11))):
        for seed in (96361, 96367):
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            for _ in range(128):
                assert rational_categorical(weights, a) == lattice_quantile(weights, b)
                assert a.bit_generator.state == b.bit_generator.state
                checks += 1
    boundaries = 0
    for seed in (96371, 96373, 96377, 96379):
        v = int(np.random.default_rng(seed).bit_generator.random_raw())
        boundary = F(2*v+1, 2**65)
        weights = (boundary, 1-boundary)
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        decision = rational_categorical(weights, a)
        assert decision == lattice_quantile(weights, b) and decision[1] == 2
        assert a.bit_generator.state == b.bit_generator.state
        try:
            rational_categorical(weights, np.random.default_rng(seed), maximum_blocks=1)
        except RuntimeError:
            boundaries += 1
        else:
            raise AssertionError('Categorical one-block cap did not abort')
    return {'independent_categorical_decisions': checks, 'two_block_and_abort_witnesses': boundaries}


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        panels = [finite_panel(*panel) for panel in PANELS]
        keys = set(panels[0])-{'rows', 'contextual', 'records', 'windows', 'degrees'}
        totals = {key: sum(panel[key] for panel in panels) for key in sorted(keys)}
        assert all(value > 0 for value in totals.values())
        categorical = categorical_qualification()
        resources = resource_report(wall, cpu)
        assert resources['wall_seconds'] <= WALL and resources['cpu_seconds'] <= CPU
        assert resources['peak_rss_bytes'] <= HOST
        require_frozen(freeze, PATHS)
        save_new(out/'result.json', {'status': 'PASS_exact_window_flux_conditional_support_and_rng',
            'freeze': freeze, 'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels,
            'totals': totals, 'categorical_checks': categorical, 'resources': resources,
            'software': {name: importlib.metadata.version(name) for name in ('numpy', 'torch')},
            'no_original_source_training_gold_or_recovery': True, 'independent_expert_review': False,
            'alternate_map_and_root_quantiles_but_legacy_enumerator_and_reference_shared': True})
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
