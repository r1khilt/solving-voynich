"""Prospective finite structural qualification, with separate direct enumeration."""
import argparse
import importlib.metadata
import itertools
import math
import signal
import time
from collections import defaultdict
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np

from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.reading_pair_rewrite import (eligible_triplets, pair_rewrite_candidate, pair_rewrite_ratio,
    reading_actions, rewrite_pair, triplet_from_rank, valid_pair_rewrite_ratio)
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState

EXP = 'READING-PAIR-REWRITE-THEORY-001'
RECORDS = (((0, 1), (0, 1)), ((0, 0), (0, 0)), ((0, 1, 0, 1), (0, 1)))
PATHS = ('src/voynich/reading_pair_rewrite.py', 'src/voynich/reading_regrowth.py',
    'src/voynich/source_action_proposal.py', 'src/voynich/joint_key_proposal.py',
    'scripts/check_reading_pair_rewrite001.py', 'scripts/audit_reading_pair_rewrite001.py',
    'tests/test_reading_pair_rewrite.py', 'scripts/run_joint_key_train001.py',
    'scripts/run_blind_channel_dev001.py', 'scripts/run_latin_source_model001.py',
    'scripts/run_blind_channel_dev004.py', 'docs/research/global-pair-rewrite-proposal-2026-10-02.md',
    'docs/research/reading-pair-rewrite-2026-10-08.md',
    'docs/experiments/READING-PAIR-REWRITE-THEORY-001.md')


def source_fixture(rows, contextual):
    base = [1, 2, 1] if rows == 3 else [1, 2, 3, 2]
    p = np.array([base[i:]+base[:i] for i in range(rows)], dtype=np.float64)/sum(base)
    t = np.array([[(i+j+1) % rows if contextual else 0 for j in range(rows)]
                  for i in range(rows)], dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    return SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)


def enumerate_complete(env):
    """Recordwise recursion, independent of environment actions or source policy."""
    states = []

    def descend(record, offset, key, texts):
        if record == len(env.records):
            states.append(ReadingState(key, tuple(map(len, env.records)), texts))
            return
        if offset == len(env.records[record]):
            descend(record+1, 0, key, texts)
            return
        for row in range(env.rows):
            for length in (1, 2):
                unit = env.records[record][offset:offset+length]
                if len(unit) != length or (key[row] >= 0 and env.pool[key[row]] != unit):
                    continue
                new_key = key[:row]+(env.pool.index(unit),)+key[row+1:]
                new_texts = texts[:record]+(texts[record]+(row,),)+texts[record+1:]
                descend(record, offset+length, new_key, new_texts)

    descend(0, 0, (-1,)*env.rows, ((),)*len(env.records))
    assert len(set(states)) == len(states)
    return states


def direct_rewrite(env, state, triplet, *, unsafe_split=False):
    """A second map using indexed match sets and independently derived visited key."""
    a, b, c = triplet
    units = tuple(None if k < 0 else env.pool[k] for k in state.key)
    positions = [{i for i in range(len(t)-1) if t[i:i+2] == (b, c)} for t in state.texts]
    if units[a] is None and units[b] is not None and units[c] is not None:
        if len(units[b]) != 1 or len(units[c]) != 1 or not any(positions):
            return state, 'identity', 0
        transformed = tuple(tuple(a if i in matches else row for i, row in enumerate(text)
                                  if i-1 not in matches)
                            for text, matches in zip(state.texts, positions, strict=True))
        branch, number = 'merge', sum(map(len, positions))
    elif (units[a] is not None and len(units[a]) == 2
          and units[b] in (None, units[a][:1]) and units[c] in (None, units[a][1:])
          and (unsafe_split or not any(positions))):
        transformed = tuple(tuple(j for row in t for j in ((b, c) if row == a else (row,)))
                            for t in state.texts)
        branch, number = 'split', sum(t.count(a) for t in state.texts)
    else:
        return state, 'identity', 0
    # Reconstruct from observed slices and transformed units, not production's
    # release/bind bookkeeping. Existing unrelated unit assignments are retained.
    expected_units = list(units)
    if branch == 'merge':
        expected_units[a] = units[b]+units[c]
    else:
        expected_units[b], expected_units[c] = units[a][:1], units[a][1:]
    new_key = [-1]*env.rows
    for text, observation in zip(transformed, env.records, strict=True):
        offset = 0
        for row in text:
            unit = expected_units[row]
            assert observation[offset:offset+len(unit)] == unit
            index = env.pool.index(unit)
            assert new_key[row] in (-1, index)
            new_key[row] = index
            offset += len(unit)
        assert offset == len(observation)
    return ReadingState(tuple(new_key), state.offsets, transformed), branch, number


def direct_actions(env, state):
    lengths = [len(r) for r in env.records]
    offsets, indices, result = [0]*len(lengths), [0]*len(lengths), []
    while unfinished := [i for i, n in enumerate(lengths) if offsets[i] < n]:
        i = min(unfinished, key=lambda j: (F(offsets[j], lengths[j]), j))
        row = state.texts[i][indices[i]]
        size = len(env.pool[state.key[row]])
        result.append(2*row+size-1)
        indices[i] += 1
        offsets[i] += size
    return tuple(result)


def direct_target(source, env, state, stop):
    mass = F(1, len(env.pool))**len(set(itertools.chain.from_iterable(state.texts)))
    for text in state.texts:
        mass *= stop*(1-stop)**len(text)
        context = 0
        for row in text:
            mass *= F(float(source.probabilities[context, row]))
            context = int(source.transitions[context, row])
    return mass


def finite_panel(rows, contextual, records):
    source = source_fixture(rows, contextual)
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    config = RegrowthConfig(stop=F(1, 3), grid_bits=8, two_glyph_bias=F(6))
    sampler = SourceRegrowth(source, env, config)
    states = enumerate_complete(env)
    triplets = tuple(itertools.permutations(range(rows), 3))
    assert {triplet_from_rank(rows, r) for r in range(len(triplets))} == set(triplets)
    paths, targets, policy, neighbors, degrees = {}, {}, {}, {}, {}
    counters = dict.fromkeys(('triplet_checks', 'merge_edges', 'split_edges', 'identity_edges',
        'repeated_replacements', 'cross_record_changes', 'visited_count_changes',
        'preexisting_pair_noninvolution_witnesses', 'omitted_prior_flux_failures',
        'omitted_continuation_flux_failures', 'wrong_policy_flux_failures',
        'uncorrected_valid_selection_flux_failures', 'corrected_valid_selection_edges'), 0)
    for state in states:
        actions = direct_actions(env, state)
        assert reading_actions(env, state) == actions
        path = sampler.path(forced_actions=actions)
        assert path.state == state and path.complete
        paths[state] = path
        targets[state] = direct_target(source, env, state, config.stop)
        assert F(*sampler.target_integers(path)) == targets[state]
        policy[state] = math.prod(F(c, 2**config.grid_bits) for c in path.counts)
        neighbors[state] = [direct_rewrite(env, state, t) for t in triplets]
        degrees[state] = sum(branch != 'identity' for _, branch, _ in neighbors[state])
        independently_eligible = tuple(t for t, (_, branch, _) in zip(triplets, neighbors[state], strict=True)
                                       if branch != 'identity')
        assert eligible_triplets(env, state) == independently_eligible
    incoming, valid_incoming = defaultdict(F), defaultdict(F)
    selection = F(1, len(triplets))
    for state in states:
        tx, path = targets[state], paths[state]
        stay, outgoing = F(0), F(0)
        valid_stay, valid_outgoing = (F(1) if not degrees[state] else F(0)), F(0)
        for triplet, (destination, branch, number) in zip(triplets, neighbors[state], strict=True):
            change = rewrite_pair(env, state, *triplet)
            assert (change.state, change.branch, change.replacements) == (destination, branch, number)
            inverse = rewrite_pair(env, change.state, *triplet)
            assert inverse.state == state
            assert direct_rewrite(env, destination, triplet)[0] == state
            counters['triplet_checks'] += 1
            counters[branch+'_edges'] += 1
            # Unsafe split is a negative control, not an alternate production kernel.
            unsafe, unsafe_branch, _ = direct_rewrite(env, state, triplet, unsafe_split=True)
            if branch == 'identity' and unsafe_branch == 'split':
                counters['preexisting_pair_noninvolution_witnesses'] += (
                    direct_rewrite(env, unsafe, triplet)[0] != state)
            ty = targets[destination]
            candidate, actual = pair_rewrite_candidate(sampler, path, *triplet)
            assert actual == change and candidate == paths[destination]
            ratio = F(*pair_rewrite_ratio(sampler, path, candidate))
            assert ratio == ty/tx
            acceptance, reverse = min(F(1), ratio), min(F(1), 1/ratio)
            assert tx*selection*acceptance == ty*selection*reverse
            incoming[destination] += tx*selection*acceptance
            stay += selection*(1-acceptance)
            outgoing += selection*acceptance
            if branch == 'identity':
                continue
            assert len(candidate.actions) != len(path.actions)
            counters['repeated_replacements'] += number > 1
            counters['cross_record_changes'] += sum(x != y for x, y in zip(
                state.texts, destination.texts, strict=True)) > 1
            mx = sum(k >= 0 for k in state.key)
            my = sum(k >= 0 for k in destination.key)
            counters['visited_count_changes'] += mx != my
            nx, ny = len(path.actions), len(candidate.actions)
            for name, bad in (('omitted_prior_flux_failures', ratio*len(env.pool)**(my-mx)
                              if my >= mx else ratio/F(len(env.pool)**(mx-my))),
                             ('omitted_continuation_flux_failures', ratio/(1-config.stop)**(ny-nx)),
                             ('wrong_policy_flux_failures', ratio*policy[state]/policy[destination])):
                counters[name] += tx*min(F(1), bad) != ty*min(F(1), 1/bad)
            dx, dy = degrees[state], degrees[destination]
            assert dx > 0 and dy > 0
            counters['uncorrected_valid_selection_flux_failures'] += (
                tx*acceptance/dx != ty*reverse/dy)
            corrected = ratio*F(dx, dy)
            assert F(*valid_pair_rewrite_ratio(sampler, path, candidate, dx, dy)) == corrected
            assert tx*min(F(1), corrected)/dx == ty*min(F(1), 1/corrected)/dy
            valid_incoming[destination] += tx*min(F(1), corrected)/dx
            valid_outgoing += min(F(1), corrected)/dx
            valid_stay += (1-min(F(1), corrected))/dx
            counters['corrected_valid_selection_edges'] += 1
        assert outgoing+stay == 1
        assert valid_outgoing+valid_stay == 1
        incoming[state] += tx*stay
        valid_incoming[state] += tx*valid_stay
    assert all(incoming[state] == targets[state] for state in states)
    assert all(valid_incoming[state] == targets[state] for state in states)
    assert counters['merge_edges'] == counters['split_edges'] > 0
    return {'rows': rows, 'contextual': contextual, 'records': [list(r) for r in records],
            'states': len(states), 'triplets_per_state': len(triplets),
            'stationarity_equations_per_selector': len(states), **counters}


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    limit_resources(900, 800)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        panels = [finite_panel(rows, context, records) for rows, context, records in
                  itertools.product((3, 4), (False, True), RECORDS)]
        counts = tuple(k for k in panels[0] if k not in ('rows', 'contextual', 'records', 'triplets_per_state'))
        totals = {k: sum(p[k] for p in panels) for k in counts}
        assert len(panels) == 12 and all(v > 0 for v in totals.values())
        resources = resource_report(wall, cpu)
        assert resources['wall_seconds'] <= 900 and resources['cpu_seconds'] <= 800
        assert resources['peak_rss_bytes'] <= 512*1024**2
        save_new(out/'result.json', {'status': 'PASS_exact_global_pair_involution_and_flux', 'freeze': freeze,
            'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels, 'totals': totals,
            'resources': resources, 'paid_spend_usd': 0, 'no_original_source_or_recovery': True,
            'software': {name: importlib.metadata.version(name) for name in ('numpy', 'torch')},
            'independent_expert_review': False, 'valid_only_selector_implemented_and_checked': True})
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
