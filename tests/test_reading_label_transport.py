"""Independent exact finite label flux and label/regrowth mixture qualification.

Preparation fixtures only: no original corpus, learned model or Gold reading.
"""
import itertools
import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

from voynich.reading_label_transport import (label_transport_step, pair_probability,
    reference_reading, score_label_transport)
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment


def source_fixture(rows, contextual):
    p = np.array([[.75, .25], [.25, .75], [.5, .5]] if rows == 2 else
                 [[.5, .375, .125], [.125, .5, .375], [.25, .125, .625]], dtype=np.float64)
    t = np.array([list(range(rows))]*3 if contextual else [[0]*rows]*3, dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    return SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)


def exact_target(source, env, state, stop):
    mass = F(1, len(env.pool))**sum(k >= 0 for k in state.key)
    for text in state.texts:
        context = 0
        mass *= stop*(1-stop)**len(text)
        for row in text:
            mass *= F(float(source.probabilities[context, row]))
            context = int(source.transitions[context, row])
    return mass


def exact_terminal_law(source, env, config):
    """Separate largest-remainder arithmetic and literal environment expansion."""
    terminal, agenda = [], [(env.initial, (), (), F(1))]
    while agenda:
        state, actions, counts, probability = agenda.pop()
        record = env.selected_record(state)
        legal = env.legal_actions(state)
        if not legal:
            terminal.append((state, actions, counts, probability, record is None))
            continue
        context = 0
        for row in state.texts[record]:
            context = int(source.transitions[context, row])
        weights = []
        for action in legal:
            row, length = action//2, action%2+1
            w = F(float(source.probabilities[context, row]))
            if state.key[row] < 0:
                w /= len(env.pool)
                if length == 2:
                    w *= config.two_glyph_bias
            if state.offsets[record]+length == len(env.records[record]):
                w *= config.stop
            weights.append(w)
        grid = 2**config.grid_bits
        allocation = [(grid-len(legal))*w/sum(weights) for w in weights]
        units = [1+int(w) for w in allocation]
        order = sorted(range(len(legal)), key=lambda i: (-(allocation[i]-int(allocation[i])), i))
        for i in order[:grid-sum(units)]:
            units[i] += 1
        for action, unit in zip(legal, units, strict=True):
            agenda.append((env.advance(state, action), actions+(action,), counts+(unit,),
                           probability*F(unit, grid)))
    assert sum(row[3] for row in terminal) == 1
    return terminal


def finite_label_panel(rows, contextual, bias):
    source = source_fixture(rows, contextual)
    env = ReadingEnvironment(((0,), (0, 0)), rows=rows, glyphs=2)
    config = RegrowthConfig(stop=F(1, 3), root_mass=F(1, 8), two_glyph_bias=bias, grid_bits=8)
    sampler = SourceRegrowth(source, env, config)
    terminal = exact_terminal_law(source, env, config)
    paths, target, q = {}, {}, {}
    for state, actions, counts, probability, complete in terminal:
        replay = sampler.path(forced_actions=actions)
        assert (replay.state, replay.counts, replay.complete) == (state, counts, complete)
        if complete:
            paths[actions] = replay
            target[actions] = exact_target(source, env, state, config.stop)
            q[actions] = probability
    label, regrowth, wrong_policy = {}, {}, {}
    unbound = duplicates = wrong_flux = changed_contexts = 0
    for x, path in paths.items():
        row = {y: F(0) for y in paths}
        bad = row.copy()
        probabilities = []
        used = sum(k >= 0 for k in path.state.key)
        for a, b in itertools.combinations(range(rows), 2):
            marginal = F(int(path.state.key[a] >= 0)+int(path.state.key[b] >= 0), used*(rows-1))
            assert marginal == pair_probability(path, a, b)
            probabilities.append(marginal)
            point = score_label_transport(sampler, path, a, b)
            replay = reference_reading(sampler, point)
            y = point.actions
            assert y in paths and replay == paths[y]
            assert F(point.numerator, point.denominator) == target[y]
            assert abs(point.log_potential()-math.log(float(target[y]))) < 2e-14
            inverse = score_label_transport(sampler, replay, a, b)
            assert inverse.actions == x and inverse.state == path.state
            assert F(inverse.numerator, inverse.denominator) == target[x]
            assert sorted(path.state.key) == sorted(point.state.key)
            assert path.state.offsets == point.state.offsets
            assert pair_probability(replay, a, b) == marginal
            unbound += (path.state.key[a] < 0) != (path.state.key[b] < 0)
            duplicates += path.state.key[a] == path.state.key[b] >= 0 and point.state.texts != path.state.texts
            changed_contexts += any(t != u and old == new for t, u, old, new in
                zip(path.source_terms, point.source_terms, x, y, strict=True))
            acceptance = min(F(1), target[y]/target[x])
            wrong = min(F(1), target[y]*q[x]/(target[x]*q[y]))
            row[y] += marginal*acceptance
            row[x] += marginal*(1-acceptance)
            bad[y] += marginal*wrong
            bad[x] += marginal*(1-wrong)
        assert sum(probabilities) == sum(row.values()) == sum(bad.values()) == 1
        label[x], wrong_policy[x] = row, bad
        row = {y: F(0) for y in paths}
        for cut in range(len(x)):
            cx = (1-config.root_mass)/len(x)+(config.root_mass if cut == 0 else 0)
            prefix = math.prod(F(c, 2**config.grid_bits) for c in path.counts[:cut])
            for _, y, _, probability, complete in terminal:
                if y[:cut] != x[:cut]:
                    continue
                proposal = cx*probability/prefix
                if not complete:
                    row[x] += proposal
                    continue
                cy = (1-config.root_mass)/len(y)+(config.root_mass if cut == 0 else 0)
                ratio = target[y]*cy*q[x]/(target[x]*cx*q[y])
                acceptance = min(F(1), ratio)
                row[y] += proposal*acceptance
                row[x] += proposal*(1-acceptance)
        assert sum(row.values()) == 1
        regrowth[x] = row
    assert unbound > 0 and duplicates > 0
    if contextual and rows == 3:
        assert changed_contexts > 0
    mixture = {x: {y: (label[x][y]+regrowth[x][y])/2 for y in paths} for x in paths}
    composition = {x: {y: sum(label[x][z]*regrowth[z][y] for z in paths) for y in paths} for x in paths}
    for kernel in (label, regrowth, mixture, composition):
        for x in paths:
            assert sum(kernel[x].values()) == 1
            assert sum(target[y]*kernel[y][x] for y in paths) == target[x]
            if kernel is not composition:
                for y in paths:
                    assert target[x]*kernel[x][y] == target[y]*kernel[y][x]
    for x in paths:
        for y in paths:
            wrong_flux += target[x]*wrong_policy[x][y] != target[y]*wrong_policy[y][x]
    assert wrong_flux > 0
    return {'states': len(paths), 'balance_pairs_per_reversible_kernel': len(paths)**2,
            'unbound_transports': unbound, 'duplicate_code_transports': duplicates,
            'unchanged_action_context_changes': changed_contexts, 'wrong_policy_flux_failures': wrong_flux,
            'composition_nonreversible_pairs': sum(target[x]*composition[x][y] != target[y]*composition[y][x]
                for x in paths for y in paths)}


@pytest.mark.parametrize('rows', [2, 3])
@pytest.mark.parametrize('contextual', [False, True])
@pytest.mark.parametrize('bias', [F(1), F(6)])
def test_exact_label_kernel_and_fixed_mixture(rows, contextual, bias):
    assert finite_label_panel(rows, contextual, bias)['states'] > 0


def test_partial_binding_replay_and_rng():
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 1), (0, 1)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env)
    old = sampler.path(forced_actions=(1, 1))
    point = score_label_transport(sampler, old, 0, 2)
    assert point.state.key[0] == -1 and point.state.key[2] == old.state.key[0]
    assert point.actions == (5, 5) and point.state.texts == ((2,), (2,))
    assert pair_probability(old, 0, 2) == F(1, 2)
    assert pair_probability(old, 1, 2) == 0
    rng1, rng2 = np.random.default_rng(95201), np.random.default_rng(95201)
    for _ in range(40):
        retained, candidate, info = label_transport_step(sampler, old, rng1)
        other, other_candidate, other_info = label_transport_step(sampler, old, rng2)
        assert retained == other and candidate == other_candidate and info == other_info
        assert retained == sampler.path(forced_actions=retained.actions)
        # Retained policy counts are usable in a subsequent regrowth correction.
        new = sampler.path(forced_actions=(0, 0, 2, 2))
        assert sampler.ratio(retained, new, 0)[0] > 0
        old = retained


def test_interleaved_records_keep_action_order_and_whole_contexts():
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 1), (1, 0, 1)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env)
    old = sampler.path(forced_actions=(0, 4, 0, 4, 4))
    point = score_label_transport(sampler, old, 0, 1)
    replay = reference_reading(sampler, point)
    assert replay.actions == (2, 4, 2, 4, 4)
    assert point.source_terms == tuple((F(p).numerator, F(p).denominator.bit_length()-1)
                                     for p in (.375, .125, .125, .375, .375))
    assert old.source_terms[3] != point.source_terms[3]  # Unchanged row, changed earlier context.
    assert F(point.numerator, point.denominator) == exact_target(source, env, point.state, sampler.config.stop)
    with pytest.raises(ValueError):
        score_label_transport(sampler, replace(old, actions=old.actions[:-1]), 0, 1)


def test_guard_foreign_invalid_pair_and_rng():
    env = ReadingEnvironment(((0, 1),), rows=3, glyphs=2)
    sampler = SourceRegrowth(source_fixture(3, True), env)
    old = sampler.path(forced_actions=(1,))
    foreign = SourceRegrowth(source_fixture(3, True), env).path(forced_actions=(1,))
    with pytest.raises(ValueError):
        score_label_transport(sampler, foreign, 0, 1)
    with pytest.raises(ValueError):
        score_label_transport(sampler, replace(old, complete=False), 0, 1)
    for pair in ((0, 0), (1, 0), (0, 3), (False, 1)):
        with pytest.raises(ValueError):
            score_label_transport(sampler, old, *pair)
    with pytest.raises(ValueError):
        label_transport_step(sampler, old, np.random.Generator(np.random.MT19937(1)))
    point = score_label_transport(sampler, old, 0, 1)
    with pytest.raises(ValueError):
        reference_reading(sampler, replace(point, law_token=object()))
    with pytest.raises(ArithmeticError):
        reference_reading(sampler, replace(point, numerator=point.numerator+1))


def test_finite_driver_and_receipt_schema(tmp_path, monkeypatch):
    from scripts import check_reading_label_transport001 as run
    from scripts.audit_reading_label_transport001 import audit

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run, 'resource_report', lambda *_: {'wall_seconds': .1, 'cpu_seconds': .1,
                                                         'peak_rss_bytes': 1024, 'paid_spend_usd': 0})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_new(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(value, handle)
        return artifact(path)

    monkeypatch.setattr(run, 'artifact', artifact)
    monkeypatch.setattr(run, 'save_new', save_new)
    totals = run.check('artificial-fixture-only')
    assert all(v > 0 for v in totals.values())
    assert audit('artificial-fixture-only') == 'PASS_receipt_hash_and_arithmetic_closure'
    result = json.loads((tmp_path/'results'/run.EXP/'result.json').read_text())
    closure = json.loads((tmp_path/'results'/run.EXP/'audit.json').read_text())
    assert closure['result'] == artifact(tmp_path/'results'/run.EXP/'result.json')
    assert result['totals'] == totals and len(result['panels']) == 8
    # Exclusive namespaces refuse an accidental second invocation before work.
    with pytest.raises(FileExistsError):
        run.check('artificial-fixture-only')
