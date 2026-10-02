"""Independent literal enumeration, rational target and transition flux checks."""
import itertools
import math
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

from voynich.reading_regrowth import (RegrowthConfig, SourceRegrowth, cut_probability,
                                      exact_bernoulli, quantized_counts)
from voynich.source_action_proposal import ReadingEnvironment


def fixture_source(contextual=False):
    p = np.array([[.75, .25], [.25, .75], [.5, .5]], dtype=np.float64)
    t = np.array([[1, 2]]*3 if contextual else [[0, 0]]*3, dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    return SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)


def independent_target(source, env, state, stop):
    result = F(1, len(env.pool))**len(set(itertools.chain.from_iterable(state.texts)))
    for text in state.texts:
        result *= stop*(1-stop)**len(text)
        context = 0
        for row in text:
            result *= F(float(source.probabilities[context, row]))
            context = int(source.transitions[context, row])
    return result


def independent_counts(source, env, state, config):
    record = env.selected_record(state)
    if record is None:
        return {}
    context = 0
    for row in state.texts[record]:
        context = int(source.transitions[context, row])
    actions = env.legal_actions(state)
    if not actions:
        return {}
    weights = []
    for a in actions:
        row, length = a//2, a%2+1
        w = F(float(source.probabilities[context, row]))
        if state.key[row] < 0:
            w /= len(env.pool)
            if length == 2:
                w *= config.two_glyph_bias
        if state.offsets[record]+length == len(env.records[record]):
            w *= config.stop
        weights.append(w)
    grid = 2**config.grid_bits
    allocations = [(grid-len(actions))*w/sum(weights) for w in weights]
    counts = [1+int(x) for x in allocations]
    order = sorted(range(len(actions)), key=lambda j: (-(allocations[j]-int(allocations[j])), j))
    for j in order[:grid-sum(counts)]:
        counts[j] += 1
    return dict(zip(actions, counts, strict=True))


def enumerate_paths(source, env, config):
    terminal, agenda = [], [(env.initial, (), (), F(1))]
    while agenda:
        state, actions, counts, q = agenda.pop()
        allocation = independent_counts(source, env, state, config)
        if not allocation:
            terminal.append((state, actions, counts, q, env.selected_record(state) is None))
        for a, count in allocation.items():
            agenda.append((env.advance(state, a), actions+(a,), counts+(count,),
                           q*F(count, 2**config.grid_bits)))
    assert sum(r[3] for r in terminal) == 1
    return terminal


def finite_regrowth_panel(records, contextual, bias, root_mass, *, rows=2):
    source = fixture_source(contextual)
    if rows == 1:
        p = np.ones((1, 1), dtype=np.float64)
        t = np.zeros((1, 1), dtype=np.uint32)
        p.flags.writeable = t.flags.writeable = False
        source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    config = RegrowthConfig(stop=F(1, 3), root_mass=root_mass, two_glyph_bias=bias, grid_bits=8)
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    impl = SourceRegrowth(source, env, config)
    terminal = enumerate_paths(source, env, config)
    paths, target, proposal, failures = {}, {}, {}, []
    for state, actions, counts, q, complete in terminal:
        replay = impl.path(forced_actions=actions)
        assert replay.state == state and replay.counts == counts and replay.complete == complete
        assert abs(replay.policy_log_probability()-math.log(float(q))) <= 2e-13
        if complete:
            paths[actions] = replay
            target[actions] = independent_target(source, env, state, config.stop)
            assert F(*impl.target_integers(replay)) == target[actions]
            proposal[actions] = q
        else:
            failures.append((actions, q))
    assert len({p.state for p in paths.values()}) == len(paths)
    # A separate full-key sum checks visited-row collapse without action logic.
    collapsed = {}
    for key in itertools.product(range(len(env.pool)), repeat=rows):
        options = []
        for record in env.records:
            options.append([text for n in range(1, len(record)+1)
                            for text in itertools.product(range(rows), repeat=n)
                            if tuple(g for a in text for g in env.pool[key[a]]) == record])
        for texts in itertools.product(*options):
            result = F(1, len(env.pool))**rows
            for text in texts:
                context = 0
                result *= config.stop*(1-config.stop)**len(text)
                for a in text:
                    result *= F(float(source.probabilities[context, a]))
                    context = int(source.transitions[context, a])
            used = set(itertools.chain.from_iterable(texts))
            signature = (tuple(k if a in used else -1 for a, k in enumerate(key)), texts)
            collapsed[signature] = collapsed.get(signature, F(0))+result
    assert collapsed == {(p.state.key, p.state.texts): target[a] for a, p in paths.items()}
    transition, bad_cut, bad_prior = {}, {}, {}
    cut_witnesses = prior_witnesses = inventory_edges = length_edges = failed_branches = 0
    no_prior = {a: t*len(env.pool)**sum(k >= 0 for k in paths[a].state.key) for a, t in target.items()}
    for x, px in paths.items():
        row, cut_bad, prior_bad = {y: F(0) for y in paths}, {y: F(0) for y in paths}, {y: F(0) for y in paths}
        for cut in range(len(x)):
            # Independent rational cut selection, rather than the implementation helper.
            cx = (1-root_mass)/len(x)+(root_mass if cut == 0 else 0)
            assert cx == cut_probability(len(x), cut, root_mass)
            if not cx:
                continue
            prefix_q = math.prod(F(c, 2**config.grid_bits) for c in px.counts[:cut])
            eligible = [(a, q, complete) for _, a, _, q, complete in terminal if a[:cut] == x[:cut]]
            assert sum(q/prefix_q for _, q, _ in eligible) == 1
            for y, qy, complete in eligible:
                mass = cx*qy/prefix_q
                if not complete:
                    row[x] += mass
                    cut_bad[x] += mass
                    prior_bad[x] += mass
                    failed_branches += 1
                    continue
                cy = (1-root_mass)/len(y)+(root_mass if cut == 0 else 0)
                ratio = target[y]*cy*proposal[x]/(target[x]*cx*qy)
                assert F(*impl.ratio(px, paths[y], cut)) == ratio
                a = min(F(1), ratio)
                b = min(F(1), target[y]*proposal[x]/(target[x]*qy))
                c = min(F(1), no_prior[y]*cy*proposal[x]/(no_prior[x]*cx*qy))
                for dest, acceptance in ((row, a), (cut_bad, b), (prior_bad, c)):
                    dest[y] += mass*acceptance
                    dest[x] += mass*(1-acceptance)
                if x != y:
                    inventory_edges += set(px.state.key)-{-1} != set(paths[y].state.key)-{-1}
                    length_edges += len(x) != len(y)
            assert sum(row.values()) <= 1
        assert sum(row.values()) == sum(cut_bad.values()) == sum(prior_bad.values()) == 1
        transition[x], bad_cut[x], bad_prior[x] = row, cut_bad, prior_bad
    for x, tx in target.items():
        assert sum(target[y]*transition[y][x] for y in target) == tx
        for y, ty in target.items():
            assert tx*transition[x][y] == ty*transition[y][x]
            cut_witnesses += tx*bad_cut[x][y] != ty*bad_cut[y][x]
            prior_witnesses += tx*bad_prior[x][y] != ty*bad_prior[y][x]
    return {'states': len(paths), 'failed_paths': len(failures), 'failed_mass': str(sum(q for _, q in failures)),
            'balance_pairs': len(paths)**2, 'omitted_cut_flux_failures': cut_witnesses,
            'omitted_prior_flux_failures': prior_witnesses, 'inventory_changing_edges': inventory_edges,
            'different_length_edges': length_edges, 'failed_branch_selfloops': failed_branches}


@pytest.mark.parametrize('contextual', [False, True])
@pytest.mark.parametrize('bias', [F(1), F(6)])
@pytest.mark.parametrize('root', [F(0), F(1, 8), F(1)])
def test_finite_kernel_and_schedule(contextual, bias, root):
    # Unequal lengths explicitly catch the draft's wrong raw-offset scheduler.
    result = finite_regrowth_panel(((0,), (0, 1)), contextual, bias, root)
    assert result['states'] > 0 and result['inventory_changing_edges'] > 0


def test_failed_draws_in_diagonal():
    result = finite_regrowth_panel(((0, 1, 0, 1),), False, F(6), F(1, 8), rows=1)
    assert result['failed_paths'] and result['failed_branch_selfloops']


def test_quantizer_extremes_and_ties():
    assert quantized_counts((1, 1, 1), 8) == (3, 3, 2)
    assert quantized_counts((1, 10**1000), 256) == (1, 255)
    assert quantized_counts((4, 3), 2) == (1, 1)
    for grid in range(2, 24):
        for weights in itertools.product(range(1, 5), repeat=2):
            result = quantized_counts(weights, grid)
            assert sum(result) == grid and min(result) >= 1
    for weights, grid in (((), 2), ((0,), 2), ((True,), 2), ((1, 1), 1)):
        with pytest.raises(ValueError):
            quantized_counts(weights, grid)


class ScriptedBits(np.random.PCG64):
    def __init__(self, values):
        super().__init__(0)
        self.values = iter(values)

    def random_raw(self):
        return next(self.values)


def test_lazy_bernoulli_tiny_probability_and_equal_block():
    def rng(values):
        return np.random.Generator(ScriptedBits(values))
    assert exact_bernoulli(1, 2**128, rng([0, 0])) == (True, 2)
    assert exact_bernoulli(1, 2**128, rng([0, 1])) == (False, 2)
    assert exact_bernoulli(1, 2, rng([2**63])) == (False, 1)
    assert exact_bernoulli(2, 1, rng([17])) == (True, 1)
    with pytest.raises(RuntimeError):
        exact_bernoulli(1, 2**128, rng([0]), maximum_blocks=1)
    with pytest.raises(ValueError):
        exact_bernoulli(1, 2, np.random.Generator(np.random.MT19937(1)))


def test_regrowth_releases_future_inventory_and_replays_rng():
    env = ReadingEnvironment(((0, 1), (0, 1)), rows=2, glyphs=2)
    s = SourceRegrowth(fixture_source(), env)
    old = s.path(forced_actions=(1, 1))  # row0 holds (0,1); row1 is unbound.
    short = s.path(forced_actions=(0, 0, 2, 2))  # rows0/1 now hold (0)/(1).
    assert old.state.key != short.state.key and F(*s.ratio(old, short, 0)) > 0
    rng1, rng2 = np.random.default_rng(41), np.random.default_rng(41)
    for _ in range(30):
        a, candidate, info = s.step(old, rng1)
        b, replay, other = s.step(old, rng2)
        assert a == b and candidate == replay and info == other
        cut = info['cut']
        assert candidate.actions[:cut] == old.actions[:cut]
        assert candidate == s.path(forced_actions=candidate.actions)
        old = a
    with pytest.raises(ValueError):
        s.path(forced_actions=(0,))
    with pytest.raises(ValueError):
        s.path(forced_actions=(False,))
    foreign = SourceRegrowth(fixture_source(), env).path(forced_actions=(1, 1))
    with pytest.raises(ValueError):
        s.ratio(old, foreign, 0)
    root = SourceRegrowth(fixture_source(), env, RegrowthConfig(root_mass=F(1)))
    p = root.path(forced_actions=(1, 1))
    with pytest.raises(ValueError):
        root.ratio(p, p, 1)
