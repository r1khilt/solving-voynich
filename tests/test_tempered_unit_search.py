"""Exhaustive transition checks, a genuine likelihood trap, and accounting."""
import itertools
import math
from collections import defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from voynich.sparse_suffix_source import decode
from voynich.tempered_unit_search import (
    TemperedConfig, log_acceptance, log_swap_acceptance, propose, search_tempered,
)


class Tape:
    """Enumerate public RNG outcomes, not implementation probabilities."""
    def __init__(self, ranges=(), choices=(), samples=()):
        self.ranges, self.choices, self.samples = iter(ranges), iter(choices), iter(samples)

    def randrange(self, n):
        value = next(self.ranges)
        assert value in range(n)
        return value

    def choice(self, values):
        value = next(self.choices)
        assert value in values
        return value

    def sample(self, population, k):
        value = next(self.samples)
        assert len(set(value)) == len(value) == k and set(value) <= set(population)
        return value


def kernel(key, pool, config):
    result = defaultdict(F)
    n, m, total = len(key), len(pool), sum(config.proposal_weights)
    for draw in range(total):
        if draw < config.proposal_weights[0]:
            for row in range(n):
                for unit in pool:
                    if unit != key[row]:
                        following, meta = propose(key, pool, Tape((draw, row), (unit,)), config)
                        assert meta == {'kind': 'replace', 'rows': (row,), 'new_units': (unit,)}
                        result[following] += F(1, total*n*(m-1))
        elif draw < sum(config.proposal_weights[:2]):
            for rows in itertools.combinations(range(n), 2):
                following, meta = propose(key, pool, Tape((draw,), samples=(rows,)), config)
                assert meta['kind'] == 'swap' and meta['rows'] == rows
                result[following] += F(1, total*math.comb(n, 2))
        else:
            sizes = [size for size in config.block_sizes if size <= n]
            for size in sizes:
                for rows in itertools.combinations(range(n), size):
                    for units in itertools.product(pool, repeat=size):
                        tape = Tape((draw,), (size, *units), (rows,))
                        following, meta = propose(key, pool, tape, config)
                        assert meta == {'kind': 'block', 'rows': rows, 'new_units': units}
                        result[following] += F(1, total*len(sizes)*math.comb(n, size)*m**size)
    assert sum(result.values()) == 1
    return result


@pytest.mark.parametrize('weights', [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 1)])
def test_actual_proposals_are_exactly_symmetric_for_every_small_dictionary(weights):
    pool = ('x', 'y', 'xy')
    states = list(itertools.product(pool, repeat=3))  # Includes duplicate rows.
    config = TemperedConfig(proposal_weights=weights, block_sizes=(2, 3, 6))
    kernels = {key: kernel(key, pool, config) for key in states}
    for left in states:
        for right in states:
            assert kernels[left][right] == kernels[right][left]
    if weights[1] or weights[2]:
        assert kernels[('x',)*3][('x',)*3] > 0


@pytest.mark.parametrize('temperature', [.5, 1., 7.])
def test_metropolis_matrix_satisfies_detailed_balance_and_preserves_target(temperature):
    pool = ('x', 'y')
    states = list(itertools.product(pool, repeat=3))
    config = TemperedConfig(block_sizes=(2, 3))
    scores = {s: float(sum(i+1 for i, u in enumerate(s) if u == 'x')) for s in states}
    scores[states[-1]] = -math.inf
    target = np.array([math.exp(scores[s]/temperature) for s in states])
    target /= target.sum()
    transition = np.zeros((len(states), len(states)))
    for i, left in enumerate(states):
        for j, right in enumerate(states):
            transition[i, j] = float(kernel(left, pool, config)[right])*math.exp(
                log_acceptance(scores[left], scores[right], temperature))
        transition[i, i] += 1-transition[i].sum()
    flow = target[:, None]*transition
    np.testing.assert_allclose(flow, flow.T, atol=1e-16, rtol=1e-14)
    np.testing.assert_allclose(target@transition, target, atol=2e-16, rtol=1e-14)
    np.testing.assert_allclose(transition.sum(axis=1), 1., atol=2e-16)


def test_replica_exchange_preserves_product_target_and_sign():
    scores, temperatures = (-4., -1., 2.), (1., 4.)
    states = list(itertools.product(range(3), repeat=2))
    target = np.array([math.exp(scores[a]/temperatures[0]+scores[b]/temperatures[1]) for a, b in states])
    target /= target.sum()
    transition = np.zeros((len(states), len(states)))
    for i, (a, b) in enumerate(states):
        j = states.index((b, a))
        acceptance = math.exp(log_swap_acceptance(scores[a], scores[b], *temperatures))
        transition[i, j] += acceptance
        transition[i, i] += 1-acceptance
    np.testing.assert_allclose(target[:, None]*transition, (target[:, None]*transition).T, atol=1e-16)
    np.testing.assert_allclose(target@transition, target, atol=1e-16)
    assert log_swap_acceptance(-4., 2., 1., 4.) == 0.
    assert log_swap_acceptance(2., -4., 1., 4.) == -4.5


def test_escapes_real_marginal_likelihood_barrier_that_greedy_moves_cannot_cross():
    class Source:
        alphabet = ('a', 'b')
        probabilities = np.array([[.99, .01], [.01, .99], [.99, .01]])
        def state(self, history): return 0
        def step(self, state, letter): return letter+1
    pool = ('x', 'y', 'xx', 'xy', 'yx', 'yy')
    def score(key):
        value = decode(Source(), key, 'xyxy', .5).log_likelihood
        return value - (2+sum(map(len, key)))*math.log(2)
    start = ('x', 'y')
    neighbors = {s for s in itertools.product(pool, repeat=2)
                 if sum(a != b for a, b in zip(s, start)) <= 1} | {start[::-1]}
    assert all(score(s) <= score(start) for s in neighbors)
    optimum = max(itertools.product(pool, repeat=2), key=score)
    assert score(optimum) > score(start)
    result = search_tempered(score, [start], pool, config=TemperedConfig(
        seed=17, temperatures=(1., 8., 64.), iterations=300, max_scored_keys=100, block_sizes=(2,)))
    assert result['best_score'] == score(optimum)
    assert any(e['accepted'] and result['bank'][e['candidate_index']]['score'] <
               result['bank'][e['previous_index']]['score'] for e in result['events'] if e['event'] == 'proposal'
               and result['bank'][e['candidate_index']]['score'] is not None)
    assert not result['global_optimality_claimed'] and not result['stationary_distribution_claimed']


def test_replay_callbacks_cache_and_event_accounting_are_exact():
    calls, scored, events = [], [], []
    def score(key):
        calls.append(key)
        return None if key == ('y', 'y') else float(key.count('x'))
    config = TemperedConfig(seed=321, temperatures=(1., 2., 4.), iterations=30, swap_interval=2, block_sizes=(2,))
    first = search_tempered(score, [('x', 'x'), ('x', 'y'), ('x', 'x')], ('x', 'y'),
                            config=config, on_score=scored.append, on_event=events.append)
    second = search_tempered(score, [('x', 'x'), ('x', 'y'), ('x', 'x')], ('x', 'y'), config=config)
    for name in first.keys()-{'wall_seconds'}:
        assert first[name] == second[name]
    assert len(calls) == 2*first['scored_keys'] and len(set(calls)) == first['scored_keys']
    assert first['bank'] == scored and first['events'] == events
    assert first['completed_iterations'] == 30 and first['proposals'] == 90
    assert first['exchange_counts']['proposed'] == 15
    assert sum(r['proposed'] for r in first['proposal_counts'].values()) == 90
    assert first['cache_hits'] == 3+90-first['scored_keys']
    unsupported = next(row['index'] for row in scored if row['score'] is None)
    assert all(not e['accepted'] for e in events if e['event'] == 'proposal' and e['candidate_index'] == unsupported)
    replicas = list(first['initial_replica_indices'])
    for event in events:
        if event['event'] == 'proposal':
            assert replicas[event['replica']] == event['previous_index']
            if event['accepted']:
                replicas[event['replica']] = event['candidate_index']
        else:
            left, right = event['left'], event['right']
            assert (replicas[left], replicas[right]) == event['previous_indices']
            if event['accepted']:
                replicas[left], replicas[right] = replicas[right], replicas[left]
    assert tuple(replicas) == first['final_replica_indices']


def test_callbacks_cannot_modify_retained_scores_or_events():
    def modify(row):
        row.clear()
    result = search_tempered(lambda k: float(k.count('x')), [('x', 'x')], ('x', 'y'),
                            config=TemperedConfig(iterations=10, block_sizes=(2,)),
                            on_score=modify, on_event=modify)
    assert result['bank'] and all(row['units'] for row in result['bank'])
    assert result['events'] and all('event' in e for e in result['events'])


def test_score_cap_stops_without_counting_partial_round_or_scoring_extra_key():
    calls = []
    def score(key):
        calls.append(key)
        return 0.
    result = search_tempered(score, [('x', 'x')], ('x', 'y'), config=TemperedConfig(
        temperatures=(1., 2.), max_scored_keys=1, proposal_weights=(1, 0, 0), block_sizes=(2,)))
    assert result['scored_keys'] == len(calls) == 1
    assert result['completed_iterations'] == result['proposals'] == 0
    assert result['stop_reason'] == 'score_limit'


def test_cooperative_time_limit_and_initialization_overrun_are_explicit():
    class Clock:
        now = 0.
        def __call__(self): return self.now
    clock = Clock()
    def slow(key):
        clock.now += 2.
        return 0.
    config = TemperedConfig(max_seconds=1., block_sizes=(2,))
    with pytest.raises(RuntimeError, match='Initialization'):
        search_tempered(slow, [('x', 'x')], ('x', 'y'), config=config, clock=clock)
    clock.now = 0.
    def moderate(key):
        clock.now += .4
        return 0.
    result = search_tempered(moderate, [('x', 'x')], ('x', 'y'), config=config, clock=clock)
    assert result['stop_reason'] == 'time_limit' and result['wall_seconds'] < 1.5
    clock.now = 0.
    result = search_tempered(moderate, [('x', 'x')], ('x', 'y'), clock=clock, config=TemperedConfig(
        temperatures=(1.,), iterations=1, max_seconds=.7, proposal_weights=(1, 0, 0), block_sizes=(2,)))
    assert result['wall_seconds'] == .8 and result['stop_reason'] == 'time_limit'
    assert result['completed_iterations'] == 1  # Fully evaluated; termination cause remains time.


@pytest.mark.parametrize('score', [True, math.nan, math.inf, 'bad'])
def test_numerical_failure_is_not_reclassified_as_unsupported(score):
    with pytest.raises(ValueError):
        search_tempered(lambda k: score, [('x', 'y')], ('x', 'y'))


def test_impossible_start_and_unrepresentable_arithmetic_fail():
    with pytest.raises(ValueError, match='initial replicas'):
        search_tempered(lambda k: None, [('x', 'y')], ('x', 'y'))
    with pytest.raises(ArithmeticError):
        log_acceptance(-1e308, 1e308, 1.)
    with pytest.raises(ArithmeticError):
        log_acceptance(0., -1e308, 1e-308)
    assert log_acceptance(None, 1., 1.) == 0.
    assert log_acceptance(None, None, 1.) == -math.inf


@pytest.mark.parametrize('changes', [
    {'seed': True}, {'iterations': 0}, {'swap_interval': 0}, {'max_scored_keys': 0},
    {'temperatures': ()}, {'temperatures': (1., 1.)}, {'temperatures': (True,)},
    {'max_seconds': math.inf}, {'proposal_weights': (0, 0, 0)}, {'proposal_weights': (1, -1, 1)},
    {'block_sizes': (1,)}, {'block_sizes': (2, 2)},
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        TemperedConfig(**changes)
