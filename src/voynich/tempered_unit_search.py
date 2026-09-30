"""Bounded replica search with symmetric dictionary proposals and exact scores.

Every target score comes from the caller. No source fitting, decoding, answer
access, approximate likelihood, convergence or global optimum is implied.
"""
import math
import random
import time
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class TemperedConfig:
    seed: int = 0
    temperatures: tuple = (1., 2., 4., 8., 16., 32., 64., 128.)
    iterations: int = 10_000
    swap_interval: int = 4
    max_scored_keys: int = 100_000
    max_seconds: float = 600.
    proposal_weights: tuple = (2, 1, 1)  # replacement, swap, block replacement
    block_sizes: tuple = (2, 3, 6)

    def __post_init__(self):
        for name in ('seed', 'iterations', 'swap_interval', 'max_scored_keys'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'seed' else 1):
                raise ValueError('Invalid integer search setting: '+name)
        temps = tuple(self.temperatures)
        if (not temps or any(isinstance(t, bool) or not isinstance(t, Real)
                             or not math.isfinite(t) or t <= 0 for t in temps)
                or any(a >= b for a, b in zip(temps, temps[1:]))):
            raise ValueError('Temperatures must be finite, positive and increasing')
        if (isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, Real)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError('Positive cooperative time limit required')
        weights = tuple(self.proposal_weights)
        if len(weights) != 3 or any(type(w) is not int or w < 0 for w in weights) or not sum(weights):
            raise ValueError('Nonnegative integer mixture weights required')
        blocks = tuple(self.block_sizes)
        if not blocks or len(set(blocks)) != len(blocks) or any(type(n) is not int or n < 2 for n in blocks):
            raise ValueError('Distinct block sizes of at least two required')
        object.__setattr__(self, 'temperatures', temps)
        object.__setattr__(self, 'proposal_weights', weights)
        object.__setattr__(self, 'block_sizes', blocks)


def _score(value):
    if value is None:
        return -math.inf
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) or value == math.inf:
        raise ValueError('Numerical failure cannot become unsupported probability')
    return float(value)


def log_acceptance(previous, candidate, temperature):
    """Symmetric Metropolis ratio; impossible candidates stay impossible."""
    previous, candidate = _score(previous), _score(candidate)
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Invalid temperature')
    if candidate == -math.inf:
        return -math.inf
    if previous == -math.inf:
        return 0.
    difference = candidate - previous
    if not math.isfinite(difference):
        raise ArithmeticError('Unrepresentable score difference')
    ratio = difference / temperature
    if not math.isfinite(ratio):
        raise ArithmeticError('Unrepresentable acceptance ratio')
    return min(0., ratio)


def log_swap_acceptance(left_score, right_score, left_temperature, right_temperature):
    left_score, right_score = _score(left_score), _score(right_score)
    if (not math.isfinite(left_score) or not math.isfinite(right_score)
            or any(isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or t <= 0
                   for t in (left_temperature, right_temperature))):
        raise ValueError('Swaps require supported states and positive temperatures')
    ratio = (1/left_temperature - 1/right_temperature) * (right_score - left_score)
    if not math.isfinite(ratio):
        raise ArithmeticError('Unrepresentable replica exchange ratio')
    return min(0., ratio)


def propose(key, pool, rng, config):
    """Each component is symmetric, including self loops and duplicate units.

    Block rows are chosen independently of the key. New units are uniform over
    the whole pool, INCLUDING old units. Conditioning on actual change would
    require a separate symmetry argument and is deliberately not done here.
    """
    key, pool = tuple(key), tuple(pool)
    if (len(key) < 2 or len(pool) < 2 or len(set(pool)) != len(pool)
            or any(not isinstance(u, str) or not u for u in pool) or any(u not in pool for u in key)):
        raise ValueError('Invalid dictionary proposal family')
    sizes = tuple(n for n in config.block_sizes if n <= len(key))
    if not sizes and config.proposal_weights[2]:
        raise ValueError('No legal block size')
    draw = rng.randrange(sum(config.proposal_weights))
    replacement, swap, _ = config.proposal_weights
    changed = list(key)
    if draw < replacement:
        kind = 'replace'
        rows = (rng.randrange(len(key)),)
        changed[rows[0]] = rng.choice(tuple(u for u in pool if u != key[rows[0]]))
    elif draw < replacement + swap:
        kind = 'swap'
        rows = tuple(sorted(rng.sample(range(len(key)), 2)))
        changed[rows[0]], changed[rows[1]] = key[rows[1]], key[rows[0]]
    else:
        kind = 'block'
        rows = tuple(sorted(rng.sample(range(len(key)), rng.choice(sizes))))
        for row in rows:
            changed[row] = rng.choice(pool)
    return tuple(changed), {'kind': kind, 'rows': rows, 'new_units': tuple(changed[r] for r in rows)}


def search_tempered(score_key, starts, pool, *, config=None, on_score=None, on_event=None, clock=None):
    """Retain every fully scored key; callers must freeze empirical retention.

    A deterministic scan composes invariant Metropolis/exchange kernels. It is
    not a claim that the whole composed scan is reversible, that finite output
    is stationary, or that visitation counts provide prior/posterior weights.
    Initialization counts against score/time budgets and must fully complete.
    """
    config = TemperedConfig() if config is None else config
    clock = time.monotonic if clock is None else clock
    starts, pool = tuple(tuple(s) for s in starts), tuple(pool)
    if len(starts) not in (1, len(config.temperatures)):
        raise ValueError('Supply one start or one per temperature')
    if (not starts or len(starts[0]) < 2 or len(pool) < 2 or len(set(pool)) != len(pool)
            or any(not isinstance(u, str) or not u for u in pool)
            or any(len(s) != len(starts[0]) or any(u not in pool for u in s) for s in starts)
            or len(set(starts)) > config.max_scored_keys
            or (config.proposal_weights[2] and not any(n <= len(starts[0]) for n in config.block_sizes))):
        raise ValueError('Invalid starts or literal unit pool')
    started = clock()
    deadline = started + config.max_seconds
    rng = random.Random(config.seed)
    indices, bank, scores, events = {}, [], [], []
    best = None
    hits = 0

    def evaluate(key):
        nonlocal best, hits
        if key in indices:
            hits += 1
            return indices[key]
        value = _score(score_key(key))
        index = len(bank)
        row = {'index': index, 'units': key, 'score': None if value == -math.inf else value}
        indices[key] = index
        bank.append(row)
        scores.append(value)
        if best is None or value > scores[best]:
            best = index
        if on_score is not None:
            on_score(dict(row))
        return index

    def emit(event):
        events.append(event)
        if on_event is not None:
            on_event(dict(event))

    def uniform_log():
        draw = rng.random()
        return math.log(draw) if draw else -math.inf

    replicas = []
    for start in starts:
        if clock() >= deadline:
            raise RuntimeError('Initialization did not complete before time limit')
        index = evaluate(start)
        if clock() >= deadline:
            raise RuntimeError('Initialization did not complete before time limit')
        if not math.isfinite(scores[index]):
            raise ValueError('All initial replicas must support the observations')
        replicas.append(index)
    if len(replicas) == 1:
        replicas *= len(config.temperatures)
    initial_replicas = tuple(replicas)
    counts = {k: {'proposed': 0, 'accepted': 0} for k in ('replace', 'swap', 'block')}
    exchanges = {'proposed': 0, 'accepted': 0}
    completed = proposals = self_proposals = exchange_round = 0
    reason = 'iteration_limit'
    stop = False
    for iteration in range(config.iterations):
        for replica, temperature in enumerate(config.temperatures):
            if clock() >= deadline:
                reason, stop = 'time_limit', True
                break
            previous = replicas[replica]
            key, metadata = propose(bank[previous]['units'], pool, rng, config)
            if key not in indices and len(bank) >= config.max_scored_keys:
                reason, stop = 'score_limit', True
                break
            candidate = evaluate(key)
            threshold = log_acceptance(scores[previous], scores[candidate], temperature)
            draw = uniform_log()
            accepted = draw < threshold
            if accepted:
                replicas[replica] = candidate
            kind = metadata['kind']
            counts[kind]['proposed'] += 1
            counts[kind]['accepted'] += int(accepted)
            proposals += 1
            self_proposals += int(candidate == previous)
            emit({'event': 'proposal', 'iteration': iteration, 'replica': replica,
                  'previous_index': previous, 'candidate_index': candidate, **metadata,
                  'log_acceptance': None if threshold == -math.inf else threshold,
                  'log_uniform': None if draw == -math.inf else draw, 'accepted': accepted})
        if stop:
            break
        if (iteration + 1) % config.swap_interval == 0:
            for left in range(exchange_round % 2, len(replicas)-1, 2):
                right = left + 1
                old_left, old_right = replicas[left], replicas[right]
                threshold = log_swap_acceptance(scores[old_left], scores[old_right],
                                                config.temperatures[left], config.temperatures[right])
                draw = uniform_log()
                accepted = draw < threshold
                if accepted:
                    replicas[left], replicas[right] = old_right, old_left
                exchanges['proposed'] += 1
                exchanges['accepted'] += int(accepted)
                emit({'event': 'exchange', 'iteration': iteration, 'left': left, 'right': right,
                      'previous_indices': (old_left, old_right), 'log_acceptance': threshold,
                      'log_uniform': None if draw == -math.inf else draw, 'accepted': accepted})
            exchange_round += 1
        completed += 1
    finished = clock()
    if finished >= deadline:
        reason = 'time_limit'
    return {'bank': bank, 'events': events, 'best_index': best, 'best_units': bank[best]['units'],
            'best_score': scores[best], 'initial_replica_indices': initial_replicas,
            'final_replica_indices': tuple(replicas), 'temperatures': config.temperatures,
            'completed_iterations': completed, 'proposals': proposals, 'self_proposals': self_proposals,
            'scored_keys': len(bank), 'cache_hits': hits, 'proposal_counts': counts, 'exchange_counts': exchanges,
            'stop_reason': reason, 'wall_seconds': finished-started,
            'global_optimality_claimed': False, 'stationary_distribution_claimed': False}
