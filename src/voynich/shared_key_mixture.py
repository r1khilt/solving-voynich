"""Exact finite-state decoding with one uncertain key shared across records.

Determinize key compatibility by tracking every key's observed offset under the
same plaintext prefix. Sum key mass only at the final stop. This finds the MAP
text tuple, not the MAP key/text path. A state cap fails without beam fallback.
The supplied source state must be sufficient for its future probabilities.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real


def _logsum(values):
    values = tuple(values)
    if not values:
        return -math.inf
    high = max(values)
    return high + math.log(math.fsum(math.exp(v - high) for v in values)) if high != -math.inf else high


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low - high))


@dataclass(frozen=True)
class SharedKeyReading:
    plaintexts: tuple[str, ...] | None
    log_likelihood: float | None
    joint_log_probability: float | None
    compatible_key_indices: tuple[int, ...]
    reachable_nodes: int
    edges: int
    bank_size: int
    records: int
    maximum_offset_groups: int


def decode_shared_keys(source, keys, records, rho, *, log_weights=None, max_nodes=500_000):
    alphabet = tuple(source.alphabet)
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)
            or isinstance(keys, (str, bytes)) or isinstance(records, (str, bytes))):
        raise ValueError('Ordered character alphabet, distinct keys and separate records required')
    keys = tuple(keys)
    if any(isinstance(key, (str, bytes)) for key in keys):
        raise ValueError('A key must be a sequence of complete units, not a string')
    keys = tuple(tuple(key) for key in keys)
    records = tuple(records)
    if (not keys or any(len(key) != len(alphabet) for key in keys)
            or any(not isinstance(u, str) or not u for key in keys for u in key)
            or not records or any(not isinstance(r, str) for r in records)
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or type(max_nodes) is not int or max_nodes < 1):
        raise ValueError('Invalid finite key-bank decoding settings')
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate keys need explicit mass aggregation before decoding')
    weights = tuple(0. for _ in keys) if log_weights is None else tuple(log_weights)
    if (len(weights) != len(keys) or any(isinstance(w, bool) or not isinstance(w, Real)
                                       or math.isnan(w) or w == math.inf for w in weights)
            or all(w == -math.inf for w in weights)):
        raise ValueError('Finite relative log weights or negative infinity required; some mass must remain')
    high = max(weights)
    shifted = tuple(w - high for w in weights)
    if any(math.isfinite(w) and not math.isfinite(s) for w, s in zip(weights, shifted)):
        raise ArithmeticError('Relative log-weight range exceeds floating representation')
    normalizer = _logsum(shifted)
    weights = tuple(w - normalizer for w in shifted)
    if any(math.isnan(w) or w == math.inf for w in weights):
        raise ArithmeticError('Unrepresentable normalized key weights')
    root = source.state('')
    initial_mask = sum(1 << i for i, w in enumerate(weights) if w != -math.inf)
    # Keys at the same observed offset share a bit set. This is an exact
    # compressed representation of the per-key offset vector, not pruning.
    start = (0, root, ((0, initial_mask),))
    states, index = [start], {start: 0}
    forward, best, back = [0.], [0.], [None]
    stop, cont = math.log(rho), math.log1p(-rho)

    def rank(state):
        record, _, groups = state
        live = sum(mask.bit_count() for _, mask in groups)
        return record, len(keys) - live, sum(offset * mask.bit_count() for offset, mask in groups)

    heap = [(*rank(start), 0)]
    evidence, maximum, terminal_node, surviving = -math.inf, -math.inf, None, ()
    edges = 0
    maximum_groups = 1
    unit_masks = [dict() for _ in alphabet]
    for i, key in enumerate(keys):
        for letter, unit in enumerate(key):
            unit_masks[letter][unit] = unit_masks[letter].get(unit, 0) | (1 << i)
    unit_lengths = [tuple(sorted({len(unit) for unit in masks})) for masks in unit_masks]

    @lru_cache(maxsize=200_000)
    def source_row(state):
        row = tuple(float(source.probabilities[state, i]) for i in range(len(alphabet)))
        if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
                or not math.isclose(math.fsum(row), 1., rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Invalid normalized source row')
        return row

    @lru_cache(maxsize=200_000)
    def advance(record, groups, letter):
        observed = records[record]
        following = {}
        for offset, live in groups:
            for length in unit_lengths[letter]:
                end = offset + length
                if end <= len(observed):
                    retained = live & unit_masks[letter].get(observed[offset:end], 0)
                    if retained:
                        following[end] = following.get(end, 0) | retained
        return tuple(sorted(following.items()))

    while heap:
        _, _, _, node = heapq.heappop(heap)
        state = states[node]
        record, context, groups = state
        complete_mask = next((mask for offset, mask in groups if offset == len(records[record])), 0)

        def offer(following, weight, label):
            nonlocal edges, maximum_groups
            if rank(following) <= rank(state):
                raise AssertionError('Compatibility graph must be acyclic')
            target = index.get(following)
            if target is None:
                if len(states) >= max_nodes:
                    raise RuntimeError('Shared-key exact state cap exceeded; no approximation')
                target = len(states)
                states.append(following)
                index[following] = target
                forward.append(-math.inf)
                best.append(-math.inf)
                back.append(None)
                maximum_groups = max(maximum_groups, len(following[2]))
                heapq.heappush(heap, (*rank(following), target))
            forward[target] = _logadd(forward[target], forward[node] + weight)
            candidate = best[node] + weight
            if candidate > best[target]:
                best[target] = candidate
                back[target] = node, label
            edges += 1

        if complete_mask:
            if record == len(records) - 1:
                complete = tuple(i for i in range(len(keys)) if complete_mask & (1 << i))
                terminal_weight = stop + _logsum(weights[i] for i in complete)
                evidence = _logadd(evidence, forward[node] + terminal_weight)
                score = best[node] + terminal_weight
                if score > maximum:
                    maximum, terminal_node, surviving = score, node, complete
            else:
                following = (record + 1, root, ((0, complete_mask),))
                offer(following, stop, None)
        for letter, probability in enumerate(source_row(context)):
            if probability == 0:
                continue
            following_groups = advance(record, groups, letter)
            if not following_groups:
                continue
            offer((record, source.step(context, letter), following_groups), cont + math.log(probability), alphabet[letter])
    if terminal_node is None:
        return SharedKeyReading(None, None, None, (), len(states), edges, len(keys), len(records), maximum_groups)
    if maximum > evidence + 1e-8 or evidence > 1e-8:
        raise ArithmeticError('Invalid joint MAP/evidence ordering')
    actions = []
    node = terminal_node
    while back[node] is not None:
        node, label = back[node]
        actions.append(label)
    texts = ['']
    for label in reversed(actions):
        if label is None:
            texts.append('')
        else:
            texts[-1] += label
    if len(texts) != len(records):
        raise AssertionError('Record-boundary traceback changed')
    return SharedKeyReading(tuple(texts), evidence, maximum, surviving, len(states), edges, len(keys), len(records), maximum_groups)
