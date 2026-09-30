"""Exact finite-state k-best readings with a bound for unseen shared-key tuples.

Per-key acyclic lattices supply exact completion heuristics. Prefix A* lists
distinct texts; Cartesian enumeration preserves one key across records. The
mixture sums every bank key's support for each proposed text tuple. Unseen
tuples are bounded by the weighted sum of per-key next-best scores.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from voynich.shared_key_mixture import _logsum


@dataclass(frozen=True)
class KeyList:
    readings: tuple[tuple[tuple[str, ...], float], ...]
    next_log_probability: float
    log_likelihood: float
    nodes: int
    edges: int
    expanded: int


def record_kbest(source, units, observed, rho, count, *, max_nodes=500_000,
                 max_edges=2_000_000, max_expanded=200_000):
    alphabet = tuple(source.alphabet)
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)
            or len(units) != len(alphabet) or any(not isinstance(u, str) or not u for u in units)
            or not isinstance(observed, str) or isinstance(rho, bool)
            or not math.isfinite(rho) or not 0 < rho < 1
            or any(type(v) is not int or v < 1 for v in (count, max_nodes, max_edges, max_expanded))):
        raise ValueError('Invalid k-best lattice settings')
    n, root = len(observed), source.state('')
    graph = [{} for _ in range(n + 1)]
    graph[0][root] = None
    nodes, edges = 1, 0
    cont, stop = math.log1p(-rho), math.log(rho)
    transitions, rows = {}, {}
    for offset in range(n + 1):
        matches = [(i, offset + len(u)) for i, u in enumerate(units) if observed.startswith(u, offset)]
        for state in graph[offset]:
            outgoing = []
            if state not in rows:
                row = tuple(float(source.probabilities[state, i]) for i in range(len(alphabet)))
                if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
                        or not math.isclose(math.fsum(row), 1., abs_tol=1e-12, rel_tol=0)):
                    raise ValueError('Invalid source row')
                rows[state] = row
            for letter, end in matches:
                probability = rows[state][letter]
                if not probability:
                    continue
                key = state, letter
                if key not in transitions:
                    transitions[key] = source.step(state, letter), cont + math.log(probability)
                following, weight = transitions[key]
                if following not in graph[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('K-best lattice node cap; no pruning')
                    graph[end][following] = None
                outgoing.append((letter, end, following, weight))
                edges += 1
                if edges > max_edges:
                    raise RuntimeError('K-best lattice edge cap; no pruning')
            graph[offset][state] = outgoing
    best, marginal = {}, {}
    for offset in range(n, -1, -1):
        for state, outgoing in graph[offset].items():
            if offset == n:
                best[offset, state] = marginal[offset, state] = stop
            else:
                best[offset, state] = max((w + best[end, following] for _, end, following, w in outgoing), default=-math.inf)
                marginal[offset, state] = _logsum(w + marginal[end, following] for _, end, following, w in outgoing)
    if best[0, root] == -math.inf:
        return [], marginal[0, root], nodes, edges, 0
    serial = itertools.count()
    agenda = [(-best[0, root], next(serial), 0, root, 0., '')]
    found, expanded = [], 0
    while agenda and len(found) < count:
        _, _, offset, state, prefix, text = heapq.heappop(agenda)
        if offset == n:
            found.append((text, prefix + stop))
            continue
        expanded += 1
        if expanded > max_expanded:
            raise RuntimeError('K-best prefix cap; no pruning')
        for letter, end, following, weight in graph[offset][state]:
            bound = prefix + weight + best[end, following]
            if bound != -math.inf:
                heapq.heappush(agenda, (-bound, next(serial), end, following,
                                       prefix + weight, text + alphabet[letter]))
    return found, marginal[0, root], nodes, edges, expanded


def key_kbest(source, units, records, rho, k, **limits):
    records = tuple(records)
    if not records or type(k) is not int or k < 1:
        raise ValueError('Nonempty records and positive k required')
    lists, evidence, nodes, edges, expanded = [], 0., 0, 0, 0
    for record in records:
        values, total, ns, es, xs = record_kbest(source, units, record, rho, k + 1, **limits)
        lists.append(values)
        evidence += total
        nodes, edges, expanded = nodes + ns, edges + es, expanded + xs
    if any(not values for values in lists):
        return KeyList((), -math.inf, -math.inf, nodes, edges, expanded)
    origin = (0,) * len(lists)

    def score(indices):
        return math.fsum(values[i][1] for values, i in zip(lists, indices))

    agenda, seen, found = [(-score(origin), origin)], {origin}, []
    while agenda and len(found) <= k:
        negative, indices = heapq.heappop(agenda)
        found.append((tuple(values[i][0] for values, i in zip(lists, indices)), -negative))
        for r in range(len(lists)):
            following = list(indices)
            following[r] += 1
            following = tuple(following)
            if following[r] < len(lists[r]) and following not in seen:
                seen.add(following)
                heapq.heappush(agenda, (-score(following), following))
    return KeyList(tuple(found[:k]), found[k][1] if len(found) > k else -math.inf,
                   evidence, nodes, edges, expanded)


def _matcher(alphabet, keys):
    masks = {char: {} for char in alphabet}
    for i, key in enumerate(keys):
        for char, unit in zip(alphabet, key, strict=True):
            masks[char][unit] = masks[char].get(unit, 0) | (1 << i)
    lengths = {char: tuple({len(u) for u in row}) for char, row in masks.items()}
    def match(records, texts):
        live = (1 << len(keys)) - 1
        for text, record in zip(texts, records, strict=True):
            groups = {0: live}
            for char in text:
                following = {}
                for offset, group in groups.items():
                    for length in lengths[char]:
                        end = offset + length
                        if end <= len(record):
                            keep = group & masks[char].get(record[offset:end], 0)
                            if keep:
                                following[end] = following.get(end, 0) | keep
                groups = following
            live = groups.get(len(record), 0)
            if not live:
                break
        return live
    return match


def compatible_mask(alphabet, keys, records, texts):
    """Literal deterministic compatibility, compressed across equal offsets."""
    return _matcher(alphabet, keys)(records, texts)


def bounded_mixture(source, keys, records, rho, *, k=8, log_weights=None, progress=None, **limits):
    keys, records = tuple(tuple(key) for key in keys), tuple(records)
    if not keys or len(set(keys)) != len(keys) or not records:
        raise ValueError('Distinct nonempty key bank and records required')
    weights = tuple(0. for _ in keys) if log_weights is None else tuple(log_weights)
    if (len(weights) != len(keys) or any(isinstance(w, bool) or math.isnan(w) or w == math.inf for w in weights)
            or all(w == -math.inf for w in weights)):
        raise ValueError('Invalid mixture weights')
    shifted = tuple(w - max(weights) for w in weights)
    if any(math.isfinite(w) and not math.isfinite(s) for w, s in zip(weights, shifted)):
        raise ArithmeticError('Relative log-weight range exceeds floating representation')
    normalizer = _logsum(shifted)
    weights = tuple(w - normalizer for w in shifted)
    candidates, bound_terms, evidence_terms, lists = {}, [], [], []
    for i, (key, weight) in enumerate(zip(keys, weights)):
        values = key_kbest(source, key, records, rho, k, **limits)
        lists.append(values)
        if weight != -math.inf:
            evidence_terms.append(weight + values.log_likelihood)
            bound_terms.append(weight + values.next_log_probability)
            for texts, score in values.readings:
                if texts in candidates and abs(candidates[texts] - score) > 1e-8:
                    raise ArithmeticError('Same source text has inconsistent scores')
                candidates[texts] = score
        if progress is not None:
            progress(i, values)
    best_texts, best_score, best_keys = None, -math.inf, ()
    match = _matcher(source.alphabet, keys)
    for texts, score in candidates.items():
        mask = match(records, texts)
        compatible = tuple(i for i, w in enumerate(weights) if w != -math.inf and mask & (1 << i))
        joint = score + _logsum(weights[i] for i in compatible)
        if joint > best_score:
            best_texts, best_score, best_keys = texts, joint, compatible
    upper, evidence = _logsum(bound_terms), _logsum(evidence_terms)
    tolerance = 1e-8 * max(1, sum(map(len, records)))
    return {'plaintexts': best_texts, 'joint_log_probability': best_score,
            'log_likelihood': evidence, 'unseen_log_probability_upper_bound': upper,
            'floating_bound_separated': best_texts is not None and (upper == -math.inf or best_score > upper + tolerance),
            'not_interval_arithmetic_proof': True, 'compatible_key_indices': best_keys,
            'candidate_tuples': len(candidates), 'k': k,
            'per_key': lists}
