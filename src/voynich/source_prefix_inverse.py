"""Cipher-only source-prefix search with a growing shared iid dictionary.

Unused rows are analytically integrated. Ordering heuristics affect exploration,
never the probability law. Frontier/evicted subtrees retain probability bounds.
This is a finite visited-bank method unless the entire frontier is exhausted
without pruning; floating bounds are not interval-arithmetic certificates.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = max(a, b), min(a, b)
    return high+math.log1p(math.exp(low-high))


def logsum(values):
    value = -math.inf
    for v in values:
        value = logadd(value, v)
    return value


@dataclass(frozen=True)
class Node:
    completed: tuple
    prefix: tuple
    offset: int
    key: tuple
    score: float


class BoundedQueue:
    """Two heaps plus live IDs; lazy entries compacted to bounded storage."""
    def __init__(self, capacity):
        self.capacity, self.serial = capacity, 0
        self.best, self.worst, self.live = [], [], {}
        self.pruned_bound, self.pruned_count, self.maximum = -math.inf, 0, 0

    def push(self, node, priority, bound):
        token = self.serial
        self.serial += 1
        self.live[token] = (node, priority, bound)
        heapq.heappush(self.best, (-priority, token))
        heapq.heappush(self.worst, (priority, -token))
        if len(self.live) > self.capacity:
            while self.worst:
                _, negative = heapq.heappop(self.worst)
                victim = -negative
                if victim in self.live:
                    _, _, dropped = self.live.pop(victim)
                    self.pruned_bound = logadd(self.pruned_bound, dropped)
                    self.pruned_count += 1
                    break
        self.maximum = max(self.maximum, len(self.live))
        # Deleted entries must not grow with the whole search history.
        if max(len(self.best), len(self.worst)) > 4*self.capacity+16:
            self.best = [(-p, token) for token, (_, p, _) in self.live.items()]
            self.worst = [(p, -token) for token, (_, p, _) in self.live.items()]
            heapq.heapify(self.best)
            heapq.heapify(self.worst)

    def pop(self):
        while self.best:
            _, token = heapq.heappop(self.best)
            if token in self.live:
                return self.live.pop(token)[0]
        raise IndexError("Empty active queue")

    def unresolved(self):
        return logadd(self.pruned_bound, logsum(b for _, _, b in self.live.values()))


def subtree_bound(node, cipher, rho):
    """Remaining source lengths are between ceil(glyphs/2) and glyphs.

    Sum the geometric length law across that entire interval, not just the
    probability of one EOS/length. It bounds all matching descendants together.
    """
    value = node.score
    cont = math.log1p(-rho)
    for r in range(len(node.completed), len(cipher)):
        remaining = len(cipher[r])-(node.offset if r == len(node.completed) else 0)
        low, high = (remaining+1)//2, remaining
        value += low*cont+math.log(-math.expm1((high-low+1)*cont))
    return value


def search_source_prefix(cipher, next_probabilities, *, rows=23, glyphs=6,
                         rho=1/225, max_expanded=20_000, max_frontier=4096,
                         max_terminals=4096, progress_bonus=0., canonical=False,
                         numerical_margin=1e-9, observe=None):
    """No plaintext, key, source length or unit boundaries are input arguments.

    next_probabilities sees the current hypothesized source prefix only and
    must supply a deterministic normalized source distribution, reset per record.
    """
    cipher = tuple(map(tuple, cipher))
    if (any(type(v) is not int or v < 1 for v in (rows, glyphs, max_expanded, max_frontier, max_terminals))
            or not cipher or any(not r or any(type(g) is not int or not 0 <= g < glyphs for g in r) for r in cipher)
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or isinstance(progress_bonus, bool) or not math.isfinite(progress_bonus) or progress_bonus < 0
            or type(canonical) is not bool or not math.isfinite(numerical_margin) or numerical_margin < 0):
        raise ValueError("Invalid observation, declared prior or bounded search configuration")
    seen = tuple(dict.fromkeys(g for record in cipher for g in record))
    if canonical and seen != tuple(range(len(seen))):
        raise ValueError("Canonical observations require global first-occurrence names")
    orbit = math.factorial(glyphs)//math.factorial(glyphs-len(seen)) if canonical else 1
    frame = math.log(orbit)
    penalty, continuation, stop = math.log(glyphs+glyphs**2), math.log1p(-rho), math.log(rho)
    queue, terminals = BoundedQueue(max_frontier), []
    expanded, source_calls, source_requests, generated = 0, 0, 0, 0
    # Source histories can be shared only when the full reset prefix matches.
    # No key/cipher-state merging or Markov-only history truncation is performed.
    source_cache = {}

    def enqueue(node):
        nonlocal generated
        consumed = sum(map(len, cipher[:len(node.completed)]))+node.offset
        bound = subtree_bound(node, cipher, rho)
        priority = bound+progress_bonus*consumed
        queue.push(node, priority, bound)
        generated += 1

    enqueue(Node((), (), 0, (None,)*rows, 0.))
    while queue.live and expanded < max_expanded and len(terminals) < max_terminals:
        node = queue.pop()
        expanded += 1
        record = len(node.completed)
        if record == len(cipher):
            terminals.append(node)
        elif node.offset == len(cipher[record]):
            enqueue(Node(node.completed+(node.prefix,), (), 0, node.key, node.score+stop))
        else:
            source_requests += 1
            if node.prefix not in source_cache:
                probabilities = tuple(next_probabilities(node.prefix))
                source_calls += 1
                if (len(probabilities) != rows or any(isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                        or abs(math.fsum(probabilities)-1.) > 1e-12):
                    raise ValueError("Source callback must return normalized finite row probabilities")
                source_cache[node.prefix] = probabilities
            probabilities = source_cache[node.prefix]
            for row, probability in enumerate(probabilities):
                if probability == 0:
                    continue
                base = node.score+continuation+math.log(probability)
                existing = node.key[row]
                if existing is not None:
                    if cipher[record][node.offset:node.offset+len(existing)] == existing:
                        enqueue(Node(node.completed, node.prefix+(row,), node.offset+len(existing), node.key, base))
                else:
                    for length in (1, 2):
                        if node.offset+length <= len(cipher[record]):
                            key = list(node.key)
                            key[row] = cipher[record][node.offset:node.offset+length]
                            enqueue(Node(node.completed, node.prefix+(row,), node.offset+length, tuple(key), base-penalty))
        if observe is not None and expanded % 100 == 0:
            observe({"expanded": expanded, "generated": generated, "active": len(queue.live),
                     "terminals": len(terminals), "pruned": queue.pruned_count})
    groups = {}
    if len({(n.completed, n.key) for n in terminals}) != len(terminals):
        raise AssertionError("Each source/used-dictionary leaf must be counted once")
    for node in terminals:
        group = groups.setdefault(node.completed, {"log_mass": -math.inf, "visited_assignments": 0})
        group["log_mass"] = logadd(group["log_mass"], node.score+frame)
        group["visited_assignments"] += 1
    ordered = sorted(groups.items(), key=lambda pair: (-pair[1]["log_mass"], pair[0]))
    found = logsum(v["log_mass"] for _, v in ordered)
    unresolved = queue.unresolved()
    if unresolved != -math.inf:
        unresolved += frame
    second = ordered[1][1]["log_mass"] if len(ordered) > 1 else -math.inf
    separated = bool(ordered and ordered[0][1]["log_mass"] > logadd(second, unresolved)+numerical_margin)
    complete = not queue.live and queue.pruned_count == 0
    if complete:
        reason = "exhaustive"
    elif not queue.live:
        reason = "frontier_exhausted_with_pruning"
    elif expanded == max_expanded:
        reason = "expansion_cap"
    else:
        reason = "terminal_cap"
    return {"complete_search": complete, "stop_reason": reason,
            "readings": [{"source_records": text, **value} for text, value in ordered],
            "terminals": [{"source_records": n.completed, "used_key": n.key, "log_mass": n.score+frame} for n in terminals],
            "found_log_mass": found, "unresolved_log_mass_upper": unresolved,
            "evidence_log_upper": logadd(found, unresolved), "reading_bound_separated": separated,
            "interval_arithmetic_certificate": False, "expanded": expanded, "generated": generated,
            "source_calls": source_calls, "source_requests": source_requests,
            "pruned_states": queue.pruned_count,
            "maximum_active_states": queue.maximum, "canonical_orbit_factor": orbit,
            "probability_frame": "canonical_glyphs" if canonical else "literal_glyphs",
            "ordering_progress_bonus": progress_bonus,
            "scope": "Shared iid dictionary/normalized source/geometric lengths. Visited masses are lower bounds unless exhaustive; no historical identification."}
