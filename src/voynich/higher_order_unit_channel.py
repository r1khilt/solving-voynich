"""Exact fixed-unit inference for normalized finite-order character sources.

Source order is capped at three for this diagnostic. This module does not learn
channel units, run beam search, or infer a source language. All record paths
consume a nonempty unit; likelihood is the sum of every compatible plaintext.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


MAX_CONTEXTS = 100_000
MAX_NODES = 3_000_000


def contexts_for(alphabet: tuple[str, ...], order: int) -> tuple[str, ...]:
    if type(order) is not int or not 0 <= order <= 3:
        raise ValueError("Order must be an integer in [0, 3]")
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(char, str) or len(char) != 1 for char in alphabet)):
        raise ValueError("Alphabet must contain unique Unicode characters")
    if sum(len(alphabet) ** depth for depth in range(order + 1)) > MAX_CONTEXTS:
        raise ValueError("Source exceeds context cap")
    return tuple("".join(chars) for depth in range(order + 1)
                 for chars in itertools.product(alphabet, repeat=depth))


@dataclass(frozen=True)
class MarkovSource:
    alphabet: tuple[str, ...]
    order: int
    probabilities: Mapping[str, Mapping[str, float]]

    def __post_init__(self):
        alphabet = tuple(self.alphabet)
        contexts = contexts_for(alphabet, self.order)
        if set(self.probabilities) != set(contexts):
            raise ValueError("Source must give all contexts up to its order")
        copied = {}
        for context in contexts:
            row = self.probabilities[context]
            if set(row) != set(alphabet):
                raise ValueError("Source row support differs from alphabet")
            values = {char: float(row[char]) for char in alphabet}
            if any(isinstance(row[char], bool) or not math.isfinite(value) or not 0 <= value <= 1
                   for char, value in values.items()):
                raise ValueError("Source probabilities must be finite and in [0, 1]")
            if not math.isclose(math.fsum(values.values()), 1., rel_tol=0., abs_tol=1e-12):
                raise ValueError("Source row is not normalized")
            copied[context] = MappingProxyType(values)
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "probabilities", MappingProxyType(copied))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "alphabet": list(self.alphabet), "order": self.order,
                "probabilities": {context: dict(row) for context, row in self.probabilities.items()}}

    @classmethod
    def from_dict(cls, raw: Mapping) -> MarkovSource:
        if (set(raw) != {"schema_version", "alphabet", "order", "probabilities"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
            raise ValueError("Unexpected source schema")
        return cls(tuple(raw["alphabet"]), raw["order"], raw["probabilities"])


def estimate_source(records: Sequence[str], alphabet: tuple[str, ...], order: int,
                    tau: float) -> MarkovSource:
    """Suffix-recursive Dirichlet interpolation; never join record boundaries."""
    alphabet = tuple(alphabet)
    contexts = contexts_for(alphabet, order)
    if isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Dirichlet mass must be positive and finite")
    if not records or any(not row or set(row) - set(alphabet) for row in records):
        raise ValueError("Training requires nonempty in-alphabet records")
    counts = {context: Counter() for context in contexts}
    for record in records:
        for index, char in enumerate(record):
            for depth in range(min(order, index) + 1):
                context = record[index - depth:index] if depth else ""
                counts[context][char] += 1
    total = sum(counts[""].values()) + .5 * len(alphabet)
    rows = {"": {char: (counts[""][char] + .5) / total for char in alphabet}}
    for context in contexts[1:]:
        total = sum(counts[context].values())
        lower = rows[context[1:]]
        rows[context] = {char: (counts[context][char] + tau * lower[char]) / (total + tau)
                         for char in alphabet}
    return MarkovSource(alphabet, order, rows)


def source_bits(source: MarkovSource, records: Sequence[str]) -> float:
    scores = []
    for record in records:
        if set(record) - set(source.alphabet):
            raise ValueError("Source text outside alphabet")
        context = ""
        for char in record:
            probability = source.probabilities[context][char]
            if probability == 0:
                return math.inf
            scores.append(-math.log2(probability))
            context = (context + char)[-source.order:] if source.order else ""
    return math.fsum(scores)


@dataclass(frozen=True)
class UnitDecode:
    log_likelihood: float
    joint_log_probability: float
    plaintext: str | None
    reachable_nodes: int
    edges: int

    def to_dict(self) -> dict:
        return {"log_likelihood": self.log_likelihood if math.isfinite(self.log_likelihood) else None,
                "joint_log_probability": self.joint_log_probability if math.isfinite(self.joint_log_probability) else None,
                "plaintext": self.plaintext, "reachable_nodes": self.reachable_nodes, "edges": self.edges}


def _logadd(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def decode_units(source: MarkovSource, units: Sequence[str], record: str,
                 stop_probability: float, *, max_nodes: int = MAX_NODES) -> UnitDecode:
    """Log-space sparse dynamic programming over offset and source history.

    Duplicate units remain separate source-letter alternatives. Empty and
    impossible observations are explicit. Ties keep the first DP-discovered
    best path. A finite state cap fails loudly, without silently pruning.
    """
    units = tuple(units)
    if len(units) != len(source.alphabet) or any(not isinstance(unit, str) or not unit for unit in units):
        raise ValueError("Need one nonempty unit per source character")
    if not isinstance(record, str):
        raise ValueError("Observed record must be a string")
    if (isinstance(stop_probability, bool) or not math.isfinite(stop_probability)
            or not 0 < stop_probability < 1):
        raise ValueError("Stop probability must be in (0, 1)")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("Node cap must be positive")
    n = len(record)
    forward: list[dict[str, float]] = [{} for _ in range(n + 1)]
    maximum: list[dict[str, float]] = [{} for _ in range(n + 1)]
    back: dict[tuple[int, str], tuple[int, str, str]] = {}
    forward[0][""] = maximum[0][""] = 0.
    nodes, edges = 1, 0
    log_continue, log_stop = math.log1p(-stop_probability), math.log(stop_probability)
    transition_cache: dict[tuple[str, str], tuple[str, float]] = {}
    for offset in range(n):
        matches = [(char, offset + len(unit)) for char, unit in zip(source.alphabet, units, strict=True)
                   if record.startswith(unit, offset)]
        for context, prefix in forward[offset].items():
            for char, end in matches:
                key = (context, char)
                transition = transition_cache.get(key)
                if transition is None:
                    probability = source.probabilities[context][char]
                    following = (context + char)[-source.order:] if source.order else ""
                    transition = (following, log_continue + math.log(probability)
                                  if probability else -math.inf)
                    transition_cache[key] = transition
                following, log_weight = transition
                if log_weight == -math.inf:
                    continue
                edges += 1
                if following not in forward[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError("Exact lattice exceeds node cap; no pruning performed")
                forward[end][following] = _logadd(forward[end].get(following, -math.inf), prefix + log_weight)
                candidate = maximum[offset][context] + log_weight
                if candidate > maximum[end].get(following, -math.inf):
                    maximum[end][following] = candidate
                    back[end, following] = (offset, context, char)
    if not forward[n]:
        return UnitDecode(-math.inf, -math.inf, None, nodes, edges)
    score = -math.inf
    for value in forward[n].values():
        score = _logadd(score, value)
    context = max(maximum[n], key=maximum[n].get)
    best = maximum[n][context] + log_stop
    offset, letters = n, []
    while offset:
        previous, prior, char = back[offset, context]
        letters.append(char)
        offset, context = previous, prior
    return UnitDecode(score + log_stop, best, "".join(reversed(letters)), nodes, edges)
