"""Finite, suffix-closed source automata with exact variable-unit decoding.

Unobserved contexts back off without allocating the complete alphabet**order
space. This is fixed recursive interpolation, not a Bayesian posterior sampler
or PPM implementation. Learned probabilities never adapt to ciphertext.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from voynich.higher_order_unit_channel import UnitDecode


MAX_ORDER = 12
MAX_CONTEXTS = 1_200_000


def collect_counts(records: Sequence[str], alphabet: Sequence[str], order: int,
                   *, max_contexts: int = MAX_CONTEXTS) -> dict[str, dict[str, int]]:
    alphabet = tuple(alphabet)
    _settings(alphabet, order, 1.)
    if type(max_contexts) is not int or max_contexts < 1:
        raise ValueError("Positive context cap required")
    if not records or any(not isinstance(r, str) or not r or set(r) - set(alphabet) for r in records):
        raise ValueError("Nonempty in-alphabet training records required")
    counts: dict[str, Counter] = {}
    for record in records:
        for index, char in enumerate(record):
            for depth in range(min(order, index) + 1):
                context = record[index - depth:index]
                if context not in counts:
                    if len(counts) >= max_contexts:
                        raise RuntimeError("Source context cap exceeded; no pruning")
                    counts[context] = Counter()
                counts[context][char] += 1
    return {context: dict(row) for context, row in counts.items()}


def _settings(alphabet, order, tau):
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)):
        raise ValueError("Alphabet requires unique characters")
    if type(order) is not int or not 0 <= order <= MAX_ORDER:
        raise ValueError("Order outside [0, 12]")
    if isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Positive finite interpolation mass required")


class SuffixSource:
    def __init__(self, alphabet: Sequence[str], order: int, tau: float,
                 counts: Mapping[str, Mapping[str, int]]):
        alphabet = tuple(alphabet)
        _settings(alphabet, order, tau)
        if "" not in counts or not counts[""] or len(counts) > MAX_CONTEXTS:
            raise ValueError("Source requires root counts within context cap")
        clean = {}
        for context, row in counts.items():
            if (not isinstance(context, str) or len(context) > order or set(context) - set(alphabet)
                    or not row or set(row) - set(alphabet)
                    or any(type(n) is not int or n <= 0 for n in row.values())):
                raise ValueError("Invalid source counts")
            if context and (context[1:] not in counts or context[:-1] not in counts):
                raise ValueError("Contexts must be both prefix and suffix closed")
            clean[context] = MappingProxyType(dict(row))
        self.alphabet, self.order, self.tau = alphabet, order, float(tau)
        self.counts = MappingProxyType(clean)
        self.contexts = tuple(sorted(clean, key=lambda c: (len(c), c)))
        self.indices = MappingProxyType({c: i for i, c in enumerate(self.contexts)})
        self.letters = MappingProxyType({c: i for i, c in enumerate(alphabet)})
        probabilities = np.empty((len(clean), len(alphabet)), dtype=np.float64)
        for index, context in enumerate(self.contexts):
            row = clean[context]
            values = np.array([row.get(c, 0) for c in alphabet], dtype=np.float64)
            total = sum(row.values())
            if not context:
                probabilities[index] = (values + .5) / (total + .5 * len(alphabet))
            else:
                lower = probabilities[self.indices[context[1:]]]
                probabilities[index] = (values + tau * lower) / (total + tau)
        if not np.all(np.isfinite(probabilities)) or not np.all(probabilities > 0):
            raise ValueError("Nonpositive or nonfinite source")
        if not np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-12):
            raise ValueError("Unnormalized source")
        probabilities.setflags(write=False)
        self.probabilities = probabilities

    def state(self, history: str) -> int:
        if not isinstance(history, str) or set(history) - set(self.alphabet):
            raise ValueError("History outside source alphabet")
        history = history[-self.order:] if self.order else ""
        while history not in self.indices:
            history = history[1:]
        return self.indices[history]

    def step(self, state: int, letter: int) -> int:
        following = self.contexts[state] + self.alphabet[letter]
        if len(following) > self.order:
            following = following[1:]
        while following not in self.indices:
            following = following[1:]
        return self.indices[following]

    def bits(self, records: Sequence[str]) -> float:
        values = []
        for record in records:
            if not isinstance(record, str) or set(record) - set(self.alphabet):
                raise ValueError("Text outside source alphabet")
            state = 0
            for char in record:
                letter = self.letters[char]
                values.append(-math.log2(self.probabilities[state, letter]))
                state = self.step(state, letter)
        return math.fsum(values)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "alphabet": list(self.alphabet), "order": self.order,
                "tau": self.tau, "counts": {c: dict(self.counts[c]) for c in self.contexts}}

    @classmethod
    def from_dict(cls, raw):
        if (set(raw) != {"schema_version", "alphabet", "order", "tau", "counts"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
            raise ValueError("Invalid suffix source schema")
        return cls(raw["alphabet"], raw["order"], raw["tau"], raw["counts"])


def decode(source: SuffixSource, units: Sequence[str], record: str, rho: float,
           *, max_nodes: int = 3_000_000) -> UnitDecode:
    """Exact sum and MAP on observed offsets times suffix-automaton states."""
    if len(units) != len(source.alphabet) or any(not isinstance(u, str) or not u for u in units):
        raise ValueError("One nonempty unit per source letter required")
    if not isinstance(record, str):
        raise ValueError("Observed record must be a string")
    if isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("Stopping probability outside (0, 1)")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("Positive lattice cap required")
    n = len(record)
    forward, best = [{} for _ in range(n + 1)], [{} for _ in range(n + 1)]
    forward[0][0] = best[0][0] = 0.
    back, transitions = {}, {}
    nodes, edges = 1, 0
    cont, stop = math.log1p(-rho), math.log(rho)
    for offset in range(n):
        matches = [(i, offset + len(u)) for i, u in enumerate(units) if record.startswith(u, offset)]
        for state, prefix in forward[offset].items():
            for letter, end in matches:
                key = state, letter
                if key not in transitions:
                    transitions[key] = (source.step(state, letter), cont + math.log(source.probabilities[state, letter]))
                following, weight = transitions[key]
                edges += 1
                if following not in forward[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError("Exact lattice cap exceeded; no pruning")
                    forward[end][following] = prefix + weight
                else:
                    a, b = forward[end][following], prefix + weight
                    high, low = max(a, b), min(a, b)
                    forward[end][following] = high + math.log1p(math.exp(low - high))
                candidate = best[offset][state] + weight
                if candidate > best[end].get(following, -math.inf):
                    best[end][following] = candidate
                    back[end, following] = offset, state, letter
    if not forward[n]:
        return UnitDecode(-math.inf, -math.inf, None, nodes, edges)
    peak = max(forward[n].values())
    total = peak + math.log(math.fsum(math.exp(x - peak) for x in forward[n].values())) + stop
    state = max(best[n], key=best[n].get)
    maximum = best[n][state] + stop
    offset, text = n, []
    while offset:
        offset, state, letter = back[offset, state]
        text.append(source.alphabet[letter])
    return UnitDecode(total, maximum, "".join(reversed(text)), nodes, edges)
