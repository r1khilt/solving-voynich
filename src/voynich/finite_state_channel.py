"""Exact finite-state likelihoods for a normalized variable-emission channel.

This module evaluates GIVEN source/channel parameters; it does not learn keys,
units, or states. Sources have order zero or one. Each source character emits
at least one glyph, so the observed-string lattice is finite and acyclic.
Forward sums all plaintext/state/segmentation paths; Viterbi selects the best
JOINT path, which need not give the most probable plaintext after path sums.
"Exact" means unpruned finite inference; scores still use floating-point
arithmetic, and probability normalization is checked to the declared tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


NORMALIZATION_TOLERANCE = 1e-12
NEGATIVE_INFINITY = -math.inf
Node = tuple[str, str]  # source context, channel state (at a given glyph offset)
EmissionKey = tuple[str, str, int]  # starting state, source letter, row index


def _probability(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not probabilities")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return number


def _names(values: tuple[str, ...], *, characters: bool, name: str) -> tuple[str, ...]:
    values = tuple(values)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique strings")
    if characters and any(len(value) != 1 for value in values):
        raise ValueError(f"{name} entries must be single Unicode codepoints")
    return values


def _distribution(values: Mapping[str, float], keys: tuple[str, ...]) -> Mapping[str, float]:
    if set(values) != set(keys):
        raise ValueError("Distribution keys must exactly match the declared support")
    copied = {key: _probability(values[key]) for key in keys}
    if not math.isclose(math.fsum(copied.values()), 1.0, rel_tol=0.0,
                        abs_tol=NORMALIZATION_TOLERANCE):
        raise ValueError("Distribution must sum to one; no automatic normalization")
    return MappingProxyType(copied)


def _check_fields(value: Mapping[str, Any], fields: set[str]) -> None:
    if set(value) != fields or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unexpected serialized fields or schema version")


@dataclass(frozen=True)
class SourceModel:
    """Fixed normalized character probabilities with an explicit start context.

    Order 0 requires only context ''. Order 1 also requires one row per letter.
    No fitting, smoothing, fallback, or normalization is performed implicitly.
    """

    alphabet: tuple[str, ...]
    order: int
    probabilities: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        alphabet = _names(self.alphabet, characters=True, name="Source alphabet")
        if type(self.order) is not int or self.order not in (0, 1):
            raise ValueError("Only source order zero or one is implemented")
        contexts = ("",) if self.order == 0 else ("", *alphabet)
        if set(self.probabilities) != set(contexts):
            raise ValueError("Source context rows must exactly match the declared order")
        rows = {context: _distribution(self.probabilities[context], alphabet) for context in contexts}
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "probabilities", MappingProxyType(rows))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "alphabet": list(self.alphabet), "order": self.order,
                "probabilities": {context: dict(row) for context, row in self.probabilities.items()}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceModel:
        _check_fields(value, {"schema_version", "alphabet", "order", "probabilities"})
        return cls(tuple(value["alphabet"]), value["order"], value["probabilities"])


@dataclass(frozen=True)
class Emission:
    """One latent emission alternative; duplicate alternatives retain row identities."""

    next_state: str
    glyphs: str
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.next_state, str) or not self.next_state:
            raise ValueError("Emission destination must be a nonempty state name")
        if not isinstance(self.glyphs, str) or not self.glyphs:
            raise ValueError("Epsilon/empty glyph emissions are not supported")
        object.__setattr__(self, "probability", _probability(self.probability))

    def to_dict(self) -> dict[str, Any]:
        return {"next_state": self.next_state, "glyphs": self.glyphs, "probability": self.probability}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Emission:
        if set(value) != {"next_state", "glyphs", "probability"}:
            raise ValueError("Unexpected serialized emission fields")
        return cls(value["next_state"], value["glyphs"], value["probability"])


@dataclass(frozen=True)
class Channel:
    """A full normalized channel, including unobserved alternatives and stopping.

    Before every source draw, stop with probability rho. Otherwise draw a source
    letter, then one emission from rows[(state, letter)]. The final stop costs
    rho even for empty ciphertext. No emitted glyph can be omitted or trimmed.
    """

    states: tuple[str, ...]
    glyph_alphabet: tuple[str, ...]
    initial: Mapping[str, float]
    rows: Mapping[tuple[str, str], tuple[Emission, ...]]
    stop_probability: float
    max_emission_length: int = 4

    def __post_init__(self) -> None:
        states = _names(self.states, characters=False, name="Channel states")
        glyphs = _names(self.glyph_alphabet, characters=True, name="Glyph alphabet")
        rho = _probability(self.stop_probability)
        if not 0.0 < rho < 1.0:
            raise ValueError("Stop probability must be strictly between zero and one")
        if type(self.max_emission_length) is not int or self.max_emission_length < 1:
            raise ValueError("Maximum emission length must be a positive integer")
        if not self.rows:
            raise ValueError("Channel must define source-letter rows")
        for key in self.rows:
            if (not isinstance(key, tuple) or len(key) != 2 or key[0] not in states
                    or not isinstance(key[1], str) or len(key[1]) != 1):
                raise ValueError("Channel row keys must be (known state, source codepoint)")
        letters = {key[1] for key in self.rows}
        if set(self.rows) != {(state, letter) for state in states for letter in letters}:
            raise ValueError("Every state needs every source-letter row")
        rows = {}
        for key, supplied in self.rows.items():
            row = tuple(supplied)
            if not row or any(not isinstance(item, Emission) for item in row):
                raise ValueError("Each channel row needs nonempty Emission alternatives")
            for item in row:
                if item.next_state not in states:
                    raise ValueError("Unknown emission destination state")
                if len(item.glyphs) > self.max_emission_length or not set(item.glyphs) <= set(glyphs):
                    raise ValueError("Emission exceeds its length or glyph-alphabet support")
            if not math.isclose(math.fsum(item.probability for item in row), 1.0, rel_tol=0.0,
                                abs_tol=NORMALIZATION_TOLERANCE):
                raise ValueError("Every complete emission row must sum to one")
            rows[key] = row
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "glyph_alphabet", glyphs)
        object.__setattr__(self, "initial", _distribution(self.initial, states))
        object.__setattr__(self, "rows", MappingProxyType(rows))
        object.__setattr__(self, "stop_probability", rho)

    @property
    def source_alphabet(self) -> frozenset[str]:
        return frozenset(letter for _, letter in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "states": list(self.states), "glyph_alphabet": list(self.glyph_alphabet),
                "initial": dict(self.initial), "stop_probability": self.stop_probability,
                "max_emission_length": self.max_emission_length,
                "rows": [{"state": state, "letter": letter,
                          "emissions": [item.to_dict() for item in self.rows[(state, letter)]]}
                         for state, letter in sorted(self.rows)]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Channel:
        _check_fields(value, {"schema_version", "states", "glyph_alphabet", "initial", "rows",
                              "stop_probability", "max_emission_length"})
        rows = {}
        for row in value["rows"]:
            if set(row) != {"state", "letter", "emissions"}:
                raise ValueError("Unexpected serialized channel-row fields")
            key = row["state"], row["letter"]
            if key in rows:
                raise ValueError("Duplicate serialized channel row")
            rows[key] = tuple(Emission.from_dict(item) for item in row["emissions"])
        return cls(tuple(value["states"]), tuple(value["glyph_alphabet"]), value["initial"], rows,
                   value["stop_probability"], value["max_emission_length"])


@dataclass(frozen=True)
class ViterbiResult:
    """Best JOINT latent path; plaintext None denotes an impossible observation."""

    log_probability: float
    plaintext: str | None
    states: tuple[str, ...]
    glyph_chunks: tuple[str, ...]
    emission_indices: tuple[int, ...]


@dataclass(frozen=True)
class PosteriorResult:
    """Conditional path counts; impossible observations have no posterior.

    For a possible observation every declared emission alternative has an entry,
    including zero-probability/unobserved alternatives with expected count zero.
    For impossible observations counts are empty and expected length is None.
    """

    log_likelihood: float
    expected_emissions: dict[EmissionKey, float]
    expected_source_length: float | None


def _validate_pair(source: SourceModel, channel: Channel, ciphertext: str) -> bool:
    if set(source.alphabet) != channel.source_alphabet:
        raise ValueError("Source and channel alphabets must match exactly")
    if not isinstance(ciphertext, str):
        raise TypeError("Ciphertext must be a Unicode string")
    return set(ciphertext) <= set(channel.glyph_alphabet)


def _logadd(first: float, second: float) -> float:
    if first == NEGATIVE_INFINITY:
        return second
    if second == NEGATIVE_INFINITY:
        return first
    high, low = max(first, second), min(first, second)
    return high + math.log1p(math.exp(low - high))


def _logsum(values) -> float:
    answer = NEGATIVE_INFINITY
    for value in values:
        answer = _logadd(answer, value)
    return answer


def _transitions(source: SourceModel, channel: Channel, ciphertext: str, offset: int, node: Node):
    context, state = node
    non_stop = math.log1p(-channel.stop_probability)
    for letter in source.alphabet:
        probability = source.probabilities[context][letter]
        if probability == 0.0:
            continue
        source_log = math.log(probability)
        for index, emission in enumerate(channel.rows[(state, letter)]):
            if emission.probability == 0.0 or not ciphertext.startswith(emission.glyphs, offset):
                continue
            end = offset + len(emission.glyphs)
            next_node = (letter if source.order == 1 else "", emission.next_state)
            step = non_stop + source_log + math.log(emission.probability)
            yield end, next_node, letter, index, emission.glyphs, step


def _forward(source: SourceModel, channel: Channel, ciphertext: str) -> tuple[list[dict[Node, float]], float]:
    alpha: list[dict[Node, float]] = [{} for _ in range(len(ciphertext) + 1)]
    for state in channel.states:
        if channel.initial[state] > 0.0:
            alpha[0][("", state)] = math.log(channel.initial[state])
    for offset in range(len(ciphertext)):
        for node, value in alpha[offset].items():
            for end, next_node, _, _, _, step in _transitions(source, channel, ciphertext, offset, node):
                previous = alpha[end].get(next_node, NEGATIVE_INFINITY)
                alpha[end][next_node] = _logadd(previous, value + step)
    total = _logsum(alpha[-1].values()) + math.log(channel.stop_probability)
    return alpha, total


def forward_log_probability(source: SourceModel, channel: Channel, ciphertext: str) -> float:
    """Log marginal p(ciphertext), summing all legal latent paths without pruning."""
    if not _validate_pair(source, channel, ciphertext):
        return NEGATIVE_INFINITY
    return _forward(source, channel, ciphertext)[1]


def viterbi_decode(source: SourceModel, channel: Channel, ciphertext: str) -> ViterbiResult:
    """Return the highest-probability joint path, not a MAP-plaintext guarantee.

    Exact score ties retain the first traversed path; model ordering can affect
    that representative. Likelihood and best-path score are order invariant.
    """
    impossible = ViterbiResult(NEGATIVE_INFINITY, None, (), (), ())
    if not _validate_pair(source, channel, ciphertext):
        return impossible
    best: list[dict[Node, float]] = [{} for _ in range(len(ciphertext) + 1)]
    back: dict[tuple[int, Node], tuple[int, Node, str, int, str]] = {}
    for state in channel.states:
        if channel.initial[state] > 0.0:
            best[0][("", state)] = math.log(channel.initial[state])
    for offset in range(len(ciphertext)):
        for node, value in best[offset].items():
            for end, next_node, letter, index, glyphs, step in _transitions(
                    source, channel, ciphertext, offset, node):
                candidate = value + step
                if candidate > best[end].get(next_node, NEGATIVE_INFINITY):
                    best[end][next_node] = candidate
                    back[(end, next_node)] = offset, node, letter, index, glyphs
    if not best[-1]:
        return impossible
    node = max(best[-1], key=best[-1].__getitem__)
    score = best[-1][node] + math.log(channel.stop_probability)
    offset = len(ciphertext)
    states, letters, chunks, indices = [node[1]], [], [], []
    while offset:
        offset, node, letter, index, glyphs = back[(offset, node)]
        states.append(node[1])
        letters.append(letter)
        indices.append(index)
        chunks.append(glyphs)
    return ViterbiResult(score, "".join(reversed(letters)), tuple(reversed(states)),
                         tuple(reversed(chunks)), tuple(reversed(indices)))


def posterior_expected_counts(source: SourceModel, channel: Channel, ciphertext: str) -> PosteriorResult:
    """Expected per-row-alternative use given ciphertext, via exact forward/backward.

    Counts marginalize plaintext, segmentation, initial state, and channel paths.
    Their sum is the expected source length; it need not equal a Viterbi length.
    """
    if not _validate_pair(source, channel, ciphertext):
        return PosteriorResult(NEGATIVE_INFINITY, {}, None)
    alpha, total = _forward(source, channel, ciphertext)
    if total == NEGATIVE_INFINITY:
        return PosteriorResult(total, {}, None)
    beta: list[dict[Node, float]] = [{} for _ in alpha]
    beta[-1] = {node: math.log(channel.stop_probability) for node in alpha[-1]}
    for offset in range(len(ciphertext) - 1, -1, -1):
        for node in alpha[offset]:
            continuation = NEGATIVE_INFINITY
            for end, next_node, _, _, _, step in _transitions(source, channel, ciphertext, offset, node):
                continuation = _logadd(continuation, step + beta[end].get(next_node, NEGATIVE_INFINITY))
            beta[offset][node] = continuation
    counts = {(state, letter, index): 0.0 for (state, letter), row in channel.rows.items()
              for index in range(len(row))}
    for offset in range(len(ciphertext)):
        for node, prefix in alpha[offset].items():
            for end, next_node, letter, index, _, step in _transitions(source, channel, ciphertext, offset, node):
                suffix = beta[end].get(next_node, NEGATIVE_INFINITY)
                if suffix != NEGATIVE_INFINITY:
                    counts[(node[1], letter, index)] += math.exp(prefix + step + suffix - total)
    return PosteriorResult(total, counts, math.fsum(counts.values()))
