"""Fixed-support EM and explicit finite-grid codes for candidate channels.

EM learns emission probabilities, not support, states, source language or stop
law. The separate quantizer is not an EM step and need not improve likelihood.
The code length is conditional on a shared, declared CodingContext; it must not
be compared across uncharged changes to that context. None of these operations
establishes plaintext identifiability or a globally optimal channel.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from voynich.finite_state_channel import (
    Channel, Emission, SourceModel, forward_log_probability, posterior_expected_counts,
)


@dataclass(frozen=True)
class EMResult:
    channel: Channel
    initial_log_likelihood: float
    log_likelihood: float
    trace: list[dict]
    iterations: int
    stop_reason: str
    seconds: float


def _records(records: Sequence[str]) -> tuple[str, ...]:
    if isinstance(records, (str, bytes)):
        raise ValueError("Supply a sequence of separate ciphertext records")
    result = tuple(records)
    if not result or any(not isinstance(record, str) for record in result):
        raise ValueError("Need at least one ciphertext record, each a string")
    return result


def corpus_log_likelihood(source: SourceModel, channel: Channel, records: Sequence[str]) -> float:
    """Records are independent draws with fresh source/initial-state contexts."""
    return math.fsum(forward_log_probability(source, channel, record) for record in _records(records))


def fit_em(source: SourceModel, channel: Channel, records: Sequence[str], *,
           max_iterations: int = 50, tolerance: float = 1e-8,
           max_seconds: float = 60.) -> EMResult:
    """Exact E/M updates on a supplied support, with fixed source/initial/stop.

    Positive alternatives can become zero; zero probabilities cannot revive.
    Rows with zero posterior occupancy keep their old probabilities. No prior
    or smoothing is silently added. The likelihood, not decoded accuracy,
    controls stopping. Deadline checks occur between exact per-record calls;
    initial scoring and one such call can exceed the requested time limit.
    An interrupted update returns the last completely scored channel.
    """
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("Iterations must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Need finite nonnegative tolerance and positive time budget")
    records = _records(records)
    started = time.monotonic()
    deadline = started + max_seconds
    initial = score = corpus_log_likelihood(source, channel, records)
    if not math.isfinite(score):
        raise ValueError("Initial channel assigns zero probability to an observed record")
    trace = []
    reason = "iteration_limit"
    for iteration in range(1, max_iterations + 1):
        counts = {(*row, index): 0. for row, alternatives in channel.rows.items()
                  for index in range(len(alternatives))}
        for record in records:
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            posterior = posterior_expected_counts(source, channel, record)
            if not math.isfinite(posterior.log_likelihood):
                raise ArithmeticError("EM incumbent lost support for a training record")
            for event, count in posterior.expected_emissions.items():
                counts[event] += count
        if reason == "time_limit":
            break
        rows = {}
        for row, alternatives in channel.rows.items():
            total = math.fsum(counts[(*row, index)] for index in range(len(alternatives)))
            rows[row] = (alternatives if total == 0 else tuple(
                Emission(item.next_state, item.glyphs, counts[(*row, index)] / total)
                for index, item in enumerate(alternatives)))
        candidate = Channel(channel.states, channel.glyph_alphabet, channel.initial, rows,
                            channel.stop_probability, channel.max_emission_length)
        scores = []
        for record in records:
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            scores.append(forward_log_probability(source, candidate, record))
        if reason == "time_limit":
            break
        updated = math.fsum(scores)
        numerical_slack = 1e-10 * max(1., abs(score))
        if not math.isfinite(updated) or updated < score - numerical_slack:
            raise ArithmeticError("Exact EM likelihood decreased beyond numerical tolerance")
        gain = updated - score
        channel, score = candidate, updated
        trace.append({"iteration": iteration, "log_likelihood": score, "gain": gain})
        if gain <= tolerance:
            reason = "converged"
            break
    return EMResult(channel, initial, score, trace, len(trace), reason, time.monotonic() - started)


def _width(cardinality: int) -> int:
    if cardinality < 1:
        raise ValueError("A coding alphabet cannot be empty")
    return (cardinality - 1).bit_length()


def _field(value: int, cardinality: int) -> str:
    if type(value) is not int or not 0 <= value < cardinality:
        raise ValueError("Code value outside declared range")
    width = _width(cardinality)
    return format(value, f"0{width}b") if width else ""


def _composition_rank(counts: Sequence[int], total: int, positive: bool) -> tuple[int, int]:
    """Lexicographic enumerative rank among integer compositions of total."""
    floor = int(positive)
    if not counts or any(type(count) is not int or count < floor for count in counts) or sum(counts) != total:
        raise ValueError("Invalid weight composition")
    values = [count - floor for count in counts]
    remainder = total - floor * len(counts)
    cardinality = math.comb(remainder + len(counts) - 1, len(counts) - 1)
    rank = 0
    for index, count in enumerate(values[:-1]):
        remaining_slots = len(values) - index - 1
        for smaller in range(count):
            rank += math.comb(remainder - smaller + remaining_slots - 1, remaining_slots - 1)
        remainder -= count
    return rank, cardinality


def _round_counts(probabilities: Sequence[float], denominator: int, positive: bool) -> list[int]:
    floor = int(positive)
    if denominator < floor * len(probabilities):
        raise ValueError("Grid too coarse to retain every positive alternative")
    targets = [float(value) * denominator for value in probabilities]
    counts = [max(floor, math.floor(target)) for target in targets]
    while sum(counts) < denominator:
        index = max(range(len(counts)), key=lambda i: (targets[i] - counts[i], -i))
        counts[index] += 1
    while sum(counts) > denominator:
        eligible = [i for i, count in enumerate(counts) if count > floor]
        index = max(eligible, key=lambda i: (counts[i] - targets[i], -i))
        counts[index] -= 1
    return counts


def quantize_channel(channel: Channel, denominator: int) -> Channel:
    """Separate deterministic grid conversion; no monotonicity claim.

    Merge duplicate destination/string alternatives and discard exactly zero
    emissions. Round positive emission weights with count >=1; initial weights
    allow zero. Greedy adjustment of floor counts uses largest residual first
    with stable ties. This is not optimization of the data likelihood or MDL.
    """
    if type(denominator) is not int or denominator < 1:
        raise ValueError("Denominator must be a positive integer")
    initial_counts = _round_counts([channel.initial[state] for state in channel.states], denominator, False)
    initial = {state: count / denominator for state, count in zip(channel.states, initial_counts, strict=True)}
    rows = {}
    state_index = {state: i for i, state in enumerate(channel.states)}
    glyph_index = {glyph: i for i, glyph in enumerate(channel.glyph_alphabet)}
    for row, alternatives in channel.rows.items():
        pooled = {}
        for item in alternatives:
            key = item.next_state, item.glyphs
            pooled[key] = pooled.get(key, 0.) + item.probability
        support = sorted((key for key, value in pooled.items() if value > 0),
                         key=lambda key: (state_index[key[0]], tuple(glyph_index[g] for g in key[1])))
        counts = _round_counts([pooled[key] for key in support], denominator, True)
        rows[row] = tuple(Emission(state, glyphs, count / denominator)
                          for (state, glyphs), count in zip(support, counts, strict=True))
    return Channel(channel.states, channel.glyph_alphabet, initial, rows,
                   channel.stop_probability, channel.max_emission_length)


@dataclass(frozen=True)
class CodingContext:
    """Shared side information fixed BEFORE comparing model descriptions."""
    source_alphabet: tuple[str, ...]
    glyph_alphabet: tuple[str, ...]
    denominator: int
    max_states: int
    max_emission_length: int
    max_alternatives: int
    source_count: int = 1
    stop_probability: float = .1

    def __post_init__(self):
        for name in ("source_alphabet", "glyph_alphabet"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values) or any(not isinstance(v, str) or len(v) != 1 for v in values):
                raise ValueError("Coding alphabets must contain distinct Unicode codepoints")
            object.__setattr__(self, name, values)
        for name in ("denominator", "max_states", "max_emission_length", "max_alternatives", "source_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Coding bounds must be positive integers")
        if not math.isfinite(self.stop_probability) or not 0 < self.stop_probability < 1:
            raise ValueError("Coding context needs a fixed proper stop law")


def _grid_counts(probabilities: Sequence[float], denominator: int, positive: bool) -> list[int]:
    counts = [round(value * denominator) for value in probabilities]
    if (sum(counts) != denominator or any(count < int(positive) for count in counts)
            or any(abs(value * denominator - count) > 1e-9
                   for value, count in zip(probabilities, counts, strict=True))):
        raise ValueError("Probabilities are not on the declared grid; quantize explicitly")
    return counts


def encode_channel(channel: Channel, context: CodingContext, source_index: int = 0) -> str:
    """Construct an actual self-delimiting model description in binary.

    State and alphabet labels are indices in declared order, not free semantic
    labels. Headers specify state count, then each row's number of alternatives
    and each emission's length. Probability vectors use finite enumerative
    codes; no floating-point weights or uncharged literal emission strings.
    The context, including source models and stop law, is conditioned on.
    """
    if (channel.source_alphabet != set(context.source_alphabet)
            or channel.glyph_alphabet != context.glyph_alphabet
            or channel.stop_probability != context.stop_probability
            or len(channel.states) > context.max_states):
        raise ValueError("Channel does not match the fixed coding context")
    parts = [_field(source_index, context.source_count), _field(len(channel.states) - 1, context.max_states)]
    initial = _grid_counts([channel.initial[state] for state in channel.states], context.denominator, False)
    parts.append(_field(*_composition_rank(initial, context.denominator, False)))
    state_index = {state: i for i, state in enumerate(channel.states)}
    glyph_index = {glyph: i for i, glyph in enumerate(context.glyph_alphabet)}
    for state in channel.states:
        for letter in context.source_alphabet:
            row = channel.rows[(state, letter)]
            if len(row) > context.max_alternatives or len({(item.next_state, item.glyphs) for item in row}) != len(row):
                raise ValueError("Too many or duplicate emission alternatives")
            counts = _grid_counts([item.probability for item in row], context.denominator, True)
            parts.append(_field(len(row) - 1, context.max_alternatives))
            for item in row:
                parts.append(_field(state_index[item.next_state], len(channel.states)))
                parts.append(_field(len(item.glyphs) - 1, context.max_emission_length))
                parts.extend(_field(glyph_index[glyph], len(context.glyph_alphabet)) for glyph in item.glyphs)
            parts.append(_field(*_composition_rank(counts, context.denominator, True)))
    return "".join(parts)


def channel_code_bits(channel: Channel, context: CodingContext, source_index: int = 0) -> int:
    return len(encode_channel(channel, context, source_index))


def two_part_score(source: SourceModel, channel: Channel, records: Sequence[str],
                   context: CodingContext, source_index: int = 0) -> dict:
    """Conditional model bits plus exact observed-data bits, at grid weights."""
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses different ordered coding alphabet")
    bits = channel_code_bits(channel, context, source_index)
    log_likelihood = corpus_log_likelihood(source, channel, records)
    data_bits = -log_likelihood / math.log(2)
    return {"model_bits": bits, "data_bits": data_bits, "total_bits": bits + data_bits,
            "log_likelihood": log_likelihood}
