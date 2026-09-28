"""Exact sufficient-statistic scoring and bounded search of bijective channels.

This deliberately narrow family has one state and a deterministic one-to-one
single-glyph map. A score and a complete pair-swap neighborhood are exact;
hill climbing with restarts is not a globally optimal general decipherer.
No ciphertext files, target text, or known keys are read by this module.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import CodingContext, channel_code_bits, two_part_score
from voynich.finite_state_channel_search import _source_frequencies


@dataclass(frozen=True)
class CiphertextStatistics:
    glyph_alphabet: tuple[str, ...]
    glyph_counts: tuple[int, ...]
    start_counts: tuple[int, ...]
    pair_counts: tuple[tuple[int, ...], ...]
    records: int
    glyphs: int


def ciphertext_statistics(records: Sequence[str], glyph_alphabet: Sequence[str]) -> CiphertextStatistics:
    """Count separate records, including empty records and no cross-record pairs."""
    records = _records(records)
    glyphs = tuple(glyph_alphabet)
    if (not glyphs or len(set(glyphs)) != len(glyphs)
            or any(not isinstance(glyph, str) or len(glyph) != 1 for glyph in glyphs)):
        raise ValueError("Need distinct single-codepoint glyph labels")
    index = {glyph: i for i, glyph in enumerate(glyphs)}
    counts = [0] * len(glyphs)
    starts = [0] * len(glyphs)
    pairs = [[0] * len(glyphs) for _ in glyphs]
    for record in records:
        if not set(record) <= set(glyphs):
            raise ValueError("Ciphertext contains glyphs outside the declared alphabet")
        if record:
            starts[index[record[0]]] += 1
        for glyph in record:
            counts[index[glyph]] += 1
        for left, right in zip(record, record[1:]):
            pairs[index[left]][index[right]] += 1
    return CiphertextStatistics(glyphs, tuple(counts), tuple(starts),
                                tuple(tuple(row) for row in pairs), len(records), sum(counts))


def _records(records: Sequence[str]) -> tuple[str, ...]:
    if isinstance(records, (str, bytes)):
        raise ValueError("Supply separate ciphertext records")
    records = tuple(records)
    if not records or any(not isinstance(record, str) for record in records):
        raise ValueError("Need at least one ciphertext record, each a string")
    return records


def _log(probability: float) -> float:
    return math.log(probability) if probability > 0 else -math.inf


def _tables(source: SourceModel) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    initial = tuple(_log(source.probabilities[""][letter]) for letter in source.alphabet)
    transitions = (() if source.order == 0 else
                   tuple(tuple(_log(source.probabilities[previous][letter]) for letter in source.alphabet)
                         for previous in source.alphabet))
    return initial, transitions


def _permutation(source: SourceModel, stats: CiphertextStatistics, mapping: Mapping[str, str]) -> tuple[int, ...]:
    if (len(source.alphabet) != len(stats.glyph_alphabet) or set(mapping) != set(stats.glyph_alphabet)
            or set(mapping.values()) != set(source.alphabet)):
        raise ValueError("Mapping must be a bijection between the complete declared alphabets")
    source_index = {letter: index for index, letter in enumerate(source.alphabet)}
    return tuple(source_index[mapping[glyph]] for glyph in stats.glyph_alphabet)


def _likelihood(stats: CiphertextStatistics, permutation: tuple[int, ...], initial: tuple[float, ...],
                transitions: tuple[tuple[float, ...], ...], rho: float) -> float:
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("Need a fixed proper stop probability")
    counts = stats.start_counts if transitions else stats.glyph_counts
    terms = [count * initial[permutation[index]] for index, count in enumerate(counts) if count]
    if transitions:
        terms.extend(count * transitions[permutation[left]][permutation[right]]
                     for left, row in enumerate(stats.pair_counts) for right, count in enumerate(row) if count)
    terms.extend((stats.glyphs * math.log1p(-rho), stats.records * math.log(rho)))
    return math.fsum(terms)


def mapping_log_likelihood(source: SourceModel, stats: CiphertextStatistics, mapping: Mapping[str, str],
                           stop_probability: float) -> float:
    """Full normalized marginal; bijectivity leaves exactly one latent path."""
    permutation = _permutation(source, stats, mapping)
    initial, transitions = _tables(source)
    return _likelihood(stats, permutation, initial, transitions, stop_probability)


def _delta(stats: CiphertextStatistics, permutation: tuple[int, ...], initial: tuple[float, ...],
           transitions: tuple[tuple[float, ...], ...], left: int, right: int) -> float:
    """O(K) difference, counting each affected directed pair exactly once."""
    if left == right:
        return 0.
    switched = {left: permutation[right], right: permutation[left]}
    counts = stats.start_counts if transitions else stats.glyph_counts
    old = [counts[index] * initial[permutation[index]] for index in (left, right) if counts[index]]
    new = [counts[index] * initial[switched[index]] for index in (left, right) if counts[index]]
    if transitions:
        # Entire affected rows, followed by affected columns in OTHER rows.
        # The four intersections and the two self-pairs are not double counted.
        pairs = [(i, j) for i in (left, right) for j in range(len(permutation))]
        pairs.extend((i, j) for i in range(len(permutation)) if i not in (left, right) for j in (left, right))
        for i, j in pairs:
            count = stats.pair_counts[i][j]
            if count:
                old.append(count * transitions[permutation[i]][permutation[j]])
                new.append(count * transitions[switched.get(i, permutation[i])][switched.get(j, permutation[j])])
    old_sum, new_sum = math.fsum(old), math.fsum(new)
    if old_sum == new_sum == -math.inf:
        raise ValueError("Swap delta is undefined when both affected likelihoods are zero")
    return new_sum - old_sum


def swap_log_likelihood_delta(source: SourceModel, stats: CiphertextStatistics, mapping: Mapping[str, str],
                              left_glyph: str, right_glyph: str) -> float:
    """Exact swap difference for supported incumbents; zero-probability new keys yield -inf.

    The stopping law cancels because record/character counts do not change.
    If both affected old and new factors have zero probability the numerical
    difference is undefined and this helper raises, rather than returning NaN.
    """
    permutation = _permutation(source, stats, mapping)
    initial, transitions = _tables(source)
    try:
        left, right = stats.glyph_alphabet.index(left_glyph), stats.glyph_alphabet.index(right_glyph)
    except ValueError as error:
        raise ValueError("Swap glyph is outside the declared alphabet") from error
    return _delta(stats, permutation, initial, transitions, left, right)


def channel_from_mapping(source: SourceModel, context: CodingContext, mapping: Mapping[str, str]) -> Channel:
    """Create an engine-compatible normalized one-state deterministic channel."""
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses a different ordered coding alphabet")
    empty = CiphertextStatistics(context.glyph_alphabet, (), (), (), 0, 0)
    _permutation(source, empty, mapping)
    inverse = {letter: glyph for glyph, letter in mapping.items()}
    return Channel(("s0",), context.glyph_alphabet, {"s0": 1.},
                   {("s0", letter): (Emission("s0", inverse[letter], 1.),) for letter in source.alphabet},
                   context.stop_probability, context.max_emission_length)


@dataclass(frozen=True)
class BijectiveSearchConfig:
    seed: int = 0
    restarts: int = 16
    max_sweeps: int = 100
    max_seconds: float = 60.
    frequency_initialization: bool = True
    improvement_tolerance_bits: float = 1e-10

    def __post_init__(self) -> None:
        if type(self.seed) is not int or type(self.restarts) is not int or self.restarts < 1:
            raise ValueError("Need integer seed and positive integer restart count")
        if type(self.max_sweeps) is not int or self.max_sweeps < 0:
            raise ValueError("Sweep limit must be a nonnegative integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Need a finite positive time budget")
        if type(self.frequency_initialization) is not bool:
            raise ValueError("frequency_initialization must be boolean")
        if not math.isfinite(self.improvement_tolerance_bits) or self.improvement_tolerance_bits < 0:
            raise ValueError("Need a finite nonnegative score tolerance")


@dataclass(frozen=True)
class BijectiveSearchResult:
    channel: Channel | None
    score: dict | None
    mapping: dict[str, str] | None
    statistics: CiphertextStatistics
    trace: tuple[dict, ...]
    stop_reason: str
    evaluated_neighbors: int
    completed_sweeps: int
    pair_local_optima: int
    best_is_certified_pair_local_optimum: bool
    seconds: float
    config: BijectiveSearchConfig
    source_index: int

    def to_dict(self) -> dict:
        return {"schema_version": 1, "candidate_family": "one_state_deterministic_bijection",
                "channel": None if self.channel is None else self.channel.to_dict(), "score": self.score,
                "mapping": self.mapping, "statistics": asdict(self.statistics), "trace": list(self.trace),
                "stop_reason": self.stop_reason, "evaluated_neighbors": self.evaluated_neighbors,
                "completed_sweeps": self.completed_sweeps, "pair_local_optima": self.pair_local_optima,
                "best_is_certified_pair_local_optimum": self.best_is_certified_pair_local_optimum,
                "seconds": self.seconds, "config": asdict(self.config), "source_index": self.source_index}


def search_bijective_channel(source: SourceModel, records: Sequence[str], context: CodingContext, *,
                             config: BijectiveSearchConfig = BijectiveSearchConfig(),
                             source_index: int = 0) -> BijectiveSearchResult:
    """Seeded full-neighborhood steepest ascent within the bijective subfamily.

    Each completed sweep evaluates every unordered pair exactly once. Only a
    completed sweep without sufficient gain establishes pair-local optimality
    to the declared tolerance. An interrupted sweep may accept its best finished
    neighbor but cannot claim that certificate. The best finished key from any
    restart is retained and checked once with the original forward engine.

    The time budget is cooperative: preprocessing, one O(K) swap evaluation,
    serialization, and the final original-engine corpus check are atomic. Zero-
    probability initial mappings are reported and skipped, not repaired using
    an oracle. No supported start yields channel/score/mapping None.
    """
    started = time.monotonic()
    deadline = started + config.max_seconds
    records = _records(records)
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses a different ordered coding alphabet")
    if len(source.alphabet) != len(context.glyph_alphabet):
        raise ValueError("Bijective search requires equal complete alphabet sizes")
    if type(source_index) is not int or not 0 <= source_index < context.source_count:
        raise ValueError("Source index is outside the shared coding context")
    stats = ciphertext_statistics(records, context.glyph_alphabet)
    initial, transitions = _tables(source)
    frequencies = _source_frequencies(source, context.stop_probability) if config.frequency_initialization else None
    master = random.Random(config.seed)
    best_key, best_likelihood = None, -math.inf
    trace = []
    neighbors = sweeps = optima = 0
    size = len(source.alphabet)
    pairs = tuple((left, right) for left in range(size) for right in range(left + 1, size))
    tolerance = config.improvement_tolerance_bits * math.log(2)

    def labels(permutation):
        return {glyph: source.alphabet[letter] for glyph, letter in zip(stats.glyph_alphabet, permutation, strict=True)}

    for restart in range(config.restarts):
        if time.monotonic() >= deadline:
            break
        restart_seed = master.getrandbits(64)
        rng = random.Random(restart_seed)
        key = list(range(size))
        if restart == 0 and frequencies is not None:
            glyph_rank = sorted(range(size), key=lambda index: -stats.glyph_counts[index])
            source_rank = sorted(range(size), key=lambda index: -frequencies[source.alphabet[index]])
            for glyph, letter in zip(glyph_rank, source_rank, strict=True):
                key[glyph] = letter
            method = "frequency"
        else:
            rng.shuffle(key)
            method = "random"
        key = tuple(key)
        current = _likelihood(stats, key, initial, transitions, context.stop_probability)
        supported = math.isfinite(current)
        trace.append({"event": "restart", "restart": restart, "seed": restart_seed, "method": method,
                      "mapping": labels(key), "log_likelihood": current if supported else None,
                      "status": "scored" if supported else "unsupported"})
        if not supported:
            continue
        if current > best_likelihood:
            best_key, best_likelihood = key, current
        for sweep in range(config.max_sweeps):
            if time.monotonic() >= deadline:
                break
            scored = []
            next_key, next_score, selected_pair = key, current, None
            for left, right in pairs:
                if time.monotonic() >= deadline:
                    break
                delta = _delta(stats, key, initial, transitions, left, right)
                candidate_score = current + delta
                # Recompute a prospective best from all counts to avoid drift
                # from accumulated delta arithmetic across many accepted swaps.
                candidate_key = None
                if candidate_score > next_score:
                    changed = list(key)
                    changed[left], changed[right] = changed[right], changed[left]
                    candidate_key = tuple(changed)
                    candidate_score = _likelihood(stats, candidate_key, initial, transitions, context.stop_probability)
                    if candidate_score > next_score:
                        next_key, next_score, selected_pair = candidate_key, candidate_score, (left, right)
                scored.append({"pair": [stats.glyph_alphabet[left], stats.glyph_alphabet[right]],
                               "log_likelihood": candidate_score if math.isfinite(candidate_score) else None})
                neighbors += 1
            complete = len(scored) == len(pairs)
            sweeps += int(complete)
            if next_score > best_likelihood:
                best_key, best_likelihood = next_key, next_score
            accept = next_score - current > tolerance
            reason = "improved" if accept else ("pair_local_optimum" if complete else "time_limit")
            optima += int(complete and not accept)
            trace.append({"event": "sweep", "restart": restart, "sweep": sweep,
                          "parent_mapping": labels(key), "parent_log_likelihood": current,
                          "neighbors": scored, "complete": complete, "accepted": accept,
                          "selected_pair": (None if selected_pair is None else
                                            [stats.glyph_alphabet[index] for index in selected_pair]),
                          "best_neighbor_log_likelihood": next_score, "stop_reason": reason})
            if accept:
                key, current = next_key, next_score
            if not complete or not accept:
                break
    expired = time.monotonic() >= deadline
    reason = "time_limit" if expired else ("budget_exhausted" if best_key is not None else "no_supported_initialization")
    mapping = None if best_key is None else labels(best_key)
    channel = None if mapping is None else channel_from_mapping(source, context, mapping)
    score = None
    if channel is not None:
        # Constant across the family, but charged using the actual encoder.
        model_bits = channel_code_bits(channel, context, source_index)
        score = two_part_score(source, channel, records, context, source_index)
        slack = 1e-10 * max(1., abs(best_likelihood))
        if abs(score["log_likelihood"] - best_likelihood) > slack or score["model_bits"] != model_bits:
            raise ArithmeticError("Sufficient-statistic and original-engine scores disagree")
    certified = mapping is not None and any(event["event"] == "sweep" and event["complete"]
                                            and not event["accepted"] and event["parent_mapping"] == mapping
                                            for event in trace)
    return BijectiveSearchResult(channel, score, mapping, stats, tuple(trace), reason, neighbors,
                                 sweeps, optima, certified, time.monotonic() - started, config, source_index)
