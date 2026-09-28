"""Bounded complete-neighborhood search of deterministic variable-length units.

One state, one nonempty unit per source letter, and duplicate units are allowed.
The literal unit pool is complete within the declared alphabet/length bound.
Exact marginals and literal model-code costs do not make local search global.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from voynich.batched_unit_channel import batch_log_likelihood
from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import CodingContext, channel_code_bits, two_part_score


@dataclass(frozen=True)
class UnitSearchConfig:
    seed: int = 0
    restarts: int = 16
    max_sweeps: int = 80
    max_seconds: float = 300.
    batch_size: int = 256
    max_units: int = 256
    improvement_tolerance_bits: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("seed", "restarts", "max_sweeps", "batch_size", "max_units"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.restarts < 1 or self.max_sweeps < 0 or self.batch_size < 1 or self.max_units < 1:
            raise ValueError("Need positive restarts/batch size/unit cap and nonnegative sweeps")
        if isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Need a finite positive time budget")
        if (isinstance(self.improvement_tolerance_bits, bool)
                or not math.isfinite(self.improvement_tolerance_bits) or self.improvement_tolerance_bits < 0):
            raise ValueError("Need a finite nonnegative score tolerance")


def literal_unit_pool(context: CodingContext, max_units: int = 256) -> tuple[str, ...]:
    """Complete length-then-declared-glyph-order pool; check capacity first."""
    if type(max_units) is not int or max_units < 1:
        raise ValueError("Unit-pool capacity must be a positive integer")
    total, count = 0, 1
    for _ in range(context.max_emission_length):
        count *= len(context.glyph_alphabet)
        total += count
        if total > max_units:
            raise ValueError(f"Complete literal unit pool exceeds max_units={max_units}")
    return tuple("".join(unit) for length in range(1, context.max_emission_length + 1)
                 for unit in itertools.product(context.glyph_alphabet, repeat=length))


def channel_from_units(source: SourceModel, context: CodingContext, units: Sequence[str]) -> Channel:
    """Engine-compatible deterministic channel, with distinct source contexts."""
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses a different ordered coding alphabet")
    if isinstance(units, (str, bytes)) or len(units) != len(source.alphabet):
        raise ValueError("Need one unit for each source letter in declared order")
    return Channel(("s0",), context.glyph_alphabet, {"s0": 1.},
                   {("s0", letter): (Emission("s0", unit, 1.),)
                    for letter, unit in zip(source.alphabet, units, strict=True)},
                   context.stop_probability, context.max_emission_length)


def unit_neighbors(units: Sequence[str], pool: Sequence[str]) -> Iterator[tuple[dict, tuple[str, ...]]]:
    """Every nonidentity swap, then every row replacement, ordered and unique.

    Different unequal-row swaps change distinct pairs of positions; a row
    replacement changes one position. Thus these two lists cannot overlap and
    do not need a memory-growing deduplication set. Duplicate pool entries are
    removed in their first-occurrence order for callers of this public helper.
    """
    units, pool = tuple(units), tuple(dict.fromkeys(pool))
    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            if units[left] != units[right]:
                changed = list(units)
                changed[left], changed[right] = changed[right], changed[left]
                yield {"kind": "swap", "rows": [left, right]}, tuple(changed)
    for row in range(len(units)):
        for unit in pool:
            if unit != units[row]:
                changed = list(units)
                changed[row] = unit
                yield {"kind": "replace", "row": row, "unit": unit}, tuple(changed)


def _initial_units(source: SourceModel, records: tuple[str, ...], context: CodingContext,
                   pool: tuple[str, ...], restart: int, rng: random.Random) -> tuple[tuple[str, ...], str]:
    rows, glyphs = list(range(len(source.alphabet))), list(context.glyph_alphabet)
    if restart == 0:
        counts = Counter(glyph for record in records for glyph in record)
        rows.sort(key=lambda row: -source.probabilities[""][source.alphabet[row]])
        glyphs.sort(key=lambda glyph: -counts[glyph])
        # All rows receive singleton units; ranked remaining rows cycle through
        # the ranked glyphs. This is only an initialization heuristic.
        units = [""] * len(rows)
        for rank, row in enumerate(rows):
            units[row] = glyphs[rank % len(glyphs)]
        return tuple(units), "source_start_and_glyph_frequency_singletons"
    rng.shuffle(rows)
    rng.shuffle(glyphs)
    units = [""] * len(rows)
    for row, glyph in zip(rows, glyphs):
        units[row] = glyph
    for row in rows[len(glyphs):]:
        units[row] = rng.choice(pool)
    return tuple(units), "random_singleton_coverage_plus_uniform_pool"


@dataclass(frozen=True)
class UnitSearchResult:
    channel: Channel | None
    score: dict | None
    units: tuple[str, ...] | None
    trace: tuple[dict, ...]
    stop_reason: str
    evaluated_neighbors: int
    completed_sweeps: int
    local_optima: int
    best_is_certified_local_optimum: bool
    seconds: float
    config: UnitSearchConfig
    source_index: int
    unit_pool: tuple[str, ...]
    kernel_fallbacks: int
    scored_initializations: int

    def to_dict(self) -> dict:
        return {"schema_version": 1, "candidate_family": "one_state_deterministic_literal_units",
                "channel": None if self.channel is None else self.channel.to_dict(), "score": self.score,
                "units": None if self.units is None else list(self.units), "trace": list(self.trace),
                "stop_reason": self.stop_reason, "evaluated_neighbors": self.evaluated_neighbors,
                "completed_sweeps": self.completed_sweeps, "local_optima": self.local_optima,
                "best_is_certified_local_optimum": self.best_is_certified_local_optimum,
                "seconds": self.seconds, "config": asdict(self.config), "source_index": self.source_index,
                "unit_pool": list(self.unit_pool), "kernel_fallbacks": self.kernel_fallbacks,
                "scored_initializations": self.scored_initializations}


def search_unit_channel(source: SourceModel, records: Sequence[str], context: CodingContext, *,
                        config: UnitSearchConfig = UnitSearchConfig(), source_index: int = 0) -> UnitSearchResult:
    """Seeded steepest descent in exact conditional model-plus-data bits.

    Complete sweeps test every unequal-row pair swap and every single-row
    replacement from the complete literal pool. Locality is within this family
    and these moves, to the declared tolerance; it is not global optimality.
    Initializations preserve singleton glyph coverage, but proposals need not.
    A zero-probability initialization is traced and skipped.

    The deadline is cooperative between batches. Every completed batch is
    eligible for best-candidate retention even if it finishes after the
    deadline. A partial sweep cannot certify its parent. Preprocessing, kernel
    calls, trace creation and the mandatory final original-engine recheck can
    overrun the budget. No finished supported candidate yields a None result.
    """
    started = time.monotonic()
    deadline = started + config.max_seconds
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses a different ordered coding alphabet")
    if len(source.alphabet) < len(context.glyph_alphabet):
        raise ValueError("Singleton-coverage initialization requires source alphabet size >= glyph alphabet size")
    if type(source_index) is not int or not 0 <= source_index < context.source_count:
        raise ValueError("Source index is outside the shared coding context")
    if isinstance(records, (str, bytes)):
        raise ValueError("Supply separate ciphertext records")
    records = tuple(records)
    if not records or any(not isinstance(record, str) for record in records):
        raise ValueError("Need at least one ciphertext record, each a string")
    if any(not set(record) <= set(context.glyph_alphabet) for record in records):
        raise ValueError("Ciphertext contains glyphs outside the declared alphabet")
    pool = literal_unit_pool(context, config.max_units)
    size = len(source.alphabet)
    baseline_units = (context.glyph_alphabet[0],) * size
    baseline_bits = channel_code_bits(channel_from_units(source, context, baseline_units), context, source_index)
    glyph_width = (len(context.glyph_alphabet) - 1).bit_length()
    master = random.Random(config.seed)
    trace = []
    best_units, best_score = None, None
    neighbors = sweeps = optima = fallbacks = initializations = 0

    def evaluate(candidates: list[tuple[str, ...]]) -> tuple[list[dict | None], dict]:
        nonlocal fallbacks
        diagnostics = {}
        likelihoods = batch_log_likelihood(source, candidates, records, context.stop_probability,
                                           context.max_emission_length, diagnostics=diagnostics)
        fallbacks += diagnostics["log_domain_fallbacks"]
        scores = []
        for units, likelihood in zip(candidates, likelihoods, strict=True):
            if not math.isfinite(likelihood):
                if likelihood != -math.inf:
                    raise ArithmeticError("Kernel returned an invalid likelihood")
                scores.append(None)
                continue
            # All code fields except the literal glyph indices have identical
            # width in this deterministic one-state family, including the
            # length headers and singleton finite-grid weight compositions.
            bits = baseline_bits + glyph_width * (sum(map(len, units)) - size)
            data_bits = -float(likelihood) / math.log(2)
            scores.append({"model_bits": bits, "data_bits": data_bits, "total_bits": bits + data_bits,
                           "log_likelihood": float(likelihood)})
        return scores, diagnostics

    def consider(units: tuple[str, ...], score: dict | None) -> None:
        nonlocal best_units, best_score
        if score is not None and (best_score is None or score["total_bits"] < best_score["total_bits"]):
            best_units, best_score = units, score

    for restart in range(config.restarts):
        if time.monotonic() >= deadline:
            break
        restart_seed = master.getrandbits(64)
        units, method = _initial_units(source, records, context, pool, restart, random.Random(restart_seed))
        if time.monotonic() >= deadline:
            break
        scores, diagnostics = evaluate([units])
        current = scores[0]
        initializations += 1
        consider(units, current)
        trace.append({"event": "restart", "restart": restart, "seed": restart_seed, "method": method,
                      "units": list(units), "score": current, "status": "scored" if current else "unsupported",
                      "kernel_diagnostics": diagnostics})
        if current is None:
            continue
        for sweep in range(config.max_sweeps):
            if time.monotonic() >= deadline:
                break
            expected = size * (len(pool) - 1) + sum(units[i] != units[j]
                       for i in range(size) for j in range(i + 1, size))
            iterator = unit_neighbors(units, pool)
            scored, batches = [], []
            selected_units, selected_score, selected_move = units, current, None
            while len(scored) < expected:
                if time.monotonic() >= deadline:
                    break
                batch = list(itertools.islice(iterator, config.batch_size))
                if not batch:
                    raise ArithmeticError("Neighbor enumeration disagrees with its exact cardinality")
                if time.monotonic() >= deadline:
                    break
                scores, diagnostics = evaluate([candidate for _, candidate in batch])
                batches.append({"first_neighbor": len(scored), "count": len(batch),
                                "kernel_diagnostics": diagnostics})
                for (move, candidate), score in zip(batch, scores, strict=True):
                    consider(candidate, score)
                    if score is not None and score["total_bits"] < selected_score["total_bits"]:
                        selected_units, selected_score, selected_move = candidate, score, move
                    scored.append({"move": move, "units": list(candidate), "score": score,
                                   "status": "scored" if score else "unsupported"})
                neighbors += len(batch)
            complete = len(scored) == expected
            sweeps += int(complete)
            accepted = current["total_bits"] - selected_score["total_bits"] > config.improvement_tolerance_bits
            optima += int(complete and not accepted)
            reason = ("time_limit" if not complete else ("improved" if accepted else "local_optimum"))
            trace.append({"event": "sweep", "restart": restart, "sweep": sweep,
                          "parent_units": list(units), "parent_score": current, "neighbors": scored,
                          "batches": batches, "expected_neighbors": expected, "evaluated_neighbors": len(scored),
                          "complete": complete, "accepted": accepted, "selected_units": list(selected_units),
                          "selected_score": selected_score, "selected_move": selected_move, "stop_reason": reason})
            if accepted:
                units, current = selected_units, selected_score
            if not complete or not accepted:
                break
    reason = ("time_limit" if time.monotonic() >= deadline else
              ("budget_exhausted" if best_units is not None else "no_supported_initialization"))
    channel = None if best_units is None else channel_from_units(source, context, best_units)
    score = None
    if channel is not None:
        score = two_part_score(source, channel, records, context, source_index)
        slack = 1e-10 * max(1., abs(best_score["log_likelihood"]))
        if (abs(score["log_likelihood"] - best_score["log_likelihood"]) > slack
                or score["model_bits"] != best_score["model_bits"]):
            raise ArithmeticError("Batched/derived and original-engine/encoded scores disagree")
    certified = best_units is not None and any(event["event"] == "sweep" and event["complete"]
                    and not event["accepted"] and tuple(event["parent_units"]) == best_units for event in trace)
    return UnitSearchResult(channel, score, best_units, tuple(trace), reason, neighbors, sweeps, optima,
                            certified, time.monotonic() - started, config, source_index, pool,
                            fallbacks, initializations)
