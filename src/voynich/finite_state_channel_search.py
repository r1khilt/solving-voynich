"""Bounded ciphertext-only local search for literal finite-state channels.

The supplied source model and CodingContext are conditioned-on assistance.
Search is seeded greedy optimization with restarts, not posterior sampling or
exhaustive optimization. Every retained candidate pays its actual binary model
code and sums every latent path at its actual, quantized probabilities.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from voynich.finite_state_channel import Channel, Emission, SourceModel, forward_log_probability
from voynich.finite_state_channel_fit import (
    CodingContext, channel_code_bits, fit_em, quantize_channel,
)


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 0
    state_counts: tuple[int, ...] = (1, 2)
    restarts_per_state: int = 2
    proposals_per_restart: int = 50
    em_iterations: int = 2
    max_seconds: float = 60.
    initialization_attempts: int = 8
    max_units: int = 256
    frequency_initialization: bool = True
    em_tolerance: float = 1e-8
    improvement_tolerance_bits: float = 1e-9

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        if type(self.frequency_initialization) is not bool:
            raise ValueError("frequency_initialization must be boolean")
        states = tuple(self.state_counts)
        if (not states or len(set(states)) != len(states)
                or any(type(count) is not int or count < 1 for count in states)):
            raise ValueError("State counts must be distinct positive integers")
        object.__setattr__(self, "state_counts", states)
        for name in ("restarts_per_state", "initialization_attempts", "max_units"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("proposals_per_restart", "em_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Need a finite positive time budget")
        for name in ("em_tolerance", "improvement_tolerance_bits"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class SearchResult:
    channel: Channel | None
    score: dict | None
    trace: tuple[dict, ...]
    stop_reason: str
    proposals: int
    completed_candidates: int
    seconds: float
    config: SearchConfig
    source_index: int
    unit_pool: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-compatible audit record, including every proposed/scored model."""
        return {"schema_version": 1, "channel": None if self.channel is None else self.channel.to_dict(),
                "score": self.score, "trace": list(self.trace), "stop_reason": self.stop_reason,
                "proposals": self.proposals, "completed_candidates": self.completed_candidates,
                "seconds": self.seconds, "config": asdict(self.config),
                "source_index": self.source_index, "unit_pool": list(self.unit_pool)}


def observed_units(records: Sequence[str], context: CodingContext, max_units: int = 256) -> tuple[str, ...]:
    """Declared singleton coverage plus visible substrings; never cross records.

    All glyphs in the shared public alphabet are retained, even when absent in
    fitting ciphertext. Remaining slots take overlapping visible substrings
    substrings by descending count, shorter length, then declared glyph order.
    This pool is a search restriction, not a free dictionary in the model code.
    """
    records = _records(records)
    if type(max_units) is not int or max_units < 1:
        raise ValueError("max_units must be a positive integer")
    visible = set("".join(records))
    if not visible <= set(context.glyph_alphabet):
        raise ValueError("Ciphertext contains glyphs outside the shared coding context")
    singletons = context.glyph_alphabet
    if len(singletons) > max_units:
        raise ValueError("max_units cannot discard declared singleton glyphs")
    frequencies = Counter(record[start:end] for record in records for start in range(len(record))
                          for end in range(start + 2, min(len(record), start + context.max_emission_length) + 1))
    glyph_index = {glyph: index for index, glyph in enumerate(context.glyph_alphabet)}
    ordered = sorted(frequencies, key=lambda unit: (-frequencies[unit], len(unit),
                                                   tuple(glyph_index[glyph] for glyph in unit)))
    return (*singletons, *ordered[:max_units - len(singletons)])


def _records(records: Sequence[str]) -> tuple[str, ...]:
    if isinstance(records, (str, bytes)):
        raise ValueError("Supply a sequence of separate ciphertext records")
    result = tuple(records)
    if not result or any(not isinstance(record, str) for record in result):
        raise ValueError("Need at least one ciphertext record, each a string")
    return result


def _source_frequencies(source: SourceModel, stop_probability: float) -> dict[str, float]:
    """Source-only ranking heuristic: 256 survival-weighted letter marginals.

    This truncation only seeds a search; it is never used as a data likelihood.
    The source is fixed, and no target plaintext or empirical source corpus is
    read. Source order zero needs no approximation.
    """
    if source.order == 0:
        return dict(source.probabilities[""])
    marginal = dict(source.probabilities[""])
    counts = {letter: 0. for letter in source.alphabet}
    survival = 1.
    for _ in range(256):
        for letter in source.alphabet:
            counts[letter] += survival * marginal[letter]
        marginal = {letter: math.fsum(marginal[previous] * source.probabilities[previous][letter]
                                     for previous in source.alphabet) for letter in source.alphabet}
        survival *= 1 - stop_probability
    total = math.fsum(counts.values())
    return {letter: value / total for letter, value in counts.items()}


def _initial_channel(source: SourceModel, context: CodingContext, singletons: tuple[str, ...],
                     state_count: int, rng: random.Random, source_frequencies: dict | None = None,
                     glyph_frequencies: Counter | None = None) -> Channel:
    """Cover every declared singleton at every state with self-loop emissions.

    This guarantees parse support for a strictly positive source. Sources with
    zero transitions are accepted only after exact record scoring succeeds.
    """
    states = tuple(f"s{index}" for index in range(state_count))
    rows = {}
    for state in states:
        letters, glyphs = list(source.alphabet), list(singletons)
        if source_frequencies is None:
            rng.shuffle(letters)
            rng.shuffle(glyphs)
        else:
            letters.sort(key=lambda letter: -source_frequencies[letter])
            glyphs.sort(key=lambda glyph: -glyph_frequencies[glyph])
        assigned = {letter: [] for letter in letters}
        for index, glyph in enumerate(glyphs):
            assigned[letters[index % len(letters)]].append(glyph)
        for letter in letters:
            units = assigned[letter] or [rng.choice(singletons)]
            rows[(state, letter)] = tuple(Emission(state, unit, 1 / len(units)) for unit in units)
    channel = Channel(states, context.glyph_alphabet, {state: 1 / state_count for state in states}, rows,
                      context.stop_probability, context.max_emission_length)
    return quantize_channel(channel, context.denominator)


MOVES = ("swap_rows", "swap_letters", "replace_unit", "add_emission", "remove_emission",
         "retarget", "row_weights", "initial_weights")


def _transfer(counts: list[int], floor: int, rng: random.Random) -> tuple[list[int], dict] | None:
    donors = [index for index, count in enumerate(counts) if count > floor]
    if not donors or len(counts) < 2:
        return None
    donor = rng.choice(donors)
    recipient = rng.choice([index for index in range(len(counts)) if index != donor])
    amount = rng.randint(1, counts[donor] - floor)
    updated = list(counts)
    updated[donor] -= amount
    updated[recipient] += amount
    return updated, {"donor": donor, "recipient": recipient, "count": amount}


def _propose(channel: Channel, context: CodingContext, units: tuple[str, ...], rng: random.Random
             ) -> tuple[Channel | None, dict]:
    """One seeded proposal, including explicitly reported inapplicable moves."""
    kind = rng.choice(MOVES)
    move = {"kind": kind}
    rows = dict(channel.rows)
    initial = dict(channel.initial)
    state = rng.choice(channel.states)
    letter = rng.choice(context.source_alphabet)
    key = state, letter
    row = list(rows[key])
    if kind in ("swap_rows", "swap_letters"):
        if len(context.source_alphabet) < 2:
            return None, move | {"reason": "fewer_than_two_source_letters"}
        other = rng.choice([value for value in context.source_alphabet if value != letter])
        changed_states = channel.states if kind == "swap_letters" else (state,)
        for changed in changed_states:
            rows[(changed, letter)], rows[(changed, other)] = rows[(changed, other)], rows[(changed, letter)]
        move.update(states=list(changed_states), letters=[letter, other])
    elif kind == "initial_weights":
        transfer = _transfer([round(initial[value] * context.denominator) for value in channel.states], 0, rng)
        if transfer is None:
            return None, move | {"reason": "no_initial_weight_transfer"}
        counts, details = transfer
        initial = {value: count / context.denominator for value, count in zip(channel.states, counts, strict=True)}
        move.update(details)
    else:
        move.update(state=state, letter=letter)
        index = rng.randrange(len(row))
        if kind == "replace_unit":
            options = [unit for unit in units if unit != row[index].glyphs]
            if not options:
                return None, move | {"reason": "no_different_unit"}
            unit = rng.choice(options)
            move.update(index=index, old_unit=row[index].glyphs, unit=unit)
            row[index] = Emission(row[index].next_state, unit, row[index].probability)
        elif kind == "add_emission":
            if len(row) >= min(context.denominator, context.max_alternatives):
                return None, move | {"reason": "alternative_limit"}
            support = {(item.next_state, item.glyphs) for item in row}
            choices = [(destination, unit) for destination in channel.states for unit in units
                       if (destination, unit) not in support]
            if not choices:
                return None, move | {"reason": "support_exhausted"}
            destination, unit = rng.choice(choices)
            weight = 1 / (len(row) + 1)
            row = [Emission(item.next_state, item.glyphs, item.probability * (1 - weight)) for item in row]
            row.append(Emission(destination, unit, weight))
            move.update(destination=destination, unit=unit, initial_weight=weight)
        elif kind == "remove_emission":
            if len(row) == 1:
                return None, move | {"reason": "cannot_empty_row"}
            removed = row.pop(index)
            row = [Emission(item.next_state, item.glyphs, item.probability / (1 - removed.probability))
                   for item in row]
            move.update(index=index, unit=removed.glyphs, destination=removed.next_state)
        elif kind == "retarget":
            destinations = [value for value in channel.states if value != row[index].next_state]
            if not destinations:
                return None, move | {"reason": "single_state"}
            destination = rng.choice(destinations)
            move.update(index=index, old_destination=row[index].next_state, destination=destination)
            row[index] = Emission(destination, row[index].glyphs, row[index].probability)
        elif kind == "row_weights":
            transfer = _transfer([round(item.probability * context.denominator) for item in row], 1, rng)
            if transfer is None:
                return None, move | {"reason": "no_row_weight_transfer"}
            counts, details = transfer
            row = [Emission(item.next_state, item.glyphs, count / context.denominator)
                   for item, count in zip(row, counts, strict=True)]
            move.update(details)
        rows[key] = tuple(row)
    candidate = Channel(channel.states, channel.glyph_alphabet, initial, rows,
                        channel.stop_probability, channel.max_emission_length)
    candidate = quantize_channel(candidate, context.denominator)
    if candidate == channel:
        return None, move | {"reason": "unchanged_channel"}
    return candidate, move


def _score(source: SourceModel, channel: Channel, records: tuple[str, ...], context: CodingContext,
           source_index: int, deadline: float) -> dict:
    stage = {"channel": channel.to_dict(), "score": None}
    model_bits = channel_code_bits(channel, context, source_index)
    scores = []
    for record in records:
        if time.monotonic() >= deadline:
            return stage | {"status": "time_limit", "records_scored": len(scores)}
        value = forward_log_probability(source, channel, record)
        if not math.isfinite(value):
            return stage | {"status": "unsupported", "records_scored": len(scores) + 1}
        scores.append(value)
    log_likelihood = math.fsum(scores)
    data_bits = -log_likelihood / math.log(2)
    return stage | {"status": "scored", "records_scored": len(scores),
                    "score": {"model_bits": model_bits, "data_bits": data_bits,
                              "total_bits": model_bits + data_bits, "log_likelihood": log_likelihood}}


def _evaluate(source: SourceModel, channel: Channel, records: tuple[str, ...], context: CodingContext,
              config: SearchConfig, source_index: int, deadline: float) -> tuple[Channel | None, dict | None, dict]:
    raw = _score(source, channel, records, context, source_index, deadline) | {"phase": "raw"}
    detail = {"stages": [raw]}
    if raw["status"] != "scored":
        return None, None, detail
    best, score = channel, raw["score"]
    remaining = deadline - time.monotonic()
    # With one alternative per row EM cannot update any row parameter; its
    # initial/stop/source laws are fixed. Avoid repeating exact lattice work.
    has_free_row_weights = any(len(row) > 1 for row in channel.rows.values())
    if config.em_iterations and remaining > 0 and has_free_row_weights:
        fitted = fit_em(source, channel, records, max_iterations=config.em_iterations,
                        tolerance=config.em_tolerance, max_seconds=remaining)
        detail["em"] = {"iterations": fitted.iterations, "stop_reason": fitted.stop_reason,
                        "initial_log_likelihood": fitted.initial_log_likelihood,
                        "log_likelihood": fitted.log_likelihood, "trace": fitted.trace}
        quantized = quantize_channel(fitted.channel, context.denominator)
        if quantized != channel:
            refined = _score(source, quantized, records, context, source_index, deadline) | {"phase": "refined"}
            detail["stages"].append(refined)
            if refined["status"] == "scored" and refined["score"]["total_bits"] < score["total_bits"]:
                best, score = quantized, refined["score"]
    return best, score, detail


def search_channel(source: SourceModel, records: Sequence[str], context: CodingContext, *,
                   config: SearchConfig = SearchConfig(), source_index: int = 0) -> SearchResult:
    """Search literal supports/mappings using only supplied ciphertext and source.

    The global budget includes initialization, scoring and EM. Deadline checks
    occur between record scores; an atomic record score may overrun. fit_em's
    initial likelihood pass is also atomic to this wrapper and may overrun by
    one corpus pass. Interrupted candidate scores never replace a completed
    incumbent. With no completed supported initialization, channel/score are
    None and the trace reports the failure; there is no hidden fallback model.

    State count is fixed within a restart. Restart index is outermost, so both
    requested counts are attempted before a second round, budget permitting.
    The ordered source list and context must stay fixed across language runs.
    This function performs no filesystem access or empirical evaluation.
    """
    started = time.monotonic()
    deadline = started + config.max_seconds
    records = _records(records)
    if source.alphabet != context.source_alphabet:
        raise ValueError("Source model uses different ordered coding alphabet")
    if type(source_index) is not int or not 0 <= source_index < context.source_count:
        raise ValueError("Source index is outside the shared coding context")
    if max(config.state_counts) > context.max_states:
        raise ValueError("Requested state count exceeds the shared coding context")
    units = observed_units(records, context, config.max_units)
    singletons = tuple(unit for unit in units if len(unit) == 1)
    if len(singletons) > len(source.alphabet) * min(context.denominator, context.max_alternatives):
        raise ValueError("Singleton initialization capacity is smaller than the declared glyph inventory")
    frequencies = (_source_frequencies(source, context.stop_probability)
                   if config.frequency_initialization and 1 in config.state_counts else None)
    glyph_frequencies = Counter("".join(records))
    master = random.Random(config.seed)
    best_channel, best_score = None, None
    trace = []
    proposals = completed = 0
    for restart in range(config.restarts_per_state):
        for state_count in config.state_counts:
            if time.monotonic() >= deadline:
                break
            restart_seed = master.getrandbits(64)
            rng = random.Random(restart_seed)
            incumbent, incumbent_score = None, None
            for attempt in range(config.initialization_attempts):
                if time.monotonic() >= deadline:
                    break
                frequency_start = frequencies is not None and restart == 0 and state_count == 1 and attempt == 0
                initial = _initial_channel(source, context, singletons, state_count, rng,
                                           frequencies if frequency_start else None, glyph_frequencies)
                channel, score, detail = _evaluate(source, initial, records, context, config, source_index, deadline)
                event = {"event": "initialization", "restart": restart, "state_count": state_count,
                         "restart_seed": restart_seed, "attempt": attempt, **detail,
                         "method": "frequency" if frequency_start else "random",
                         "accepted": channel is not None}
                completed += sum(stage["status"] == "scored" for stage in detail["stages"])
                if channel is None:
                    event["rejection_reason"] = detail["stages"][0]["status"]
                else:
                    incumbent, incumbent_score = channel, score
                    if best_score is None or score["total_bits"] < best_score["total_bits"]:
                        best_channel, best_score = channel, score
                    event["selected_channel"] = channel.to_dict()
                    event["selected_score"] = score
                event["best_total_bits"] = None if best_score is None else best_score["total_bits"]
                trace.append(event)
                if incumbent is not None:
                    break
            if incumbent is None:
                continue
            for step in range(config.proposals_per_restart):
                if time.monotonic() >= deadline:
                    break
                proposals += 1
                proposed, move = _propose(incumbent, context, units, rng)
                event = {"event": "proposal", "restart": restart, "state_count": state_count,
                         "restart_seed": restart_seed, "step": step, "move": move,
                         "parent_score": incumbent_score, "stages": [], "accepted": False}
                if proposed is None:
                    event["rejection_reason"] = move["reason"]
                else:
                    channel, score, detail = _evaluate(source, proposed, records, context, config, source_index, deadline)
                    event.update(detail)
                    completed += sum(stage["status"] == "scored" for stage in detail["stages"])
                    if channel is None:
                        event["rejection_reason"] = detail["stages"][0]["status"]
                    else:
                        # Global best means every finished candidate, even a
                        # sub-tolerance improvement rejected by this restart.
                        if best_score is None or score["total_bits"] < best_score["total_bits"]:
                            best_channel, best_score = channel, score
                        event["selected_channel"], event["selected_score"] = channel.to_dict(), score
                        gain = incumbent_score["total_bits"] - score["total_bits"]
                        if gain > config.improvement_tolerance_bits:
                            incumbent, incumbent_score = channel, score
                            event["accepted"] = True
                        else:
                            event["rejection_reason"] = "no_sufficient_score_improvement"
                event["best_total_bits"] = best_score["total_bits"]
                trace.append(event)
    expired = time.monotonic() >= deadline
    reason = "time_limit" if expired else ("budget_exhausted" if best_channel is not None else "no_supported_initialization")
    return SearchResult(best_channel, best_score, tuple(trace), reason, proposals, completed,
                        time.monotonic() - started, config, source_index, units)
