"""Bounded dictionary refinement under a fixed higher-order source model.

One existing ciphertext-only dictionary is the disclosed warm start. This is
best-improving exact local search, not a posterior sampler or global optimizer.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence

from voynich.finite_state_channel_fit import CodingContext, channel_code_bits
from voynich.higher_order_unit_channel import MarkovSource, decode_units
from voynich.unit_channel_search import channel_from_units, literal_unit_pool, unit_neighbors


def refine_units(source: MarkovSource, records: Sequence[str], context: CodingContext,
                 initial: Sequence[str], *, max_sweeps: int = 20,
                 max_seconds: float = 300., tolerance: float = 1e-8) -> dict:
    """Scan every unequal-row swap and every one-row unit replacement per sweep.

    Keep the best completed candidate even if a deadline interrupts a sweep.
    A local certificate belongs only to a fully scanned, unimprovable returned
    parent. Numerical inference errors fail the run instead of dropping a case.
    """
    if (type(max_sweeps) is not int or max_sweeps < 0 or isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds) or max_seconds <= 0
            or isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Invalid bounded-search configuration")
    if tuple(context.source_alphabet) != source.alphabet:
        raise ValueError("Source and coding alphabet differ")
    records, initial = tuple(records), tuple(initial)
    pool = literal_unit_pool(context)
    if (not records or any(not isinstance(record, str) or set(record) - set(context.glyph_alphabet)
                           for record in records)
            or len(initial) != len(source.alphabet) or any(unit not in pool for unit in initial)):
        raise ValueError("Invalid records or warm-start dictionary")
    started = time.monotonic()
    deadline = started + max_seconds

    def score(units):
        values = [decode_units(source, units, record, context.stop_probability).log_likelihood
                  for record in records]
        if any(value == -math.inf for value in values):
            return None
        if not all(math.isfinite(value) for value in values):
            raise ArithmeticError("Nonfinite likelihood other than unsupported")
        likelihood = math.fsum(values)
        bits = channel_code_bits(channel_from_units(source, context, units), context)
        return {"log_likelihood": likelihood, "model_bits": bits,
                "data_bits": -likelihood / math.log(2), "total_bits": bits - likelihood / math.log(2)}

    current_units, current_score = initial, score(initial)
    if current_score is None:
        raise ValueError("Warm start does not support every record")
    best_units, best_score = current_units, current_score
    trace, evaluated, complete_sweeps, certificate = [], 0, 0, False
    stop_reason = "sweep_limit"
    for sweep in range(max_sweeps):
        if time.monotonic() >= deadline:
            stop_reason = "time_limit"
            break
        selected_units, selected_score = current_units, current_score
        expected = len(initial) * (len(pool) - 1) + sum(current_units[i] != current_units[j]
                    for i in range(len(initial)) for j in range(i + 1, len(initial)))
        neighbors = []
        for move, candidate in unit_neighbors(current_units, pool):
            if time.monotonic() >= deadline:
                break
            candidate_score = score(candidate)
            evaluated += 1
            neighbors.append({"move": move, "units": list(candidate), "score": candidate_score})
            if candidate_score is not None and candidate_score["total_bits"] < selected_score["total_bits"]:
                selected_units, selected_score = candidate, candidate_score
            if candidate_score is not None and candidate_score["total_bits"] < best_score["total_bits"]:
                best_units, best_score = candidate, candidate_score
        complete = len(neighbors) == expected
        complete_sweeps += int(complete)
        accepted = current_score["total_bits"] - selected_score["total_bits"] > tolerance
        trace.append({"sweep": sweep, "parent_units": list(current_units), "parent_score": current_score,
                      "neighbors": neighbors, "expected_neighbors": expected, "complete": complete,
                      "accepted": accepted, "selected_units": list(selected_units), "selected_score": selected_score})
        if not complete:
            stop_reason = "time_limit"
            break
        if not accepted:
            certificate = best_units == current_units
            stop_reason = "local_optimum"
            break
        current_units, current_score = selected_units, selected_score
    return {"units": list(best_units), "score": best_score, "initial_units": list(initial),
            "trace": trace, "evaluated_neighbors": evaluated, "completed_sweeps": complete_sweeps,
            "best_is_certified_local_optimum": certificate, "stop_reason": stop_reason,
            "seconds": time.monotonic() - started,
            "config": {"max_sweeps": max_sweeps, "max_seconds": max_seconds, "tolerance": tolerance}}
