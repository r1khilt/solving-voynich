"""Locally adapted shared-key particles with deterministic record scheduling.

The source and iid key prior are separate from the proposal. No plaintext,
dictionary, segmentation or source length is supplied. Finite particle outputs
are approximate posterior samples, not exhaustive support or unbiased posterior
probabilities. The evidence recursion has the usual exact-arithmetic SMC law;
this implementation does not claim an interval-arithmetic certificate.
"""
from __future__ import annotations

import math

import numpy as np


def select_records(offsets, closed, lengths, schedule):
    """Exactly one open record per particle; ties select the lowest index."""
    count, records = offsets.shape
    chosen = np.full(count, -1, dtype=np.int16)
    for record in range(records):
        allowed = ~closed[:, record]
        if schedule == "sequential":
            change = allowed & (chosen < 0)
        else:
            previous = np.maximum(chosen, 0)
            prior_offset = offsets[np.arange(count), previous].astype(np.int64)
            change = allowed & ((chosen < 0) | (
                offsets[:, record].astype(np.int64)*lengths[previous]
                < prior_offset*int(lengths[record])))
        chosen[change] = record
    return chosen


def coefficients(keys, offsets, contexts, closed, observations, lengths,
                 probabilities, glyphs, rho, schedule):
    """Rows × two unit lengths, then EOS and absorption; first binding once."""
    count, rows = keys.shape
    records = select_records(offsets, closed, lengths, schedule)
    values = np.zeros((count, 2*rows+2), dtype=np.float64)
    absorbed = records < 0
    values[absorbed, 2*rows+1] = 1.
    active = np.flatnonzero(~absorbed)
    chosen = records[active]
    positions = offsets[active, chosen]
    ending = positions == lengths[chosen]
    values[active[ending], 2*rows] = rho
    active, chosen, positions = active[~ending], chosen[~ending], positions[~ending]
    singles = np.full(count, -1, dtype=np.int32)
    doubles = np.full(count, -1, dtype=np.int32)
    singles[active] = observations[chosen, positions]
    has_double = positions+1 < lengths[chosen]
    doubles[active[has_double]] = (glyphs
        + glyphs*observations[chosen[has_double], positions[has_double]]
        + observations[chosen[has_double], positions[has_double]+1])
    source = probabilities[contexts[active, chosen]]*(1-rho)
    unbound = keys[active] < 0
    penalty = 1/(glyphs+glyphs**2)
    values[active, :2*rows:2] = source*np.where(
        unbound, penalty, keys[active] == singles[active, None])
    values[active, 1:2*rows:2] = source*np.where(
        unbound, penalty, keys[active] == doubles[active, None])*(doubles[active, None] >= 0)
    return values, records, singles, doubles


def advance(keys, offsets, contexts, closed, records, singles, doubles,
            parents, actions, transitions):
    """Apply selected parent/action pairs, preserving each record's source state."""
    rows = keys.shape[1]
    keys, offsets = keys[parents].copy(), offsets[parents].copy()
    contexts, closed = contexts[parents].copy(), closed[parents].copy()
    selected = records[parents]
    letter = actions < 2*rows
    indices = np.flatnonzero(letter)
    row, length = actions[indices]//2, actions[indices]%2+1
    record = selected[indices]
    units = np.where(length == 1, singles[parents[indices]], doubles[parents[indices]])
    missing = keys[indices, row] < 0
    keys[indices[missing], row[missing]] = units[missing]
    offsets[indices, record] += length
    contexts[indices, record] = transitions[contexts[indices, record], row]
    ending = np.flatnonzero(actions == 2*rows)
    closed[ending, selected[ending]] = True
    labels = np.where(letter, actions//2, np.where(actions == 2*rows, -1, -2)).astype(np.int16)
    return keys, offsets, contexts, closed, selected, labels


def categorical(weights, uniforms):
    """Multinomial draws, including boundaries that must skip zero mass."""
    weights = np.asarray(weights, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if (weights.ndim not in (1, 2) or not weights.size or uniforms.ndim != 1
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or np.any(~np.isfinite(uniforms)) or np.any(uniforms < 0) or np.any(uniforms >= 1)
            or (weights.ndim == 2 and len(uniforms) != len(weights))):
        raise ValueError("Finite nonnegative weights and uniforms in [0,1) required")
    if weights.ndim == 1:
        total = float(weights.sum())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("Positive finite categorical mass required")
        cumulative = np.minimum(np.cumsum(weights), total)
        cumulative[-1] = total
        points = np.minimum(uniforms*total, np.nextafter(total, 0.))
        return np.searchsorted(cumulative, points, side="right").astype(np.int32)
    totals = weights.sum(axis=1)
    if np.any(totals <= 0) or np.any(~np.isfinite(totals)):
        raise ValueError("Positive finite row categorical mass required")
    cumulative = np.minimum(np.cumsum(weights, axis=1), totals[:, None])
    cumulative[:, -1] = totals
    points = np.minimum(uniforms*totals, np.nextafter(totals, 0.))
    return np.sum(cumulative <= points[:, None], axis=1).astype(np.int32)


def validate_source(probabilities, transitions):
    if (not isinstance(probabilities, np.ndarray) or probabilities.ndim != 2
            or probabilities.dtype != np.float64 or not 0 < probabilities.shape[1] < 32768
            or not 0 < probabilities.shape[0] <= np.iinfo(np.uint32).max
            or not isinstance(transitions, np.ndarray) or transitions.shape != probabilities.shape
            or not np.issubdtype(transitions.dtype, np.integer)):
        raise ValueError("Float64 probability/finite integer transition tables required")
    for lo in range(0, len(probabilities), 4096):
        p, t = probabilities[lo:lo+4096], transitions[lo:lo+4096]
        if (np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1)
                or np.any(np.abs(p.sum(axis=1)-1.) > 1e-12)
                or np.any(t < 0) or np.any(t >= len(probabilities))):
            raise ValueError("Normalized finite source with valid reset/goto states required")


def run_particles(cipher, probabilities, transitions, *, glyphs=6, rho=1/225,
                  particles=4096, seed=0, schedule="balanced",
                  max_ancestry_bytes=128*1024**2, max_work_array_bytes=256*1024**2, observe=None):
    """No hidden-source/key/length inputs. Context0 resets every record.

    At each fixed-horizon step: compute local mass, multiply mean mass into the
    evidence estimate, multinomial-resample parents by mass, sample their local
    action. Never refill an extinct population or drop early completed paths.
    """
    cipher = tuple(map(tuple, cipher))
    if (type(glyphs) is not int or not 0 < glyphs < 32768
            or type(particles) is not int or not 0 < particles <= np.iinfo(np.int32).max
            or type(seed) is not int or seed < 0 or schedule not in ("sequential", "balanced")
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or type(max_ancestry_bytes) is not int or max_ancestry_bytes < 1
            or type(max_work_array_bytes) is not int or max_work_array_bytes < 1
            or not 0 < len(cipher) < 32768
            or any(not c or len(c) > np.iinfo(np.int32).max
                or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)):
        raise ValueError("Invalid observation, prior, seed or bounded particle configuration")
    horizon = sum(map(len, cipher))+len(cipher)
    ancestry_bytes = 8*horizon*particles
    if ancestry_bytes > max_ancestry_bytes:
        raise MemoryError("Declared ancestry allocation cap exceeded")
    validate_source(probabilities, transitions)
    rows, records = probabilities.shape[1], len(cipher)
    work_envelope = 4*records*max(map(len, cipher))+particles*(128*rows+64*records+128)
    if work_envelope > max_work_array_bytes:
        raise MemoryError("Declared observation/particle work array envelope exceeded")
    lengths = np.array(list(map(len, cipher)), dtype=np.int64)
    observations = np.full((records, int(max(lengths))), -1, dtype=np.int32)
    for i, c in enumerate(cipher):
        observations[i, :len(c)] = c
    keys = np.full((particles, rows), -1, dtype=np.int32)
    offsets = np.zeros((particles, records), dtype=np.int32)
    contexts = np.zeros((particles, records), dtype=np.uint32)
    closed = np.zeros((particles, records), dtype=bool)
    parents_history = np.empty((horizon, particles), dtype=np.int32)
    records_history = np.empty((horizon, particles), dtype=np.int16)
    labels_history = np.empty((horizon, particles), dtype=np.int16)
    scores = np.zeros(particles, dtype=np.float64)
    rng, log_evidence, trace = np.random.default_rng(seed), 0., []
    completed_steps, extinct = 0, False
    for step in range(horizon):
        values, selected, singles, doubles = coefficients(keys, offsets, contexts, closed,
            observations, lengths, probabilities, glyphs, rho, schedule)
        masses = values.sum(axis=1)
        total = float(masses.sum())
        if total == 0:
            extinct, log_evidence = True, -math.inf
            trace.append({"step": step+1, "positive_parents": 0, "extinct": True})
            break
        normalized = masses/total
        log_evidence += math.log(total)-math.log(particles)
        parents = categorical(masses, rng.random(particles))
        actions = categorical(values[parents], rng.random(particles))
        selected_values = values[parents, actions]
        if np.any(selected_values <= 0) or np.any(~np.isfinite(selected_values)):
            raise ArithmeticError("Unsupported categorical action selected")
        scores = scores[parents]+np.log(selected_values)
        keys, offsets, contexts, closed, selected, labels = advance(
            keys, offsets, contexts, closed, selected, singles, doubles, parents, actions, transitions)
        parents_history[step], records_history[step], labels_history[step] = parents, selected, labels
        completed_steps += 1
        row = {"step": step+1, "positive_parents": int(np.count_nonzero(masses)),
               "ess_before_resampling": float(1/np.square(normalized).sum()),
               "distinct_selected_parents": int(len(np.unique(parents))),
               "completed_particles": int(np.count_nonzero(closed.all(axis=1))),
               "log_evidence_estimate": log_evidence}
        trace.append(row)
        if observe is not None and (step+1) % 20 == 0:
            observe(row)
    if not extinct and (not closed.all() or not np.array_equal(offsets, np.tile(lengths, (particles, 1)))):
        raise AssertionError("Every surviving fixed-horizon particle must finish all records")
    return {"summary": {"status": "extinct" if extinct else "complete_particles",
                "particles": particles, "seed": seed, "schedule": schedule,
                "horizon": horizon, "transitions": completed_steps,
                "log_evidence_estimate": None if extinct else log_evidence,
                "ancestry_allocated_bytes": ancestry_bytes,
                "work_array_envelope_bytes": work_envelope,
                "posterior_is_finite_approximation": True,
                "exhaustive_support": False, "interval_arithmetic_certificate": False},
            "parents": parents_history[:completed_steps], "records": records_history[:completed_steps],
            "labels": labels_history[:completed_steps], "keys": keys, "scores": scores,
            "offsets": offsets, "contexts": contexts, "closed": closed, "trace": trace}


def reconstruct(result, index, records):
    """Recover source letters through ancestry, ignoring EOS/absorption labels."""
    if result["summary"]["status"] != "complete_particles":
        raise ValueError("Extinct particles do not define completed posterior readings")
    if type(index) is not int or not 0 <= index < result["summary"]["particles"]:
        raise ValueError("Invalid particle index")
    texts, eos = [[] for _ in range(records)], [0]*records
    for step in range(result["summary"]["transitions"]-1, -1, -1):
        record, row = int(result["records"][step, index]), int(result["labels"][step, index])
        if row >= 0:
            texts[record].append(row)
        elif row == -1:
            eos[record] += 1
        index = int(result["parents"][step, index])
    if eos != [1]*records:
        raise AssertionError("Each reconstructed record must have exactly one EOS")
    return tuple(tuple(reversed(t)) for t in texts)


def terminal_groups(result):
    """Group by actual terminal ancestor, never just context/key similarity."""
    if result["summary"]["status"] != "complete_particles":
        return [], 0
    count = result["summary"]["particles"]
    cursor, origins = np.arange(count), np.full(count, -1, dtype=np.int64)
    for step in range(result["summary"]["transitions"]-1, -1, -1):
        labels = result["labels"][step, cursor]
        capture = (origins < 0) & (labels != -2)
        if np.any(labels[capture] != -1):
            raise AssertionError("Last semantic action must close a record")
        origins[capture] = step*count+cursor[capture]
        cursor = result["parents"][step, cursor]
    if np.any(origins < 0):
        raise AssertionError("Completed particles need terminal ancestors")
    groups = []
    for origin in np.unique(origins):
        indices = np.flatnonzero(origins == origin)
        representative = int(indices[0])
        if (not np.all(result["keys"][indices] == result["keys"][representative])
                or not np.all(result["scores"][indices] == result["scores"][representative])):
            raise AssertionError("One terminal ancestor must retain its key and score")
        groups.append((representative, int(len(indices))))
    return groups, int(len(np.unique(cursor)))
