"""Batched exact-path marginals for one-state deterministic unit channels.

Each candidate supplies one nonempty unit per source letter, in source.alphabet
order. Different letters may emit the same unit; their distinct source contexts
are summed, not identified. This is a likelihood kernel, not a unit/key search.

The fast recurrence uses float64 matrix products and one scale per candidate
shared by ALL pending glyph offsets. Conservative underflow-risk detection sends
individual candidate/record pairs to a rolling log-space recurrence. "Exact"
means no path pruning, not exact real-number arithmetic.
"""
from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence

import numpy as np

from voynich.finite_state_channel import SourceModel


_LOG_SAFE_PRODUCT = math.log(np.finfo(np.float64).tiny) + math.log(16.)


def _validated_inputs(source: SourceModel, candidate_units: Sequence[Sequence[str]],
                      records: Sequence[str], stop_probability: float, max_emission_length: int
                      ) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], float]:
    if (type(max_emission_length) is not int or max_emission_length < 1
            or isinstance(stop_probability, bool)):
        raise ValueError("Need a positive integer emission bound and a proper stopping probability")
    rho = float(stop_probability)
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("Stopping probability must be finite and strictly between zero and one")
    if isinstance(records, (str, bytes)) or isinstance(candidate_units, (str, bytes)):
        raise ValueError("Supply separate records and a sequence of candidate unit tuples")
    records = tuple(records)
    if any(not isinstance(record, str) for record in records):
        raise ValueError("Every ciphertext record must be a Unicode string")
    candidates = []
    for row in candidate_units:
        if isinstance(row, (str, bytes)):
            raise ValueError("A candidate must supply one string per source letter")
        row = tuple(row)
        if len(row) != len(source.alphabet):
            raise ValueError("Candidate units must match the complete ordered source alphabet")
        if any(not isinstance(unit, str) or not 1 <= len(unit) <= max_emission_length for unit in row):
            raise ValueError("Every emission must be a nonempty string within the declared bound")
        candidates.append(row)
    return tuple(candidates), records, rho


def _source_matrix(source: SourceModel) -> np.ndarray:
    # Contexts 0..A-1 are source letters; A is a start-only context. No emitted
    # character ever returns to the extra start context.
    contexts = (*source.alphabet, "")
    return np.asarray([[source.probabilities[context if source.order else ""][letter]
                        for letter in source.alphabet] for context in contexts], dtype=np.float64)


def _log_domain_record(source: SourceModel, units: tuple[str, ...], record: str,
                       rho: float, maximum_length: int) -> float:
    # Keep the fallback memory bounded too: the existing engine is an external
    # validation reference, rather than a full-lattice dependency at runtime.
    probabilities = _source_matrix(source)
    log_probabilities = np.full_like(probabilities, -np.inf)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    alphabet, width = len(source.alphabet), maximum_length + 1
    pending = np.full((width, alphabet + 1), -np.inf, dtype=np.float64)
    pending[0, alphabet] = 0.
    for offset in range(len(record)):
        slot = offset % width
        current = pending[slot].copy()
        pending[slot].fill(-np.inf)
        next_letter = np.logaddexp.reduce(current[:, None] + log_probabilities, axis=0) + math.log1p(-rho)
        for letter, unit in enumerate(units):
            if record.startswith(unit, offset):
                destination = (offset + len(unit)) % width
                pending[destination, letter] = np.logaddexp(pending[destination, letter], next_letter[letter])
        if not np.isfinite(pending).any():
            break
    return float(np.logaddexp.reduce(pending[len(record) % width])) + math.log(rho)


def _record_likelihood(source: SourceModel, candidates: tuple[tuple[str, ...], ...], units: np.ndarray,
                       lengths: np.ndarray, probabilities: np.ndarray, record: str, rho: float,
                       maximum_length: int) -> tuple[np.ndarray, int]:
    batch, alphabet = units.shape
    width = maximum_length + 1
    pending = np.zeros((width, batch, alphabet + 1), dtype=np.float64)
    pending[0, :, alphabet] = 1.
    log_scale = np.zeros(batch, dtype=np.float64)
    fallback = np.zeros(batch, dtype=np.bool_)
    minimum_source = float(probabilities[probabilities > 0].min())
    log_minimum_factor = math.log(minimum_source) + math.log1p(-rho)

    for offset in range(len(record)):
        slot = offset % width
        current = pending[slot].copy()
        pending[slot].fill(0.)

        # A tiny context can later uniquely support a suffix, even while another
        # context dominates total mass. Detect risky products BEFORE a dot
        # product can round them away; waiting for a final -inf is insufficient.
        minimum_live = np.where(current > 0, current, np.inf).min(axis=1)
        risky = (~fallback) & (np.log(minimum_live) + log_minimum_factor < _LOG_SAFE_PRODUCT)
        fallback |= risky
        if np.any(risky):
            pending[:, risky, :] = 0.
        current[fallback, :] = 0.

        letter_mass = (current @ probabilities) * (1 - rho)
        for length in range(1, min(maximum_length, len(record) - offset) + 1):
            # Explicit lengths also distinguish trailing U+0000 from NumPy's
            # fixed-width Unicode padding.
            matches = (lengths == length) & (units == record[offset:offset + length])
            pending[(offset + length) % width, :, :alphabet] += letter_mass * matches

        # Every latent path keeps its correct relative weight across different
        # future offsets. Normalizing only the current or destination slice
        # would silently change the variable-length model.
        total = pending.sum(axis=(0, 2))
        live = total > 0
        log_scale[live] += np.log(total[live])
        pending /= np.where(live, total, 1.)[None, :, None]
        if not np.any(live):
            break

    terminal = pending[len(record) % width].sum(axis=1)
    possible = terminal > 0
    result = np.full(batch, -np.inf, dtype=np.float64)
    result[possible] = log_scale[possible] + np.log(terminal[possible]) + math.log(rho)
    for index in np.flatnonzero(fallback):
        result[index] = _log_domain_record(source, candidates[index], record, rho, maximum_length)
    return result, int(np.count_nonzero(fallback))


def batch_log_likelihood(source: SourceModel, candidate_units: Sequence[Sequence[str]],
                         records: Sequence[str], stop_probability: float, max_emission_length: int = 2,
                         *, diagnostics: MutableMapping | None = None) -> np.ndarray:
    """Return one corpus log likelihood per candidate, using independent records.

    Candidate j is a tuple (unit_for_source.alphabet[0], ...). Units may repeat,
    contain arbitrary Unicode codepoints and have different nonzero lengths.
    Each source character emits its candidate's unit with probability one. The
    single channel state self-loops, source/stop parameters are fixed, and every
    plaintext segmentation path is included. Impossible observations yield -inf.

    Empty records have probability rho. An empty record collection has log
    likelihood zero for each candidate; an empty candidate collection returns
    an empty float64 array. Inputs are copied/validated and are not mutated.

    Space is O(B*(L+1)*(A+1) + A*A + B*A*L), with batch B, source size A, maximum
    actual unit length L. No record-length lattice, beam or candidate mixing is
    used. Each underflow-risk candidate/record is recomputed by a scalar rolling
    log-space recurrence with O((L+1)*(A+1) + A*A) additional space. The existing
    engine is a validation reference, not a full-lattice runtime dependency.
    If supplied, diagnostics is updated with dimensions and fallback count.
    This synchronous kernel has no cooperative deadline; bound batch/input sizes
    and enforce any hard external process limit in the caller.
    """
    candidates, records, rho = _validated_inputs(source, candidate_units, records,
                                                stop_probability, max_emission_length)
    batch, alphabet = len(candidates), len(source.alphabet)
    actual_length = max((len(unit) for row in candidates for unit in row), default=0)
    result = np.zeros(batch, dtype=np.float64)
    fallback_count = 0
    if batch and records:
        units = np.asarray(candidates, dtype=f"U{actual_length}")
        lengths = np.asarray([[len(unit) for unit in row] for row in candidates], dtype=np.intp)
        probabilities = _source_matrix(source)
        for record in records:
            scores, count = _record_likelihood(source, candidates, units, lengths, probabilities,
                                               record, rho, actual_length)
            result += scores
            fallback_count += count
    if diagnostics is not None:
        diagnostics.update(candidate_count=batch, record_count=len(records), source_order=source.order,
                           source_alphabet_size=alphabet, maximum_actual_unit_length=actual_length,
                           candidate_record_pairs=batch * len(records), log_domain_fallbacks=fallback_count,
                           rolling_buffer_bytes=(actual_length + 1) * batch * (alphabet + 1) * 8)
    return result
