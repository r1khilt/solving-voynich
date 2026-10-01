"""Future-guided SMC, with a positive surrogate and exact target corrections.

The surrogate uses iid source letters and resamples unbound emission units at
EVERY occurrence. It is neither the true shared-key likelihood nor an upper
bound. Only the proposal changes; original leaf scores remain unchanged.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from voynich.source_key_particles import advance, coefficients, validate_source


def log_categorical(log_weights, rng, count=None):
    """Gumbel-max draws without exponentiating tiny categorical probabilities.

    In exact arithmetic independent Gumbels give categorical softmax draws.
    Floating RNG/arithmetic are still finite; this is not a support certificate.
    1-D parent draws use blocks of 64 to bound the temporary allocation.
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if (log_weights.ndim not in (1, 2) or not log_weights.size
            or np.any(np.isnan(log_weights)) or np.any(log_weights == math.inf)
            or np.any(~np.isfinite(np.max(log_weights, axis=-1)))):
        raise ValueError("Finite or minus-infinite log weights, with positive row mass required")
    if log_weights.ndim == 2:
        u = np.maximum(rng.random(log_weights.shape), np.nextafter(0., 1.))
        return np.argmax(log_weights-np.log(-np.log(u)), axis=1).astype(np.int32)
    if type(count) is not int or count < 1:
        raise ValueError("Positive draw count required")
    result = np.empty(count, dtype=np.int32)
    for start in range(0, count, 64):
        u = np.maximum(rng.random((min(64, count-start), len(log_weights))), np.nextafter(0., 1.))
        result[start:start+len(u)] = np.argmax(log_weights[None, :]-np.log(-np.log(u)), axis=1)
    return result


class IidSuffixGuide:
    """Exact backward sums for an explicitly relaxed iid surrogate.

    Each unfinished record includes its remaining EOS. Closed records have
    guide one. A floor on the TOTAL log guide preserves positive support even
    for surrogate-impossible states. All-complete guide is exactly one.
    """
    def __init__(self, cipher, row, *, glyphs=6, rho=1/225, log_floor=-2000.,
                 cache_entries=1024, max_tables=200_000, max_cache_bytes=64*1024**2):
        self.cipher = tuple(map(tuple, cipher))
        row = np.asarray(row, dtype=np.float64)
        if (row.ndim != 1 or not row.size or np.any(~np.isfinite(row)) or np.any(row < 0)
                or abs(float(row.sum())-1) > 1e-12 or type(glyphs) is not int or glyphs < 1
                or not self.cipher or any(not c or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in self.cipher)
                or not math.isfinite(rho) or not 0 < rho < 1
                or not math.isfinite(log_floor) or log_floor >= 0
                or any(type(v) is not int or v < 1 for v in (cache_entries, max_tables, max_cache_bytes))):
            raise ValueError("Invalid bounded iid guide")
        self.row = (row+1e-8)/(1+len(row)*1e-8)
        self.glyphs, self.rho, self.log_floor = glyphs, rho, log_floor
        self.units = glyphs+glyphs**2
        self.lengths = np.array(list(map(len, self.cipher)), dtype=np.int64)
        self.width = int(max(self.lengths))+2
        self.cache_entries, self.max_tables = cache_entries, max_tables
        if 8*len(self.cipher)*self.width*(cache_entries+64) > max_cache_bytes:
            raise MemoryError("Guide cache plus 64-table build envelope exceeds cap")
        self.cache, self.tables_built = OrderedDict(), 0
        self.single_codes = [np.array(c, dtype=np.int32) for c in self.cipher]
        self.double_codes = [glyphs+glyphs*np.array(c[:-1])+np.array(c[1:]) for c in self.cipher]

    def build(self, keys):
        if self.tables_built+len(keys) > self.max_tables:
            raise RuntimeError("Declared surrogate-table work cap exceeded; no extension")
        self.tables_built += len(keys)
        unknown = keys < 0
        weights = np.repeat(((unknown*self.row).sum(axis=1)/self.units)[:, None], self.units, axis=1)
        batch, letters = np.nonzero(~unknown)
        np.add.at(weights, (batch, keys[batch, letters]), self.row[letters])
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)+math.log1p(-self.rho)
        dp = np.full((len(keys), len(self.cipher), self.width), -math.inf)
        for r, c in enumerate(self.cipher):
            dp[:, r, len(c)] = math.log(self.rho)
            for pos in range(len(c)-1, -1, -1):
                single = log_weights[:, self.single_codes[r][pos]]+dp[:, r, pos+1]
                double = (log_weights[:, self.double_codes[r][pos]]+dp[:, r, pos+2]
                          if pos+1 < len(c) else -math.inf)
                dp[:, r, pos] = np.logaddexp(single, double)
        return dp

    def __call__(self, keys, offsets, closed):
        if (keys.ndim != 2 or keys.shape[1] != len(self.row)
                or offsets.shape != (len(keys), len(self.cipher)) or closed.shape != offsets.shape
                or np.any(keys < -1) or np.any(keys >= self.units)
                or np.any(offsets < 0) or np.any(offsets > self.lengths)):
            raise ValueError("Malformed guide state")
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        groups = [[] for _ in unique]
        for i, u in enumerate(inverse):
            groups[u].append(i)
        result = np.empty(len(keys), dtype=np.float64)
        missing = []

        def assign(u, table):
            indices = np.asarray(groups[u], dtype=np.int64)
            terms = table[np.arange(len(self.cipher))[None, :], offsets[indices]]
            result[indices] = np.maximum(np.where(closed[indices], 0., terms).sum(axis=1), self.log_floor)

        for u, key in enumerate(unique):
            ident = key.tobytes()
            if ident in self.cache:
                table = self.cache.pop(ident)
                self.cache[ident] = table
                assign(u, table)
            else:
                missing.append(u)
        for start in range(0, len(missing), 64):
            batch = missing[start:start+64]
            tables = self.build(unique[batch])
            for u, table in zip(batch, tables, strict=True):
                assign(u, table)
                # Copy: a view would retain the whole 64-table backing allocation.
                self.cache[unique[u].tobytes()] = table.copy()
                if len(self.cache) > self.cache_entries:
                    self.cache.popitem(last=False)
        if np.any(~np.isfinite(result)) or np.any(result > 1e-12):
            raise ArithmeticError("Positive finite log guide required")
        return result


class ConstantGuide:
    """Matched transport control; h is one for every state."""
    log_floor, tables_built = 0., 0

    def __call__(self, keys, offsets, closed):
        return np.zeros(len(keys), dtype=np.float64)


def guided_actions(keys, offsets, contexts, closed, observations, lengths,
                   probabilities, transitions, glyphs, rho, schedule, guide):
    """log b=log a+log h(child)-log h(parent), on original support only."""
    values, records, singles, doubles = coefficients(keys, offsets, contexts, closed,
        observations, lengths, probabilities, glyphs, rho, schedule)
    log_h = guide(keys, offsets, closed)
    parent, action = np.nonzero(values > 0)
    changed = advance(keys, offsets, contexts, closed, records, singles, doubles, parent, action, transitions)
    child_h = guide(changed[0], changed[1], changed[3])
    log_b = np.full_like(values, -math.inf)
    log_b[parent, action] = np.log(values[parent, action])+child_h-log_h[parent]
    return values, records, singles, doubles, log_b


def run_guided_particles(cipher, probabilities, transitions, *, glyphs=6, rho=1/225,
                         particles=512, seed=0, schedule="balanced", observe=None,
                         cache_entries=1024, max_tables=200_000, guidance="iid",
                         max_ancestry_bytes=128*1024**2, max_work_array_bytes=256*1024**2):
    cipher = tuple(map(tuple, cipher))
    if (type(glyphs) is not int or not 0 < glyphs < 32768
            or type(particles) is not int or not 0 < particles <= np.iinfo(np.int32).max
            or type(seed) is not int or seed < 0 or schedule not in ("sequential", "balanced")
            or guidance not in ("none", "iid")
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or any(type(v) is not int or v < 1 for v in (max_ancestry_bytes, max_work_array_bytes))
            or not 0 < len(cipher) < 32768
            or any(not c or len(c) > np.iinfo(np.int32).max
                or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)):
        raise ValueError("Invalid bounded guided particle configuration")
    validate_source(probabilities, transitions)
    rows, records = probabilities.shape[1], len(cipher)
    horizon = sum(map(len, cipher))+records
    ancestry_bytes = 8*horizon*particles
    # Includes up to 2*rows+2 children, unique-key sorting and temporary state copies.
    envelope = 4*records*max(map(len, cipher))+particles*(2*rows+2)*(32*rows+48*records+128)
    if ancestry_bytes > max_ancestry_bytes:
        raise MemoryError("Declared ancestry allocation cap exceeded")
    if envelope > max_work_array_bytes:
        raise MemoryError("Declared guided work-array envelope exceeded")
    guide = (IidSuffixGuide(cipher, probabilities[0], glyphs=glyphs, rho=rho,
                           cache_entries=cache_entries, max_tables=max_tables)
             if guidance == "iid" else ConstantGuide())
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
    rng = np.random.default_rng(seed)
    initial_h = float(guide(keys[:1], offsets[:1], closed[:1])[0])
    log_evidence, trace, completed_steps, extinct = initial_h, [], 0, False
    for step in range(horizon):
        values, selected, singles, doubles, log_b = guided_actions(keys, offsets, contexts, closed,
            observations, lengths, probabilities, transitions, glyphs, rho, schedule, guide)
        peak = float(np.max(log_b))
        if peak == -math.inf:
            extinct, log_evidence = True, -math.inf
            trace.append({"step": step+1, "positive_parents": 0, "extinct": True})
            break
        log_masses = np.logaddexp.reduce(log_b, axis=1)
        log_total = float(np.logaddexp.reduce(log_masses))
        normalized = np.exp(log_masses-log_total)  # Diagnostics only; never used for sampling.
        log_evidence += log_total-math.log(particles)
        parents = log_categorical(log_masses, rng, particles)
        actions = log_categorical(log_b[parents], rng)
        chosen = values[parents, actions]
        if np.any(chosen <= 0):
            raise ArithmeticError("Unsupported original action selected")
        scores = scores[parents]+np.log(chosen)
        keys, offsets, contexts, closed, selected, labels = advance(keys, offsets, contexts, closed,
            selected, singles, doubles, parents, actions, transitions)
        parents_history[step], records_history[step], labels_history[step] = parents, selected, labels
        completed_steps += 1
        row = {"step": step+1, "positive_parents": int(np.count_nonzero(np.isfinite(log_masses))),
            "ess_before_resampling": float(1/np.square(normalized).sum()),
            "distinct_selected_parents": int(len(np.unique(parents))),
            "completed_particles": int(np.count_nonzero(closed.all(axis=1))),
            "log_evidence_estimate": log_evidence, "guide_tables_built": guide.tables_built}
        trace.append(row)
        if observe is not None and (step+1) % 20 == 0:
            observe(row)
    if not extinct and (not closed.all() or not np.array_equal(offsets, np.tile(lengths, (particles, 1)))):
        raise AssertionError("Every surviving particle must finish all records")
    return {"summary": {"status": "extinct" if extinct else "complete_particles",
        "particles": particles, "seed": seed, "schedule": schedule, "horizon": horizon,
        "transitions": completed_steps, "log_evidence_estimate": None if extinct else log_evidence,
        "ancestry_allocated_bytes": ancestry_bytes, "work_array_envelope_bytes": envelope,
        "guide": "iid_root_smoothed_fresh_units_each_occurrence" if guidance == "iid" else "constant_one",
        "categorical_method": "blocked_log_gumbel_max",
        "initial_log_guide": initial_h, "guide_log_floor": guide.log_floor,
        "guide_tables_built": guide.tables_built, "guide_cache_entries": cache_entries,
        "guide_max_tables": max_tables, "posterior_is_finite_approximation": True,
        "exhaustive_support": False, "interval_arithmetic_certificate": False},
        "parents": parents_history[:completed_steps], "records": records_history[:completed_steps],
        "labels": labels_history[:completed_steps], "keys": keys, "scores": scores,
        "offsets": offsets, "contexts": contexts, "closed": closed, "trace": trace}
