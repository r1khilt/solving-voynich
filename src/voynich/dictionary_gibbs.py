"""Exact random-row conditional updates inside the observation-prefix SMC law."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from voynich.dictionary_smc import bridge_schedule
from voynich.source_key_particles import categorical


def gibbs_update(bridge,keys,values,free,cuts,closed,rng):
    """Enumerate every code for one uniformly chosen free row per particle.

    Uses actual contextual likelihoods, not a surrogate. Given other rows, the
    uniform key prior cancels. Its positive old code makes the denominator >0.
    A graph/work cap aborts the update; there is no pruning or mass repair.
    """
    n,u = len(keys),len(bridge.units)
    selected = rng.choice(free,size=n)
    bank = np.repeat(keys,u,axis=0)
    bank[np.arange(n*u),np.repeat(selected,u)] = np.tile(np.arange(u),n)
    logs = bridge.log_values(bank,cuts,closed=closed).reshape(n,u)
    old_codes = keys[np.arange(n),selected]
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(logs[np.arange(n),old_codes],values,rtol=0,atol=1e-7)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ArithmeticError("Conditional target cannot contain NaN/infinite positive scores")
    totals = np.logaddexp.reduce(logs,axis=1)
    if np.any(~np.isfinite(totals)):
        raise ArithmeticError("Positive parent missing from complete Gibbs conditional")
    weights = np.exp(logs-totals[:,None])
    codes = categorical(weights,rng.random(n))
    updated = keys.copy()
    updated[np.arange(n),selected] = codes
    return updated,logs[np.arange(n),codes],{
        "accepted_proposals":n,"changed_proposals":int(np.count_nonzero(codes!=old_codes)),
        "proposals":n,"conditional_target_tables":n*u,
        "minimum_positive_codes":int(np.min(np.count_nonzero(np.isfinite(logs),axis=1))),
        "maximum_conditional_weight":float(weights.max())}


def run_dictionary_gibbs(bridge,partial,*,particles=32,seed=0,stride=16,
                         max_particle_bytes=32*1024**2,observe=None):
    """Weight at the old dictionary, resample, then ONE exact row Gibbs update.

    Finite populations can become extinct or miss posterior modes. Enumeration
    of a conditional is exact; posterior recovery or mixing is not guaranteed.
    """
    partial = np.asarray(partial)
    if (partial.shape!=(bridge.rows,) or partial.dtype.kind not in "iu"
            or np.any(partial< -1) or np.any(partial>=len(bridge.units))
            or type(particles) is not int or not 1<=particles<=4096//len(bridge.units)
            or type(seed) is not int or seed<0
            or type(max_particle_bytes) is not int or max_particle_bytes<1):
        raise ValueError("Bounded literal conditional Gibbs configuration required")
    stages = bridge_schedule(bridge.lengths,stride)
    if particles*len(bridge.units)*(8*bridge.rows+64)+len(stages)*4096>max_particle_bytes:
        raise MemoryError("Gibbs candidate/particle/trace allocation envelope exceeds cap")
    free = np.flatnonzero(partial<0)
    rng = np.random.default_rng(seed)
    keys = rng.integers(len(bridge.units),size=(particles,bridge.rows),dtype=np.int32)
    keys[:,partial>=0] = partial[partial>=0]
    old,log_evidence,trace = np.zeros(particles),0.,[]
    for stage,(cuts,closed) in enumerate(stages):
        new = bridge.log_values(keys,cuts,closed=closed)
        if np.any(~np.isfinite(old)) or np.any(np.isnan(new)) or np.any(np.isposinf(new)):
            raise ArithmeticError("Resampled parents must have positive preceding target")
        increment = new-old
        log_total = float(np.logaddexp.reduce(increment))
        if log_total==-math.inf:
            trace.append({"stage":stage,"cuts":cuts,"closed":closed,"extinct":True})
            return {"status":"extinct","log_evidence_estimate":None,"keys":keys,
                "key_log_likelihoods":new,"trace":trace,"tables":bridge.tables,"edges":bridge.edges,
                "posterior_is_finite_approximation":True}
        weights = np.exp(increment-log_total)
        log_evidence += log_total-math.log(particles)
        parents = categorical(weights,rng.random(particles))
        keys,old = keys[parents].copy(),new[parents].copy()
        move = {"accepted_proposals":0,"changed_proposals":0,"proposals":0,
            "conditional_target_tables":0,"minimum_positive_codes":None,"maximum_conditional_weight":None}
        if len(free):
            keys,old,move = gibbs_update(bridge,keys,old,free,cuts,closed,rng)
        row = {"stage":stage,"cuts":cuts,"closed":closed,"extinct":False,
            "positive_pre_resampling":int(np.count_nonzero(np.isfinite(new))),
            "incremental_ess":float(1/np.square(weights).sum()),
            "maximum_incremental_weight":float(weights.max()),
            "distinct_selected_parents":int(len(np.unique(parents))),
            "distinct_keys_after_mutation":int(len(np.unique(keys,axis=0))),**move,
            "increment_log_mean":log_total-math.log(particles),"log_evidence_estimate":log_evidence,
            "key_bank_sha256":hashlib.sha256(keys.tobytes()).hexdigest()}
        trace.append(row)
        if observe is not None:
            observe(row)
    return {"status":"complete_particles","log_evidence_estimate":log_evidence,
        "keys":keys,"key_log_likelihoods":old,"trace":trace,"tables":bridge.tables,"edges":bridge.edges,
        "posterior_is_finite_approximation":True}
