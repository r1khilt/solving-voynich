"""Corrected full-support initialization and symmetric coordinated key moves.

These are exact finite proposal laws, not a guarantee of posterior recovery.
The target remains the uniform dictionary prior times the actual source bridge.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

from voynich.dictionary_smc import bridge_schedule
from voynich.source_key_particles import categorical


PAIR_OPERATIONS = ("swap", "head", "tail", "slide", "motif")
OPERATION_TICKETS = ("row",)*16+("pair",)*4+("swap",)*3+("head",)*2+("tail",)*2+("slide",)*2+("motif",)*2+("refresh",)


@lru_cache(maxsize=1024)
def completion_count(n, missing, units):
    """Number of n-code suffixes containing each of missing specified codes."""
    if (any(type(v) is not int for v in (n, missing, units)) or not 0<=n<=23
            or not 0<=missing<=units<=42 or units<1):
        raise ValueError("Bounded integer completion-count arguments required")
    if not n:
        return int(missing==0)
    return ((units-missing)*completion_count(n-1,missing,units)
            +missing*completion_count(n-1,missing-1,units) if missing
            else units**n)


def randbelow(rng, bound):
    """Unbiased integer draw from [0,bound), with explicit bounded rejection.

    No modulo reduction or floating probabilities for >64-bit counts. Failure
    aborts rather than substituting a biased draw. A 64-attempt cap is recorded.
    """
    if type(bound) is not int or not 1<=bound<=42**23:
        raise ValueError("Bounded positive integer random range required")
    bits = (bound-1).bit_length()
    if not bits:
        return 0
    for _ in range(64):
        candidate = int.from_bytes(rng.bytes((bits+7)//8),"little") & ((1<<bits)-1)
        if candidate<bound:
            return candidate
    raise RuntimeError("Exact integer sampler rejection budget exhausted")


def covered_completion(partial, glyphs, units, rng):
    """Uniform completion conditional on including every single-glyph code."""
    key = partial.copy()
    free = np.flatnonzero(key<0)
    missing = set(range(glyphs))-set(map(int,key[key>=0]))
    if not completion_count(len(free),len(missing),units):
        raise ValueError("Coverage event has zero prior probability")
    for i,row in enumerate(free):
        remaining = len(free)-i-1
        counts = [completion_count(remaining,len(missing)-int(code in missing),units)
                  for code in range(units)]
        total = sum(counts)
        assert total==completion_count(remaining+1,len(missing),units)
        draw = randbelow(rng,total)
        for code,count in enumerate(counts):
            if draw<count:
                key[row] = code
                missing.discard(code)
                break
            draw -= count
        else:
            raise AssertionError("Integer cumulative sampler lost probability mass")
    assert not missing
    return key


def initialization(partial, particles, glyphs, units, rng, covered):
    """q = prior/2 + (prior | coverage)/2; return exact prior/q corrections.

    If covered=False, q is the unchanged prior. The coverage component does
    not become a prior constraint, and mutations need not preserve coverage.
    """
    free = np.flatnonzero(partial<0)
    missing = set(range(glyphs))-set(map(int,partial[partial>=0]))
    event_probability = Fraction(completion_count(len(free),len(missing),units),units**len(free))
    if covered and not event_probability:
        raise ValueError("Positive coverage event required for this mixture")
    keys = rng.integers(units,size=(particles,len(partial)),dtype=np.int32)
    keys[:,partial>=0] = partial[partial>=0]
    if covered:
        for i in range(particles):
            if rng.random()<.5:
                keys[i] = covered_completion(partial,glyphs,units,rng)
    correction = np.zeros(particles)
    covered_count = 0
    for i,key in enumerate(keys):
        event = set(range(glyphs))<=set(map(int,key))
        covered_count += int(event)
        if covered:
            density_ratio = Fraction(1,2)+(Fraction(1,2)/event_probability if event else 0)
            correction[i] = -math.log(float(density_ratio))
    return keys,correction,{"coverage_probability":str(event_probability),
        "initial_covered_particles":covered_count,"mixture_prior_fraction":.5 if covered else 1.,
        "initial_key_bank_sha256":hashlib.sha256(keys.tobytes()).hexdigest()}


def pair_involution(a,b,units,operation):
    """Self-inverse transformation of an ordered code pair; no hidden repairs."""
    if operation not in PAIR_OPERATIONS:
        raise ValueError("Registered pair involution required")
    x,y = units[int(a)],units[int(b)]
    if operation=="swap":
        left,right = y,x
    elif operation=="head":
        left,right = (y[0],)+x[1:],(x[0],)+y[1:]
    elif operation=="tail":
        left,right = x[:-1]+(y[-1],),y[:-1]+(x[-1],)
    elif operation=="slide" and len(x)+len(y)==3:
        joined = x+y
        split = len(y)
        left,right = joined[:split],joined[split:]
    elif operation=="motif" and len(x)==len(y)==1:
        left,right = x+y,y+x
    elif operation=="motif" and len(x)==len(y)==2 and x==y[::-1]:
        left,right = (x[0],),(y[0],)
    else:
        left,right = x,y
    return units.index(left),units.index(right)


def proposals(keys,free,units,rng,kernel):
    """State-independent mixture of symmetric proposals (including self)."""
    if kernel not in ("row","coordinated") or not len(free):
        raise ValueError("Nonempty free rows and registered kernel required")
    proposal = keys.copy()
    counts = dict.fromkeys(sorted(set(OPERATION_TICKETS)),0)
    for i in range(len(keys)):
        operation = "row" if kernel=="row" else OPERATION_TICKETS[int(rng.integers(32))]
        counts[operation] += 1
        if operation=="refresh":
            proposal[i,free] = rng.integers(len(units),size=len(free))
        elif operation in ("row","pair"):
            chosen = rng.choice(free,size=1 if operation=="row" else min(2,len(free)),replace=False)
            proposal[i,chosen] = rng.integers(len(units),size=len(chosen))
        elif len(free)>=2:
            a,b = rng.choice(free,size=2,replace=False)
            proposal[i,a],proposal[i,b] = pair_involution(keys[i,a],keys[i,b],units,operation)
        # With <2 free rows the pair involutions are identity, not renormalized.
    return proposal,counts


def run_coordinated_smc(bridge,partial,*,particles=32,seed=0,stride=4,
                        mutations=4,covered=False,kernel="row",observe=None):
    """First weight G1*prior/q; subsequent weights Gt/Gprev; then resample/MH."""
    partial = np.asarray(partial)
    if (partial.shape!=(bridge.rows,) or partial.dtype.kind not in "iu"
            or np.any(partial< -1) or np.any(partial>=len(bridge.units))
            or type(particles) is not int or not 1<=particles<=128
            or type(seed) is not int or seed<0 or type(mutations) is not int or not 0<=mutations<=8
            or type(covered) is not bool or kernel not in ("row","coordinated")):
        raise ValueError("Bounded coordinated dictionary configuration required")
    stages = bridge_schedule(bridge.lengths,stride)
    if particles*(32*bridge.rows+128)+len(stages)*8192>32*1024**2:
        raise MemoryError("Particle/trace allocation envelope exceeds32MiB")
    assert len(bridge.units)==bridge.glyphs+bridge.glyphs**2
    free = np.flatnonzero(partial<0)
    rng = np.random.default_rng(seed)
    keys,correction,initial = initialization(partial,particles,bridge.glyphs,len(bridge.units),rng,covered)
    old,log_evidence,trace = np.zeros(particles),0.,[]
    for stage,(cuts,closed) in enumerate(stages):
        new = bridge.log_values(keys,cuts,closed=closed)
        if np.any(~np.isfinite(old)) or np.any(np.isnan(new)) or np.any(np.isposinf(new)):
            raise ArithmeticError("Invalid target or unsupported resampled parent")
        increment = new-old+(correction if stage==0 else 0.)
        log_total = float(np.logaddexp.reduce(increment))
        if log_total==-math.inf:
            trace.append({"stage":stage,"cuts":cuts,"closed":closed,"extinct":True})
            return {"status":"extinct","keys":keys,"key_log_likelihoods":new,
                "log_evidence_estimate":None,"trace":trace,"tables":bridge.tables,
                "edges":bridge.edges,"initialization":initial}
        weights = np.exp(increment-log_total)
        log_evidence += log_total-math.log(particles)
        parents = categorical(weights,rng.random(particles))
        keys,old = keys[parents].copy(),new[parents].copy()
        accepted = changed = proposed = 0
        operation_counts = dict.fromkeys(sorted(set(OPERATION_TICKETS)),0)
        for _ in range(mutations if len(free) else 0):
            candidate,counts = proposals(keys,free,bridge.units,rng,kernel)
            scores = bridge.log_values(candidate,cuts,closed=closed)
            if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
                raise ArithmeticError("Invalid proposed target score")
            with np.errstate(divide="ignore"):
                draws = np.log(rng.random(particles))
            accept = np.isfinite(scores)&(draws<np.minimum(0.,scores-old))
            accepted += int(np.count_nonzero(accept))
            changed += int(np.count_nonzero(accept&np.any(candidate!=keys,axis=1)))
            proposed += particles
            keys[accept],old[accept] = candidate[accept],scores[accept]
            for name,count in counts.items():
                operation_counts[name] += count
        row = {"stage":stage,"cuts":cuts,"closed":closed,"extinct":False,
            "positive_pre_resampling":int(np.count_nonzero(np.isfinite(new))),
            "incremental_ess":float(1/np.square(weights).sum()),
            "maximum_incremental_weight":float(weights.max()),
            "distinct_selected_parents":int(len(np.unique(parents))),
            "distinct_keys_after_mutation":int(len(np.unique(keys,axis=0))),
            "accepted_proposals":accepted,"changed_proposals":changed,"proposals":proposed,
            "operation_counts":operation_counts,"increment_log_mean":log_total-math.log(particles),
            "log_evidence_estimate":log_evidence,"key_bank_sha256":hashlib.sha256(keys.tobytes()).hexdigest()}
        trace.append(row)
        if observe is not None:
            observe(row)
    return {"status":"complete_particles","keys":keys,"key_log_likelihoods":old,
        "log_evidence_estimate":log_evidence,"trace":trace,"tables":bridge.tables,
        "edges":bridge.edges,"initialization":initial,"posterior_is_finite_approximation":True}
