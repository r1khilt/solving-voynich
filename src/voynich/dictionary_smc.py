"""Whole-dictionary resample--move with exact fixed-key observation bridges.

Source strings are marginalized; a sampled dictionary is shared by all records.
Finite populations can collapse or miss modes. No posterior/translation certificate.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

from voynich.source_key_particles import categorical, validate_source


class FixedKeyBridge:
    """Conditional source likelihood, with mid-unit cuts and final EOS closure.

    A supplied offset at the original record end denotes an already paid EOS.
    Intermediate cuts at that end still integrate possible source continuations.
    The source tables are copied, not replaced with an IID approximation.
    """

    def __init__(self, cipher, probabilities, transitions, *, glyphs=6, rho=1/225,
                 offsets=None, contexts=None, max_tables=1_000_000,
                 max_edges=100_000_000, max_states_per_table=100_000,
                 max_work_bytes=128*1024**2):
        validate_source(probabilities, transitions)
        self.cipher = tuple(map(tuple, cipher))
        if (type(glyphs) is not int or not 1<=glyphs<=6 or not 1<=len(self.cipher)<=2
                or probabilities.shape[1]>23
                or any(not c or len(c)>4096 or any(type(g) is not int or not 0<=g<glyphs for g in c)
                       for c in self.cipher)
                or isinstance(rho, bool) or not math.isfinite(rho) or not 0<rho<1
                or any(type(n) is not int or n<1 for n in
                       (max_tables,max_edges,max_states_per_table,max_work_bytes))):
            raise ValueError("Bounded literal observations/source configuration required")
        self.offsets = (0,)*len(self.cipher) if offsets is None else tuple(offsets)
        self.contexts = (0,)*len(self.cipher) if contexts is None else tuple(contexts)
        if (len(self.offsets)!=len(self.cipher) or len(self.contexts)!=len(self.cipher)
                or any(type(n) is not int or not 0<=n<=len(c) for n,c in zip(self.offsets,self.cipher,strict=True))
                or any(type(n) is not int or not 0<=n<len(probabilities) for n in self.contexts)):
            raise ValueError("Valid conditional offsets and contexts required")
        envelope = probabilities.nbytes+transitions.nbytes
        if envelope>max_work_bytes:
            raise MemoryError("Source copies exceed work cap")
        self.p = probabilities.copy()
        self.goto = transitions.copy()
        self.p.setflags(write=False)
        self.goto.setflags(write=False)
        self.source_bytes = envelope
        self.units = tuple(u for n in (1,2) for u in itertools.product(range(glyphs),repeat=n))
        self.glyphs, self.rho = glyphs, rho
        self.rows = self.p.shape[1]
        self.lengths = tuple(len(c)-o for c,o in zip(self.cipher,self.offsets,strict=True))
        self.max_tables, self.max_edges = max_tables, max_edges
        self.max_states, self.max_work_bytes = max_states_per_table, max_work_bytes
        self.tables = self.edges = 0
        # Only an actual one-state source uses the specialized IID computation.
        self.iid = len(self.p)==1

    def _stage(self, cuts, closed):
        cuts = tuple(cuts)
        if (type(closed) is not bool or len(cuts)!=len(self.lengths)
                or any(type(c) is not int or not 0<=c<=n for c,n in zip(cuts,self.lengths,strict=True))
                or (closed and cuts!=self.lengths)):
            raise ValueError("Relative prefix cuts or exact final closed stage required")
        return cuts

    def _budget(self, edges):
        if self.edges+edges>self.max_edges:
            raise RuntimeError("Bridge edge-work cap exhausted")
        self.edges += edges

    def _iid_logs(self, keys, cuts, closed):
        n = len(keys)
        self._budget(n*self.rows*sum(cuts))  # Conservative row-operation accounting.
        weights = np.zeros((n,len(self.units)))
        np.add.at(weights,(np.arange(n)[:,None],keys),self.p[0][None,:])
        with np.errstate(divide="ignore"):
            logw = np.log(weights)+math.log1p(-self.rho)
        values = np.zeros(n)
        for cipher,offset,cut,remaining in zip(self.cipher,self.offsets,cuts,self.lengths,strict=True):
            if not remaining or not cut:
                continue
            y = cipher[offset:offset+cut]
            dp = np.full((n,cut+2),-math.inf)
            dp[:,cut] = math.log(self.rho) if closed else 0.
            for pos in range(cut-1,-1,-1):
                single = logw[:,y[pos]]+dp[:,pos+1]
                if pos+1<cut:
                    dual = logw[:,self.glyphs+self.glyphs*y[pos]+y[pos+1]]+dp[:,pos+2]
                elif closed:
                    dual = -math.inf
                else:
                    start = self.glyphs+self.glyphs*y[pos]
                    dual = np.logaddexp.reduce(logw[:,start:start+self.glyphs],axis=1)
                dp[:,pos] = np.logaddexp(single,dual)
            values += dp[:,0]
        return values

    def _markov_log(self, key, cuts, closed, reserved):
        total = 0.
        continuation = math.log1p(-self.rho)
        for cipher,offset,cut,ctx,remaining in zip(
                self.cipher,self.offsets,cuts,self.contexts,self.lengths,strict=True):
            if not remaining or not cut:
                continue
            y = cipher[offset:offset+cut]
            layers = {0:{ctx:0.}}
            answer, allocated = -math.inf, 1
            for pos in range(cut):
                for context,mass in layers.pop(pos,{}).items():
                    self._budget(self.rows)
                    for row,code in enumerate(key):
                        probability = float(self.p[context,row])
                        unit = self.units[int(code)]
                        visible = min(len(unit),cut-pos)
                        if probability==0 or unit[:visible]!=y[pos:pos+visible]:
                            continue
                        end = pos+len(unit)
                        value = mass+continuation+math.log(probability)
                        if end>=cut:
                            if closed and end!=cut:
                                continue
                            answer = float(np.logaddexp(answer,value+(math.log(self.rho) if closed else 0.)))
                        else:
                            target = layers.setdefault(end,{})
                            next_ctx = int(self.goto[context,row])
                            if next_ctx not in target:
                                allocated += 1
                                if (allocated>self.max_states
                                        or reserved+256*allocated>self.max_work_bytes):
                                    raise MemoryError("Bridge source-state allocation cap exhausted")
                            target[next_ctx] = float(np.logaddexp(target.get(next_ctx,-math.inf),value))
            total += answer
        return total

    def log_values(self, keys, cuts, *, closed=False):
        cuts = self._stage(cuts,closed)
        keys = np.asarray(keys)
        if (keys.ndim!=2 or keys.shape[1]!=self.rows or not 1<=len(keys)<=4096
                or keys.dtype.kind not in "iu" or np.any(keys<0) or np.any(keys>=len(self.units))):
            raise ValueError("Bounded complete dictionary bank required")
        # Covers full bank, output, and <=64-row DP/scratch; source copies included.
        envelope = self.source_bytes+keys.nbytes+8*len(keys)+8*min(len(keys),64)*(
            4*len(self.units)+4*self.rows+4*(max(cuts)+2)+32)
        if envelope>self.max_work_bytes:
            raise MemoryError("Bridge bank/DP scratch envelope exceeds work cap")
        if self.tables+len(keys)>self.max_tables:
            raise RuntimeError("Bridge dictionary-table cap exhausted")
        self.tables += len(keys)
        output = np.empty(len(keys))
        if self.iid:
            for start in range(0,len(keys),64):
                output[start:start+64] = self._iid_logs(keys[start:start+64],cuts,closed)
        else:
            for index,key in enumerate(keys):
                output[index] = self._markov_log(key,cuts,closed,envelope)
        return output


def bridge_schedule(lengths, stride=1):
    """Balanced relative observation cuts; a separate final EOS stage."""
    if (type(stride) is not int or stride<1 or not 1<=len(lengths)<=2
            or any(type(n) is not int or not 0<=n<=4096 for n in lengths)):
        raise ValueError("Finite lengths and positive stride required")
    cuts, stages = [0]*len(lengths), []
    while tuple(cuts)!=tuple(lengths):
        active = [r for r,n in enumerate(lengths) if cuts[r]<n]
        chosen = active[0]
        for r in active[1:]:
            if cuts[r]*lengths[chosen]<cuts[chosen]*lengths[r]:
                chosen = r
        cuts[chosen] = min(lengths[chosen],cuts[chosen]+stride)
        stages.append((tuple(cuts),False))
    stages.append((tuple(lengths),True))
    return tuple(stages)


def propose(keys, free, units, rng, kernel):
    """State-independent symmetric mixtures, with uniform codes incl. self.

    joint: 3/4 single, 3/16 two-row, 1/16 all-free-row proposals. The all-row
    component supplies irreducibility on positive finite-key support, but says
    nothing about practical mixing speed.
    """
    if kernel not in ("row","joint") or not len(free):
        raise ValueError("Nonempty free rows and registered symmetric kernel required")
    proposal = keys.copy()
    for index in range(len(keys)):
        draw = float(rng.random()) if kernel=="joint" else 0.
        size = 1 if draw<.75 else min(2,len(free)) if draw<.9375 else len(free)
        selected = rng.choice(free,size=size,replace=False)
        proposal[index,selected] = rng.integers(units,size=size)
    return proposal


def run_dictionary_smc(bridge, partial, *, particles=128, seed=0, stride=1,
                       mutations=2, kernel="joint", max_particle_bytes=32*1024**2,
                       observe=None):
    """Prior initialization -> weight -> multinomial resample -> invariant MH.

    Every stage resamples. Zero population likelihood returns explicit extinction;
    never refill. This estimates the supplied conditional law, not a full joint
    prefix likelihood. No gold key, source reading, or source length is an input.
    """
    partial = np.asarray(partial)
    if (partial.shape!=(bridge.rows,) or partial.dtype.kind not in "iu"
            or np.any(partial< -1) or np.any(partial>=len(bridge.units))
            or type(particles) is not int or not 1<=particles<=4096
            or type(seed) is not int or seed<0 or type(mutations) is not int or not 0<=mutations<=64
            or kernel not in ("row","joint") or type(max_particle_bytes) is not int or max_particle_bytes<1):
        raise ValueError("Bounded conditional dictionary/population/kernel required")
    stages = bridge_schedule(bridge.lengths,stride)
    if particles*(32*bridge.rows+128)+len(stages)*4096>max_particle_bytes:
        raise MemoryError("Particle/small trace allocation envelope exceeds cap")
    free = np.flatnonzero(partial<0)
    rng = np.random.default_rng(seed)
    keys = rng.integers(len(bridge.units),size=(particles,bridge.rows),dtype=np.int32)
    keys[:,partial>=0] = partial[partial>=0]
    old, log_evidence, trace = np.zeros(particles), 0., []
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
        keys, old = keys[parents].copy(),new[parents].copy()
        accepted = changed = proposals = 0
        for _ in range(mutations if len(free) else 0):
            candidate = propose(keys,free,len(bridge.units),rng,kernel)
            candidate_values = bridge.log_values(candidate,cuts,closed=closed)
            # Random uniforms are in [0,1); log(0) legitimately accepts any
            # positive ratio, but zero-target proposals must always be rejected.
            with np.errstate(divide="ignore"):
                log_uniform = np.log(rng.random(particles))
            accept = np.isfinite(candidate_values)&(log_uniform<np.minimum(0.,candidate_values-old))
            changed += int(np.count_nonzero(accept&np.any(candidate!=keys,axis=1)))
            accepted += int(np.count_nonzero(accept))
            proposals += particles
            keys[accept],old[accept] = candidate[accept],candidate_values[accept]
        row = {"stage":stage,"cuts":cuts,"closed":closed,"extinct":False,
            "positive_pre_resampling":int(np.count_nonzero(np.isfinite(new))),
            "incremental_ess":float(1/np.square(weights).sum()),
            "maximum_incremental_weight":float(weights.max()),
            "distinct_selected_parents":int(len(np.unique(parents))),
            "distinct_keys_after_mutation":int(len(np.unique(keys,axis=0))),
            "accepted_proposals":accepted,"changed_proposals":changed,"proposals":proposals,
            "increment_log_mean":log_total-math.log(particles),"log_evidence_estimate":log_evidence,
            "key_bank_sha256":hashlib.sha256(keys.tobytes()).hexdigest()}
        trace.append(row)
        if observe is not None:
            observe(row)
    return {"status":"complete_particles","log_evidence_estimate":log_evidence,
        "keys":keys,"key_log_likelihoods":old,"trace":trace,"tables":bridge.tables,"edges":bridge.edges,
        "posterior_is_finite_approximation":True}
