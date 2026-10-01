"""Shared-dictionary IID future sums; source histories integrated by fixed-key DP.

This preserves unknown row reuse but relaxes the context-dependent source law.
No floor, smoothing or finite-bank value is an evidence/support certificate.
"""

from __future__ import annotations

import math

import numpy as np


class SharedDictionaryIid:
    def __init__(self, cipher, row, *, glyphs=6, rho=1/225, epsilon=1e-8,
                 max_tables=100_000, max_work_bytes=128*1024**2):
        self.cipher = tuple(map(tuple, cipher))
        p = np.asarray(row, dtype=np.float64)
        if (p.ndim != 1 or not 1 <= len(p) <= 23 or np.any(~np.isfinite(p))
                or np.any(p < 0) or abs(float(p.sum())-1) > 1e-12
                or type(glyphs) is not int or not 1 <= glyphs <= 6
                or not 1 <= len(self.cipher) <= 2
                or any(not c or len(c)>4096 or any(type(g) is not int or not 0<=g<glyphs for g in c) for c in self.cipher)
                or isinstance(rho, bool) or not math.isfinite(rho) or not 0<rho<1
                or isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0<=epsilon<=1e-3
                or any(type(v) is not int or v<1 for v in (max_tables,max_work_bytes))):
            raise ValueError("Bounded normalized IID source and observed records required")
        self.row = (p+epsilon)/(1+len(p)*epsilon)
        self.glyphs, self.units = glyphs, glyphs+glyphs**2
        self.rho, self.epsilon = rho, epsilon
        self.max_tables, self.tables = max_tables, 0
        self.max_work_bytes = max_work_bytes

    def _validate(self, partial, offsets, bank):
        key = np.asarray(partial)
        bank = np.asarray(bank)
        if (key.shape != self.row.shape or key.dtype.kind not in "iu"
                or np.any(key < -1) or np.any(key >= self.units)
                or len(offsets) != len(self.cipher)
                or any(type(n) is not int or not 0<=n<=len(c) for n,c in zip(offsets,self.cipher,strict=True))
                or bank.ndim != 2 or bank.shape[1] != len(key) or not 1<=len(bank)<=4096
                or bank.dtype.kind not in "iu" or np.any(bank<0) or np.any(bank>=self.units)):
            raise ValueError("Valid partial dictionary, offsets and bounded full dictionary bank required")
        return key, bank

    def _fixed_logs(self, keys, offsets, reserved_bytes=0):
        """One whole dictionary per row, reused across every future occurrence."""
        n = len(keys)
        if self.tables+n>self.max_tables:
            raise RuntimeError("Declared shared-dictionary table work exhausted")
        # All arrays here are <=64 rows; conservative scratch-array envelope.
        envelope = 8*n*(2*len(self.row)+4*self.units+4*(max(map(len,self.cipher))+2)+32)
        if envelope+reserved_bytes>self.max_work_bytes:
            raise MemoryError("Fixed-key DP scratch allocation envelope exceeds cap")
        self.tables += n
        weights = np.zeros((n,self.units),dtype=np.float64)
        np.add.at(weights,(np.arange(n)[:,None],keys),self.row[None,:])
        with np.errstate(divide="ignore"):
            weights = np.log(weights)+math.log1p(-self.rho)
        values = np.zeros(n,dtype=np.float64)
        for cipher, offset in zip(self.cipher,offsets,strict=True):
            if offset==len(cipher):
                continue  # Its EOS was already included before this state.
            dp = np.full((n,len(cipher)+2),-math.inf)
            dp[:,len(cipher)] = math.log(self.rho)
            for pos in range(len(cipher)-1,offset-1,-1):
                single = weights[:,cipher[pos]]+dp[:,pos+1]
                dual = (weights[:,self.glyphs+self.glyphs*cipher[pos]+cipher[pos+1]]+dp[:,pos+2]
                        if pos+1<len(cipher) else -math.inf)
                dp[:,pos] = np.logaddexp(single,dual)
            values += dp[:,offset]
        return values

    def estimate(self, partial, offsets, bank, *, integrate_row=None):
        """Average raw likelihoods, optionally integrate one free row exactly.

        A uniform iid bank gives an unbiased arithmetic likelihood estimate for
        THIS IID surrogate. User-supplied deterministic banks have no automatic
        sampling guarantee. Rao-Blackwell averaging groups a fixed base sample.
        """
        key, bank = self._validate(partial,offsets,bank)
        if integrate_row is not None and (type(integrate_row) is not int
                or not 0<=integrate_row<len(key) or key[integrate_row]>=0):
            raise ValueError("Only an unknown dictionary row may be integrated")
        span = self.units if integrate_row is not None else 1
        logical = len(bank)*span
        envelope = bank.nbytes+8*len(bank)*(span+len(key)+8)
        if envelope>self.max_work_bytes:
            raise MemoryError("Sample bank/result allocation envelope exceeds cap")
        if all(offset==len(c) for offset,c in zip(offsets,self.cipher,strict=True)):
            return {"log_mean":0.,"base_samples":len(bank),"integrated_row":integrate_row,
                "fixed_dictionary_tables":0,"positive_fixed_dictionaries":logical,
                "positive_base_groups":len(bank),"likelihood_contribution_ess":float(len(bank)),
                "maximum_base_contribution_share":1/len(bank),"base_log_likelihoods":[0.]*len(bank),
                "source_is_iid_surrogate":True,"epsilon":self.epsilon,"zero_is_not_target_impossibility":True}
        if self.tables+logical>self.max_tables:
            raise RuntimeError("Declared whole-call table work exhausted")
        values = np.empty(logical,dtype=np.float64)
        began = self.tables
        for start in range(0,logical,64):
            index = np.arange(start,min(start+64,logical))
            keys = np.where(key[None,:]>=0,key[None,:],bank[index//span]).copy()
            if integrate_row is not None:
                keys[:,integrate_row] = index%span
            values[index] = self._fixed_logs(keys,offsets,reserved_bytes=envelope)
        grouped = np.logaddexp.reduce(values.reshape(len(bank),span),axis=1)-math.log(span)
        total = float(np.logaddexp.reduce(grouped))
        if total==-math.inf:
            estimate, ess, maximum_share = None, 0., 0.
        else:
            estimate = total-math.log(len(bank))
            shares = np.exp(grouped-total)
            ess, maximum_share = float(1/np.square(shares).sum()),float(shares.max())
        return {"log_mean":estimate,"base_samples":len(bank),"integrated_row":integrate_row,
            "fixed_dictionary_tables":self.tables-began,
            "positive_fixed_dictionaries":int(np.count_nonzero(np.isfinite(values))),
            "positive_base_groups":int(np.count_nonzero(np.isfinite(grouped))),
            "likelihood_contribution_ess":ess,"maximum_base_contribution_share":maximum_share,
            "base_log_likelihoods":[float(v) if math.isfinite(v) else None for v in grouped],
            "source_is_iid_surrogate":True,"epsilon":self.epsilon,"zero_is_not_target_impossibility":True}
