"""Censored observation prefixes through the unchanged native suffix scorer.

For an unclosed cut, matching units may cross it. Clamp their destination to the
cut, then remove the native scorer's terminal rho. Source rows/gotos are intact.
Only strictly positive dense source tables are supported by the legacy kernel.
"""

from __future__ import annotations

import ctypes
import itertools
import math

import numpy as np

from voynich.native_suffix_marginal import NativeMarginal, _Result
from voynich.source_key_particles import validate_source


class CensoredNativeBridge:
    """Exact conditional fixed-key bridges; pins existing immutable source arrays.

    max_work_bytes bounds owned bank/matches/native graph envelopes, excluding
    the already materialized pinned source. A process RSS cap remains separate.
    Source-state caps raise errors; they never produce zero or partial scores.
    """

    def __init__(self, cipher, native, *, glyphs=6, rho=1/225, offsets=None, contexts=None,
                 max_tables=1_000_000, max_edges=100_000_000,
                 max_nodes_per_record=300_000, max_edges_per_record=2_000_000,
                 max_work_bytes=128*1024**2):
        if not isinstance(native,NativeMarginal):
            raise ValueError("Verified native marginal scorer required")
        self.native = native
        self.p,self.goto = native.probabilities,native.transitions
        if (self.p.dtype!=np.float64 or self.goto.dtype!=np.uint32
                or not self.p.flags.c_contiguous or not self.goto.flags.c_contiguous
                or ctypes.cast(native.probability_pointer,ctypes.c_void_p).value!=self.p.ctypes.data
                or ctypes.cast(native.transition_pointer,ctypes.c_void_p).value!=self.goto.ctypes.data):
            raise ValueError("Native source arrays/pointers no longer agree")
        self._function = native.function
        self._probability_pointer,self._transition_pointer = native.probability_pointer,native.transition_pointer
        validate_source(self.p,self.goto)
        for start in range(0,len(self.p),4096):
            if np.any(self.p[start:start+4096]<=0):
                raise ValueError("Legacy native kernel requires strictly positive source rows")
        self.cipher = tuple(map(tuple,cipher))
        if (self.p.flags.writeable or self.goto.flags.writeable
                or type(glyphs) is not int or not 1<=glyphs<=6
                or not 1<=len(self.cipher)<=2 or not 1<=self.p.shape[1]<=23
                or any(not c or len(c)>4096 or any(type(g) is not int or not 0<=g<glyphs for g in c)
                       for c in self.cipher)
                or isinstance(rho,bool) or not math.isfinite(rho) or not 0<rho<1
                or any(type(v) is not int or not 1<=v<=2**63-1 for v in
                       (max_tables,max_edges,max_nodes_per_record,max_edges_per_record,max_work_bytes))):
            raise ValueError("Immutable bounded literal native bridge configuration required")
        self.offsets = (0,)*len(self.cipher) if offsets is None else tuple(offsets)
        self.contexts = (0,)*len(self.cipher) if contexts is None else tuple(contexts)
        if (len(self.offsets)!=len(self.cipher) or len(self.contexts)!=len(self.cipher)
                or any(type(o) is not int or not 0<=o<=len(c) for o,c in zip(self.offsets,self.cipher,strict=True))
                or any(type(c) is not int or not 0<=c<len(self.p) for c in self.contexts)):
            raise ValueError("Valid conditional offsets and source contexts required")
        self.rows,self.glyphs,self.rho = self.p.shape[1],glyphs,rho
        self.units = tuple(u for n in (1,2) for u in itertools.product(range(glyphs),repeat=n))
        self.lengths = tuple(len(c)-o for c,o in zip(self.cipher,self.offsets,strict=True))
        self.max_tables,self.max_edges = max_tables,max_edges
        self.max_nodes_per_record,self.max_edges_per_record = max_nodes_per_record,max_edges_per_record
        self.max_work_bytes = max_work_bytes
        self.tables = self.edges = self.native_calls = self.nodes = self.maximum_native_nodes = 0
        self.source_pinned_bytes = self.p.nbytes+self.goto.nbytes
        self.maximum_owned_work_envelope = 0

    def _record(self,key,text,context,closed,reserved):
        if not text:
            return 0.
        # Includes Python match lists/integers, converted NumPy arrays, native
        # layer headers and a conservative cumulative-node allocation envelope.
        envelope = reserved+128*(len(text)+1)*self.rows+128*(len(text)+1)+256*self.max_nodes_per_record
        if envelope>self.max_work_bytes:
            raise MemoryError("Native bridge match/graph allocation envelope exceeds cap")
        self.maximum_owned_work_envelope = max(self.maximum_owned_work_envelope,envelope)
        if self.edges>=self.max_edges:
            raise RuntimeError("Native bridge total edge cap exhausted")
        offsets,letters,ends = [0],[],[]
        for pos in range(len(text)):
            for row,code in enumerate(key):
                unit = self.units[int(code)]
                end = pos+len(unit)
                visible = min(len(unit),len(text)-pos)
                if (closed and end>len(text)) or unit[:visible]!=text[pos:pos+visible]:
                    continue
                letters.append(row)
                ends.append(min(end,len(text)))
            offsets.append(len(letters))
        offsets = np.array(offsets,dtype=np.uint64)
        letters,ends = np.array(letters,dtype=np.uint32),np.array(ends,dtype=np.uint32)
        output,error = _Result(),ctypes.create_string_buffer(256)
        edge_cap = min(self.max_edges_per_record,self.max_edges-self.edges)
        code = self._function(self._probability_pointer,self._transition_pointer,
            len(self.p),self.rows,context,offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            letters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ends.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(text),len(letters),self.rho,self.max_nodes_per_record,edge_cap,
            ctypes.byref(output),error,len(error))
        self.native_calls += 1
        self.edges += int(output.edges)
        self.nodes += int(output.nodes)
        self.maximum_native_nodes = max(self.maximum_native_nodes,int(output.nodes))
        if code:
            exception = {1:ValueError,2:RuntimeError,3:MemoryError,4:ArithmeticError}.get(code,RuntimeError)
            raise exception(error.value.decode("utf-8",errors="replace"))
        return float(output.log_likelihood)-(0. if closed else math.log(self.rho))

    def log_values(self,keys,cuts,*,closed=False):
        cuts = tuple(cuts)
        keys = np.asarray(keys)
        if (type(closed) is not bool or len(cuts)!=len(self.lengths)
                or any(type(c) is not int or not 0<=c<=n for c,n in zip(cuts,self.lengths,strict=True))
                or (closed and cuts!=self.lengths)
                or keys.ndim!=2 or keys.shape[1]!=self.rows or not 1<=len(keys)<=4096
                or keys.dtype.kind not in "iu" or np.any(keys<0) or np.any(keys>=len(self.units))):
            raise ValueError("Bounded complete key bank and prefix/final cuts required")
        if self.tables+len(keys)>self.max_tables:
            raise RuntimeError("Native bridge dictionary-table cap exhausted")
        reserved = keys.nbytes+8*len(keys)+4096
        if reserved>self.max_work_bytes:
            raise MemoryError("Native bridge bank allocation exceeds work cap")
        self.tables += len(keys)
        result = np.zeros(len(keys))
        for index,key in enumerate(keys):
            for text,offset,cut,context,remaining in zip(
                    self.cipher,self.offsets,cuts,self.contexts,self.lengths,strict=True):
                if not remaining or not cut:
                    continue  # Original EOS paid, or no observation yet.
                result[index] += self._record(key,text[offset:offset+cut],context,closed,reserved)
        return result
