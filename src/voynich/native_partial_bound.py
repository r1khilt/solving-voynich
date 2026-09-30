"""Relaxed match lists through the already verified exact native forward kernel.

The ABI sums arbitrary forward letter-match edges. Supplying every legal unit
for each unassigned row implements the nonnormalized relaxation without changing
source rows or native code. Source, ABI and binary remain hash-checked.
"""
import ctypes
import math

import numpy as np

from voynich.native_suffix_marginal import NativeMarginal, _Result, settings
from voynich.partial_unit_bound import family


class NativePartialBound:
    def __init__(self, source, build):
        self.native = NativeMarginal(source, build)
        # DenseSuffixAdapter validates this normally; also reject replacement
        # arrays on an object whose class is still DenseSuffixAdapter. Readonly
        # NumPy arrays remain a caller contract, not protection from malicious
        # mutation after construction.
        values = self.native.probabilities
        for start in range(0, len(values), 4096):
            rows = values[start:start+4096]
            if (np.any(~np.isfinite(rows)) or np.any(rows <= 0) or np.any(rows > 1)
                    or not np.allclose(rows.sum(axis=1), 1., rtol=0., atol=1e-12)):
                raise ValueError('Relaxation requires a normalized positive dense source')

    def record(self, partial, pool, observed, rho, *, max_nodes=500_000, max_edges=2_000_000):
        partial, pool = family(partial, pool)
        native = self.native
        settings(native, [pool[0] if u is None else u for u in partial], observed, rho, max_nodes, max_edges)
        options = [pool if unit is None else (unit,) for unit in partial]
        offsets, letters, ends = [0], [], []
        for offset in range(len(observed)):
            for letter, units in enumerate(options):
                for unit in units:
                    if observed.startswith(unit, offset):
                        letters.append(letter)
                        ends.append(offset+len(unit))
            offsets.append(len(letters))
        offsets = np.array(offsets, dtype=np.uint64)
        letters, ends = np.array(letters, dtype=np.uint32), np.array(ends, dtype=np.uint32)
        result, error = _Result(), ctypes.create_string_buffer(256)
        code = native.function(native.probability_pointer, native.transition_pointer,
            len(native.probabilities), len(native.alphabet), native.root,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            letters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), ends.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(observed), len(letters), rho, max_nodes, max_edges, ctypes.byref(result), error, len(error))
        if code:
            exception = {1: ValueError, 2: RuntimeError, 3: MemoryError, 4: ArithmeticError}.get(code, RuntimeError)
            raise exception(error.value.decode('utf-8', errors='replace'))
        return {'log_likelihood_upper_bound': None if result.log_likelihood == -math.inf else result.log_likelihood,
                'nodes': result.nodes, 'edges': result.edges, 'normalized_channel': False}
