"""Materialize the unchanged compact source as float64 rows and uint32 gotos.

The retained-context set is prefix/suffix closed. A state's failure state drops
its first letter; a missing child uses that failure state's precomputed goto.
This changes source lookup cost, not counts, smoothing, support or decoding.
"""
from __future__ import annotations

import math

import numpy as np

from voynich.compact_suffix_adapter import CompactSuffixAdapter


class DenseSuffixAdapter:
    def __init__(self, compact, tau, *, max_array_bytes=768 * 1024**2, block_rows=4096):
        if (isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0
                or type(max_array_bytes) is not int or max_array_bytes < 1
                or type(block_rows) is not int or block_rows < 1):
            raise ValueError('Invalid dense source settings')
        self.compact, self.tau = compact, tau
        self.alphabet, self.order = tuple(compact.alphabet), len(compact.levels) - 1
        self.reference = CompactSuffixAdapter(compact, tau, cache_size=1)
        offsets = self.reference.offsets
        size, width = int(self.reference.ends[-1]), len(self.alphabet)
        required = size * width * 12
        if size > np.iinfo(np.uint32).max or required > max_array_bytes:
            raise MemoryError('Dense source fixed allocation cap exceeded')
        self.transitions = np.zeros((size, width), dtype=np.uint32)
        self.probabilities = np.zeros((size, width), dtype=np.float64)
        for depth, level in enumerate(compact.levels):
            start, count = int(offsets[depth]), len(level['contexts'])
            if not count:
                continue
            totals = level['totals']
            if depth == 0:
                self.probabilities[0] = .5
            else:
                lower = compact.levels[depth - 1]['contexts']
                for lo in range(0, count, block_rows):
                    hi = min(count, lo + block_rows)
                    suffixes = level['contexts'][lo:hi] % np.uint64(width**(depth - 1))
                    indices = np.searchsorted(lower, suffixes)
                    if np.any(indices >= len(lower)) or not np.array_equal(lower[indices], suffixes):
                        raise ValueError('Missing retained suffix state')
                    failure = indices + int(offsets[depth - 1])
                    self.probabilities[start + lo:start + hi] = tau * self.probabilities[failure]
                    self.transitions[start + lo:start + hi] = self.transitions[failure]
            # Each (context, successor) is unique in the sparse count archive.
            joint = level['joints']
            row_indices = np.searchsorted(level['contexts'], joint // np.uint64(width)) + start
            letters = (joint % np.uint64(width)).astype(np.intp)
            self.probabilities[row_indices, letters] += level['frequencies']
            denominator = totals.astype(np.float64) + (width * .5 if depth == 0 else tau)
            self.probabilities[start:start + count] /= denominator[:, None]
            if depth < self.order:
                children = compact.levels[depth + 1]['contexts']
                parents = np.searchsorted(level['contexts'], children // np.uint64(width)) + start
                letters = (children % np.uint64(width)).astype(np.intp)
                self.transitions[parents, letters] = (np.arange(len(children), dtype=np.uint32)
                                                      + np.uint32(offsets[depth + 1]))
        for lo in range(0, size, block_rows):
            rows = self.probabilities[lo:lo + block_rows]
            if (np.any(~np.isfinite(rows)) or np.any(rows <= 0) or np.any(rows > 1)
                    or not np.allclose(rows.sum(axis=1), 1., atol=1e-12, rtol=0)):
                raise ValueError('Invalid dense probability rows')
        self.probabilities.setflags(write=False)
        self.transitions.setflags(write=False)
        self.array_bytes = self.probabilities.nbytes + self.transitions.nbytes

    def state(self, history):
        return self.reference.state(history)

    def row(self, state):
        if type(state) is not int or not 0 <= state < len(self.probabilities):
            raise ValueError('State outside retained contexts')
        return self.probabilities[state]

    def step(self, state, letter):
        if (type(state) is not int or not 0 <= state < len(self.transitions)
                or type(letter) is not int or not 0 <= letter < len(self.alphabet)):
            raise ValueError('State or letter outside source')
        return int(self.transitions[state, letter])
