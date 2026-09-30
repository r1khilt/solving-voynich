"""Lazy finite-state adapter for exact decoding of the compact numeric source.

Retained contexts have IDs assigned by depth and numeric code, so leading-zero
letters never collapse histories of different lengths. No dense source table.
"""
from functools import lru_cache
import math

import numpy as np

from voynich.compact_suffix_source import lookup


class _ProbabilityRows:
    def __init__(self, source):
        self.source = source

    def __getitem__(self, key):
        state, letter = key
        return self.source.row(state)[letter]


class CompactSuffixAdapter:
    def __init__(self, compact, tau, cache_size=200000):
        if not math.isfinite(tau) or tau <= 0 or isinstance(tau, bool):
            raise ValueError('Positive finite mass required')
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError('Positive bounded cache required')
        self.compact, self.tau = compact, tau
        self.alphabet, self.order = tuple(compact.alphabet), len(compact.levels) - 1
        self.ends = np.cumsum([len(level['contexts']) for level in compact.levels])
        self.offsets = np.r_[0, self.ends[:-1]]
        self.row = lru_cache(maxsize=cache_size)(self._row)
        self.step = lru_cache(maxsize=cache_size)(self._step)
        self.probabilities = _ProbabilityRows(self)

    def _parts(self, state):
        if type(state) is not int or not 0 <= state < self.ends[-1]:
            raise ValueError('State outside retained contexts')
        depth = int(np.searchsorted(self.ends, state, side='right'))
        index = state - int(self.offsets[depth])
        return depth, index, int(self.compact.levels[depth]['contexts'][index])

    def _find(self, depth, code):
        while depth:
            keys = self.compact.levels[depth]['contexts']
            index = int(np.searchsorted(keys, np.uint64(code)))
            if index < len(keys) and int(keys[index]) == code:
                return int(self.offsets[depth]) + index
            depth -= 1
            code %= len(self.alphabet)**depth
        return 0

    def state(self, history):
        if not isinstance(history, str) or set(history) - set(self.alphabet):
            raise ValueError('History outside alphabet')
        history = history[-self.order:] if self.order else ''
        code = 0
        for char in history:
            code = code * len(self.alphabet) + self.alphabet.index(char)
        return self._find(len(history), code)

    def _step(self, state, letter):
        if type(letter) is not int or not 0 <= letter < len(self.alphabet):
            raise ValueError('Letter outside alphabet')
        depth, _, code = self._parts(state)
        depth = min(depth + 1, self.order)
        code = (code * len(self.alphabet) + letter) % (len(self.alphabet)**depth)
        return self._find(depth, code)

    def _row(self, state):
        depth, index, code = self._parts(state)
        level = self.compact.levels[depth]
        queries = np.uint64(code * len(self.alphabet)) + np.arange(len(self.alphabet), dtype=np.uint64)
        counts = lookup(level['joints'], level['frequencies'], queries).astype(np.float64)
        total = int(level['totals'][index])
        if not depth:
            probabilities = (counts + .5) / (total + .5 * len(self.alphabet))
        else:
            lower = self._find(depth - 1, code % (len(self.alphabet)**(depth - 1)))
            probabilities = (counts + self.tau * self.row(lower)) / (total + self.tau)
        if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0):
            raise ValueError('Invalid lazy probability row')
        probabilities.setflags(write=False)
        return probabilities
