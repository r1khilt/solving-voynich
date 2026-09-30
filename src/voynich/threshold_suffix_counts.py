"""Exact per-context counts with a predeclared minimum support, using uint64 codes.

Filtering is based on context totals, never individual successor counts. This
preserves prefix and suffix closure, including contexts starting with zero IDs.
"""
from __future__ import annotations

import numpy as np

from voynich.recurrent_latin_source import encode
from voynich.sparse_suffix_source import MAX_CONTEXTS, MAX_ORDER


def collect_threshold_counts(records, alphabet, order=12, minimum=4, max_contexts=MAX_CONTEXTS):
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)
            or type(order) is not int or not 0 <= order <= MAX_ORDER
            or type(minimum) is not int or minimum < 1
            or type(max_contexts) is not int or max_contexts < 1
            or len(alphabet) ** (order + 1) > 2**64):
        raise ValueError('Invalid bounded count settings')
    arrays = [encode(r, alphabet).astype(np.uint64) for r in records]
    if not arrays:
        raise ValueError('Nonempty records required')
    base, result = len(alphabet), {}
    for depth in range(order + 1):
        codes = []
        for row in arrays:
            size = len(row) - depth
            if size <= 0:
                continue
            value = np.zeros(size, dtype=np.uint64)
            for pos in range(depth + 1):
                value *= np.uint64(base)
                value += row[pos:pos + size]
            codes.append(value)
        if not codes:
            continue
        unique, frequencies = np.unique(np.concatenate(codes), return_counts=True)
        del codes
        prefixes = unique // np.uint64(base)
        starts = np.r_[0, np.flatnonzero(prefixes[1:] != prefixes[:-1]) + 1]
        ends = np.r_[starts[1:], len(unique)]
        totals = np.add.reduceat(frequencies, starts)
        for start, end, total in zip(starts, ends, totals):
            if depth and total < minimum:
                continue
            code = int(prefixes[start])
            letters = [''] * depth
            for j in range(depth - 1, -1, -1):
                code, value = divmod(code, base)
                letters[j] = alphabet[value]
            context = ''.join(letters)
            if len(result) >= max_contexts:
                raise RuntimeError('Fixed context cap exceeded; do not change cutoff')
            result[context] = {alphabet[int(v % np.uint64(base))]: int(n)
                               for v, n in zip(unique[start:end], frequencies[start:end])}
    return result
