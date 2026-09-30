"""Sparse numeric storage for the same support-filtered suffix probabilities.

Unlike the string-dictionary source this stores only nonzero successor counts;
selection can query observed letters without materializing every probability.
"""
from __future__ import annotations

import math

import numpy as np

from voynich.recurrent_latin_source import chunks, encode


class CompactSuffixSource:
    def __init__(self, alphabet, levels):
        self.alphabet = alphabet
        self.levels = levels
        if not alphabet or len(set(alphabet)) != len(alphabet) or len(levels) > 13 or not levels:
            raise ValueError('Invalid compact source')
        self.validate()

    def validate(self):
        for depth, level in enumerate(self.levels):
            if set(level) != {'contexts', 'totals', 'joints', 'frequencies'}:
                raise ValueError('Invalid level fields')
            contexts, totals, joints, frequencies = (level[k] for k in
                ('contexts', 'totals', 'joints', 'frequencies'))
            if (any(a.dtype != np.uint64 or a.ndim != 1 for a in level.values())
                    or len(contexts) != len(totals) or len(joints) != len(frequencies)
                    or np.any(contexts[1:] <= contexts[:-1]) or np.any(joints[1:] <= joints[:-1])
                    or np.any(totals == 0) or np.any(frequencies == 0)
                    or np.any(contexts >= len(self.alphabet)**depth)
                    or np.any(joints >= len(self.alphabet)**(depth + 1))):
                raise ValueError('Invalid sparse numeric arrays')
            if len(joints):
                prefix = joints // np.uint64(len(self.alphabet))
                starts = np.r_[0, np.flatnonzero(prefix[1:] != prefix[:-1]) + 1]
                if (not np.array_equal(prefix[starts], contexts)
                        or not np.array_equal(np.add.reduceat(frequencies, starts), totals)):
                    raise ValueError('Context totals disagree with successors')
            elif len(contexts):
                raise ValueError('Contexts without successors')
            if depth and len(contexts):
                lower = self.levels[depth - 1]
                for keys in (contexts // np.uint64(len(self.alphabet)),
                             contexts % np.uint64(len(self.alphabet)**(depth - 1))):
                    if np.any(lookup(lower['contexts'], lower['totals'], keys) == 0):
                        raise ValueError('Context closure failed')
        if not np.array_equal(self.levels[0]['contexts'], np.array([0], dtype=np.uint64)):
            raise ValueError('Nonempty root required')

    def score(self, records, taus, length=512):
        taus = np.asarray(taus, dtype=np.float64)
        if taus.ndim != 1 or not len(taus) or np.any(~np.isfinite(taus)) or np.any(taus <= 0):
            raise ValueError('Positive finite masses required')
        parts = [encode(s, self.alphabet).astype(np.uint64) for _, s in chunks(records, length)]
        if not parts:
            raise ValueError('No selection letters')
        targets = np.concatenate(parts)
        positions = np.concatenate([np.arange(len(p)) for p in parts])
        root = self.levels[0]
        root_counts = lookup(root['joints'], root['frequencies'], targets)
        probabilities = np.repeat(((root_counts + .5) / (int(root['totals'][0]) + .5 * len(self.alphabet)))[None, :],
                                  len(taus), axis=0)
        contexts = np.zeros(len(targets), dtype=np.uint64)
        for depth, level in enumerate(self.levels[1:], 1):
            valid = positions >= depth
            preceding = np.zeros(len(targets), dtype=np.uint64)
            if depth < len(targets):
                preceding[depth:] = targets[:-depth]
            contexts = np.where(valid, preceding * np.uint64(len(self.alphabet)**(depth - 1)) + contexts, 0)
            totals = lookup(level['contexts'], level['totals'], contexts) * valid
            successors = lookup(level['joints'], level['frequencies'],
                                contexts * np.uint64(len(self.alphabet)) + targets) * valid
            probabilities = (successors[None, :] + taus[:, None] * probabilities) / (totals[None, :] + taus[:, None])
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0) or np.any(probabilities > 1):
            raise ValueError('Invalid score probabilities')
        return [{'tau': float(tau), 'selection_bits': math.fsum((-np.log2(row)).tolist()),
                 'characters': len(targets)} for tau, row in zip(taus, probabilities)]

    def save(self, path):
        arrays = {f'{depth}_{key}': value for depth, level in enumerate(self.levels) for key, value in level.items()}
        with path.open('xb') as handle:
            np.savez_compressed(handle, alphabet=np.array(self.alphabet), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            alphabet = str(archive['alphabet'])
            count = (len(archive.files) - 1) // 4
            expected = {'alphabet'} | {f'{d}_{k}' for d in range(count) for k in
                                      ('contexts', 'totals', 'joints', 'frequencies')}
            if set(archive.files) != expected:
                raise ValueError('Archive fields changed')
            levels = [{k: archive[f'{d}_{k}'] for k in ('contexts', 'totals', 'joints', 'frequencies')}
                      for d in range(count)]
        return cls(alphabet, levels)


def lookup(keys, values, query):
    query = np.asarray(query, dtype=np.uint64)
    if not len(keys):
        return np.zeros(query.shape, dtype=np.uint64)
    indices = np.searchsorted(keys, query)
    valid = indices < len(keys)
    indices = np.minimum(indices, len(keys) - 1)
    return np.where(valid & (keys[indices] == query), values[indices], 0)


def fit_compact(records, alphabet, order=12, minimum=4, max_contexts=8_000_000, max_array_bytes=2 * 1024**3):
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or type(order) is not int or not 0 <= order <= 12
            or type(minimum) is not int or minimum < 1
            or type(max_contexts) is not int or max_contexts < 1
            or len(alphabet)**(order + 1) > 2**64):
        raise ValueError('Invalid compact fit settings')
    arrays = [encode(r, alphabet).astype(np.uint64) for r in records]
    if not arrays:
        raise ValueError('Nonempty records required')
    levels, contexts_so_far, bytes_so_far = [], 0, 0
    for depth in range(order + 1):
        codes = []
        for row in arrays:
            n = len(row) - depth
            if n <= 0:
                continue
            values = np.zeros(n, dtype=np.uint64)
            for i in range(depth + 1):
                values = values * np.uint64(len(alphabet)) + row[i:i + n]
            codes.append(values)
        if codes:
            joints, frequencies = np.unique(np.concatenate(codes), return_counts=True)
            frequencies = frequencies.astype(np.uint64)
            prefix = joints // np.uint64(len(alphabet))
            starts = np.r_[0, np.flatnonzero(prefix[1:] != prefix[:-1]) + 1]
            totals = np.add.reduceat(frequencies, starts)
            keep = np.ones(len(starts), dtype=bool) if not depth else totals >= minimum
            keep_joints = np.repeat(keep, np.diff(np.r_[starts, len(joints)]))
            level = {'contexts': prefix[starts[keep]], 'totals': totals[keep],
                     'joints': joints[keep_joints], 'frequencies': frequencies[keep_joints]}
            del codes, joints, frequencies, prefix, starts, totals, keep, keep_joints
        else:
            level = {key: np.array([], dtype=np.uint64) for key in ('contexts', 'totals', 'joints', 'frequencies')}
        contexts_so_far += len(level['contexts'])
        bytes_so_far += sum(a.nbytes for a in level.values())
        if contexts_so_far > max_contexts or bytes_so_far > max_array_bytes:
            raise RuntimeError('Compact source fixed resource cap exceeded')
        levels.append(level)
    return CompactSuffixSource(alphabet, levels)
