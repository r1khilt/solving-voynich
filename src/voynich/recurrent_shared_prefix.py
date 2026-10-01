"""Full-history neural text proposals with one shared dictionary across records.

Depth beam search optimizes one text tuple's key-marginal score, not evidence.
Only identical complete record prefixes share source transitions. Every pruned
branch retains an upper bound; resource exhaustion raises instead of pruning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def logsum(values):
    values = tuple(values)
    if not values:
        return -math.inf
    high = max(values)
    return high + math.log(math.fsum(math.exp(v-high) for v in values))


def indices(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length()-1
        mask ^= low


@dataclass(frozen=True)
class Prefix:
    completed: tuple
    text: str
    groups: tuple
    score: float


class SharedUnitIndex:
    """Literal key masks and source-independent minimum completion lengths."""
    def __init__(self, alphabet, keys, records, weights, rho, *, max_channel_cells):
        self.alphabet, self.keys, self.records, self.weights = tuple(alphabet), tuple(map(tuple, keys)), tuple(records), tuple(weights)
        if (not self.alphabet or len(set(self.alphabet)) != len(self.alphabet)
                or any(not isinstance(c, str) or len(c) != 1 for c in self.alphabet)
                or not self.keys
                or any(len(k) != len(self.alphabet) or any(not isinstance(u, str) or not u for u in k) for k in self.keys)
                or len(set(self.keys)) != len(self.keys)
                or not self.records or any(not isinstance(r, str) for r in self.records)
                or len(self.weights) != len(self.keys) or any(not math.isfinite(w) for w in self.weights)
                or abs(logsum(self.weights)) > 1e-10
                or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
                or type(max_channel_cells) is not int or max_channel_cells < 1):
            raise ValueError('Invalid fixed shared-prefix channel, weights or length law')
        self.channel_cells = len(self.keys)*sum(len(r)+1 for r in self.records)
        if self.channel_cells > max_channel_cells:
            raise RuntimeError('Fixed channel-cell cap exceeded')
        self.stop, self.cont = len(self.records)*math.log(rho), math.log1p(-rho)
        self.unit_masks = []
        for letter in range(len(self.alphabet)):
            row = {}
            for k, key in enumerate(self.keys):
                row[key[letter]] = row.get(key[letter], 0) | (1 << k)
            self.unit_masks.append(tuple(row.items()))
        self.minimum = []
        for record in self.records:
            rows = []
            for key in self.keys:
                distances = [None]*len(record)+[0]
                for offset in range(len(record)-1, -1, -1):
                    options = [1+distances[offset+len(unit)] for unit in set(key)
                               if record.startswith(unit, offset) and distances[offset+len(unit)] is not None]
                    if options:
                        distances[offset] = min(options)
                rows.append(tuple(distances))
            self.minimum.append(tuple(rows))
        self.future = []
        for r in range(len(self.records)):
            self.future.append(tuple(None if any(self.minimum[j][k][0] is None for j in range(r+1, len(self.records)))
                                     else sum(self.minimum[j][k][0] for j in range(r+1, len(self.records)))
                                     for k in range(len(self.keys))))

    def viable(self, record, groups):
        return tuple((offset, sum(1 << k for k in indices(mask)
                                  if self.minimum[record][k][offset] is not None and self.future[record][k] is not None))
                     for offset, mask in groups
                     if any(self.minimum[record][k][offset] is not None and self.future[record][k] is not None for k in indices(mask)))

    def bounds(self, prefix):
        record = len(prefix.completed)
        groups = self.viable(record, prefix.groups)
        basic = logsum(self.weights[k] for _, mask in groups for k in indices(mask))
        remaining = logsum(self.weights[k]+(self.minimum[record][k][offset]+self.future[record][k])*self.cont
                           for offset, mask in groups for k in indices(mask))
        return prefix.score+self.stop+basic, prefix.score+self.stop+remaining

    def extend(self, record, groups, letter):
        following = {}
        observed = self.records[record]
        for offset, active in groups:
            if offset == len(observed):
                continue
            for unit, matching in self.unit_masks[letter]:
                mask = active & matching
                if mask and observed.startswith(unit, offset):
                    end = offset+len(unit)
                    following[end] = following.get(end, 0) | mask
        return self.viable(record, tuple(sorted(following.items())))


def decode_shared_prefix(provider, keys, records, rho, *, log_weights=None, beam_width=128,
                         max_expanded=200_000, max_source_prefixes=200_000,
                         max_channel_cells=10_000_000, numerical_margin=1e-9, observe=None):
    """Bounded text MAP within a fixed bank; source must reset at each record."""
    keys = tuple(map(tuple, keys))
    if (type(beam_width) is not int or beam_width < 1 or type(max_expanded) is not int or max_expanded < 1
            or type(max_source_prefixes) is not int or max_source_prefixes < 1
            or not math.isfinite(numerical_margin) or numerical_margin < 0):
        raise ValueError('Invalid shared-prefix search limits')
    weights = tuple(log_weights) if log_weights is not None else tuple(-math.log(len(keys)) for _ in keys)
    channel = SharedUnitIndex(provider.alphabet, keys, records, weights, rho, max_channel_cells=max_channel_cells)
    cache, source_calls, requests, expanded, proposed, discarded, maximum = {}, 0, 0, 0, 1, 0, 0
    pruned_bound = -math.inf
    best = None
    start = channel.viable(0, ((0, (1 << len(keys))-1),))
    frontier = [Prefix((), '', start, 0.)] if start else []
    for depth in range(sum(map(len, channel.records))+1):
        # Complete all zero-letter boundaries, retaining both end and extend branches.
        queue, opened = list(frontier), []
        while queue:
            row = queue.pop()
            r = len(row.completed)
            if r == len(channel.records):
                mask = row.groups[0][1]
                score = row.score+channel.stop+logsum(weights[k] for k in indices(mask))
                candidate = (score, row.completed, mask)
                if best is None or score > best[0] or (score == best[0] and row.completed < best[1]):
                    best = candidate
                continue
            groups = channel.viable(r, row.groups)
            end = dict(groups).get(len(channel.records[r]), 0)
            if end:
                following = Prefix(row.completed+(row.text,), '', ((0, end),), row.score)
                queue.append(following)
                proposed += 1
            unfinished = tuple((offset, mask) for offset, mask in groups if offset < len(channel.records[r]))
            if unfinished:
                opened.append(Prefix(row.completed, row.text, unfinished, row.score))
        if not opened:
            break
        if depth == sum(map(len, channel.records)):
            raise AssertionError('Nonempty emissions must bound total source depth')
        ranked = [(channel.bounds(row), row) for row in opened]
        ranked.sort(key=lambda value: (-value[0][1], value[1].completed, value[1].text, value[1].groups))
        maximum = max(maximum, len(ranked))
        for i, ((basic, upper), row) in enumerate(ranked):
            retained = i < beam_width
            if not retained:
                discarded += 1
                pruned_bound = max(pruned_bound, upper)
            if observe is not None:
                observe({'depth': depth, 'completed': row.completed, 'text': row.text, 'groups': row.groups,
                         'source_log_score': row.score, 'basic_upper_bound': basic, 'upper_bound': upper,
                         'retained': retained})
        retained = [row for _, row in ranked[:beam_width]]
        expanded += len(retained)
        if expanded > max_expanded:
            raise RuntimeError('Fixed shared-prefix expansion cap exceeded')
        requests += len(retained)
        missed = tuple(dict.fromkeys(row.text for row in retained if row.text not in cache))
        if len(cache)+len(missed) > max_source_prefixes:
            raise RuntimeError('Fixed source-prefix cache cap exceeded')
        batches = [(text,) for text in missed if not text]+[tuple(text for text in missed if text)]
        for texts in batches:
            if not texts:
                continue
            tokens = [len(channel.alphabet) if not text else channel.alphabet.index(text[-1]) for text in texts]
            states = [None if not text else cache[text[:-1]][1] for text in texts]
            probabilities, following = provider.advance(tokens, states)
            source_calls += 1
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if (probabilities.shape != (len(texts), len(channel.alphabet)) or len(following) != len(texts)
                    or np.any(~np.isfinite(probabilities)) or np.any(probabilities > 0)
                    or not np.allclose(np.exp(probabilities).sum(axis=1), 1., atol=1e-10, rtol=0)):
                raise ValueError('Provider needs normalized finite nonpositive full-history letter log probabilities')
            # Public scores/flags must remain native Python scalars, including
            # for NumPy-backed neural providers and strict JSON publication.
            cache.update({text: (tuple(map(float, p)), state) for text, p, state in zip(texts, probabilities, following, strict=True)})
        frontier = []
        for row in retained:
            probabilities = cache[row.text][0]
            for letter, char in enumerate(channel.alphabet):
                groups = channel.extend(len(row.completed), row.groups, letter)
                if groups:
                    frontier.append(Prefix(row.completed, row.text+char, groups,
                                           row.score+probabilities[letter]+channel.cont))
                    proposed += 1
    return {'plaintexts': None if best is None else best[1], 'joint_log_probability': None if best is None else best[0],
            'compatible_key_indices': () if best is None else tuple(indices(best[2])),
            'discarded_completion_upper_bound': None if not discarded else pruned_bound,
            'floating_map_bound_separated': best is not None and (not discarded or best[0] > pruned_bound+numerical_margin),
            'interval_certificate': False, 'exact_evidence_computed': False, 'beam_width': beam_width,
            'expanded_prefixes': expanded, 'proposed_hypotheses': proposed, 'discarded_prefixes': discarded,
            'maximum_frontier': maximum, 'source_transition_calls': source_calls,
            'distinct_source_prefixes': len(cache), 'source_cache_hits': requests-len(cache),
            'channel_cells': channel.channel_cells, 'support_empty': not start}
