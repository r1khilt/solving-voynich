"""Root-authored independent counting and reverse-recursion numeric references."""
from __future__ import annotations

import math
from collections import Counter


def counts_reference(records, order):
    result = {}
    for depth in range(order + 1):
        grams = Counter(record[i:i + depth + 1] for record in records
                        for i in range(len(record) - depth))
        for gram, count in grams.items():
            result.setdefault(gram[:-1], {})[gram[-1]] = count
    return result


def infer_reference(raw, units, observed, rho, max_nodes=500_000):
    """Build string-context states independently, then recurse right to left."""
    counts, alphabet, order, tau = raw['counts'], raw['alphabet'], raw['order'], raw['tau']
    probability_cache = {}

    def probability(context, char):
        key = context, char
        if key not in probability_cache:
            row = counts[context]
            if not context:
                value = (row.get(char, 0) + .5) / (sum(row.values()) + .5 * len(alphabet))
            else:
                value = (row.get(char, 0) + tau * probability(context[1:], char)) / (sum(row.values()) + tau)
            probability_cache[key] = value
        return probability_cache[key]

    def following(context, char):
        suffix = (context + char)[-order:] if order else ''
        while suffix not in counts:
            suffix = suffix[1:]
        return suffix

    n = len(observed)
    reachable = [set() for _ in range(n + 1)]
    reachable[0].add('')
    nodes = 1
    matches = [[(c, offset + len(u)) for c, u in zip(alphabet, units, strict=True)
                if observed.startswith(u, offset)] for offset in range(n)]
    for offset in range(n):
        for context in reachable[offset]:
            for char, end in matches[offset]:
                context_after = following(context, char)
                if context_after not in reachable[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('Reference lattice cap exceeded')
                    reachable[end].add(context_after)
    total = {(n, c): math.log(rho) for c in reachable[n]}
    best = dict(total)
    cont = math.log1p(-rho)
    for offset in range(n - 1, -1, -1):
        for context in reachable[offset]:
            terms, maxima = [], []
            for char, end in matches[offset]:
                key = end, following(context, char)
                if key in total:
                    weight = cont + math.log(probability(context, char))
                    terms.append(weight + total[key])
                    maxima.append(weight + best[key])
            if terms:
                high = max(terms)
                total[offset, context] = high + math.log(math.fsum(math.exp(v - high) for v in terms))
                best[offset, context] = max(maxima)
    return total.get((0, ''), -math.inf), best.get((0, ''), -math.inf), nodes


def reading_score(raw, units, observed, text, rho):
    if text is None:
        return -math.inf
    mapping = dict(zip(raw['alphabet'], units, strict=True))
    if ''.join(mapping[c] for c in text) != observed:
        raise ValueError('Reading fails exact re-encoding')
    values = [math.log(rho)]
    counts, tau = raw['counts'], raw['tau']
    for i, char in enumerate(text):
        probability = (counts[''].get(char, 0) + .5) / (sum(counts[''].values()) + .5 * len(raw['alphabet']))
        for depth in range(1, min(i, raw['order']) + 1):
            context = text[i - depth:i]
            if context in counts:
                row = counts[context]
                probability = (row.get(char, 0) + tau * probability) / (sum(row.values()) + tau)
        values.append(math.log1p(-rho) + math.log(probability))
    return math.fsum(values)
