"""Separate raw-count backward calculation of diagnostic moments and maxima."""
import math

from scripts.audit_suffix_reader002 import probability


def replay(raw, units, observed, rho, *, length=None, max_nodes=500_000):
    n, order = len(observed), len(raw['masses'])
    low, high = min(map(len, units)), max(map(len, units))
    def following(history, char):
        history = (history + char)[-order:] if order else ''
        while history not in raw['counts']:
            history = history[1:]
        return history
    graph = [{} for _ in range(n + 1)]
    graph[0]['', 0] = []
    nodes = 1
    weights = {}
    for offset in range(n):
        for history, used in graph[offset]:
            edges = graph[offset][history, used]
            for char, unit in zip(raw['alphabet'], units, strict=True):
                if not observed.startswith(unit, offset):
                    continue
                end = offset + len(unit)
                count = used + 1 if length is not None else 0
                if length is not None and not (count <= length and
                    (length - count) * low <= n - end <= (length - count) * high):
                    continue
                dest = following(history, char), count
                if dest not in graph[end]:
                    graph[end][dest] = []
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('Reference node cap')
                if (history, char) not in weights:
                    weights[history, char] = math.log1p(-rho) + math.log(probability(raw, history, char))
                edges.append((end, dest, weights[history, char]))
    backward = {}
    for key in graph[n]:
        if length is None or key[1] == length:
            backward[n, key] = (math.log(rho), math.log(rho), 0., 0., math.log(rho), 1)
    for offset in range(n - 1, -1, -1):
        for key, edges in graph[offset].items():
            candidates = [(weight, backward[end, dest]) for end, dest, weight in edges
                          if (end, dest) in backward]
            if not candidates:
                continue
            logs = [w + row[0] for w, row in candidates]
            peak = max(logs)
            z = peak + math.log(math.fsum(math.exp(v - peak) for v in logs))
            probabilities = [math.exp(v - z) for v in logs]
            means = math.fsum(p * (1 + row[2]) for p, (_, row) in zip(probabilities, candidates, strict=True))
            squares = math.fsum(p * (1 + 2 * row[2] + row[3])
                                for p, (_, row) in zip(probabilities, candidates, strict=True))
            log_mean = math.fsum(p * (w + row[4]) for p, (w, row) in zip(probabilities, candidates, strict=True))
            backward[offset, key] = (z, max(w + row[1] for w, row in candidates),
                                     means, squares, log_mean, sum(row[5] for _, row in candidates))
    if (0, ('', 0)) not in backward:
        raise ValueError('No supported reading')
    z, maximum, mean, square, log_mean, count = backward[0, ('', 0)]
    return {'log_likelihood': z, 'joint_log_probability': maximum,
            'expected_length': mean, 'length_variance': max(0., square - mean * mean),
            'entropy_bits': max(0., (z - log_mean) / math.log(2)), 'path_count': count}
