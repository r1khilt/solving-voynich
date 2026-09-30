"""Exact posterior moments and an explicitly supplied-length counterfactual.

No fitting, pruning, posterior sampling, or new channel is performed. Moment
merging is the normalized, log-stable form of an expectation semiring.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PosteriorDiagnostic:
    log_likelihood: float
    joint_log_probability: float
    plaintext: str
    entropy_bits: float
    expected_length: float
    length_variance: float
    reachable_nodes: int
    path_count: int

    def to_dict(self):
        return asdict(self)


def merge(a, b):
    """Each tuple holds log mass and three conditional means."""
    high = max(a[0], b[0])
    za, zb = math.exp(a[0] - high), math.exp(b[0] - high)
    z = za + zb
    return (high + math.log(z), *(math.fsum((za * x, zb * y)) / z
                                 for x, y in zip(a[1:], b[1:], strict=True)))


def diagnose(source, units, observed, rho, *, length=None, max_nodes=500_000):
    """Sum every supported plaintext; optionally require a given letter count.

    The deterministic letter-to-unit map gives exactly one path per plaintext,
    including when two different letters share a unit. Therefore path entropy
    equals plaintext entropy, under this fixed source and fixed channel only.
    """
    if (len(units) != len(source.alphabet)
            or any(not isinstance(u, str) or not u for u in units)):
        raise ValueError('One nonempty unit per letter required')
    if not isinstance(observed, str):
        raise ValueError('Observed text must be a string')
    if isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError('Invalid stopping probability')
    if length is not None and (type(length) is not int or length < 0):
        raise ValueError('Length must be a nonnegative integer')
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('Positive node cap required')
    n = len(observed)
    moments = [{} for _ in range(n + 1)]
    best = [{} for _ in range(n + 1)]
    numbers = [{} for _ in range(n + 1)]
    start = (0, 0)
    moments[0][start] = (0., 0., 0., 0.)
    best[0][start], numbers[0][start] = 0., 1
    back, transitions = {}, {}
    nodes = 1
    minimum, maximum = min(map(len, units)), max(map(len, units))
    for offset in range(n):
        matches = [(i, offset + len(u)) for i, u in enumerate(units)
                   if observed.startswith(u, offset)]
        for key, values in moments[offset].items():
            state, used = key
            for letter, end in matches:
                count = used + 1 if length is not None else 0
                if length is not None:
                    remaining = length - count
                    if remaining < 0 or not remaining * minimum <= n - end <= remaining * maximum:
                        continue
                transition = (state, letter)
                if transition not in transitions:
                    transitions[transition] = (
                        source.step(state, letter),
                        math.log1p(-rho) + math.log(source.probabilities[state, letter]))
                following, weight = transitions[transition]
                dest = (following, count)
                z, log_mean, mean, square = values
                candidate = (z + weight, log_mean + weight, mean + 1, square + 2 * mean + 1)
                if dest not in moments[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('Diagnostic node cap; no pruning or rerun')
                    moments[end][dest] = candidate
                    numbers[end][dest] = numbers[offset][key]
                else:
                    moments[end][dest] = merge(moments[end][dest], candidate)
                    numbers[end][dest] += numbers[offset][key]
                value = best[offset][key] + weight
                if value > best[end].get(dest, -math.inf):
                    best[end][dest] = value
                    back[end, dest] = offset, key, letter
    terminals = [k for k in moments[n] if length is None or k[1] == length]
    if not terminals:
        raise ValueError('No supported plaintext at requested length')
    total = moments[n][terminals[0]]
    for key in terminals[1:]:
        total = merge(total, moments[n][key])
    z, log_mean, mean, square = total
    entropy = (z - log_mean) / math.log(2)
    variance = square - mean * mean
    if entropy < -1e-8 or variance < -1e-7:
        raise ArithmeticError('Negative posterior entropy or variance')
    state = max(terminals, key=best[n].get)
    maximum_score = best[n][state] + math.log(rho)
    offset, text = n, []
    while offset:
        offset, state, letter = back[offset, state]
        text.append(source.alphabet[letter])
    return PosteriorDiagnostic(z + math.log(rho), maximum_score,
                               ''.join(reversed(text)), max(entropy, 0.), mean,
                               max(variance, 0.), nodes,
                               sum(numbers[n][k] for k in terminals))
