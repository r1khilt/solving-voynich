"""Convert a bounded posterior key population for same-observation decoding.

The shared-key decoder applies ciphertext likelihood to supplied key weights.
Posterior particles have already incorporated it: divide their empirical mass
by each whole-record likelihood before that decoder applies it again. This
preserves the empirical population, including duplicates; it does not certify
that the population approximates the true posterior.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real


def _logsum(values):
    high = max(values)
    return high + math.log(math.fsum(math.exp(v-high) for v in values))


@dataclass(frozen=True)
class PosteriorReadingWeights:
    keys: tuple[tuple[str, ...], ...]
    multiplicities: tuple[int, ...]
    posterior_log_masses: tuple[float, ...]
    decoder_log_weights: tuple[float, ...]
    expected_decoder_log_evidence: float
    particles: int


def posterior_reading_weights(keys: Sequence[Sequence[str]], log_likelihoods: Sequence[float]):
    """Same-record adapter for an equally weighted, resampled population.

    Duplicate keys retain their multiplicities; repeated likelihoods must agree
    exactly. Scores must be the marginal likelihoods of ALL records to be read,
    with the exact same source, reset, stopping and emission laws as decoding.
    The returned weights describe a derived finite mixture, not a free prior or
    a full-family evidence estimator. No Gold, proposal or new source scoring.
    """
    if isinstance(keys, (str, bytes)) or isinstance(log_likelihoods, (str, bytes)):
        raise ValueError('Explicit bounded posterior population and scores required')
    keys, scores = tuple(keys), tuple(log_likelihoods)
    if not 1<=len(keys)<=128 or len(keys)!=len(scores):
        raise ValueError('One to 128 posterior particles with matching scores required')
    if any(isinstance(k, (str, bytes)) for k in keys):
        raise ValueError('Each dictionary must be a sequence of units')
    keys = tuple(tuple(k) for k in keys)
    rows = len(keys[0])
    if not 1<=rows<=64 or any(len(k)!=rows or any(not isinstance(u,str) or not u for u in k) for k in keys):
        raise ValueError('Matching nonempty dictionaries and nonempty units required')
    if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v>0 for v in scores):
        raise ValueError('Finite complete marginal log likelihoods required')
    scores = tuple(float(v) for v in scores)
    counts, distinct_scores = {},{}
    for key,score in zip(keys,scores,strict=True):
        if key in counts and distinct_scores[key]!=score:
            raise ArithmeticError('Duplicate dictionary likelihoods differ')
        counts[key] = counts.get(key,0)+1
        distinct_scores[key] = score
    distinct = tuple(counts)
    log_masses = tuple(math.log(counts[k])-math.log(len(keys)) for k in distinct)
    # Anchor on the smallest log likelihood before adding log masses.
    # A huge common negative score must not erase duplicate multiplicities.
    anchor = min(distinct_scores.values())
    raw = tuple(m-(distinct_scores[k]-anchor) for k,m in zip(distinct,log_masses,strict=True))
    high = max(raw)
    shifted = tuple(v-high for v in raw)
    if any(not math.isfinite(v) for v in (*raw,*shifted)):
        raise ArithmeticError('Inverse-likelihood weight range is unrepresentable')
    log_total = _logsum(shifted)
    weights = tuple(v-log_total for v in shifted)
    expected = anchor-high-log_total
    if not math.isfinite(expected) or any(not math.isfinite(v) for v in weights):
        raise ArithmeticError('Derived mixture normalizer is unrepresentable')
    return PosteriorReadingWeights(distinct,tuple(counts[k] for k in distinct),log_masses,
        weights,expected,len(keys))
