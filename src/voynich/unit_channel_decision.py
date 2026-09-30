"""Posterior sampling and sampled edit-risk decisions for fixed unit channels.

The backward lattice sums every compatible plaintext; no beam, top-k, or
temperature is applied. Exact refers to unpruned finite inference, subject to
floating-point arithmetic and a finite-resolution PRNG. The sampled decision
minimizes empirical risk over a finite candidate set, not global Bayes risk.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voynich.finite_state_channel import Channel, Emission, SourceModel, viterbi_decode


def _draw_count(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _seed(value: int) -> None:
    if type(value) is not int:
        raise ValueError("seed must be an integer")


def _stream_seed(seed: int, bank: str) -> str:
    return hashlib.sha256(f"unit-channel-decision/v1:{bank}:{seed}".encode("utf-8")).hexdigest()


def _bank_hash(samples: Sequence[str]) -> str:
    raw = json.dumps(list(samples), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _masks(text: str) -> tuple[dict[str, int], int, int]:
    masks = {}
    for index, char in enumerate(text):
        masks[char] = masks.get(char, 0) | (1 << index)
    return masks, 1 << (len(text) - 1) if text else 0, len(text)


def _distance(pattern: tuple[dict[str, int], int, int], text: str) -> int:
    # Same Myers recurrence already used by scripts/evaluate_naibbe001.py,
    # with precomputed candidate masks reused across the reference bank.
    masks, highest, length = pattern
    if not length:
        return len(text)
    positive, negative, distance = ~0, 0, length
    for char in text:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        distance += bool(positive_horizontal & highest) - bool(negative_horizontal & highest)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return int(distance)


def edit_distance(left: str, right: str) -> int:
    """Unit-cost Unicode-codepoint Levenshtein distance, using bit vectors."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("Edit-distance inputs must be strings")
    return _distance(_masks(left), right)


class _BackwardLattice:
    def __init__(self, source: SourceModel, units: Sequence[str], record: str, rho: float):
        if not isinstance(source, SourceModel):
            raise ValueError("The decision kernel requires an order-zero/one SourceModel")
        if isinstance(units, (str, bytes)):
            raise ValueError("Supply one unit per source letter")
        units = tuple(units)
        if len(units) != len(source.alphabet) or any(not isinstance(u, str) or not u for u in units):
            raise ValueError("Need one nonempty unit per ordered source letter")
        if not isinstance(record, str):
            raise ValueError("The ciphertext must be a string")
        if isinstance(rho, bool):
            raise ValueError("Stopping probability must be strictly between zero and one")
        rho = float(rho)
        if not math.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("Stopping probability must be strictly between zero and one")
        self.source, self.units, self.record, self.rho = source, units, record, rho
        self.contexts = ("", *source.alphabet) if source.order else ("",)
        self.next_context = np.arange(1, len(source.alphabet) + 1) if source.order else np.zeros(
            len(source.alphabet), dtype=np.intp)
        probabilities = np.asarray([[source.probabilities[h][a] for a in source.alphabet]
                                    for h in self.contexts], dtype=np.float64)
        self.log_probabilities = np.full_like(probabilities, -np.inf)
        np.log(probabilities, out=self.log_probabilities, where=probabilities > 0)
        self.log_continue = math.log1p(-rho)
        self.matches = tuple(tuple((a, offset + len(unit)) for a, unit in enumerate(units)
                                   if record.startswith(unit, offset)) for offset in range(len(record)))
        self.beta = np.full((len(record) + 1, len(self.contexts)), -np.inf)
        self.beta[-1, :] = math.log(rho)
        for offset in range(len(record) - 1, -1, -1):
            future = np.full(len(source.alphabet), -np.inf)
            for a, end in self.matches[offset]:
                future[a] = self.beta[end, self.next_context[a]]
            self.beta[offset] = self.log_continue + np.logaddexp.reduce(
                self.log_probabilities + future[None, :], axis=1)
        self.log_likelihood = float(self.beta[0, 0])
        self._choices = {}

    def choices(self, offset: int, context: int) -> tuple[tuple[int, int, float], ...]:
        """Conditional (source-letter index, next offset, probability) edges."""
        key = (offset, context)
        if key not in self._choices:
            edges = [(a, end, self.log_continue + self.log_probabilities[context, a]
                      + self.beta[end, self.next_context[a]]) for a, end in self.matches[offset]]
            edges = [(a, end, float(value)) for a, end, value in edges if math.isfinite(value)]
            if not edges:
                raise ValueError("This lattice node has no posterior continuation")
            high = max(value for _, _, value in edges)
            weights = [math.exp(value - high) for _, _, value in edges]
            total = math.fsum(weights)
            self._choices[key] = tuple((a, end, weight / total)
                                       for (a, end, _), weight in zip(edges, weights, strict=True))
        return self._choices[key]

    def sample(self, draws: int, rng: random.Random) -> tuple[str, ...]:
        if self.log_likelihood == -math.inf:
            return ()
        samples = []
        cumulative_cache = {}
        for _ in range(draws):
            offset = context = 0
            plaintext = []
            while offset < len(self.record):
                key = (offset, context)
                edges = self.choices(offset, context)
                if key not in cumulative_cache:
                    cumulative, total = [], 0.
                    for _, _, probability in edges:
                        total += probability
                        cumulative.append(total)
                    cumulative_cache[key] = cumulative
                cumulative = cumulative_cache[key]
                index = bisect.bisect_right(cumulative, rng.random() * cumulative[-1])
                index = min(index, len(edges) - 1)  # Float endpoint rounding only.
                a, offset, _ = edges[index]
                plaintext.append(self.source.alphabet[a])
                context = int(self.next_context[a])
            samples.append("".join(plaintext))
        return tuple(samples)

    def channel(self) -> Channel:
        glyphs = tuple(dict.fromkeys("".join(self.units)))
        return Channel(("s",), glyphs, {"s": 1.},
                       {("s", a): (Emission("s", unit, 1.),)
                        for a, unit in zip(self.source.alphabet, self.units, strict=True)},
                       self.rho, max(map(len, self.units)))


@dataclass(frozen=True)
class PosteriorSamples:
    samples: tuple[str, ...]
    log_likelihood: float
    seed: int
    requested_draws: int

    def to_dict(self) -> dict:
        return {"samples": list(self.samples), "log_likelihood": self.log_likelihood
                if math.isfinite(self.log_likelihood) else None, "seed": self.seed,
                "requested_draws": self.requested_draws,
                "status": "ok" if math.isfinite(self.log_likelihood) else "impossible"}


def posterior_sample_plaintexts(source: SourceModel, units: Sequence[str], record: str,
                                stop_probability: float, *, draws: int, seed: int) -> PosteriorSamples:
    """Draw with replacement from the full fixed-channel plaintext posterior.

    Impossible observations have no conditional distribution and return no
    samples, even when draws were requested. Empty observations have the sole
    plaintext '', with marginal rho. A zero draw count is permitted.
    """
    _draw_count(draws, "draws")
    _seed(seed)
    lattice = _BackwardLattice(source, units, record, stop_probability)
    return PosteriorSamples(lattice.sample(draws, random.Random(seed)), lattice.log_likelihood, seed, draws)


@dataclass(frozen=True)
class MBRDecision:
    plaintext: str | None
    map_plaintext: str | None
    log_likelihood: float
    map_log_probability: float
    candidate_samples: tuple[str, ...]
    risk_samples: tuple[str, ...]
    candidate_risks: tuple[dict, ...]
    seed: int
    candidate_seed: str
    risk_seed: str
    candidate_draws: int
    risk_draws: int

    def to_dict(self) -> dict:
        risks = {row["plaintext"]: row["mean_edit_distance"] for row in self.candidate_risks}
        return {"status": "ok" if self.plaintext is not None else "impossible",
                "plaintext": self.plaintext, "map_plaintext": self.map_plaintext,
                "log_likelihood": self.log_likelihood if math.isfinite(self.log_likelihood) else None,
                "map_log_probability": self.map_log_probability if math.isfinite(self.map_log_probability) else None,
                "estimated_risk": risks.get(self.plaintext), "map_estimated_risk": risks.get(self.map_plaintext),
                "candidate_draws": self.candidate_draws, "risk_draws": self.risk_draws,
                "seed": self.seed, "candidate_seed": self.candidate_seed, "risk_seed": self.risk_seed,
                "candidate_samples": list(self.candidate_samples), "risk_samples": list(self.risk_samples),
                "candidate_bank_sha256": _bank_hash(self.candidate_samples),
                "risk_bank_sha256": _bank_hash(self.risk_samples),
                "candidate_risks": [dict(row) for row in self.candidate_risks],
                "tie_break": "map_first_then_candidate_first_occurrence"}


def sample_mbr_decode(source: SourceModel, units: Sequence[str], record: str, stop_probability: float,
                      *, candidate_draws: int = 32, risk_draws: int = 256, seed: int = 0) -> MBRDecision:
    """Select minimum sampled raw edit risk from MAP and sampled candidates.

    SHA-256 domain-separated PRNG streams keep candidate and risk draws
    reproducible and independent of one another's bank size. MAP is first;
    candidate duplicates are removed in first-occurrence order. Risk duplicates
    keep their multiplicities. The selected empirical risk is optimistically
    biased by selection, and no true-risk improvement or Bayes optimum is
    promised. Draw counts and external runtime/memory bounds belong to callers.
    """
    _draw_count(candidate_draws, "candidate_draws")
    _draw_count(risk_draws, "risk_draws", 1)
    _seed(seed)
    lattice = _BackwardLattice(source, units, record, stop_probability)
    candidate_seed, risk_seed = _stream_seed(seed, "candidates"), _stream_seed(seed, "risk")
    map_result = viterbi_decode(source, lattice.channel(), record)
    candidates = lattice.sample(candidate_draws, random.Random(int(candidate_seed, 16)))
    references = lattice.sample(risk_draws, random.Random(int(risk_seed, 16)))
    candidate_risks = []
    selected = None
    if map_result.plaintext is not None:
        reference_counts = Counter(references)
        for text in dict.fromkeys((map_result.plaintext, *candidates)):
            pattern = _masks(text)
            total = sum(count * _distance(pattern, reference) for reference, count in reference_counts.items())
            candidate_risks.append({"plaintext": text, "total_edit_distance": total,
                                    "mean_edit_distance": total / risk_draws})
        selected = min(candidate_risks, key=lambda row: row["total_edit_distance"])["plaintext"]
    return MBRDecision(selected, map_result.plaintext, lattice.log_likelihood, map_result.log_probability,
                       candidates, references, tuple(candidate_risks), seed, candidate_seed, risk_seed,
                       candidate_draws, risk_draws)
