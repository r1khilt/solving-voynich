"""Additive deletion diagnostics, never replacements for registered pass metrics.

The rank proxy is an offline score conditional on the original input's frequency
map. It is not a raw-symbol likelihood or a compression rate: map side
information, deletion choices and lengths are not charged. See
docs/research/RECOVERY_METRICS_AUDIT.md for applicability and sources.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def binary_mask(text, mask):
    """Strict new-API validation; legacy callers retain their padding policy."""
    values = np.asarray(mask)
    if values.shape != (len(text),) or not np.isin(values, [0, 1]).all():
        raise ValueError("mask must contain one binary decision per input character")
    return values.astype(bool)


def edit_distance(a: str, b: str) -> int:
    """Unit-cost insertion/deletion/substitution Levenshtein distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return int(previous[-1])


def reconstruction_diagnostics(target: str, text: str, mask) -> dict:
    kept = binary_mask(text, mask)
    selected = "".join(ch for ch, keep in zip(text, kept, strict=True) if keep)
    distance = edit_distance(selected, target)
    return {"aligned_similarity": 1.0 - distance / max(len(selected), len(target), 1),
            "exact_sequence_match": selected == target, "edit_distance": distance,
            "input_count": len(text), "target_count": len(target),
            "retained_count": len(selected), "deleted_count": len(text) - len(selected)}


def summarize_reconstruction(rows) -> dict:
    if not rows:
        raise ValueError("at least one reconstruction row is required")
    means = {name + "_mean": float(np.mean([row[name] for row in rows])) for name in
             ("aligned_similarity", "edit_distance", "input_count", "target_count",
              "retained_count", "deleted_count")}
    return {**means, "exact_sequence_match_rate": float(np.mean([row["exact_sequence_match"] for row in rows])),
            "evaluated_sequences": len(rows),
            "aggregation": "Equal weight per evaluated sequence; random replicates are not independent tasks."}


def frozen_rank_encoding(text: str, n_buckets: int = 8) -> tuple[int, ...]:
    """Original-input map, with first-occurrence tie breaking and separate space.

    Call once before selection; selecting this tuple must never recompute ranks.
    Renaming non-space symbols bijectively preserves this representation.
    """
    n_buckets = _integer(n_buckets, "n_buckets", 1)
    counts = Counter(ch for ch in text if ch != " ")
    ranked = [ch for ch, _ in counts.most_common()]
    mapping = {ch: min(n_buckets - 1, i * n_buckets // len(ranked)) for i, ch in enumerate(ranked)}
    return tuple(n_buckets if ch == " " else mapping[ch] for ch in text)


@dataclass(frozen=True)
class FixedRankBigram:
    """Normalized distribution over encoded strings conditional on their length.

    A separate initial distribution scores the first symbol. No end-of-sequence
    or length distribution is asserted. Parameters are immutable tuples.
    """

    n_buckets: int
    initial: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    alpha: float
    training_source: str
    training_sha256: str
    training_sequences: int
    training_symbols: int

    def __post_init__(self):
        size = _integer(self.n_buckets, "n_buckets", 1) + 1
        for values, shape in ((self.initial, (size,)), (self.transition, (size, size))):
            probabilities = np.asarray(values, dtype=float)
            if (probabilities.shape != shape or not np.isfinite(probabilities).all() or
                    np.any(probabilities <= 0) or not np.allclose(probabilities.sum(-1), 1., atol=1e-12, rtol=1e-12)):
                raise ValueError("initial and transition rows must be positive normalized probabilities")
        # Accept JSON-restored lists without leaving mutable parameter storage.
        object.__setattr__(self, "initial", tuple(map(float, self.initial)))
        object.__setattr__(self, "transition", tuple(tuple(map(float, row)) for row in self.transition))

    def score(self, sequence) -> dict:
        values = np.asarray(sequence)
        if (values.ndim != 1 or (values.size and
                (values.dtype.kind not in "iu" or values.min() < 0 or values.max() > self.n_buckets))):
            raise ValueError("encoded sequence must use the fitted fixed integer vocabulary")
        values = values.astype(np.int64)
        total = -math.log2(self.initial[values[0]]) if len(values) else 0.0
        total -= sum(math.log2(self.transition[a][b]) for a, b in zip(values, values[1:]))
        return {"negative_log2_probability": total,
                "bits_per_encoded_symbol": total / len(values) if len(values) else None,
                "symbol_count": len(values), "transition_count": max(0, len(values) - 1),
                "initial_symbol_scored": bool(len(values)),
                "status": "scored" if len(values) else "no_observations"}

    def metadata(self) -> dict:
        encoded = json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
        return {"model_sha256": hashlib.sha256(encoded).hexdigest(),
                "training_source": self.training_source, "training_sha256": self.training_sha256,
                "training_sequences": self.training_sequences, "training_symbols": self.training_symbols,
                "n_buckets": self.n_buckets, "vocabulary_size": self.n_buckets + 1, "alpha": self.alpha,
                "normalization": "Initial and every transition row sum to one over the fixed vocabulary."}


def fit_fixed_rank_bigram(texts, *, training_source: str, n_buckets: int = 8, alpha: float = .1) -> FixedRankBigram:
    """Explicitly fit on independent training strings; never fit inside scoring.

    The caller must supply appropriate training data and preserve its split.
    Fresh cipher IDs are not compared with a plaintext character vocabulary.
    """
    if isinstance(texts, str):
        raise ValueError("supply a collection of training texts, not a single string")
    texts = tuple(texts)
    n_buckets = _integer(n_buckets, "n_buckets", 1)
    if not texts or any(not isinstance(text, str) for text in texts) or not any(texts):
        raise ValueError("training texts must contain at least one nonempty string")
    if not isinstance(training_source, str) or not training_source.strip():
        raise ValueError("a training source/split identifier is required")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    size = n_buckets + 1
    initial = np.full(size, alpha, dtype=float)
    transition = np.full((size, size), alpha, dtype=float)
    for text in texts:
        sequence = frozen_rank_encoding(text, n_buckets)
        if sequence:
            initial[sequence[0]] += 1
            for a, b in zip(sequence, sequence[1:]):
                transition[a, b] += 1
    initial /= initial.sum()
    transition /= transition.sum(-1, keepdims=True)
    return FixedRankBigram(n_buckets, tuple(initial.tolist()), tuple(map(tuple, transition.tolist())),
                          float(alpha), training_source,
                          hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                          len(texts), sum(map(len, texts)))


def fixed_rank_deletion_diagnostic(text: str, mask, model: FixedRankBigram, *, n_random: int = 20,
                                   seed: int = 91021, min_retained: int = 4) -> dict:
    """Compare fixed-map selection to count- and histogram-matched deletions.

    Random masks preserve original order and have exactly the candidate's count.
    Histogram controls additionally retain the same counts of encoded buckets.
    These are descriptive reference distributions, not significance tests or
    recovery gates. No target sequence, language identity or oracle is required.
    """
    if not isinstance(model, FixedRankBigram):
        raise ValueError("supply an independently fitted FixedRankBigram")
    n_random = _integer(n_random, "n_random", 2)
    seed = _integer(seed, "seed")
    min_retained = _integer(min_retained, "min_retained", 2)
    kept = binary_mask(text, mask)
    encoded = np.asarray(frozen_rank_encoding(text, model.n_buckets), dtype=np.int64)
    selected = model.score(encoded[kept])
    count = int(kept.sum())
    rng = np.random.default_rng(seed)
    controls = {}
    for kind in ("same_retained_count", "same_retained_count_and_bucket_histogram"):
        scores, mask_ids = [], set()
        for _ in range(n_random):
            random_mask = np.zeros(len(text), dtype=bool)
            if kind == "same_retained_count":
                random_mask[rng.choice(len(text), size=count, replace=False)] = True
            else:
                for bucket in range(model.n_buckets + 1):
                    candidates = np.flatnonzero(encoded == bucket)
                    take = int(np.count_nonzero(kept & (encoded == bucket)))
                    random_mask[rng.choice(candidates, size=take, replace=False)] = True
            mask_ids.add(random_mask.tobytes())
            scores.append(model.score(encoded[random_mask]))
        values = [score["bits_per_encoded_symbol"] for score in scores]
        eligible = min_retained <= count < len(text) and len(mask_ids) > 1
        reasons = []
        if count < min_retained:
            reasons.append("Too few retained symbols for the descriptive comparison.")
        if count == len(text):
            reasons.append("No deletion; there is no alternative retained subset.")
        if len(mask_ids) < 2:
            reasons.append("Sampled controls contain fewer than two distinct masks.")
        mean = float(np.mean(values)) if count else None
        controls[kind] = {
            "replicates": n_random, "distinct_sampled_masks": len(mask_ids),
            "retained_count_each": count, "transition_count_each": max(0, count - 1),
            "bits_per_encoded_symbol": values, "mean_bits_per_encoded_symbol": mean,
            "std_bits_per_encoded_symbol": float(np.std(values)) if count else None,
            "random_minus_selected_bits_per_encoded_symbol": mean - selected["bits_per_encoded_symbol"] if count else None,
            "eligible_for_descriptive_comparison": eligible, "ineligibility_reasons": reasons,
        }
    return {"schema_version": 1, "model": model.metadata(), "seed": seed,
            "original_count": len(text), "retained_count": count, "deleted_count": len(text) - count,
            "retained_fraction": count / len(text) if text else None,
            "original_encoding_sha256": hashlib.sha256(encoded.astype("<i8").tobytes()).hexdigest(),
            "encoding_policy": "Frequency ranks fit once on the complete original input; never recomputed after deletion.",
            "full": model.score(encoded), "selected": selected, "controls": controls,
            "scope": "Offline fixed-representation proxy conditional on input frequency map and retained length; not raw-symbol likelihood, compression or recovery.",
            "limitations": ["Input-map, length and deletion-mask side information is not charged.",
                            "Count controls do not match symbol frequencies; histogram controls additionally match encoded bucket counts.",
                            "Neither control matches every position/run/copy property or corrects adaptive candidate selection.",
                            "A low score or positive contrast does not establish recovery; use independent truth and null tests."]}
