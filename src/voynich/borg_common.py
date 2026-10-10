"""Bounded common-name substitution hypothesis; not a verified Borg alphabet.

Each observed name maps injectively to one source letter.  Whitespace is not
treated as a plaintext word boundary.  Exclusion boundaries always reset the
source context.  This module does not access historical answers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


LATIN = "abcdefghilmnopqrstuvxyz"
COMMON = "01245689Mcdhimnoqvwxy"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def select_body(blob: bytes, parsed, *, leaves: range,
                excluded_leaves=frozenset((49, 99, 150)),
                excluded_lines=frozenset(), minimum=4) -> list[dict]:
    """Retain whole common-name body chunks, joining whitespace on ONE line.

    A punctuation/exception/annotation chunk, any quarantined line, or any
    physical-line/page boundary terminates a record.  Never split an exception
    chunk into a convenient certain substring.  Raw offsets remain auditable.
    """
    if minimum < 1 or digest(blob) != parsed.source_sha256:
        raise ValueError("Invalid selection inputs")
    allowed = frozenset(COMMON)
    labels = {p.block_index: p for p in parsed.pages}
    records, current = [], []
    last_line = last_block = None

    def flush():
        if current:
            symbols = "".join(x["symbols"] for x in current)
            if len(symbols) >= minimum:
                page = labels[last_block]
                records.append({"block": last_block, "page": page.raw_label,
                    "leaf": int(page.normalized_label_only[:4]),
                    "line": last_line, "symbols": symbols, "pieces": list(current),
                    "symbols_sha256": digest(symbols.encode())})
            current.clear()

    for span in parsed.spans:
        if span.start_line != last_line or span.block_index != last_block:
            flush()
            last_line, last_block = span.start_line, span.block_index
        page = labels.get(span.block_index)
        leaf = int(page.normalized_label_only[:4]) if page else None
        eligible = (page is not None and leaf in leaves and leaf not in excluded_leaves
                    and leaf > 10 and "." not in page.raw_label
                    and span.start_line not in excluded_lines and not page.explicit_cleartext)
        value = blob[span.start_byte:span.end_byte].decode("utf-8")
        if eligible and span.kind == "unresolved_body" and set(value) <= allowed:
            current.append({"start_byte": span.start_byte, "end_byte": span.end_byte,
                            "raw_sha256": span.sha256, "symbols": value})
        elif eligible and span.kind == "whitespace" and "\n" not in value and "\r" not in value:
            pass
        else:
            flush()
    flush()
    return records


def encode_records(records: list[str], alphabet: str) -> list[np.ndarray]:
    if not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet must be unique")
    index = {letter: i for i, letter in enumerate(alphabet)}
    if not records or any(not record or set(record) - set(alphabet) for record in records):
        raise ValueError("Empty or unsupported record")
    return [np.asarray([index[letter] for letter in record], dtype=np.int64)
            for record in records]


def train_source(text: str, alphabet: str = LATIN) -> list[np.ndarray]:
    """Fixed positive normalized recursive Dirichlet 1..4-gram source.

    Concentration16 follows the already source-only selected NAIBBE-003 prior;
    no choice is made on Borg or the new controls.  Counts include the frozen
    source file's preexisting artificial concatenation boundary.
    """
    values = encode_records([text], alphabet)[0]
    if len(values) < 4:
        raise ValueError("Source needs four letters")
    size, tables, lower = len(alphabet), [], None
    for order in range(1, 5):
        powers = size ** np.arange(order - 1, -1, -1)
        codes = np.lib.stride_tricks.sliding_window_view(values, order) @ powers
        counts = np.bincount(codes, minlength=size**order).reshape(-1, size)
        totals = counts.sum(axis=1, keepdims=True)
        if order == 1:
            probabilities = (counts + .1) / (totals + .1 * size)
        else:
            backoff = lower[np.arange(size ** (order - 1)) % len(lower)]
            probabilities = (counts + 16 * backoff) / (totals + 16)
        if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0):
            raise AssertionError("Unnormalized source")
        tables.append(np.log(probabilities.ravel()))
        lower = probabilities
    return tables


class Objective:
    """Full literal sequence score compressed into prefix and 4-gram counts."""

    def __init__(self, records: list[np.ndarray], tables, source_size=23,
                 observed_size=21):
        if len(tables) != 4 or not records or observed_size > source_size:
            raise ValueError("Unsupported objective")
        self.tables, self.source_size, self.observed_size = tables, source_size, observed_size
        self.rows = []
        self.characters = sum(map(len, records))
        for order in range(1, 5):
            counts = Counter()
            for sequence in records:
                if (sequence.ndim != 1 or len(sequence) == 0
                        or np.any(sequence < 0) or np.any(sequence >= observed_size)):
                    raise ValueError("Malformed record")
                if order < 4:
                    if len(sequence) >= order:
                        counts[tuple(map(int, sequence[:order]))] += 1
                elif len(sequence) >= 4:
                    counts.update(map(tuple, np.lib.stride_tricks.sliding_window_view(sequence, 4)))
            grams = np.asarray(list(counts), dtype=np.int64).reshape(-1, order)
            weight = np.asarray(list(counts.values()), dtype=float)
            self.rows.append((grams, weight, source_size ** np.arange(order - 1, -1, -1)))

    def scores(self, keys: np.ndarray) -> np.ndarray:
        if not isinstance(keys, np.ndarray):
            literal = np.asarray(keys, dtype=object)
            if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                   for value in literal.ravel()):
                raise ValueError("Key entries must be literal integers")
        keys = np.asarray(keys)
        if keys.dtype.kind not in "iu":
            raise ValueError("Key entries must be integers without coercion")
        if (keys.ndim != 2 or keys.shape[1] != self.source_size
                or any(set(key) != set(range(self.source_size)) for key in keys)):
            raise ValueError("Key must be a completed injective mapping")
        answer = np.zeros(len(keys), dtype=float)
        for order, (grams, weights, powers) in enumerate(self.rows):
            if len(grams):
                codes = (keys[:, grams] * powers).sum(axis=-1)
                answer += (self.tables[order][codes] * weights).sum(axis=-1)
        return answer


def fit(objective: Objective, *, seed: int, restarts: int = 12,
        sweeps: int = 8, guard=lambda: None) -> dict:
    """Bounded random-restart coordinate descent, not posterior sampling.

    The two unused positions complete 21->23 injections to permutations.
    Swapping them with observed positions permits changes in used letters.
    Equal scores retain the current mapping; final ties use lexicographic key.
    Every score is the complete untrimmed objective, never a local approximation.
    """
    if restarts < 1 or sweeps < 1:
        raise ValueError("Invalid finite allocation")
    rng = np.random.Generator(np.random.PCG64(seed))
    size = objective.source_size
    best = None
    endpoints, evaluated = [], 0
    for restart in range(restarts):
        guard()
        key = rng.permutation(size)
        score = float(objective.scores(key[None])[0])
        evaluated += 1
        changes = 0
        for sweep in range(sweeps):
            changed = False
            for pivot in rng.permutation(size):
                guard()
                candidates = np.repeat(key[None], size, axis=0)
                positions = np.arange(size)
                candidates[positions, pivot] = key[positions]
                candidates[positions, positions] = key[pivot]
                values = objective.scores(candidates)
                evaluated += size
                partner = int(np.argmax(values))
                if float(values[partner]) > score + 1e-10:
                    key, score = candidates[partner].copy(), float(values[partner])
                    changed = True
                    changes += 1
            if not changed:
                break
        # One extra whole-objective score detects accidental stale bookkeeping.
        recomputed = float(objective.scores(key[None])[0])
        evaluated += 1
        if not math.isclose(recomputed, score, rel_tol=0, abs_tol=1e-7):
            raise AssertionError("Stale score")
        endpoint = {"restart": restart, "key": key.tolist(), "score": score,
                    "sweeps": sweep + 1, "changes": changes}
        endpoints.append(endpoint)
        if best is None or (-score, tuple(key)) < (-best["score"], tuple(best["key"])):
            best = endpoint
    return {"key": best["key"], "score": best["score"], "seed": seed,
            "restarts": restarts, "max_sweeps": sweeps,
            "endpoints": endpoints, "full_key_scores": evaluated}


def decode(records: list[str], key: list[int], observed=COMMON, source=LATIN) -> list[str]:
    if (any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            for value in key)
            or len(key) != len(source) or set(key) != set(range(len(source)))):
        raise ValueError("Invalid completed key")
    index = {symbol: i for i, symbol in enumerate(observed)}
    return ["".join(source[key[index[symbol]]] for symbol in record) for record in records]
