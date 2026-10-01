"""Boundary-preserving supervised cipher episodes; no reader holdout access.

Two distinct sampled records are duplicated to the measured four-record GPU
shape. Identical duplication supplies no additional information: encoder blocks
are shared and memory cross-attention renormalizes the repeated pairs.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from voynich.joint_key_proposal import canonical_key_indices, canonicalize_records, unit_pool
from voynich.recurrent_latin_source import encode

GLYPHS = "ABCDEF"
MIN_LENGTH, MAX_LENGTH = 64, 224


def dictionary_code(indices):
    indices = tuple(indices)
    if len(indices) != 23 or any(type(i) is not int or not 0 <= i < 42 for i in indices):
        raise ValueError("Complete 23-row legal dictionary required")
    return bytes(indices).hex()


class EpisodeSampler:
    """Uniform length, then uniform eligible within-segment start for that length."""
    def __init__(self, texts, *, forbidden_raw=(), forbidden_canonical=()):
        self.records = [encode(text) for text in texts]
        if not self.records or any(len(r) < MAX_LENGTH for r in self.records):
            raise ValueError("Every retained segment must support all declared lengths")
        self.ends = {n: np.cumsum([len(r)-n+1 for r in self.records], dtype=np.int64)
                     for n in range(MIN_LENGTH, MAX_LENGTH+1)}
        self.forbidden_raw, self.forbidden_canonical = set(forbidden_raw), set(forbidden_canonical)
        self.rejected_keys = 0

    def draw_window(self, rng):
        length = int(rng.integers(MIN_LENGTH, MAX_LENGTH+1))
        ends = self.ends[length]
        index = int(rng.integers(int(ends[-1])))
        segment = int(np.searchsorted(ends, index, side="right"))
        start = index-int(ends[segment-1] if segment else 0)
        return {"segment": segment, "start": start, "length": length}

    def make(self, windows, raw_indices):
        pool = unit_pool(6)
        dictionary_code(raw_indices)
        if len(windows) != 2:
            raise ValueError("Exactly two original records required")
        units = tuple("".join(GLYPHS[i] for i in pool[u]) for u in raw_indices)
        source, observed = [], []
        for spec in windows:
            segment, start, length = (spec[k] for k in ("segment", "start", "length"))
            if (any(type(v) is not int for v in (segment, start, length))
                    or not 0 <= segment < len(self.records) or not MIN_LENGTH <= length <= MAX_LENGTH
                    or start < 0 or start+length > len(self.records[segment])):
                raise ValueError("Invalid boundary-preserving source window")
            text = self.records[segment][start:start+length]
            source.append(text)
            observed.append("".join(units[int(c)] for c in text))
        canonical = canonicalize_records(observed, GLYPHS)
        target = canonical_key_indices(units, canonical)
        used = sorted({int(c) for text in source for c in text})
        # The metadata is supervision/provenance. Only cipher records enter the
        # encoder; target/used/source lengths never become encoder inputs.
        metadata = {"windows": windows, "raw_indices": list(raw_indices), "canonical_key": list(target),
                    "observed_symbols": list(canonical.observed), "unseen_symbols": list(canonical.unseen),
                    "used_rows": used, "cipher_sha256": hashlib.sha256("\n".join(observed).encode()).hexdigest()}
        return canonical.records, target, metadata

    def sample(self, rng):
        windows = [self.draw_window(rng) for _ in range(2)]
        for _ in range(16):
            raw = rng.integers(42, size=23).tolist()
            records, target, metadata = self.make(windows, raw)
            if dictionary_code(raw) not in self.forbidden_raw and dictionary_code(target) not in self.forbidden_canonical:
                return records, target, metadata
            self.rejected_keys += 1
        raise RuntimeError("Finite disjoint-key draw budget exhausted; no partial episode")


def pack_episodes(episodes, *, duplicate=True):
    if not episodes or type(duplicate) is not bool:
        raise ValueError("Nonempty episode batch and explicit duplicate flag required")
    count = 4 if duplicate else 2
    records = torch.full((len(episodes), count, MAX_LENGTH*2), 6, dtype=torch.long)
    keys = []
    for i, (cipher, target, _) in enumerate(episodes):
        if len(cipher) != 2 or any(not 1 <= len(r) <= MAX_LENGTH*2 for r in cipher):
            raise ValueError("Two nonempty bounded observed records required")
        for j in range(count):
            row = torch.tensor(cipher[j%2], dtype=torch.long)
            records[i, j, :len(row)] = row
        keys.append(target)
    return records, torch.tensor(keys, dtype=torch.long)


def metadata_bytes(rows):
    return (json.dumps(rows, separators=(",", ":"), sort_keys=True, allow_nan=False)+"\n").encode()


def schedule(step, total=20_000):
    if type(step) is not int or not 1 <= step <= total or total <= 200:
        raise ValueError("Registered schedule position required")
    return 3e-4*step/200 if step <= 200 else 3e-5+.5*2.7e-4*(1+math.cos(math.pi*(step-200)/(total-200)))


@torch.inference_mode()
def validation(model, episodes, *, device, batch=4, progress=None):
    """Full-row proper density selection; free-running greedy diagnostics separate."""
    model.eval()
    values, targets, correct, used_correct, used_count = [], [], 0, 0, 0
    greedy_keys = []
    greedy_logq = []
    for start in range(0, len(episodes), batch):
        subset = episodes[start:start+batch]
        records, keys = pack_episodes(subset)
        records = records.to(device)
        memory, padding = model.encode_records(records)
        target = keys.to(device)
        logits = model.decode_partial(memory, padding, target[:, :-1]).cpu().double()
        logq = F.log_softmax(logits, -1).gather(-1, keys[..., None]).squeeze(-1).sum(-1)
        if not torch.isfinite(logq).all().item():
            raise ValueError("Nonfinite validation whole-key probability")
        proposed, proposal_logq = model.propose(memory, padding, greedy=True)
        proposed = proposed[:, 0].cpu()
        matched = proposed == keys
        correct += int(matched.sum())
        for i, episode in enumerate(subset):
            used = episode[2]["used_rows"]
            used_count += len(used)
            used_correct += int(matched[i, used].sum())
        greedy_keys.extend(proposed.tolist())
        greedy_logq.extend(proposal_logq[:, 0].tolist())
        values.extend(logq.tolist())
        targets.extend(keys.tolist())
        if progress:
            progress({"validation_episodes": start+len(subset)})
    if not values:
        raise ValueError("Nonempty fixed validation pool required")
    return {"episodes": len(values), "rows": len(values)*23, "joint_logq": values,
            "mean_nats_per_row": -math.fsum(values)/(23*len(values)),
            "free_running_greedy_keys": greedy_keys, "correct_rows": correct,
            "greedy_logq": greedy_logq,
            "used_correct_rows": used_correct, "used_rows": used_count,
            "whole_keys_exact": sum(a == b for a, b in zip(greedy_keys, targets, strict=True)),
            "used_mask_is_diagnostic_only": True}
