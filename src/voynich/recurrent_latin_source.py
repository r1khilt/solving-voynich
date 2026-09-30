"""Normalized causal letter source and boundary-preserving data utilities.

No cipher answers or reserved authors are needed by this module. A hidden state
represents a whole history; distinct states must not be merged by glyph offset.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = 'abcdefghiklmnopqrstuxyz'


class RecurrentSource(nn.Module):
    def __init__(self, alphabet=ALPHABET, embedding=96, width=768, layers=2):
        super().__init__()
        if not alphabet or len(set(alphabet)) != len(alphabet) or any(len(c) != 1 for c in alphabet):
            raise ValueError('Unique character alphabet required')
        if any(type(v) is not int or v < 1 for v in (embedding, width, layers)):
            raise ValueError('Positive dimensions required')
        self.config = dict(alphabet=alphabet, embedding=embedding, width=width, layers=layers)
        self.bos = len(alphabet)
        self.embedding = nn.Embedding(self.bos + 1, embedding)
        self.recurrent = nn.LSTM(embedding, width, num_layers=layers, batch_first=True)
        self.output = nn.Linear(width, self.bos)

    def forward(self, tokens, state=None):
        hidden, state = self.recurrent(self.embedding(tokens), state)
        return self.output(hidden), state

    def next_log_probabilities(self, tokens, state=None):
        logits, state = self(tokens, state)
        return F.log_softmax(logits, dim=-1), state


def encode(text: str, alphabet=ALPHABET) -> np.ndarray:
    table = {c: i for i, c in enumerate(alphabet)}
    if not isinstance(text, str) or not text or set(text) - set(table):
        raise ValueError('Nonempty in-alphabet text required')
    return np.fromiter((table[c] for c in text), dtype=np.int64, count=len(text))


class WindowSampler:
    """Uniform over every legal full-window start, never across segments."""
    def __init__(self, records: Sequence[str], length: int, alphabet=ALPHABET):
        if type(length) is not int or length < 1:
            raise ValueError('Positive window length required')
        self.records = [encode(r, alphabet) for r in records if len(r) >= length]
        if not self.records:
            raise ValueError('No eligible training segments')
        self.length, self.bos = length, len(alphabet)
        self.sizes = np.array([len(r) - length + 1 for r in self.records], dtype=np.int64)
        self.ends = self.sizes.cumsum()
        self.starts = np.r_[0, self.ends[:-1]]

    def at(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= self.ends[-1]):
            raise ValueError('Window indices outside legal start space')
        segments = np.searchsorted(self.ends, indices, side='right')
        offsets = indices - self.starts[segments]
        y = np.stack([self.records[s][o:o + self.length] for s, o in zip(segments, offsets)])
        x = np.concatenate((np.full((len(y), 1), self.bos, dtype=np.int64), y[:, :-1]), axis=1)
        return x, y

    def sample(self, rng, batch):
        return self.at(rng.integers(int(self.ends[-1]), size=batch))


def chunks(records, length=512):
    if type(length) is not int or length < 1:
        raise ValueError('Positive chunk length required')
    return [(i, r[j:j + length]) for i, r in enumerate(records) for j in range(0, len(r), length)]


def padded_batch(texts, alphabet=ALPHABET):
    rows = [encode(t, alphabet) for t in texts]
    if not rows:
        raise ValueError('Empty batch')
    n = max(map(len, rows))
    x = np.full((len(rows), n), len(alphabet), dtype=np.int64)
    y = np.zeros((len(rows), n), dtype=np.int64)
    mask = np.zeros((len(rows), n), dtype=bool)
    for i, row in enumerate(rows):
        y[i, :len(row)], mask[i, :len(row)] = row, True
        x[i, 1:len(row)] = row[:-1]
    return x, y, mask


@torch.inference_mode()
def score_records(model, records, device='cpu', batch=16, length=512):
    """All letters scored once; float64 host log-softmax avoids sum precision loss."""
    if not records or any(not r for r in records):
        raise ValueError('Nonempty scoring records required')
    model.eval()
    parts = chunks(records, length)
    per_record = [[] for _ in records]
    for start in range(0, len(parts), batch):
        subset = parts[start:start + batch]
        x, y, mask = padded_batch([s for _, s in subset], model.config['alphabet'])
        logits, _ = model(torch.from_numpy(x).to(device))
        logp = F.log_softmax(logits.cpu().double(), dim=-1)
        losses = -logp.gather(-1, torch.from_numpy(y).unsqueeze(-1)).squeeze(-1).numpy() / math.log(2)
        for row, (record, _) in enumerate(subset):
            per_record[record].append(float(losses[row, mask[row]].sum(dtype=np.float64)))
    totals = [math.fsum(row) for row in per_record]
    total = math.fsum(totals)
    return {'bits': total, 'characters': sum(map(len, records)),
            'bits_per_character': total / sum(map(len, records)),
            'record_bits': totals, 'chunks': len(parts)}


def learning_rate(step, total=6000, warmup=100, peak=.001, final=.0001):
    if type(step) is not int or not 1 <= step <= total or not 0 < warmup < total:
        raise ValueError('Invalid schedule position')
    if step <= warmup:
        return peak * step / warmup
    phase = (step - warmup) / (total - warmup)
    return final + .5 * (peak - final) * (1 + math.cos(math.pi * phase))
