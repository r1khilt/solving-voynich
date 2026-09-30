"""Fixed-shape, whole-history recurrent scoring; padding never earns probability."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

from voynich.recurrent_latin_source import encode


@torch.inference_mode()
def fixed_record_logps(model, texts: Sequence[str], device='cpu', *, batch=8,
                      length=270, after_batch: Callable | None = None):
    """Return source log probabilities, excluding any external length law.

    All calls use exactly (batch, length), including the final partial batch.
    A unidirectional recurrence makes right padding irrelevant to scored tokens.
    Oversized inputs are rejected, never truncated or reset midway through.
    The callback runs after synchronized host extraction and tensor release.
    """
    if type(batch) is not int or batch < 1 or type(length) is not int or length < 1:
        raise ValueError('Positive integer batch and length required')
    if not texts or any(not isinstance(t, str) or not t or len(t) > length for t in texts):
        raise ValueError('Nonempty complete records within fixed length required')
    alphabet = model.config['alphabet']
    # Validate the whole input before scoring; do not silently accept a bad suffix.
    if any(set(t) - set(alphabet) for t in texts):
        raise ValueError('Out-of-alphabet record')
    model.eval()
    values = []
    for start in range(0, len(texts), batch):
        subset = texts[start:start+batch]
        x = np.full((batch, length), model.bos, dtype=np.int64)
        y = np.zeros((batch, length), dtype=np.int64)
        sizes = []
        for row, text in enumerate(subset):
            target = encode(text, alphabet)
            sizes.append(len(target))
            y[row, :len(target)] = target
            x[row, 1:len(target)] = target[:-1]
        tokens = torch.from_numpy(x).to(device)
        logits, state = model(tokens)
        host = logits.cpu().double()
        logp = torch.log_softmax(host, dim=-1)
        picked = logp.gather(-1, torch.from_numpy(y).unsqueeze(-1)).squeeze(-1)
        for row, size in enumerate(sizes):
            value = math.fsum(picked[row, :size].tolist())
            if not math.isfinite(value):
                raise ValueError('Nonfinite complete record score')
            values.append(value)
        del tokens, logits, state, host, logp, picked
        if after_batch is not None:
            after_batch({'records': start+len(subset), 'batch_shape': [batch, length]})
    return values
