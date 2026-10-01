"""Fixed one-step batches with independent CPU-owned recurrent cache states.

Padding changes execution shape only. Each real row retains its full recurrent
history; dummy rows never contribute a score, state, hypothesis or prior mass.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


class FixedTransitionProvider:
    def __init__(self, model, device="cpu", *, batch=8, after_batch=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("Positive fixed transition batch required")
        self.model, self.device = model.eval(), device
        self.alphabet = model.config["alphabet"]
        self.batch, self.after_batch = batch, after_batch
        self.calls, self.real_rows, self.padded_rows = 0, 0, 0

    @torch.inference_mode()
    def advance(self, tokens, states):
        if (not tokens or len(tokens) != len(states)
                or any(type(t) is not int or not 0 <= t <= len(self.alphabet) for t in tokens)):
            raise ValueError("Invalid transition tokens or state inventory")
        if any(s is None for s in states):
            if states != [None] or tokens != [len(self.alphabet)]:
                raise ValueError("Only the initial BOS may lack a state")
        dtype = next(self.model.parameters()).dtype
        layers, width = self.model.config["layers"], self.model.config["width"]
        for state in states:
            if state is not None and (not isinstance(state, tuple) or len(state) != 2
                    or any(not isinstance(v, torch.Tensor) or v.shape != (layers, 1, width)
                           or v.dtype != dtype or v.device.type != "cpu" for v in state)):
                raise ValueError("Independent CPU cache states with matching dimensions required")
        scores, following = [], []
        for start in range(0, len(tokens), self.batch):
            count = min(self.batch, len(tokens)-start)
            inputs = torch.full((self.batch, 1), len(self.alphabet), dtype=torch.long, device=self.device)
            inputs[:count, 0] = torch.tensor(tokens[start:start+count], device=self.device)
            cached = tuple(torch.zeros((layers, self.batch, width), dtype=dtype, device=self.device)
                           for _ in range(2))
            for row, state in enumerate(states[start:start+count]):
                if state is not None:
                    for i in range(2):
                        cached[i][:, row:row+1] = state[i].to(self.device)
            logits, advanced = self.model(inputs, cached)
            scores.append(F.log_softmax(logits[:count, 0].cpu().double(), dim=-1))
            host = tuple(v.cpu() for v in advanced)
            # Clone even one-layer contiguous views: no real row owns a padded
            # batch's storage, and no following row aliases a cached parent.
            following.extend(tuple(v[:, row:row+1].clone() for v in host) for row in range(count))
            self.calls += 1
            self.real_rows += count
            self.padded_rows += self.batch-count
            if self.after_batch is not None:
                self.after_batch({"calls": self.calls, "real_rows": self.real_rows,
                                  "padded_rows": self.padded_rows, "shape": [self.batch, 1]})
        return torch.cat(scores).numpy(), following
