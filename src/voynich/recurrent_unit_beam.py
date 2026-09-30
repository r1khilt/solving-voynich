"""Known deterministic variable-unit reading with full-history beam states.

This is bounded approximate MAP search, not key discovery or exact marginal
inference. Every discarded prefix gets an admissible completion-score bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BeamReading:
    plaintext: str | None
    joint_log_probability: float | None
    discarded_completion_upper_bound: float | None
    score_bound_certifies_map: bool
    expanded_prefixes: int
    proposed_prefixes: int
    discarded_prefixes: int
    maximum_frontier: int
    beam_width: int
    support_empty: bool


@dataclass
class _Prefix:
    text: str
    score: float
    previous_state: Any
    last_token: int


class RecurrentProvider:
    """Advance the pending last input, then score next letters in float64."""
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        self.alphabet = model.config['alphabet']

    @torch.inference_mode()
    def advance(self, tokens, states):
        if all(state is None for state in states):
            if len(states) != 1 or tokens != [len(self.alphabet)]:
                raise ValueError('Only the initial BOS may lack a state')
            state = None
        else:
            if any(state is None for state in states):
                raise ValueError('Mixed initialized/uninitialized states')
            state = tuple(torch.cat([value[i] for value in states], dim=1) for i in range(2))
        inputs = torch.tensor(tokens, dtype=torch.long, device=self.device)[:, None]
        logits, following = self.model(inputs, state)
        logp = F.log_softmax(logits[:, 0].cpu().double(), dim=-1).numpy()
        next_states = [tuple(value[:, i:i + 1].contiguous() for value in following) for i in range(len(tokens))]
        return logp, next_states


def decode_beam(provider, units, observed, rho, *, beam_width=128, max_expanded=200_000):
    alphabet = provider.alphabet
    if (not alphabet or len(set(alphabet)) != len(alphabet) or len(units) != len(alphabet)
            or any(not isinstance(u, str) or not u for u in units) or not isinstance(observed, str)
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or type(beam_width) is not int or beam_width < 1
            or type(max_expanded) is not int or max_expanded < 1):
        raise ValueError('Invalid fixed beam/channel settings')
    n = len(observed)
    matches = [[(letter, i + len(unit)) for letter, unit in enumerate(units) if observed.startswith(unit, i)]
               for i in range(n)]
    # Channel-only backward reachability and minimum further source length.
    # Impossible suffixes can be deleted without a probabilistic approximation.
    minimum = [None] * n + [0]
    for i in range(n - 1, -1, -1):
        legal = [1 + minimum[end] for _, end in matches[i] if minimum[end] is not None]
        if legal:
            minimum[i] = min(legal)
    stop, cont = math.log(rho), math.log1p(-rho)
    if minimum[0] is None:
        return BeamReading(None, None, None, False, 0, 0, 0, 0, beam_width, True)
    pending = {0: [_Prefix('', 0., None, len(alphabet))]}
    expanded, proposed, discarded, maximum = 0, 1, 0, 1
    pruned_upper = -math.inf
    for offset in range(n + 1):
        frontier = pending.pop(offset, [])
        if not frontier:
            continue
        maximum = max(maximum, len(frontier))
        frontier.sort(key=lambda row: (-row.score, row.text))
        if offset == n:
            best = frontier[0]
            score = best.score + stop
            # This certificate is about the specified real-valued score rule.
            # Production float inference still requires separate numerical replay.
            return BeamReading(best.text, score, pruned_upper if discarded else None,
                               score >= pruned_upper, expanded, proposed, discarded, maximum, beam_width, False)
        if len(frontier) > beam_width:
            dropped = frontier[beam_width:]
            discarded += len(dropped)
            bound = dropped[0].score + minimum[offset] * cont + stop
            pruned_upper = max(pruned_upper, bound)
            frontier = frontier[:beam_width]
        expanded += len(frontier)
        if expanded > max_expanded:
            raise RuntimeError('Fixed neural beam expansion cap exceeded')
        logp, states = provider.advance([row.last_token for row in frontier],
                                        [row.previous_state for row in frontier])
        logp = np.asarray(logp, dtype=np.float64)
        if (logp.shape != (len(frontier), len(alphabet)) or len(states) != len(frontier)
                or np.any(~np.isfinite(logp)) or np.any(logp > 0)
                or not np.allclose(np.exp(logp).sum(axis=1), 1., atol=1e-10, rtol=0)):
            raise ValueError('Provider must return normalized finite nonpositive letter log probabilities')
        for row, probabilities, state in zip(frontier, logp, states):
            for letter, end in matches[offset]:
                if minimum[end] is None:
                    continue
                following = _Prefix(row.text + alphabet[letter], row.score + float(probabilities[letter]) + cont,
                                    state, letter)
                pending.setdefault(end, []).append(following)
                proposed += 1
    raise AssertionError('Channel reachability and positive source should admit a completion')
