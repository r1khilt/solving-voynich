"""Capture the unchanged sampler's actual logits; independently replay its law.

Instrumentation is local to this process and never changes a score. This checks
visited finite queries, not global floating-point support or posterior mixing.
"""

import math
import time
from unittest.mock import patch

import numpy as np
import torch

from voynich.source_action_cache import _PathCache, propose_cached


def replay_draws(environment, actions, legal_logits, *, seed):
    """Reconstruct literal masks, PCG draws and normalized path probability."""
    rng, state, terms = np.random.default_rng(seed), environment.initial, []
    if len(actions) != len(legal_logits):
        raise ValueError('Every sampled action requires its original query logits')
    for action, values in zip(actions, legal_logits, strict=True):
        legal = environment.legal_actions(state)
        scores = np.asarray(values, dtype=np.float64)
        if scores.shape != (len(legal),) or not np.isfinite(scores).all():
            raise ValueError('Finite scores for exactly the literal legal actions required')
        probabilities = np.exp(scores-scores.max())
        probabilities /= probabilities.sum()
        if not np.all(probabilities > 0):
            raise FloatingPointError('Visited softmax support underflowed; no repair')
        index = int(rng.choice(len(legal), p=probabilities))
        if legal[index] != action:
            raise ValueError('Recorded stochastic action differs from its actual law/seed')
        terms.append(math.log(float(probabilities[index])))
        state = environment.advance(state, action)
    return state, math.fsum(terms)


@torch.no_grad()
def sample_and_check(model, environment, *, seed):
    """One existing-policy draw, then one full causal reference prefix replay."""
    captured = []
    original = _PathCache.scores

    def observe(cache):
        scores = original(cache)
        legal = cache.env.legal_actions(cache.state)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[list(legal)] = True
        if not torch.equal(torch.isfinite(scores), mask):
            raise ArithmeticError('Actual sampled legal mask differs')
        captured.append(scores.detach().cpu().double()[list(legal)].tolist())
        return scores

    began = time.monotonic()
    with patch.object(_PathCache, 'scores', observe):
        prediction = propose_cached(model, environment, seed=seed, temperature=1.)
    sample_seconds = time.monotonic()-began
    state, replay_logq = replay_draws(environment, prediction['actions'], captured, seed=seed)
    if state != prediction['state'] or abs(replay_logq-prediction['path_log_probability']) > 1e-9:
        raise ArithmeticError('Actual sampled state or path density differs')
    complete = environment.selected_record(state) is None
    if prediction['status'] != ('complete_path' if complete else 'dead_end'):
        raise ArithmeticError('Sample termination differs from literal replay')
    if not complete and environment.legal_actions(state):
        raise ArithmeticError('A failed draw ended before an actual dead end')
    snapshots, prefix = [], environment.initial
    for action in prediction['actions']:
        snapshots.append(prefix)
        prefix = environment.advance(prefix, action)
    began = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        packed = model.pack([environment], [(tuple(snapshots), prediction['actions'], state)], require_complete=False)
        reference = model(packed)[0].detach().cpu().double()
        if not torch.equal(torch.isfinite(reference), packed[5][0].cpu()):
            raise ArithmeticError('Reference mask differs on the sampled prefix')
        reference_values = [reference[j, list(environment.legal_actions(s))].tolist()
                            for j, s in enumerate(snapshots)]
        delta = max(abs(a-b) for x, y in zip(captured, reference_values, strict=True)
                    for a, b in zip(x, y, strict=True))
        reference_logq = float(reference.log_softmax(-1)[torch.arange(len(snapshots)),
                                                       torch.tensor(prediction['actions'])].sum())
        if delta > .002 or abs(reference_logq-replay_logq) > .01:
            raise ArithmeticError('Full causal reference differs from sampled cache beyond fixed tolerance')
    finally:
        model.train(was_training)
    return prediction, captured, reference_values, {'sample_wall_seconds': sample_seconds,
        'reference_wall_seconds': time.monotonic()-began, 'maximum_legal_logit_delta': delta,
        'reference_path_logq': reference_logq, 'actual_path_logq': prediction['path_log_probability'],
        'maximum_path_logq_delta': abs(reference_logq-replay_logq)}
