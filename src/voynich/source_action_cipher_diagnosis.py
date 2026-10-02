"""Controlled neural cipher routes and causal-history source baselines.

These interventions retain the literal observation/task for action selection.
They do not erase all ciphertext information or identify a minimal circuit.
"""

import math
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from voynich import source_action_cache as caching
from voynich.source_action_diagnosis import action_groups
from voynich.source_action_training import greedy_reading

NEURAL_MODES = ('base', 'sham', 'shuffle', 'cross_bias', 'cross_bias_erase')
SOURCE_MODES = ('row_only', 'source_increment')


def permuted_glyph_input(values, glyphs, seed):
    """One fixed per-record suffix permutation; lengths/counts/padding retained."""
    if values.ndim != 3 or values.shape[0] != 1 or values.dtype != torch.long:
        raise ValueError('Declared one-case integer glyph packet required')
    rng, result = np.random.default_rng(seed), values.clone()
    for r in range(values.shape[1]):
        length = int((values[0, r] != glyphs).sum())
        original = values[0, r, :length].cpu().tolist()
        if (not original or any(v < 0 or v >= glyphs for v in original)
                or (values[0, r, length:] != glyphs).any()):
            raise ValueError('Only contiguous right-padded glyph records supported')
        cut = max(original.index(v) for v in set(original))+1
        suffix = rng.permutation(original[cut:]).tolist()
        result[0, r, cut:length] = torch.tensor(suffix, device=values.device)
    return result


@contextmanager
def neural_routes(model, mode, *, seed=92671):
    """Both full MHA and manual cache routes; restore every hook on exit."""
    if mode not in NEURAL_MODES or model.training or torch.is_grad_enabled():
        raise ValueError('Fixed inference-only declared neural intervention required')
    cross = {id(layer.multihead_attn) for layer in model.decoder}
    counts = {'glyph_calls': 0, 'binding_calls': 0, 'full_cross_calls': 0, 'cached_cross_calls': 0}
    handles = []

    def glyph_input(module, args):
        counts['glyph_calls'] += 1
        if mode == 'shuffle':
            return (permuted_glyph_input(args[0], model.config.glyphs, seed),)
        if mode == 'sham':
            return (args[0].clone(),)
        return None

    def binding_input(module, args):
        counts['binding_calls'] += 1
        if mode == 'cross_bias_erase':
            values = args[0]
            rows = model.config.rows
            if values.shape[-1] != rows:
                raise ValueError('Declared row slots required')
            stride = module.num_embeddings//rows
            baseline = torch.arange(rows, device=values.device)*stride
            if ((values-baseline < 0).any() or (values-baseline >= stride).any()):
                raise ValueError('Unpadded row-identity binding slots required')
            return (baseline.expand_as(values),)
        if mode == 'sham':
            return (args[0].clone(),)
        return None

    def bias_output(attention, values):
        result = torch.zeros_like(values)
        if attention.out_proj.bias is not None:
            result = result+attention.out_proj.bias
        return result

    def full_cross(module, args, output):
        counts['full_cross_calls'] += 1
        if mode in ('cross_bias', 'cross_bias_erase'):
            return (bias_output(module, output[0]), *output[1:])
        if mode == 'sham':
            return (output[0].clone(), *output[1:])
        return None

    original = caching.attend

    def cached_attention(attention, query, keys, values, allowed=None):
        if id(attention) in cross:
            counts['cached_cross_calls'] += 1
            if mode in ('cross_bias', 'cross_bias_erase'):
                shape = (query.shape[0], query.shape[2], attention.embed_dim)
                return bias_output(attention, query.new_zeros(shape))
            result = original(attention, query, keys, values, allowed)
            return result.clone() if mode == 'sham' else result
        return original(attention, query, keys, values, allowed)

    try:
        handles.extend((model.glyph.register_forward_pre_hook(glyph_input),
                        model.binding.register_forward_pre_hook(binding_input)))
        handles.extend(layer.multihead_attn.register_forward_hook(full_cross) for layer in model.decoder)
        with patch.object(caching, 'attend', cached_attention):
            yield counts
    finally:
        for handle in handles:
            handle.remove()


def source_action_scores(source, environment, state, mode, *, rho=1/225):
    """Past-only source context + original symbolic mask, no whole cipher encoder."""
    if mode not in SOURCE_MODES or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError('Declared source policy and proper geometric EOS required')
    legal = environment.legal_actions(state)
    selected = environment.selected_record(state)
    if not legal or selected is None or len(source.alphabet) != environment.rows:
        raise ValueError('Nonterminal supported source alphabet required')
    history = ''.join(source.alphabet[a] for a in state.texts[selected])
    probabilities = source.row(source.state(history))
    counts = np.bincount([a//2 for a in legal], minlength=environment.rows)
    result = torch.full((2*environment.rows,), -torch.inf, dtype=torch.float64)
    for action in legal:
        row, length = action//2, action%2+1
        value = math.log(float(probabilities[row]))
        if mode == 'row_only':
            value -= math.log(int(counts[row]))
        else:
            value += math.log1p(-rho)
            if state.key[row] < 0:
                value -= math.log(len(environment.pool))
            if state.offsets[selected]+length == len(environment.records[selected]):
                value += math.log(rho)
        result[action] = value
    if not torch.isfinite(result[list(legal)]).all():
        raise FloatingPointError('Finite source-policy legal scores required')
    return result


def source_teacher_logits(source, environment, trace, mode):
    return torch.stack([source_action_scores(source, environment, state, mode) for state in trace[0]])


def source_greedy(source, environment, mode):
    """Deterministic choice; reported probability is under its conditional policy."""
    state, actions, logq, values = environment.initial, [], 0., []
    for _ in range(sum(map(len, environment.records))):
        if environment.selected_record(state) is None:
            break
        legal = environment.legal_actions(state)
        if not legal:
            break
        logits = source_action_scores(source, environment, state, mode)
        action = int(logits.argmax())
        values.append(logits[list(legal)].tolist())
        logq += float(logits.log_softmax(-1)[action])
        actions.append(action)
        state = environment.advance(state, action)
    complete = environment.selected_record(state) is None
    if not complete and environment.legal_actions(state):
        raise ArithmeticError('A source-policy path must finish or actually die')
    return {'status': 'complete_path' if complete else 'dead_end', 'key': state.key,
        'texts': state.texts, 'actions': tuple(actions), 'greedy_path_under_model_log_probability': logq}, values


def measure(logits, trace, competitors=None):
    """Action/row/length performance and exact category-loss decomposition."""
    snapshots, actions, _ = trace
    targets = torch.tensor(actions)
    logp = logits.double().log_softmax(-1)
    prediction = logits.argmax(-1)
    target = logp[torch.arange(len(actions)), targets]
    groups = action_groups(trace)
    unbound = torch.tensor([[k < 0 for k in s.key for _ in (0, 1)] for s in snapshots])
    new = torch.tensor([g == 'first_binding' for g in groups])
    category = torch.where(new[:, None], unbound, ~unbound)
    category_logp = logp.masked_fill(~category, -torch.inf).logsumexp(-1)
    within = -target+category_logp
    forced = torch.isfinite(logits).sum(-1) == 1
    correct, row_correct, length_correct = prediction == targets, prediction//2 == targets//2, prediction%2 == targets%2
    if not torch.isfinite(target).all() or not torch.isfinite(category_logp).all():
        raise ArithmeticError('Every true action/category must remain supported')
    if competitors is None:
        alternative = logits.clone()
        alternative[torch.arange(len(actions)), targets] = -torch.inf
        candidate = alternative.argmax(-1).tolist()
        competitors = [None if bool(forced[j]) else int(a) for j, a in enumerate(candidate)]
    if len(competitors) != len(actions):
        raise ValueError('Fixed base competitor for every query required')
    margins, competitor_logq = [], []
    for j, a in enumerate(competitors):
        if a is None:
            if not bool(forced[j]):
                raise ValueError('Only forced queries may lack a competitor')
            margins.append(None)
            competitor_logq.append(None)
        else:
            if type(a) is not int or a == actions[j] or not 0 <= a < logits.shape[-1] or not torch.isfinite(logits[j, a]):
                raise ValueError('Fixed distinct legal competitor required')
            margins.append(float(logp[j, actions[j]]-logp[j, a]))
            competitor_logq.append(float(logp[j, a]))
    metrics = {}
    for group in ('first_binding', 'reuse'):
        mask = torch.tensor([g == group for g in groups])
        metrics[group] = {'actions': int(mask.sum()), 'correct_actions': int(correct[mask].sum()),
            'correct_rows': int(row_correct[mask].sum()), 'correct_lengths': int(length_correct[mask].sum()),
            'forced_actions': int(forced[mask].sum()), 'nll_sum': float(-target[mask].sum()),
            'category_nll_sum': float(-category_logp[mask].sum()),
            'within_category_nll_sum': float(within[mask].sum()),
            'fixed_margin_queries': sum(m is not None for m, g in zip(margins, groups, strict=True) if g == group),
            'fixed_margin_sum': math.fsum(m for m, g in zip(margins, groups, strict=True) if g == group and m is not None)}
    arrays = {'target_logq': target.tolist(), 'prediction': prediction.tolist(),
              'category_logq': category_logp.tolist(), 'groups': groups,
              'fixed_competitor_actions': competitors, 'competitor_logq': competitor_logq,
              'target_minus_fixed_competitor': margins}
    return {'whole_path_nll': float(-target.sum()), 'groups': metrics}, arrays


@torch.no_grad()
def neural_greedy_checked(model, environment, mode, *, seed, observe=lambda: None):
    """One greedy path with the registered intervention and full prefix replay."""
    snapshots, captured = [], []
    original = caching._PathCache.scores

    def record(cache):
        observe()
        scores = original(cache)
        legal = environment.legal_actions(cache.state)
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        allowed[list(legal)] = True
        if not torch.equal(torch.isfinite(scores), allowed):
            raise ArithmeticError('Cached legal mask differs')
        snapshots.append(cache.state)
        captured.append(scores.detach().cpu().double()[list(legal)].tolist())
        return scores

    with neural_routes(model, mode, seed=seed) as counts:
        with patch.object(caching._PathCache, 'scores', record):
            prediction = greedy_reading(model, environment)
        state = environment.initial
        for action in prediction['actions']:
            state = environment.advance(state, action)
        if state.key != prediction['key'] or state.texts != prediction['texts']:
            raise ArithmeticError('Recorded greedy output is not literal')
        packed = model.pack([environment], [(tuple(snapshots), prediction['actions'], state)], require_complete=False)
        reference = model(packed)[0].detach().cpu().double()
        if not torch.equal(torch.isfinite(reference), packed[5][0].cpu()):
            raise ArithmeticError('Reference prefix mask differs')
        selected = reference[torch.arange(len(snapshots)), torch.tensor(prediction['actions'])]
        deficit = float((reference.max(-1).values-selected).max())
        reference_logq = float(reference.log_softmax(-1)[torch.arange(len(snapshots)),
                                                        torch.tensor(prediction['actions'])].sum())
        reference_values = [reference[j, list(environment.legal_actions(s))].tolist()
                            for j, s in enumerate(snapshots)]
        delta = max(abs(a-b) for x, y in zip(captured, reference_values, strict=True)
                    for a, b in zip(x, y, strict=True))
        qdelta = abs(reference_logq-prediction['greedy_path_under_model_log_probability'])
        layers = len(model.decoder)
        if (counts['full_cross_calls'] != layers or counts['cached_cross_calls'] != layers*len(snapshots)
                or counts['glyph_calls'] != 2 or counts['binding_calls'] != len(snapshots)+1):
            raise ArithmeticError('Intervention missed a declared full/cached route')
        if delta > .002 or qdelta > .01 or deficit > .0002:
            raise ArithmeticError('Intervened cached/full prefix differs beyond fixed tolerance')
        for action, values, s in zip(prediction['actions'], captured, snapshots, strict=True):
            legal = environment.legal_actions(s)
            if action != legal[int(np.argmax(values))]:
                raise ArithmeticError('Greedy tie/choice differs from recorded logits')
    return prediction, {'actual_legal_logits': captured, 'reference_legal_logits': reference_values}, {
        'maximum_legal_logit_delta': delta, 'maximum_path_logq_delta': qdelta,
        'maximum_reference_argmax_deficit': deficit, 'reference_path_logq': reference_logq,
        'route_counts': dict(counts)}
