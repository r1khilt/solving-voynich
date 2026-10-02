"""Local inference cache for the frozen reference reading-action transformer.

Training and all weights are unchanged. Cache lifetime is one sequential path;
it cannot be reused after updates, reordered queries or between observations.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F


def heads(value, count):
    return value.reshape(value.shape[0], value.shape[1], count, -1).transpose(1, 2)


def project(attention, value, part):
    width = attention.embed_dim
    lo = part*width
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[lo:lo+width]
    return heads(F.linear(value, attention.in_proj_weight[lo:lo+width], bias), attention.num_heads)


def attend(attention, query, keys, values, allowed=None):
    # All cached keys are past/current, so no triangular mask is needed. With
    # one query, is_causal=True would incorrectly expose only the first key.
    value = F.scaled_dot_product_attention(query, keys, values, attn_mask=allowed,
                                           dropout_p=0., is_causal=False)
    value = value.transpose(1, 2).reshape(value.shape[0], value.shape[2], attention.embed_dim)
    return F.linear(value, attention.out_proj.weight, attention.out_proj.bias)


class _PathCache:
    def __init__(self, model, environment):
        if model.training or torch.is_grad_enabled():
            raise ValueError('Inference cache requires evaluation and no-gradient scope')
        if any(not layer.norm_first or any(attention.bias_k is not None or attention.bias_v is not None
                or attention.add_zero_attn or attention.kdim != attention.embed_dim
                or attention.vdim != attention.embed_dim
                for attention in (layer.self_attn, layer.multihead_attn)) for layer in model.decoder):
            raise ValueError('Only the declared prenorm equal-dimension attention architecture is supported')
        c = model.config
        if (environment.rows != c.rows or environment.glyphs != c.glyphs
                or len(environment.records) > c.max_records
                or max(map(len, environment.records)) > c.max_glyphs):
            raise ValueError('Observed cipher exceeds the frozen model configuration')
        self.model, self.env, self.state = model, environment, environment.initial
        self.previous, self.cached_logits = 2*c.rows, None
        device = next(model.parameters()).device
        count, length = len(environment.records), max(map(len, environment.records))
        records = torch.full((1, count, length), c.glyphs, dtype=torch.long)
        for record, glyphs in enumerate(environment.records):
            records[0, record, :len(glyphs)] = torch.tensor(glyphs)
        records = records.to(device)
        padding = records == c.glyphs
        position = torch.arange(length, device=device)
        record_ids = torch.arange(count, device=device)
        memory = model.glyph(records)+model.position(position)[None, None]+model.record(record_ids)[None, :, None]
        memory = memory.reshape(count, length, -1)
        for layer in model.encoder:
            memory = layer(memory, src_key_padding_mask=padding.reshape(count, length))
        memory = model.encoder_norm(memory).reshape(1, count*length, -1)
        self.allowed = (~padding.reshape(1, count*length))[:, None, None, :]
        self.cross = [(project(layer.multihead_attn, memory, 1), project(layer.multihead_attn, memory, 2))
                      for layer in model.decoder]
        self.self_keys = [None]*len(model.decoder)
        self.self_values = [None]*len(model.decoder)

    def scores(self):
        if self.cached_logits is not None:
            return self.cached_logits
        m, env, state = self.model, self.env, self.state
        selected = env.selected_record(state)
        legal = env.legal_actions(state)
        if selected is None or not legal:
            raise ValueError('A terminal/dead state has no next-action query')
        device, c = next(m.parameters()).device, m.config
        units = len(env.pool)
        bindings = torch.tensor([[row*(units+1)+k+1 for row, k in enumerate(state.key)]], device=device)
        values = (m.previous(torch.tensor([[self.previous]], device=device))
            +m.offset(torch.tensor([[state.offsets[selected]]], device=device))
            +m.record(torch.tensor([[selected]], device=device))
            +m.binding(bindings).mean(dim=1)[:, None]*int(c.binding_input))
        for i, layer in enumerate(m.decoder):
            normalized = layer.norm1(values)
            attention = layer.self_attn
            q, k, v = (project(attention, normalized, part) for part in range(3))
            if self.self_keys[i] is not None:
                k = torch.cat((self.self_keys[i], k), dim=2)
                v = torch.cat((self.self_values[i], v), dim=2)
            self.self_keys[i], self.self_values[i] = k, v
            values = values+attend(attention, q, k, v)
            q = project(layer.multihead_attn, layer.norm2(values), 0)
            values = values+attend(layer.multihead_attn, q, *self.cross[i], self.allowed)
            normalized = layer.norm3(values)
            values = values+layer.linear2(layer.activation(layer.linear1(normalized)))
        scores = m.output(m.decoder_norm(values))[0, 0]
        allowed = torch.zeros(2*c.rows, dtype=torch.bool, device=device)
        allowed[list(legal)] = True
        self.cached_logits = scores.masked_fill(~allowed, -torch.inf)
        return self.cached_logits

    def advance(self, action):
        if self.cached_logits is None:
            raise ValueError('Every action must follow its own cached query')
        self.state = self.env.advance(self.state, action)
        self.previous, self.cached_logits = action, None


@torch.no_grad()
def cached_trace_logits(model, environment, trace):
    """Reference qualification on a supplied legal teaching trace, not recovery."""
    was_training = model.training
    model.eval()
    try:
        # Reference pack verifies every snapshot, complete ending and legal action.
        model.pack([environment], [trace])
        cache, values = _PathCache(model, environment), []
        for action in trace[1]:
            values.append(cache.scores())
            cache.advance(action)
        assert cache.state == trace[2]
        return torch.stack(values)
    finally:
        model.train(was_training)


@torch.no_grad()
def propose_cached(model, environment, *, seed=0, temperature=1.):
    """One finite sampled path; no Gold, beam, repair or success conditioning."""
    if (type(seed) is not int or seed < 0 or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Fixed seed and positive finite temperature required')
    was_training = model.training
    model.eval()
    try:
        cache, rng, actions, log_probability = _PathCache(model, environment), np.random.default_rng(seed), [], 0.
        for _ in range(sum(map(len, environment.records))):
            if environment.selected_record(cache.state) is None:
                break
            legal = environment.legal_actions(cache.state)
            if not legal:
                return {'status': 'dead_end', 'state': cache.state, 'actions': tuple(actions),
                        'path_log_probability': log_probability}
            values = cache.scores().cpu().double().numpy()[list(legal)]/temperature
            if not np.all(np.isfinite(values)):
                raise FloatingPointError('Nonfinite legal cached logits; no repaired proposal')
            values -= values.max()
            probabilities = np.exp(values)
            probabilities /= probabilities.sum()
            index = int(rng.choice(len(legal), p=probabilities))
            log_probability += math.log(float(probabilities[index]))
            action = legal[index]
            actions.append(action)
            cache.advance(action)
        if environment.selected_record(cache.state) is not None:
            raise AssertionError('Every emitted unit must consume at least one observed glyph')
        return {'status': 'complete_path', 'state': cache.state, 'actions': tuple(actions),
                'path_log_probability': log_probability}
    finally:
        model.train(was_training)
