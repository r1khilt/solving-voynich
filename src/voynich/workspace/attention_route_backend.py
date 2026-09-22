"""Direct Qwen attention-head and value-route interventions on the original model.

Only the final prompt position is edited after one block's attention operation,
before its MLP. Later blocks and generated-token steps execute normally.
"""

import numpy as np

from .route_backend import RouteWorkspace


class AttentionRouteWorkspace(RouteWorkspace):
    def heads_from_qkv(self, q, k, v):
        """Final-position outputs, one 128-dimensional vector per query head."""
        mx = self.mx
        q, k, v = (mx.array(np.asarray(item, dtype=np.float32))[None] for item in (q, k, v))
        if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[-2] != 1:
            raise ValueError('Invalid attention-route QKV shapes')
        values = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1]**-.5, mask=None)
        mx.eval(values)
        result = np.array(values[0, :, 0])
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite route heads')
        return result

    def capture_heads(self, ids, layers):
        """Capture actual per-head outputs and Q/K/V at named blocks."""
        mx = self.mx
        selected = set(layers)
        if not ids or not selected or any(type(i) is not int or not 0 <= i < self.n_layers for i in selected):
            raise ValueError('Invalid attention capture')
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        captures = {}
        for i, block in enumerate(self.layers):
            if i in selected:
                attn = block.self_attn
                normalized = block.input_layernorm(h)
                length = len(ids)
                q = attn.q_norm(attn.q_proj(normalized).reshape(1, length, attn.n_heads, -1)).transpose(0, 2, 1, 3)
                k = attn.k_norm(attn.k_proj(normalized).reshape(1, length, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                v = attn.v_proj(normalized).reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q, k = attn.rope(q), attn.rope(k)
                mx.eval(q, k, v)
                arrays = {'q': np.array(q[0, :, -1:]), 'k': np.array(k[0]), 'v': np.array(v[0])}
                heads = self.heads_from_qkv(**arrays)
                manual = attn.o_proj(mx.array(heads.reshape(1, 1, -1)))[0, 0]
                native = attn(normalized, mask='causal' if length > 1 else None)[0, -1]
                mx.eval(manual, native)
                error = float(np.max(np.abs(np.array(manual-native))))
                arrays.update({'heads': heads, 'native_error': error})
                captures[i] = arrays
            h = block(h, mask='causal' if len(ids) > 1 else None)
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        return np.array(logits), captures

    def head_delta(self, layer, source_heads, replacement_heads, selected):
        """Projected write from replacing only selected query-head outputs."""
        mx = self.mx
        source = np.asarray(source_heads, dtype=np.float32)
        replacement = np.asarray(replacement_heads, dtype=np.float32)
        attn = self.layers[layer].self_attn
        if source.shape != (attn.n_heads, attn.q_norm.weight.shape[0]) or replacement.shape != source.shape:
            raise ValueError('Head matrix shape mismatch')
        chosen = tuple(selected)
        if len(chosen) != len(set(chosen)) or any(type(i) is not int or not 0 <= i < attn.n_heads for i in chosen):
            raise ValueError('Invalid head selection')
        change = np.zeros_like(source)
        change[list(chosen)] = replacement[list(chosen)]-source[list(chosen)]
        projected = attn.o_proj(mx.array(change.reshape(1, 1, -1)))[0, 0]
        mx.eval(projected)
        result = np.array(projected)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite head edit')
        return result

    def prefill_attention_delta(self, ids, *, layer, delta):
        """Single final-site post-attention edit, preserving native prefix cache."""
        from mlx_lm.models.cache import KVCache

        mx = self.mx
        delta = np.asarray(delta, dtype=np.float32)
        if not ids or type(layer) is not int or not 0 <= layer < self.n_layers or delta.shape != (self.width,) or not np.isfinite(delta).all():
            raise ValueError('Invalid attention delta')
        cache = [KVCache() for _ in self.layers]
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        for i, (block, kv) in enumerate(zip(self.layers, cache)):
            if i == layer:
                attention = block.self_attn(block.input_layernorm(h), mask='causal' if len(ids) > 1 else None, cache=kv)
                post_attention = h+attention
                post_attention = post_attention.at[0, -1].add(mx.array(delta))
                h = post_attention+block.mlp(block.post_attention_layernorm(post_attention))
            else:
                h = block(h, mask='causal' if len(ids) > 1 else None, cache=kv)
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        result = np.array(logits)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite attention-delta logits')
        return result, cache

    def generate_attention_delta(self, ids, *, layer, delta, max_tokens=12):
        logits, cache = self.prefill_attention_delta(ids, layer=layer, delta=delta)
        initial = logits.copy()
        eos = self.config['eos_token_id']
        eos = {eos} if isinstance(eos, int) else set(eos)
        generated = []
        for _ in range(max_tokens):
            token = int(logits.argmax())
            if token in eos:
                break
            generated.append(token)
            if len(generated) < max_tokens:
                logits = self.cached_step(token, cache)
        return self.tokenizer.decode(generated), generated, initial
