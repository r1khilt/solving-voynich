"""Original-model query-head captures after a post-block residual field edit."""

import numpy as np

from .attention_route_backend import AttentionRouteWorkspace
from .path3_backend import Path3Workspace


class Path8Workspace(Path3Workspace, AttentionRouteWorkspace):
    def final_heads_after_field(self, ids, *, upstream_layer, field, capture_layers):
        """Capture actual edited-run heads; no clean-donor head substitution."""
        mx = self.mx
        field = np.asarray(field, dtype=np.float32)
        selected = set(capture_layers)
        if (not ids or field.shape != (len(ids), self.width) or not np.isfinite(field).all()
                or not selected or any(type(i) is not int or not upstream_layer < i < self.n_layers
                                    for i in selected)):
            raise ValueError('Invalid PATH-0008 head capture')
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        captures = {}
        for layer, block in enumerate(self.layers):
            if layer in selected:
                attn = block.self_attn
                normalized = block.input_layernorm(h)
                length = len(ids)
                q = attn.q_norm(attn.q_proj(normalized).reshape(
                    1, length, attn.n_heads, -1)).transpose(0, 2, 1, 3)
                k = attn.k_norm(attn.k_proj(normalized).reshape(
                    1, length, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                v = attn.v_proj(normalized).reshape(
                    1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q, k = attn.rope(q), attn.rope(k)
                mx.eval(q, k, v)
                heads = self.heads_from_qkv(np.array(q[0, :, -1:]),
                                            np.array(k[0]), np.array(v[0]))
                projected = attn.o_proj(mx.array(heads.reshape(1, 1, -1)))[0, 0]
                write = attn(normalized, mask='causal' if length > 1 else None)
                native = write[0, -1]
                mx.eval(projected, native)
                error = float(np.max(np.abs(np.array(projected-native))))
                captures[layer] = {'heads': heads, 'native_error': error,
                                   'write': np.array(native)}
                middle = h+write
                h = middle+block.mlp(block.post_attention_layernorm(middle))
            else:
                h = block(h, mask='causal' if len(ids) > 1 else None)
            if layer == upstream_layer:
                h = h+mx.array(field)[None]
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        result = np.array(logits)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite PATH-0008 head-capture logits')
        return result, captures
