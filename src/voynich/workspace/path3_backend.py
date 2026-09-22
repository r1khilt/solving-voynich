"""Original-model multi-block final-attention cuts after a value-field edit."""

import numpy as np

from .route_backend import RouteWorkspace


class Path3Workspace(RouteWorkspace):
    def clean_final_writes(self, ids, start=16):
        mx = self.mx
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        writes = {}
        for layer, block in enumerate(self.layers):
            if layer >= start:
                write = block.self_attn(block.input_layernorm(h), mask='causal' if len(ids) > 1 else None)
                mx.eval(write)
                writes[layer] = np.array(write[0, -1])
                middle = h+write
                h = middle+block.mlp(block.post_attention_layernorm(middle))
            else:
                h = block(h, mask='causal' if len(ids) > 1 else None)
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        return np.array(logits), writes

    def prefill_cut(self, ids, *, upstream_layer, upstream_field, clean_writes,
                    cut_layers=(), random_deltas=None, capture_writes=False):
        from mlx_lm.models.cache import KVCache

        mx = self.mx
        field = np.asarray(upstream_field, dtype=np.float32)
        selected = set(cut_layers)
        if not ids or field.shape != (len(ids), self.width) or not np.isfinite(field).all():
            raise ValueError('Invalid upstream edit')
        if any(layer <= upstream_layer or layer >= self.n_layers for layer in selected):
            raise ValueError('Cut must follow upstream edit')
        if selected and any(layer not in clean_writes for layer in selected):
            raise ValueError('Missing clean attention write')
        if random_deltas is not None and any(layer not in random_deltas for layer in selected):
            raise ValueError('Missing random attention delta')
        cache = [KVCache() for _ in self.layers]
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        writes = {}
        for layer, (block, kv) in enumerate(zip(self.layers, cache)):
            if layer > upstream_layer:
                write = block.self_attn(block.input_layernorm(h), mask='causal' if len(ids) > 1 else None, cache=kv)
                if capture_writes:
                    mx.eval(write)
                    writes[layer] = np.array(write[0, -1])
                if layer in selected:
                    if random_deltas is None:
                        replacement = mx.array(np.asarray(clean_writes[layer], dtype=np.float32))
                        write = write.at[0, -1].add(replacement-write[0, -1])
                    else:
                        write = write.at[0, -1].add(mx.array(random_deltas[layer]))
                middle = h+write
                h = middle+block.mlp(block.post_attention_layernorm(middle))
            else:
                h = block(h, mask='causal' if len(ids) > 1 else None, cache=kv)
                if layer == upstream_layer:
                    h = h+mx.array(field)[None]
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        result = np.array(logits)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite cut logits')
        return result, cache, writes

    def generate_cut(self, ids, **kwargs):
        logits, cache, writes = self.prefill_cut(ids, **kwargs)
        initial = logits.copy()
        eos = self.config['eos_token_id']
        eos = {eos} if isinstance(eos, int) else set(eos)
        generated = []
        for _ in range(12):
            token = int(logits.argmax())
            if token in eos:
                break
            generated.append(token)
            if len(generated) < 12:
                logits = self.cached_step(token, cache)
        return self.tokenizer.decode(generated), generated, initial, writes
