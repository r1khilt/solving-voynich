"""Full-prefix residual fields for a separate positional-routing diagnostic."""

import numpy as np

from .causal_backend import CausalWorkspace


class RouteWorkspace(CausalWorkspace):
    def prefill_field(self, ids, *, fields=None, capture_layers=()):
        from mlx_lm.models.cache import KVCache

        if not ids:
            raise ValueError('Empty input')
        fields = {} if fields is None else fields
        mx, edits = self.mx, {}
        for layer, values in fields.items():
            values = np.asarray(values, dtype=np.float32)
            if type(layer) is not int or not 0 <= layer < self.n_layers or values.shape != (len(ids), self.width) or not np.isfinite(values).all():
                raise ValueError('Invalid residual field')
            edits[layer] = mx.array(values)[None]
        cache = [KVCache() for _ in self.layers]
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        captures = {}
        for index, (block, kv) in enumerate(zip(self.layers, cache)):
            h = block(h, mask='causal' if len(ids) > 1 else None, cache=kv)
            if index in edits:
                h = h + edits[index]
            if index in capture_layers:
                captures[index] = h[0]
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits, *captures.values())
        logits = np.array(logits)
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite field logits')
        return logits, cache, {k: np.array(v) for k, v in captures.items()}

    def generate_field(self, ids, *, fields=None, capture_layers=(), max_tokens=12):
        logits, cache, captures = self.prefill_field(ids, fields=fields, capture_layers=capture_layers)
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
        return self.tokenizer.decode(generated), generated, captures, initial


def positional_field(delta, scope):
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim != 2 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Invalid positional displacement')
    result = delta.copy()
    if scope == 'last':
        result[:-1] = 0
    elif scope == 'earlier':
        result[-1] = 0
    elif scope != 'all':
        raise ValueError('Unknown scope')
    return result
