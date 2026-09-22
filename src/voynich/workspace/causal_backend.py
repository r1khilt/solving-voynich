"""Cached execution and neuron tracing of the frozen residual-edit semantics.

This extension is qualified against the original full-recomputation adapter.
It does not alter the independently running corpus-calibration implementation.
"""

import numpy as np

from .mlx_backend import QwenWorkspace


class CausalWorkspace(QwenWorkspace):
    def prefill(self, ids, *, patches=(), capture_layers=()):
        from mlx_lm.models.cache import KVCache

        mx = self.mx
        if not ids:
            raise ValueError("Empty prefix")
        edits = {}
        for p in patches:
            delta = np.asarray(p["delta"], dtype=np.float32)
            if (type(p["layer"]) is not int or not 0 <= p["layer"] < self.n_layers
                    or type(p["position"]) is not int or not 0 <= p["position"] < len(ids)
                    or delta.shape != (self.width,) or not np.isfinite(delta).all()):
                raise ValueError("Invalid cached intervention")
            edits.setdefault(p["layer"], []).append((p["position"], mx.array(delta)))
        cache = [KVCache() for _ in self.layers]
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        captures = {}
        for i, (layer, kv) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask="causal" if len(ids) > 1 else None, cache=kv)
            for position, delta in edits.get(i, ()):
                h = h.at[0, position].add(delta)
            if i in capture_layers:
                captures[i] = h[0, -1]
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits, *captures.values())
        return np.array(logits), cache, {k: np.array(v) for k, v in captures.items()}

    def cached_step(self, token, cache):
        mx = self.mx
        ids = mx.array([[token]])
        logits = self.model(ids, cache=cache, input_embeddings=self.model.model.embed_tokens(ids).astype(mx.float32))[0, 0]
        mx.eval(logits)
        value = np.array(logits)
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite cached logits")
        return value

    def greedy_cached(self, ids, *, patches=(), max_tokens=12, capture_layers=()):
        logits, cache, captures = self.prefill(ids, patches=patches, capture_layers=capture_layers)
        eos = self.config["eos_token_id"]
        eos = {eos} if isinstance(eos, int) else set(eos)
        generated = []
        for _ in range(max_tokens):
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite cached logits")
            token = int(logits.argmax())
            if token in eos:
                break
            generated.append(token)
            if len(generated) < max_tokens:
                logits = self.cached_step(token, cache)
        return self.tokenizer.decode(generated), generated, captures

    def neuron_trace(self, ids, *, patches=(), position=-1):
        """Explicit Qwen block equations, last-site SwiGLU and residual traces.

        Neurons are coordinates in the MLP expansion; activations are not claimed
        to be monosemantic. Trace includes attention and MLP residual writes.
        """
        from mlx_lm.models.activations import swiglu

        mx = self.mx
        pos = position % len(ids)
        if not -len(ids) <= position < len(ids):
            raise ValueError("Trace position out of range")
        edits = {}
        for p in patches:
            delta = np.asarray(p["delta"], dtype=np.float32)
            if not 0 <= p["layer"] < self.n_layers or not 0 <= p["position"] < len(ids) or delta.shape != (self.width,):
                raise ValueError("Invalid traced intervention")
            edits.setdefault(p["layer"], []).append((p["position"], mx.array(delta)))
        h = self.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32)
        trace = {key: [] for key in ("residual", "attention", "mlp_write", "neurons")}
        for i, layer in enumerate(self.layers):
            attention = layer.self_attn(layer.input_layernorm(h), mask="causal" if len(ids) > 1 else None)
            h = h + attention
            normalized = layer.post_attention_layernorm(h)
            neurons = swiglu(layer.mlp.gate_proj(normalized), layer.mlp.up_proj(normalized))
            write = layer.mlp.down_proj(neurons)
            h = h + write
            for p, delta in edits.get(i, ()):
                h = h.at[0, p].add(delta)
            # Materialize only the selected position, not the whole token grid.
            selected = (h[0, pos], attention[0, pos], write[0, pos], neurons[0, pos])
            mx.eval(*selected)
            for key, value in zip(trace, selected):
                trace[key].append(np.array(value))
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:]))[0, 0]
        mx.eval(logits)
        return np.array(logits), {key: np.stack(values) for key, values in trace.items()}


def qualify_cached(model):
    """Generic prompts only; no development/final label-based selection."""
    prompts = ['The quiet library contains many books about',
               'A careful reader compares two explanations before making a decision.',
               'Return only a short word. What color is a clear daytime sky?']
    rows = []
    rng = np.random.default_rng(51022)
    for p, prompt in enumerate(prompts):
        ids = model.encode(prompt, chat=True)
        delta = rng.normal(size=model.width).astype(np.float32)
        delta *= 3 / np.linalg.norm(delta)
        for condition in ("clean", "patched"):
            patches = [] if condition == "clean" else [{"layer": 15, "position": len(ids)-1, "delta": delta}]
            current = list(ids)
            logits, cache, _ = model.prefill(ids, patches=patches)
            for step in range(4):
                reference, _ = model.forward(current, patches=patches)
                error = float(np.max(np.abs(logits - reference[0])))
                rows.append({"prompt": p, "condition": condition, "step": step, "max_logit_error": error,
                             "argmax_same": int(logits.argmax()) == int(reference[0].argmax())})
                token = int(reference[0].argmax())
                current.append(token)
                logits = model.cached_step(token, cache)
        clean, _ = model.forward(ids)
        traced, _ = model.neuron_trace(ids)
        rows.append({"prompt": p, "condition": "neuron_trace", "max_logit_error": float(np.max(np.abs(clean[0]-traced))),
                     "argmax_same": int(clean[0].argmax()) == int(traced.argmax())})
    return {"passed": all(r["argmax_same"] and r["max_logit_error"] < .002 for r in rows), "rows": rows}
