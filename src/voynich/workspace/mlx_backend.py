"""Local Qwen3 residual interventions and exact selected-row Jacobian transport.

No remote model code, uploads, or automatic model downloads. Weight quantization
is preserved; residual computation is explicitly float32 throughout this adapter.
"""

from pathlib import Path
import os
import numpy as np


class QwenWorkspace:
    def __init__(self, path, *, dense_transport=False):
        # Must be set before MLX initializes its backend. The M5 default uses
        # reduced-precision matrix operations even for float32 input arrays.
        if os.environ.get("MLX_ENABLE_TF32", "0") != "0":
            raise ValueError("Workspace qualification requires MLX_ENABLE_TF32=0")
        os.environ["MLX_ENABLE_TF32"] = "0"
        import mlx.core as mx
        from mlx_lm import load

        path = Path(path).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("An existing local model snapshot is required")
        self.mx = mx
        self.model, self.tokenizer, self.config = load(
            str(path), tokenizer_config={"trust_remote_code": False}, return_config=True,
        )
        if self.config.get("model_type") != "qwen3":
            raise ValueError("This audited adapter supports Qwen3 only")
        self.model.eval()
        self.model.freeze()
        self.path = path
        self.layers = self.model.model.layers
        self.width = self.config["hidden_size"]
        self.n_layers = len(self.layers)
        self.dense_transport = dense_transport
        if dense_transport:
            # Expand the existing four-bit values, not different/full-precision
            # pretrained weights. Dense float32 avoids quantized-kernel changes
            # between primal, backward, and differently sized batches.
            import mlx.nn as nn
            from mlx.utils import tree_flatten, tree_unflatten

            for layer in self.layers:
                replacements = []
                for name, module in tree_flatten(layer.leaf_modules(), is_leaf=lambda value: isinstance(value, nn.Module)):
                    if isinstance(module, nn.QuantizedLinear):
                        weight = mx.dequantize(
                            module.weight, module.scales, module.biases,
                            group_size=module.group_size, bits=module.bits, mode=module.mode,
                        ).astype(mx.float32)
                        mx.eval(weight)
                        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                        linear.weight = weight
                        if "bias" in module:
                            linear.bias = module.bias.astype(mx.float32)
                        replacements.append((name, linear))
                layer.update_modules(tree_unflatten(replacements))
                if len(replacements) != 7:
                    raise ValueError(f"Expected seven Qwen block projections, replaced {len(replacements)}")
                layer.set_dtype(mx.float32)
            self.model.freeze()

    def encode(self, text, *, chat=False):
        if chat:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def output_rows(self, token_ids, *, with_norm_scale=True):
        """Exact dequantized selected unembedding rows, optionally folded with gamma."""
        mx, head = self.mx, self.model.lm_head
        ids = mx.array(token_ids, dtype=mx.int32)
        if hasattr(head, "scales"):
            rows = mx.dequantize(
                head.weight[ids], head.scales[ids], head.biases[ids],
                group_size=head.group_size, bits=head.bits, mode=head.mode,
            )
        else:
            rows = head.weight[ids]
        rows = rows.astype(mx.float32)
        if with_norm_scale:
            rows = rows * self.model.model.norm.weight.astype(mx.float32)
        mx.eval(rows)
        return np.array(rows)

    def forward(self, token_ids, *, capture_layers=(), capture_position=-1,
                patches=(), logit_positions=None):
        """Full causal recomputation; patches affect only explicit past positions.

        Reusing a fixed prefix-position patch while appending generated tokens is
        equivalent to persisting that intervention in prefix KV states. No patch
        is applied to newly generated positions unless explicitly specified.
        """
        mx = self.mx
        if not token_ids:
            raise ValueError("Empty input")
        position = capture_position % len(token_ids)
        if not -len(token_ids) <= capture_position < len(token_ids):
            raise ValueError("Capture position out of range")
        if any(type(i) is not int or not 0 <= i < self.n_layers for i in capture_layers):
            raise ValueError("Invalid capture layer")
        patch_map = {}
        for patch in patches:
            layer, pos = patch["layer"], patch["position"]
            delta = np.asarray(patch["delta"], dtype=np.float32)
            if not 0 <= layer < self.n_layers or not 0 <= pos < len(token_ids):
                raise ValueError("Patch location out of range")
            if delta.shape != (self.width,) or not np.isfinite(delta).all():
                raise ValueError("Malformed patch")
            patch_map.setdefault(layer, []).append((pos, mx.array(delta)))
        h = self.model.model.embed_tokens(mx.array(token_ids)[None]).astype(mx.float32)
        captures = {}
        for index, layer in enumerate(self.layers):
            h = layer(h, mask="causal" if len(token_ids) > 1 else None)
            for pos, delta in patch_map.get(index, ()):
                h = h.at[0, pos].add(delta)
            if index in capture_layers:
                captures[index] = h[0, position]
        locations = [-1] if logit_positions is None else logit_positions
        logits = self.model.lm_head(self.model.model.norm(h[:, locations]))[0]
        mx.eval(logits, *captures.values())
        if not bool(mx.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite model output")
        return np.array(logits), {k: np.array(v) for k, v in captures.items()}

    def greedy(self, token_ids, *, patches=(), max_tokens=12):
        ids, generated = list(token_ids), []
        eos = self.config["eos_token_id"]
        eos = {eos} if isinstance(eos, int) else set(eos)
        for _ in range(max_tokens):
            logits, _ = self.forward(ids, patches=patches)
            token = int(logits[-1].argmax())
            if token in eos:
                break
            generated.append(token)
            ids.append(token)
        return self.tokenizer.decode(generated), generated

    def jacobian_rows(self, token_ids, rows, source_layers, *, skip_first=16):
        """Exact R J_l for selected rows R; ALL source layers in each backward.

        Official estimator: sum cotangents across valid target positions, mean
        gradients across valid source positions. It is NOT uniform pair averaging.
        Rows include RMSNorm gamma when desired; final norm itself is not in J.
        Replicated batch elements carry independent row cotangents. Additive zero
        inputs at each source residual recover its derivative without detachment.
        """
        mx = self.mx
        rows = np.asarray(rows, dtype=np.float32)
        sources = tuple(source_layers)
        n = len(token_ids)
        if rows.ndim != 2 or rows.shape[1] != self.width or not np.isfinite(rows).all():
            raise ValueError("Invalid target directions")
        if not sources or tuple(sorted(set(sources))) != sources or sources[0] < 0 or sources[-1] >= self.n_layers - 1:
            raise ValueError("Source layers must be sorted unique and before final layer")
        if type(skip_first) is not int or skip_first < 0 or n <= skip_first + 1:
            raise ValueError("No valid source/target positions")
        batch = len(rows)
        h = self.model.model.embed_tokens(mx.array(token_ids)[None]).astype(mx.float32)
        for layer in self.layers[:sources[0] + 1]:
            h = layer(h, mask="causal")
        mx.eval(h)
        start = mx.stop_gradient(mx.broadcast_to(h, (batch, n, self.width)))
        zero = [mx.zeros_like(start) for _ in sources]
        source_index = {layer: i for i, layer in enumerate(sources)}

        def remaining(*deltas):
            value = start + deltas[0]
            for layer_index in range(sources[0] + 1, self.n_layers):
                value = self.layers[layer_index](value, mask="causal")
                if layer_index in source_index:
                    value = value + deltas[source_index[layer_index]]
            return value

        mask = mx.array((np.arange(n) >= skip_first) & (np.arange(n) < n - 1), dtype=mx.float32)
        cotangent = mx.array(rows)[:, None, :] * mask[None, :, None]
        _, gradients = mx.vjp(remaining, zero, [cotangent])
        averaged = [g[:, skip_first:n - 1].mean(axis=1) for g in gradients]
        mx.eval(*averaged)
        result = np.stack([np.array(g) for g in averaged])
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite Jacobian")
        return result

    def transport_scalar(self, token_ids, row, source_layer, direction, *, skip_first=16):
        """Independent finite-difference oracle for the temporal reduction.

        Add direction / valid-source-count at every valid source position and
        sum row projections at valid final positions. Its derivative at zero
        is the selected mean-Jacobian row dotted with direction.
        """
        mx = self.mx
        n = len(token_ids)
        if not 0 <= source_layer < self.n_layers - 1 or n <= skip_first + 1:
            raise ValueError("Invalid finite-difference configuration")
        h = self.model.model.embed_tokens(mx.array(token_ids)[None]).astype(mx.float32)
        mask = mx.array((np.arange(n) >= skip_first) & (np.arange(n) < n - 1), dtype=mx.float32)
        shift = mask[None, :, None] * mx.array(direction)[None, None, :] / (n - skip_first - 1)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask="causal")
            if i == source_layer:
                h = h + shift
        value = mx.sum(h[0] * mx.array(row)[None] * mask[:, None])
        mx.eval(value)
        return float(value)
