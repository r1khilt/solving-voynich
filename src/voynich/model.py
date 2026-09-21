"""Small dense causal transformer with explicit, replaceable activation sites.

Architecture choices and research provenance: docs/research/ARCHITECTURE.md.
Cache values are the actual *post-intervention* activations, detached from autograd.
No hooks or interventions are persisted between calls.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

Intervention = Callable[[Tensor], Tensor]


@dataclass
class ModelConfig:
    vocab_size: int = 0
    pad_id: int = 0
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 256
    dropout: float = 0.1
    rope_base: float = 10000.0
    qk_norm: bool = False
    attention_gate: bool = False
    attention_only: bool = False
    tie_embeddings: bool = False
    auxiliary_horizons: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("vocab_size", "pad_id", "d_model", "n_layers", "n_heads", "d_ff", "context_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        for name in ("qk_norm", "attention_gate", "attention_only", "tie_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("dropout", "rope_base"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        try:
            self.auxiliary_horizons = tuple(self.auxiliary_horizons)
        except TypeError as exc:
            raise ValueError("auxiliary horizons must be a sequence of integers") from exc
        if any(not isinstance(h, int) or isinstance(h, bool) for h in self.auxiliary_horizons):
            raise ValueError("auxiliary horizons must be integers")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by positive n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if min(self.n_layers, self.d_ff, self.context_length) <= 0:
            raise ValueError("layers, feedforward width, and context length must be positive")
        if not 0 <= self.dropout < 1 or self.rope_base <= 0:
            raise ValueError("invalid dropout or RoPE base")
        if any(h < 2 or h > self.context_length for h in self.auxiliary_horizons):
            raise ValueError("auxiliary horizons must lie in [2, context_length]")
        if len(set(self.auxiliary_horizons)) != len(self.auxiliary_horizons):
            raise ValueError("auxiliary horizons must be unique")

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelOutput:
    logits: Tensor
    auxiliary_logits: dict[int, Tensor] = field(default_factory=dict)
    cache: dict[str, Tensor] = field(default_factory=dict)


class ActivationContext:
    def __init__(self, names: Iterable[str] | None, interventions: Mapping[str, Intervention] | None):
        self.names = set(names or ())
        self.interventions = dict(interventions or {})
        self.cache: dict[str, Tensor] = {}
        self.visited: set[str] = set()

    @property
    def active(self):
        return bool(self.names or self.interventions)

    def __call__(self, name: str, value: Tensor) -> Tensor:
        self.visited.add(name)
        if name in self.interventions:
            changed = self.interventions[name](value)
            if changed.shape != value.shape or changed.device != value.device or changed.dtype != value.dtype:
                raise ValueError(f"Intervention at {name} must preserve shape, device, and dtype")
            value = changed
        if name in self.names or "*" in self.names:
            self.cache[name] = value.detach().clone()
        return value

    def validate(self):
        missing = ((self.names - {"*"}) | self.interventions.keys()) - self.visited
        if missing:
            raise ValueError(f"Unknown or inactive activation sites: {sorted(missing)}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotary(x: Tensor, base: float) -> Tensor:
    """x: [batch, time, head, head_dim], paired-coordinate RoPE."""
    length, width = x.shape[1], x.shape[-1]
    frequencies = base ** (-torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    angles = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    cos, sin = angles.cos()[None, :, None, :].to(x.dtype), angles.sin()[None, :, None, :].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.d_model // cfg.n_heads) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.d_model // cfg.n_heads) if cfg.qk_norm else nn.Identity()
        self.gate = nn.Linear(cfg.d_model, cfg.n_heads, bias=False) if cfg.attention_gate else None

    def forward(self, x: Tensor, valid: Tensor, ctx: ActivationContext, name: str) -> Tensor:
        batch, length, width = x.shape
        head_dim = width // self.cfg.n_heads
        shape = (batch, length, self.cfg.n_heads, head_dim)
        q = ctx(name + ".q_pre_norm", self.q(x).view(shape))
        k = ctx(name + ".k_pre_norm", self.k(x).view(shape))
        v = ctx(name + ".v", self.v(x).view(shape))
        q = ctx(name + ".q_normalized", self.q_norm(q))
        k = ctx(name + ".k_normalized", self.k_norm(k))
        q = ctx(name + ".q", rotary(q, self.cfg.rope_base)).transpose(1, 2)
        k = ctx(name + ".k", rotary(k, self.cfg.rope_base)).transpose(1, 2)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None, :, :] & valid[:, None, None, :]
        if ctx.active:
            scores = ctx(name + ".scores", (q @ k.transpose(-2, -1)) / math.sqrt(head_dim))
            pattern = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            pattern = ctx(name + ".pattern", pattern)
            pattern = F.dropout(pattern, self.cfg.dropout, self.training)
            z = (pattern @ v.transpose(1, 2)).transpose(1, 2)
        else:
            z = F.scaled_dot_product_attention(
                q, k, v.transpose(1, 2), attn_mask=mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            ).transpose(1, 2)
        z = ctx(name + ".z", z)
        if self.gate is not None:
            gate = ctx(name + ".gate", self.gate(x).sigmoid())
            z = ctx(name + ".z_gated", z * gate[..., None])
        if ctx.active:
            # W_O is split by input head; each result is an additive residual contribution.
            w_out = self.out.weight.view(width, self.cfg.n_heads, head_dim)
            result = ctx(name + ".result", torch.einsum("bthd,mhd->bthm", z, w_out))
            output = result.sum(dim=2)
        else:
            output = self.out(z.reshape(batch, length, width))
        return ctx(name + ".out", output)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        if not cfg.attention_only:
            self.mlp_norm = RMSNorm(cfg.d_model)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, valid, ctx, name):
        x = ctx(name + ".resid_pre", x)
        attn = self.attn(ctx(name + ".attn_input", self.attn_norm(x)), valid, ctx, name + ".attn")
        x = ctx(name + ".resid_mid", x + F.dropout(attn, self.cfg.dropout, self.training))
        if not self.cfg.attention_only:
            normed = ctx(name + ".mlp_input", self.mlp_norm(x))
            up = ctx(name + ".mlp.up", self.up(normed))
            gate = ctx(name + ".mlp.gate", F.silu(self.gate(normed)))
            act = ctx(name + ".mlp.act", up * gate)
            output = ctx(name + ".mlp.out", self.down(act))
            x = x + F.dropout(output, self.cfg.dropout, self.training)
        return ctx(name + ".resid_post", x)


class VoynichTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size < 2 or not 0 <= config.pad_id < config.vocab_size:
            raise ValueError("Supply a valid vocabulary and pad id from the training tokenizer")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.unembedding = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.auxiliary_heads = nn.ModuleDict({
            str(h): nn.Linear(config.d_model, config.vocab_size, bias=False)
            for h in config.auxiliary_horizons
        })
        self.apply(self._initialize)
        # Scale residual writes to control initialization variance with depth.
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            if not config.attention_only:
                nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))
        if config.tie_embeddings:
            self.unembedding.weight = self.embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: Tensor, *, cache_names=None, interventions=None) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty [batch, time] shape")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("Input exceeds configured context length")
        valid = input_ids.ne(self.config.pad_id)
        if not valid[:, 0].all():
            raise ValueError("Each sequence must start with a non-padding token; use right padding")
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Only contiguous right padding is supported")
        ctx = ActivationContext(cache_names, interventions)
        x = ctx("embed", self.embedding(input_ids))
        for i, block in enumerate(self.blocks):
            x = block(x, valid, ctx, f"blocks.{i}")
        x = ctx("final_norm", self.final_norm(x))
        logits = ctx("logits", self.unembedding(x))
        auxiliary = {int(h): head(x) for h, head in self.auxiliary_heads.items()}
        ctx.validate()
        return ModelOutput(logits, auxiliary, ctx.cache)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
