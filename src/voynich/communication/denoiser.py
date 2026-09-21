"""Conditional masked denoising proposals over fixed, typed latent slots.

PAD (0) is structural padding, MASK (1) is artificial corruption, and values
2 onward are latent categories. Observed conditioning tokens are never corrupted
or changed. This is a masked reconstruction objective and a revisable proposal
heuristic, not an implementation of an exact diffusion posterior or an ELBO.

Rationale and limitations: docs/research/world-models-diffusion-2026-09-21/
DIFFUSION_INFERENCE.md (masked objectives, explicit remasking, and constraints).
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
import inspect
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PAD = 0
MASK = 1
HiddenPatch = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class DenoiserConfig:
    vocab_size: int
    condition_vocab_size: int
    max_latent_length: int
    max_condition_length: int
    width: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    type_vocab_size: int = 8

    def __post_init__(self):
        for name in (
            "vocab_size", "condition_vocab_size", "max_latent_length", "max_condition_length",
            "width", "layers", "heads", "type_vocab_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 3:
            raise ValueError("vocab_size must include PAD, MASK, and at least one category")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (
            isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def _ids(value: Tensor, name: str, *, shape=None, device=None, minimum=0, maximum=None):
    if not isinstance(value, Tensor) or value.dtype != torch.long or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 torch.long tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if bool((value < minimum).any()) or (maximum is not None and bool((value >= maximum).any())):
        raise ValueError(f"{name} contains an out-of-domain token")


def _generator(generator):
    if generator is not None and (
        not isinstance(generator, torch.Generator) or generator.device.type != "cpu"
    ):
        raise ValueError("generator must be a CPU torch.Generator")


class ConditionalDenoiser(nn.Module):
    """Bidirectional Transformer; only noisy latents enter its prediction path.

    Separate position tables keep latent and condition coordinates independent of
    padding. An always-visible anchor prevents all-masked attention rows, including
    a fully padded example or an empty condition. Padding is never an attention key
    and its returned hidden states/logits are zero. Patches receive final normalized
    latent hidden states before the output head; returned hidden states are patched.
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        if not isinstance(config, DenoiserConfig):
            raise ValueError("config must be a DenoiserConfig")
        self.config = config
        self.latent_embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=PAD)
        self.condition_embedding = nn.Embedding(config.condition_vocab_size, config.width, padding_idx=PAD)
        self.latent_position = nn.Embedding(config.max_latent_length, config.width)
        self.condition_position = nn.Embedding(config.max_condition_length, config.width)
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.width)
        self.stream_embedding = nn.Embedding(2, config.width)
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, config.width), nn.SiLU(), nn.Linear(config.width, config.width),
        )
        self.anchor = nn.Parameter(torch.empty(1, 1, config.width))
        nn.init.normal_(self.anchor, std=0.02)
        # Construct independently, rather than cloning identically initialized layers.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.width, nhead=config.heads, dim_feedforward=4 * config.width,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.layers)
        ])
        self.final_norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size)

    def _validate(self, noisy, condition, latent_types=None):
        device = self.latent_embedding.weight.device
        _ids(noisy, "noisy", device=device, maximum=self.config.vocab_size)
        batch, length = noisy.shape
        if batch == 0 or length == 0 or length > self.config.max_latent_length:
            raise ValueError("noisy must have a nonempty batch and length within max_latent_length")
        _ids(condition, "condition", device=device, maximum=self.config.condition_vocab_size)
        if condition.shape[0] != batch or condition.shape[1] > self.config.max_condition_length:
            raise ValueError("condition batch or length does not match configuration")
        if latent_types is not None:
            _ids(
                latent_types, "latent_types", shape=noisy.shape, device=device,
                maximum=self.config.type_vocab_size,
            )

    def forward(
        self, noisy: Tensor, condition: Tensor, noise_level: Tensor, *,
        latent_types: Tensor | None = None, return_hidden: bool = False,
        hidden_patch: HiddenPatch | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate(noisy, condition, latent_types)
        batch, length = noisy.shape
        if (
            not isinstance(noise_level, Tensor) or not noise_level.is_floating_point()
            or noise_level.shape != (batch,) or noise_level.device != noisy.device
            or not bool(torch.isfinite(noise_level).all())
            or bool(((noise_level < 0) | (noise_level > 1)).any())
        ):
            raise ValueError("noise_level must be finite floating [batch] on the input device, in [0, 1]")
        if not isinstance(return_hidden, bool):
            raise ValueError("return_hidden must be a boolean")
        if hidden_patch is not None and not callable(hidden_patch):
            raise ValueError("hidden_patch must be callable")
        valid = noisy != PAD
        types = torch.zeros_like(noisy) if latent_types is None else latent_types
        noise = self.noise_embedding(noise_level.to(self.anchor.dtype)[:, None])[:, None, :]
        latent = (
            self.latent_embedding(noisy)
            + self.latent_position(torch.arange(length, device=noisy.device))[None]
            + self.type_embedding(types) + self.stream_embedding.weight[0] + noise
        ).masked_fill(~valid[..., None], 0)
        evidence = (
            self.condition_embedding(condition)
            + self.condition_position(torch.arange(condition.shape[1], device=noisy.device))[None]
            + self.stream_embedding.weight[1]
        ).masked_fill((condition == PAD)[..., None], 0)
        hidden = torch.cat((self.anchor.expand(batch, -1, -1), evidence, latent), dim=1)
        padding = torch.cat((
            torch.zeros((batch, 1), dtype=torch.bool, device=noisy.device), condition == PAD, ~valid,
        ), dim=1)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
        hidden = self.final_norm(hidden[:, -length:]).masked_fill(~valid[..., None], 0)
        if hidden_patch is not None:
            patched = hidden_patch(hidden)
            if (
                not isinstance(patched, Tensor) or patched.shape != hidden.shape
                or patched.device != hidden.device or patched.dtype != hidden.dtype
            ):
                raise ValueError("hidden_patch must preserve shape, device, and dtype")
            if not bool(torch.isfinite(patched).all()):
                raise ValueError("hidden_patch must return finite values")
            hidden = patched.masked_fill(~valid[..., None], 0)
        logits = self.head(hidden).masked_fill(~valid[..., None], 0)
        return (logits, hidden) if return_hidden else logits


def _support(allowed: Tensor | None, tokens: Tensor, vocab_size: int) -> Tensor:
    """Special IDs are structural, never ordinary sampled categories."""
    shape = (*tokens.shape, vocab_size)
    if allowed is not None:
        if (
            not isinstance(allowed, Tensor) or allowed.dtype != torch.bool
            or tuple(allowed.shape) != shape or allowed.device != tokens.device
        ):
            raise ValueError("allowed must be boolean [batch, length, vocab_size] on the input device")
        support = allowed.clone()
    else:
        support = torch.ones(shape, dtype=torch.bool, device=tokens.device)
    support[..., :2] = False
    active = tokens != PAD
    if bool((active & ~support.any(dim=-1)).any()):
        raise ValueError("every active latent slot needs nonempty category support (IDs >= 2)")
    return support


def _check_supported(tokens: Tensor, support: Tensor, name: str):
    concrete = tokens >= 2
    admitted = support.gather(-1, tokens.clamp_min(0)[..., None]).squeeze(-1)
    if bool((concrete & ~admitted).any()):
        raise ValueError(f"{name} contains a category outside allowed support")


def masked_loss(
    model: ConditionalDenoiser, clean: Tensor, condition: Tensor, *,
    generator: torch.Generator | None = None, latent_types: Tensor | None = None,
    allowed: Tensor | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    """Mean CE on masked, nonpadding slots, with no clean-target model input.

    For each nonempty row, choose a corruption count uniformly from 1..n, then
    choose that many positions uniformly without replacement. Thus every such row
    supplies a learning target and fully masked examples have positive probability.
    This explicitly specified objective is unweighted masked reconstruction; it is
    not a diffusion likelihood bound. Randomness comes from a CPU generator even
    when tensors/model are on MPS. Empty rows contribute no targets.
    """
    _generator(generator)
    model._validate(clean, condition, latent_types)
    if bool((clean == MASK).any()):
        raise ValueError("clean must not contain MASK")
    valid_cpu = (clean != PAD).detach().cpu()
    if not bool(valid_cpu.any()):
        raise ValueError("clean must contain at least one nonpadding target")
    support = _support(allowed, clean, model.config.vocab_size)
    _check_supported(clean, support, "clean")
    masked_cpu = torch.zeros_like(valid_cpu)
    for row in range(clean.shape[0]):
        positions = valid_cpu[row].nonzero(as_tuple=False).flatten()
        count = positions.numel()
        if count:
            number = int(torch.randint(1, count + 1, (1,), generator=generator))
            chosen = positions[torch.randperm(count, generator=generator)[:number]]
            masked_cpu[row, chosen] = True
    masked = masked_cpu.to(clean.device)
    noise_level = (
        masked_cpu.sum(-1).float() / valid_cpu.sum(-1).clamp_min(1)
    ).to(clean.device)
    noisy = clean.masked_fill(masked, MASK)
    logits = model(noisy, condition, noise_level, latent_types=latent_types)
    masked_logits = logits[masked].masked_fill(~support[masked], -torch.inf)
    targets = clean[masked]
    loss = F.cross_entropy(masked_logits, targets)
    metrics = {
        "loss": float(loss.detach()),
        "masked_accuracy": float((masked_logits.detach().argmax(-1) == targets).float().mean()),
        "masked_tokens": int(masked_cpu.sum()),
        "active_tokens": int(valid_cpu.sum()),
        "mean_noise_level": float(noise_level.detach().mean()),
    }
    return loss, metrics


def _step_patch(patch, step):
    if patch is None:
        return None
    if not callable(patch):
        raise ValueError("hidden_patch must be callable")
    try:
        signature = inspect.signature(patch)
    except (ValueError, TypeError) as exc:
        raise ValueError("sample hidden_patch must have an inspectable call signature") from exc
    try:
        signature.bind(step, None)
    except TypeError:
        try:
            signature.bind(None)
        except TypeError as exc:
            raise ValueError("sample hidden_patch must accept hidden or (step, hidden)") from exc
        return patch
    return lambda hidden: patch(step, hidden)


@torch.no_grad()
def sample(
    model: ConditionalDenoiser, condition: Tensor, *, length: int, steps: int = 16,
    generator: torch.Generator | None = None, latent_types: Tensor | None = None,
    allowed: Tensor | None = None, fixed: Tensor | None = None, initial: Tensor | None = None,
    return_trace: bool = False, hidden_patch: Callable | None = None,
) -> Tensor | tuple[Tensor, list[dict]]:
    """Constrained proposals with explicit revision of previously filled guesses.

    Every pass fills its masked slots. Later passes remask a random subset of ALL
    mutable slots, with a linearly decreasing count, so earlier guesses may change.
    Fixed values use -1 for unknown and 0 or IDs >=2 for constants; fixed MASK is
    invalid. PAD in ``initial`` is also immutable. Other initial values are guesses:
    their existing masks set the first update; an entirely filled initial row has
    half its mutable positions remasked to start refinement.

    ``allowed`` restricts ordinary categories; its PAD/MASK bits are ignored.
    Structural PAD requires fixed=0 or initial=0. A supplied initial/fixed category
    must be allowed, and conflicting concrete initial/fixed values are errors.
    Trace entries contain post-update tokens, the actual pre-update noisy_tokens,
    remasked boolean positions, and per-row noise_level, all detached CPU copies.
    A patch accepts hidden, or (zero-based step, hidden). Sampling temporarily uses
    evaluation mode and restores every module's prior mode, even after an error.

    This bounded remasking heuristic is not an exact posterior sampler. CPU random
    draws enable seeded repeatability on a given device; cross-device bit identity
    is not promised because floating point probabilities can differ.
    """
    _generator(generator)
    for name, value in (("length", length), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if length > model.config.max_latent_length:
        raise ValueError("length exceeds max_latent_length")
    if not isinstance(return_trace, bool):
        raise ValueError("return_trace must be a boolean")
    device = model.latent_embedding.weight.device
    _ids(condition, "condition", device=device, maximum=model.config.condition_vocab_size)
    shape = (condition.shape[0], length)
    if initial is None:
        tokens = torch.full(shape, MASK, dtype=torch.long, device=device)
    else:
        _ids(initial, "initial", shape=shape, device=device, maximum=model.config.vocab_size)
        tokens = initial.clone()
    model._validate(tokens, condition, latent_types)
    if fixed is None:
        constants = torch.full(shape, -1, dtype=torch.long, device=device)
    else:
        _ids(fixed, "fixed", shape=shape, device=device, minimum=-1, maximum=model.config.vocab_size)
        if bool((fixed == MASK).any()):
            raise ValueError("fixed must not contain MASK")
        constants = fixed.clone()
    if initial is not None:
        conflict = (constants >= 0) & (tokens != MASK) & (tokens != constants)
        if bool(conflict.any()):
            raise ValueError("initial and fixed contain conflicting concrete values")
    constants = torch.where(tokens == PAD, torch.zeros_like(constants), constants)
    tokens = torch.where(constants >= 0, constants, tokens)
    support = _support(allowed, tokens, model.config.vocab_size)
    _check_supported(tokens, support, "initial/fixed")
    mutable_cpu = (constants < 0).cpu()
    first_masks = ((tokens == MASK) & (constants < 0)).cpu()
    base_counts = first_masks.sum(-1)
    # For fully specified warm starts, make refinement meaningful without erasing
    # the entire initial proposal at its first step.
    base_counts = torch.where(base_counts > 0, base_counts, (mutable_cpu.sum(-1) + 1) // 2)
    active_counts = (tokens != PAD).sum(-1).clamp_min(1).cpu()
    trace = []
    prior_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for step in range(steps):
            remasked_cpu = torch.zeros_like(mutable_cpu)
            for row in range(shape[0]):
                positions = mutable_cpu[row].nonzero(as_tuple=False).flatten()
                if step == 0 and bool(first_masks[row].any()):
                    remasked_cpu[row] = first_masks[row]
                elif positions.numel():
                    count = max(1, math.ceil(int(base_counts[row]) * (steps - step) / steps))
                    chosen = positions[torch.randperm(positions.numel(), generator=generator)[:count]]
                    remasked_cpu[row, chosen] = True
            remasked = remasked_cpu.to(device)
            noisy = tokens.masked_fill(remasked, MASK)
            noise_level = (remasked_cpu.sum(-1).float() / active_counts).to(device)
            logits = model(
                noisy, condition, noise_level, latent_types=latent_types,
                hidden_patch=_step_patch(hidden_patch, step),
            )
            if bool(remasked_cpu.any()):
                selected = logits[remasked].masked_fill(~support[remasked], -torch.inf)
                probabilities = selected.float().softmax(-1).cpu()
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("model produced nonfinite sampling probabilities")
                draws = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1).to(device)
                tokens = tokens.clone()
                tokens[remasked] = draws
            if return_trace:
                trace.append({
                    "step": step, "tokens": tokens.detach().cpu().clone(),
                    "noisy_tokens": noisy.detach().cpu().clone(), "remasked": remasked_cpu.clone(),
                    "noise_level": noise_level.detach().cpu().clone(),
                })
    finally:
        for module, training in prior_modes:
            module.training = training
    return (tokens, trace) if return_trace else tokens
