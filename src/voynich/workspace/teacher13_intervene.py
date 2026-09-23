"""Position-aware causal interventions for prospective TEACH-0013 analyses."""

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .teacher12_models import LoopedRawClassifier, RawClassifier, model_for_arm
from .teacher12_tasks import Episode, pad_batch
from .teacher13_tasks import SemanticLayout, semantic_layout


RAW_ARMS = ("raw_shallow", "raw_looped", "raw_deep", "raw_null")
_MATERIALIZATION_PROBE = ContextVar("teacher13_materialization_probe", default=None)


@contextmanager
def materialized_tensor_counter(probe):
    """Count returned and intervention tensors inside one bounded campaign block."""
    token = _MATERIALIZATION_PROBE.set(probe)
    try:
        yield
    finally:
        _MATERIALIZATION_PROBE.reset(token)


def record_materialized(*tensors: Tensor) -> None:
    """Charge each explicitly materialized tensor once at its actual dtype."""
    probe = _MATERIALIZATION_PROBE.get()
    if probe is not None:
        probe(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def residual_sites(arm: str) -> tuple[str, ...]:
    """All post-block residual cuts, in execution order."""
    if arm == "raw_looped":
        return ("embed",) + tuple(f"passes.{pass_index}.blocks.{layer}.resid_post"
                                  for pass_index in range(3) for layer in range(4))
    if arm in ("raw_shallow", "raw_null"):
        return ("embed",) + tuple(f"blocks.{layer}.resid_post" for layer in range(4))
    if arm == "raw_deep":
        return ("embed",) + tuple(f"blocks.{layer}.resid_post" for layer in range(12))
    raise ValueError(f"Arm has no raw residual stream: {arm}")


def attention_sites(arm: str, suffix: str = "result") -> tuple[str, ...]:
    """Sequence-first attention sites safe for position patching.

    Post-RoPE q/k, score and pattern caches are head-first and therefore require a
    different axis-aware intervention API.  Excluding them here prevents silently
    patching a head index as if it were a token position.
    """
    if suffix not in ("q_pre_norm", "k_pre_norm", "v", "q_normalized", "k_normalized",
                      "z", "result", "out"):
        raise ValueError(f"Unsupported attention site suffix: {suffix}")
    return tuple(site.removesuffix("resid_post") + "attn." + suffix
                 for site in residual_sites(arm)[1:])


def load_raw_checkpoint(path: Path | str, *, device: str | torch.device = "cpu"
                        ) -> tuple[nn.Module, dict]:
    """Load a frozen TEACH-0012 raw checkpoint with metadata checks."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "config", "replicate", "arm", "step", "parameters"}
    if set(checkpoint) != required:
        raise ValueError("Unexpected checkpoint fields")
    arm = checkpoint["arm"]
    if arm not in RAW_ARMS:
        raise ValueError(f"Not a raw checkpoint: {arm}")
    net = model_for_arm(arm)
    if checkpoint["parameters"] != net.parameter_count:
        raise ValueError("Checkpoint parameter count does not match architecture")
    net.load_state_dict(checkpoint["model"], strict=True)
    net.to(device).eval()
    return net, checkpoint


def _unique_position(layout: SemanticLayout, name: str, *, semantic_label: bool) -> int:
    positions = ((layout.label_position(name),) if semantic_label
                 else layout.positions(name))
    if len(positions) != 1:
        kind = "Label" if semantic_label else "Role"
        raise ValueError(f"{kind} {name!r} has {len(positions)} positions, expected one")
    return positions[0]


def _validate_basis(basis: Tensor, width: int) -> None:
    if basis.ndim != 2 or basis.shape[0] != width or basis.shape[1] == 0:
        raise ValueError("Basis must have shape [activation_width, positive_rank]")
    gram = basis.float().T @ basis.float()
    eye = torch.eye(basis.shape[1], device=gram.device)
    if not torch.allclose(gram, eye, atol=1e-5, rtol=1e-5):
        raise ValueError("Subspace basis must have orthonormal columns")


def position_patch(donor: Tensor, base_layouts: Iterable[SemanticLayout],
                   donor_layouts: Iterable[SemanticLayout], *, base_role: str,
                   donor_role: str | None = None, heads: tuple[int, ...] | None = None,
                   basis: Tensor | None = None, component: str = "full",
                   semantic_label: bool = False):
    """Create a dynamic-position patch for residual or per-head activation tensors.

    `component` can transfer the full vector, its projection into `basis`, or its
    orthogonal complement.  Subspace operations are restricted to rank-3 residuals.
    """
    bases, donors = tuple(base_layouts), tuple(donor_layouts)
    record_materialized(donor)
    donor_role = base_role if donor_role is None else donor_role
    if donor.shape[0] != len(bases) or len(bases) != len(donors):
        raise ValueError("Donor batch and semantic layouts must have equal batch size")
    source_positions = tuple(_unique_position(
        layout, donor_role, semantic_label=semantic_label) for layout in donors)
    target_positions = tuple(_unique_position(
        layout, base_role, semantic_label=semantic_label) for layout in bases)
    if component not in ("full", "subspace", "complement"):
        raise ValueError(f"Unknown patch component: {component}")
    if component != "full" and basis is None:
        raise ValueError("Subspace/complement patch requires a basis")
    if basis is not None and donor.ndim != 3:
        raise ValueError("Subspace patching is defined only for rank-3 residual streams")
    if heads is not None and donor.ndim != 4:
        raise ValueError("Head selection requires rank-4 per-head activations")

    def intervene(value: Tensor) -> Tensor:
        if value.ndim != donor.ndim or value.shape[0] != donor.shape[0] \
                or value.shape[2:] != donor.shape[2:]:
            raise ValueError("Base and donor activation shapes are incompatible")
        if max(source_positions) >= donor.shape[1] or max(target_positions) >= value.shape[1]:
            raise ValueError("Semantic position exceeds activation sequence length")
        changed = value.clone()
        donor_local = donor.to(device=value.device, dtype=value.dtype)
        for index, (target, source) in enumerate(zip(
                target_positions, source_positions, strict=True)):
            if heads is not None:
                if not heads or min(heads) < 0 or max(heads) >= value.shape[2]:
                    raise ValueError("Head index outside activation shape")
                changed[index, target, list(heads)] = donor_local[index, source, list(heads)]
                continue
            source_value, base_value = donor_local[index, source], value[index, target]
            if basis is None or component == "full":
                changed[index, target] = source_value
                continue
            local_basis = basis.to(device=value.device, dtype=value.dtype)
            _validate_basis(local_basis, value.shape[-1])
            delta = source_value - base_value
            projected = (delta @ local_basis) @ local_basis.T
            changed[index, target] = base_value + (
                projected if component == "subspace" else delta - projected)
        return changed

    return intervene


def raw_forward(net: nn.Module, episodes: list[Episode] | tuple[Episode, ...], *,
                device: str | torch.device = "cpu", cache_names=(), interventions=None):
    """Run a raw arm on a right-padded batch."""
    if not episodes:
        raise ValueError("At least one episode is required")
    padded, _ = pad_batch(list(episodes))
    ids = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.inference_mode():
        output = net(ids, cache_names=cache_names, interventions=interventions)
    record_materialized(output.logits, *output.cache.values())
    return output


@dataclass(frozen=True)
class PatchScores:
    predictions: tuple[int, ...]
    target_probabilities: tuple[float, ...]
    exact_accuracy: float
    maximum_logit_error: float | None = None


def score_logits(logits: Tensor, targets: Iterable[int], *, reference: Tensor | None = None
                 ) -> PatchScores:
    targets = tuple(targets)
    if logits.ndim != 2 or logits.shape[0] != len(targets):
        raise ValueError("Logits and targets have incompatible shapes")
    target = torch.tensor(targets, dtype=torch.long, device=logits.device)
    prediction = logits.argmax(-1)
    probabilities = logits.float().softmax(-1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    maximum_error = None if reference is None else float(
        (logits.float() - reference.float()).abs().max().item())
    return PatchScores(
        tuple(int(value) for value in prediction.cpu()),
        tuple(float(value) for value in probabilities[rows, target].cpu()),
        float(prediction.eq(target).float().mean().item()), maximum_error)


def cached_position_patch(net: nn.Module, base: tuple[Episode, ...], donor: tuple[Episode, ...],
                          *, site: str, role: str, donor_role: str | None = None,
                          device: str | torch.device = "cpu", heads=None, basis=None,
                          component: str = "full", semantic_label: bool = False):
    """Cache one donor site and patch it into dynamic semantic positions of base runs."""
    donor_output = raw_forward(net, donor, device=device, cache_names=(site,))
    layouts_base = tuple(semantic_layout(episode) for episode in base)
    layouts_donor = tuple(semantic_layout(episode) for episode in donor)
    patch = position_patch(
        donor_output.cache[site], layouts_base, layouts_donor, base_role=role,
        donor_role=donor_role, heads=heads, basis=basis, component=component,
        semantic_label=semantic_label)
    return raw_forward(net, base, device=device, interventions={site: patch})


@dataclass(frozen=True)
class TwoSiteOutput:
    propagated_logits: Tensor
    path_logits: Tensor
    propagated_state: Tensor


def ordered_two_site_patch(net: nn.Module, base: tuple[Episode, ...],
                           donor: tuple[Episode, ...], *, early_site: str,
                           early_label: str, late_site: str, late_label: str,
                           device: str | torch.device = "cpu") -> TwoSiteOutput:
    """Propagate an early donor write, then isolate its later state in a clean base run."""
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in donor)
    donor_output = raw_forward(net, donor, device=device, cache_names=(early_site,))
    early = position_patch(
        donor_output.cache[early_site], base_layouts, donor_layouts,
        base_role=early_label, semantic_label=True)
    propagated = raw_forward(
        net, base, device=device, cache_names=(late_site,),
        interventions={early_site: early})
    late = position_patch(
        propagated.cache[late_site], base_layouts, base_layouts,
        base_role=late_label, semantic_label=True)
    isolated = raw_forward(net, base, device=device, interventions={late_site: late})
    return TwoSiteOutput(propagated.logits, isolated.logits, propagated.cache[late_site])


def numerical_reconstruction(net: RawClassifier | LoopedRawClassifier,
                             episodes: tuple[Episode, ...], *, site: str,
                             device: str | torch.device = "cpu") -> float:
    """Measure fused-versus-instrumented forward error before interpreting patches."""
    native = raw_forward(net, episodes, device=device).logits
    explicit = raw_forward(net, episodes, device=device, cache_names=(site,)).logits
    return float((native.float() - explicit.float()).abs().max().item())
