"""Finite causal subspace interventions at a discovery-frozen TEACH-0013 mediator."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .teacher12_tasks import Episode
from .teacher13_confirm import (
    MediatorSpec,
    diagnostic_rows,
    mediator_logits,
    summarize_confirmation_rows,
)
from .teacher13_geometry import (
    ContrastGeometry,
    OrthogonalFactorGeometry,
    blocked_contrast_geometry,
    orthogonal_factor_geometry,
)
from .teacher13_intervene import position_patch, raw_forward, record_materialized
from .teacher13_tasks import semantic_layout


@dataclass(frozen=True)
class MediatorStatePairs:
    """Base and full-donor endpoint states in one common residual coordinate system."""

    base: Tensor
    donor: Tensor
    site: str
    label: str
    endpoint: str


@dataclass(frozen=True)
class FactorStatePanel:
    """A complete within-group factor-by-nuisance endpoint-state panel."""

    states: Tensor
    factor: tuple
    nuisance: tuple
    blocks: tuple
    metadata: tuple[dict, ...]


@dataclass(frozen=True)
class RegisteredFactorGeometries:
    content: ContrastGeometry
    binding: ContrastGeometry
    order: ContrastGeometry
    orthogonal: OrthogonalFactorGeometry


FACTOR_VARIANTS = {
    "content": (
        ("marked", "base", "donor"),
        ("marker_free", "marker_free_base", "marker_free_donor"),
        ("reordered", "reordered_base", "reordered_donor"),
        ("format", "format_base", "format_donor"),
        ("distractor", "distractor_base", "distractor_donor"),
    ),
    "binding": (
        ("marked", "binding_base", "binding_donor"),
        ("marker_free", "binding_marker_free_base", "binding_marker_free_donor"),
        ("reordered", "binding_reordered_base", "binding_reordered_donor"),
        ("format", "binding_format_base", "binding_format_donor"),
        ("distractor", "binding_distractor_base", "binding_distractor_donor"),
    ),
    "order": (
        ("f0_marked", "base", "reordered_base"),
        ("f1_marked", "donor", "reordered_donor"),
        ("f0_marker_free", "marker_free_base", "marker_free_reordered_base"),
        ("f1_marker_free", "marker_free_donor", "marker_free_reordered_donor"),
        ("f0_format", "format_base", "format_reordered_base"),
        ("f1_format", "format_donor", "format_reordered_donor"),
        ("f0_distractor", "distractor_base", "distractor_reordered_base"),
        ("f1_distractor", "distractor_donor", "distractor_reordered_donor"),
    ),
}


def _positions(episodes: tuple[Episode, ...], label: str) -> tuple[int, ...]:
    return tuple(semantic_layout(episode).label_position(label) for episode in episodes)


def _gather(activation: Tensor, positions: tuple[int, ...]) -> Tensor:
    if activation.ndim != 3 or activation.shape[0] != len(positions):
        raise ValueError("Residual activation and semantic positions are incompatible")
    return torch.stack([activation[index, position]
                        for index, position in enumerate(positions)])


def mediator_state_pairs(net: nn.Module, base: tuple[Episode, ...],
                         donor: tuple[Episode, ...], spec: MediatorSpec, *,
                         device="cpu") -> MediatorStatePairs:
    """Capture the actual full mediator endpoint used by the frozen transfer protocol."""
    spec.validate()
    if not base or len(base) != len(donor):
        raise ValueError("Mediator state batches must be nonempty and paired")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in donor)
    if spec.kind == "single":
        base_output = raw_forward(net, base, device=device, cache_names=(spec.site,))
        donor_output = raw_forward(net, donor, device=device, cache_names=(spec.site,))
        return MediatorStatePairs(
            _gather(base_output.cache[spec.site], _positions(base, spec.label)),
            _gather(donor_output.cache[spec.site], _positions(donor, spec.label)),
            spec.site, spec.label, "single")

    base_output = raw_forward(net, base, device=device, cache_names=(spec.late_site,))
    donor_output = raw_forward(net, donor, device=device, cache_names=(spec.early_site,))
    early = position_patch(
        donor_output.cache[spec.early_site], base_layouts, donor_layouts,
        base_role=spec.source_label, semantic_label=True)
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={spec.early_site: early})
    positions = _positions(base, spec.destination_label)
    return MediatorStatePairs(
        _gather(base_output.cache[spec.late_site], positions),
        _gather(propagated.cache[spec.late_site], positions),
        spec.late_site, spec.destination_label, "path_destination")


def mediator_subspace_logits(net: nn.Module, base: tuple[Episode, ...],
                             donor: tuple[Episode, ...], spec: MediatorSpec,
                             basis: Tensor, *, component: str = "subspace",
                             device="cpu") -> Tensor:
    """Transfer only a frozen basis or its complement through the selected mediator."""
    spec.validate()
    if component not in ("subspace", "complement"):
        raise ValueError("Causal subspace component must be subspace or complement")
    if not base or len(base) != len(donor):
        raise ValueError("Causal subspace batches must be nonempty and paired")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in donor)
    if spec.kind == "single":
        donor_output = raw_forward(net, donor, device=device, cache_names=(spec.site,))
        patch = position_patch(
            donor_output.cache[spec.site], base_layouts, donor_layouts,
            base_role=spec.label, basis=basis, component=component,
            semantic_label=True)
        return raw_forward(
            net, base, device=device, interventions={spec.site: patch}).logits

    donor_output = raw_forward(net, donor, device=device, cache_names=(spec.early_site,))
    early = position_patch(
        donor_output.cache[spec.early_site], base_layouts, donor_layouts,
        base_role=spec.source_label, semantic_label=True)
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={spec.early_site: early})
    late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, basis=basis, component=component,
        semantic_label=True)
    return raw_forward(net, base, device=device,
                       interventions={spec.late_site: late}).logits


def mediator_endpoint_logits(net: nn.Module, base: tuple[Episode, ...],
                             spec: MediatorSpec, states: Tensor, *,
                             device="cpu") -> Tensor:
    """Replace the frozen single/path endpoint with explicitly supplied states."""
    spec.validate()
    if not base or states.ndim != 2 or states.shape[0] != len(base):
        raise ValueError("Endpoint states must be [batch, width] and match base episodes")
    site = spec.site if spec.kind == "single" else spec.late_site
    label = spec.label if spec.kind == "single" else spec.destination_label
    layouts = tuple(semantic_layout(episode) for episode in base)
    record_materialized(states)

    def replace(value: Tensor) -> Tensor:
        if value.ndim != 3 or value.shape[0] != states.shape[0] \
                or value.shape[-1] != states.shape[-1]:
            raise ValueError("Endpoint state and residual activation shapes are incompatible")
        changed = value.clone()
        local = states.to(device=value.device, dtype=value.dtype)
        for index, layout in enumerate(layouts):
            changed[index, layout.label_position(label)] = local[index]
        return changed

    return raw_forward(net, base, device=device, interventions={site: replace}).logits


def mean_ablation_states(states: Tensor, mean: Tensor, basis: Tensor) -> Tensor:
    """Set selected coordinates to their discovery mean while preserving the complement."""
    if states.ndim != 2 or mean.ndim != 1 or basis.ndim != 2 \
            or states.shape[1] != mean.shape[0] or basis.shape[0] != mean.shape[0]:
        raise ValueError("Mean-ablation geometry has incompatible shapes")
    local_states, local_mean = states.double(), mean.double()
    local_basis = basis.double()
    centered = local_states - local_mean
    return (local_states - (centered @ local_basis) @ local_basis.T).to(states.dtype)


def equal_energy_offspace_states(base: Tensor, donor: Tensor, basis: Tensor, *,
                                 seed: int) -> Tensor:
    """Add random complement perturbations matching each selected donor-delta norm."""
    if base.shape != donor.shape or base.ndim != 2 or basis.ndim != 2 \
            or basis.shape[0] != base.shape[1] or basis.shape[1] == 0:
        raise ValueError("Equal-energy state geometry has incompatible shapes")
    local_basis = basis.double()
    delta = donor.double() - base.double()
    selected = (delta @ local_basis) @ local_basis.T
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(base.shape, generator=generator, dtype=torch.double)
    noise = noise - (noise @ local_basis) @ local_basis.T
    noise_norm = noise.norm(dim=1, keepdim=True)
    if (noise_norm <= 1e-12).any():
        raise ValueError("Degenerate equal-energy complement draw")
    scaled = noise / noise_norm * selected.norm(dim=1, keepdim=True)
    return (base.double() + scaled).to(base.dtype)


def equal_norm_component_corruption_states(states: Tensor, mean: Tensor, basis: Tensor, *,
                                           seed: int) -> Tensor:
    """Replace native selected coordinates by an equal-norm complement perturbation."""
    if states.ndim != 2 or mean.ndim != 1 or basis.ndim != 2 \
            or states.shape[1] != mean.shape[0] or basis.shape[0] != mean.shape[0]:
        raise ValueError("Component-corruption geometry has incompatible shapes")
    local, center, local_basis = states.double(), mean.double(), basis.double()
    centered = local - center
    selected = (centered @ local_basis) @ local_basis.T
    complement = centered - selected
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(states.shape, generator=generator, dtype=torch.double)
    noise = noise - (noise @ local_basis) @ local_basis.T
    noise_norm = noise.norm(dim=1, keepdim=True)
    if (noise_norm <= 1e-12).any():
        raise ValueError("Degenerate component-corruption complement draw")
    replacement = noise / noise_norm * selected.norm(dim=1, keepdim=True)
    return (center + complement + replacement).to(states.dtype)


def factor_state_panel(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                       spec: MediatorSpec, *, factor_name: str, episode_loader,
                       device="cpu", batch_size: int = 32) -> FactorStatePanel:
    """Capture registered content, binding or order contrasts on discovery groups."""
    if not groups or batch_size <= 0:
        raise ValueError("Factor panel and batch size must be nonempty")
    if factor_name == "content":
        pairs = FACTOR_VARIANTS["content"]
    elif factor_name == "binding":
        pairs = FACTOR_VARIANTS["binding"]
    elif factor_name == "order":
        pairs = FACTOR_VARIANTS["order"]
    else:
        raise ValueError(f"Unknown factor panel: {factor_name}")

    bases, donors, pair_metadata = [], [], []
    for group in groups:
        for nuisance_name, base_name, donor_name in pairs:
            base_rows = tuple(episode_loader(row) for row in group[base_name])
            donor_rows = tuple(episode_loader(row) for row in group[donor_name])
            if len(base_rows) != 3 or len(donor_rows) != 3:
                raise ValueError("Factor geometry requires three recipient rows")
            for recipient, (base, donor) in enumerate(zip(
                    base_rows, donor_rows, strict=True)):
                base_layout, donor_layout = semantic_layout(base), semantic_layout(donor)
                endpoint_label = spec.label if spec.kind == "single" \
                    else spec.destination_label
                source_label = spec.label if spec.kind == "single" else spec.source_label
                bases.append(base)
                donors.append(donor)
                pair_metadata.append({"logical_group_id": group["group_id"],
                                      "nuisance": (nuisance_name, recipient),
                                      "recipient": recipient,
                                      "base_variant": base_name,
                                      "donor_variant": donor_name,
                                      "base_logical_id": base.logical_id,
                                      "donor_logical_id": donor.logical_id,
                                      "base_render_id": base.render_id,
                                      "donor_render_id": donor.render_id,
                                      "endpoint_label": endpoint_label,
                                      "base_endpoint_position":
                                          base_layout.label_position(endpoint_label),
                                      "donor_endpoint_position": (
                                          donor_layout.label_position(endpoint_label)
                                          if spec.kind == "single"
                                          else base_layout.label_position(endpoint_label)),
                                      "source_label": source_label,
                                      "donor_source_position":
                                          donor_layout.label_position(source_label)})
    base_states, donor_states = [], []
    for offset in range(0, len(bases), batch_size):
        captured = mediator_state_pairs(
            net, tuple(bases[offset:offset + batch_size]),
            tuple(donors[offset:offset + batch_size]), spec, device=device)
        base_states.append(captured.base.detach().float().cpu())
        donor_states.append(captured.donor.detach().float().cpu())
    base_states, donor_states = torch.cat(base_states), torch.cat(donor_states)
    states, factor, nuisance, blocks, metadata = [], [], [], [], []
    for index, item in enumerate(pair_metadata):
        for level, tensor in ((0, base_states[index]), (1, donor_states[index])):
            states.append(tensor)
            factor.append(level)
            nuisance.append(item["nuisance"])
            blocks.append(item["logical_group_id"])
            metadata.append({**item, "factor": factor_name, "level": level,
                             "state_source": "native_base" if level == 0
                             else "donor_mediated"})
    return FactorStatePanel(torch.stack(states), tuple(factor), tuple(nuisance),
                            tuple(blocks), tuple(metadata))


def fit_registered_factor_geometries(content: FactorStatePanel,
                                     binding: FactorStatePanel,
                                     order: FactorStatePanel, *,
                                     maximum_rank: int = 64) -> RegisteredFactorGeometries:
    """Fit blocked contrast covariances and both registered orthogonalization orders."""
    geometries = tuple(blocked_contrast_geometry(
        panel.states, panel.factor, panel.nuisance, panel.blocks,
        maximum_rank=maximum_rank) for panel in (content, binding, order))
    names = ("content", "binding", "order")
    orthogonal = orthogonal_factor_geometry(
        *(geometry.basis for geometry in geometries),
        eigenvalues={name: geometry.eigenvalues for name, geometry in zip(
            names, geometries, strict=True)})
    return RegisteredFactorGeometries(*geometries, orthogonal)


def factor_rank_screen(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                       spec: MediatorSpec, basis: Tensor, *, factor_name: str,
                       episode_loader, candidate_ranks=(1, 2, 4, 8, 16, 32, 64),
                       device="cpu", logit_sink=None) -> list[dict]:
    """Measure full and prefix-rank finite effects for one frozen factor basis."""
    if basis.ndim != 2 or basis.shape[1] == 0:
        raise ValueError("Factor rank screen requires a nonempty width-by-rank basis")
    batches = []
    if factor_name == "content":
        for surface, base_name, donor_name in (
                ("marked", "base", "donor"),
                ("marker_free", "marker_free_base", "marker_free_donor"),
                ("reordered", "reordered_base", "reordered_donor"),
                ("format", "format_base", "format_donor"),
                ("distractor", "distractor_base", "distractor_donor")):
            bases, donors, metadata = [], [], []
            for group in groups:
                recipients = tuple(episode_loader(row) for row in group[base_name])
                source = episode_loader(group[donor_name][0])
                for recipient, episode in enumerate(recipients):
                    bases.append(episode)
                    donors.append(source)
                    metadata.append({"logical_group_id": group["group_id"],
                                     "recipient": recipient,
                                     "target": group["recipient_answers"][recipient],
                                     "fixed_donor_answer": source.answer,
                                     "base_render_id": episode.render_id,
                                     "donor_render_id": source.render_id})
            batches.append((surface, tuple(bases), tuple(donors), tuple(metadata)))
    elif factor_name == "binding":
        for surface, base_name, donor_name in (
                ("marked", "binding_base", "binding_donor"),
                ("marker_free", "binding_marker_free_base", "binding_marker_free_donor"),
                ("reordered", "binding_reordered_base", "binding_reordered_donor"),
                ("format", "binding_format_base", "binding_format_donor"),
                ("distractor", "binding_distractor_base", "binding_distractor_donor")):
            bases, donors, metadata = [], [], []
            for group in groups:
                base_rows = tuple(episode_loader(row) for row in group[base_name])
                donor_rows = tuple(episode_loader(row) for row in group[donor_name])
                source = donor_rows[0]
                for recipient, base in enumerate(base_rows):
                    bases.append(base)
                    donors.append(source)
                    metadata.append({"logical_group_id": group["group_id"],
                                     "recipient": recipient,
                                     "target": group["recipient_answers"][recipient],
                                     "fixed_donor_answer": source.answer,
                                     "base_render_id": base.render_id,
                                     "donor_render_id": source.render_id})
            batches.append((surface, tuple(bases), tuple(donors), tuple(metadata)))
    elif factor_name == "order":
        for assignment, base_name, donor_name in (
                ("f0_marked", "base", "reordered_base"),
                ("f1_marked", "donor", "reordered_donor"),
                ("f0_marker_free", "marker_free_base", "marker_free_reordered_base"),
                ("f1_marker_free", "marker_free_donor", "marker_free_reordered_donor"),
                ("f0_format", "format_base", "format_reordered_base"),
                ("f1_format", "format_donor", "format_reordered_donor"),
                ("f0_distractor", "distractor_base", "distractor_reordered_base"),
                ("f1_distractor", "distractor_donor", "distractor_reordered_donor")):
            bases, donors, metadata = [], [], []
            for group in groups:
                base_rows = tuple(episode_loader(row) for row in group[base_name])
                donor_rows = tuple(episode_loader(row) for row in group[donor_name])
                for recipient, (base, donor) in enumerate(zip(
                        base_rows, donor_rows, strict=True)):
                    bases.append(base)
                    donors.append(donor)
                    metadata.append({"logical_group_id": group["group_id"],
                                     "recipient": recipient, "target": base.answer,
                                     "fixed_donor_answer": donor_rows[0].answer,
                                     "base_render_id": base.render_id,
                                     "donor_render_id": donor.render_id})
            batches.append((assignment, tuple(bases), tuple(donors), tuple(metadata)))
    else:
        raise ValueError(f"Unknown factor rank screen: {factor_name}")

    rows = []
    ranks = tuple(rank for rank in candidate_ranks if 0 < rank <= basis.shape[1])
    if not ranks:
        raise ValueError("No registered candidate rank fits the factor basis")

    def tagged(metadata, condition, direction):
        return tuple({**row, "item_id": (
            f"{factor_name}:{condition}:{direction}:{row['logical_group_id']}:"
            f"{row['recipient']}")} for row in metadata)

    for direction, bases, donors, metadata in batches:
        clean = raw_forward(net, bases, device=device).logits
        full = mediator_logits(net, bases, donors, spec, device=device)
        full_condition = f"{factor_name}_rank_full"
        full_metadata = tagged(metadata, full_condition, direction)
        full_rows = diagnostic_rows(
            full, clean, bases, metadata=full_metadata,
            condition=full_condition, direction=direction,
            logit_sink=logit_sink)
        for row in full_rows:
            row.update({"factor": factor_name, "rank": "full"})
        rows.extend(full_rows)
        for rank in ranks:
            rank_condition = f"{factor_name}_rank_{rank}"
            rank_metadata = tagged(metadata, rank_condition, direction)
            logits = mediator_subspace_logits(
                net, bases, donors, spec, basis[:, :rank], device=device)
            rank_rows = diagnostic_rows(
                logits, clean, bases, metadata=rank_metadata,
                condition=rank_condition, direction=direction,
                logit_sink=logit_sink)
            for row in rank_rows:
                row.update({"factor": factor_name, "rank": rank})
            rows.extend(rank_rows)
    return rows


def summarize_factor_ranks(rows: list[dict]) -> dict:
    """Recompute grouped finite-effect metrics for every full/prefix rank cell."""
    by_rank = {}
    for row in rows:
        by_rank.setdefault(row["rank"], []).append(row)
    if "full" not in by_rank:
        raise ValueError("Factor rank rows lack a full-state reference")
    return {str(rank): summarize_confirmation_rows(selected)
            for rank, selected in sorted(
                by_rank.items(), key=lambda item: (-1 if item[0] == "full" else item[0]))}


def factor_transfer_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                         spec: MediatorSpec, basis: Tensor, *, factor_name: str,
                         condition_prefix: str, episode_loader, component="subspace",
                         surfaces: tuple[str, ...] | None = None, device="cpu",
                         logit_sink=None) -> list[dict]:
    """Apply one frozen factor basis bidirectionally on complete confirmation strata."""
    if factor_name not in FACTOR_VARIANTS or not groups:
        raise ValueError("Unknown or empty factor confirmation panel")
    selected = [pair for pair in FACTOR_VARIANTS[factor_name]
                if surfaces is None or pair[0] in surfaces]
    if not selected:
        raise ValueError("No requested factor confirmation surfaces exist")
    rows = []
    for surface, direction, base_tuple, donor_tuple, metadata in factor_episode_batches(
            groups, factor_name=factor_name, condition_prefix=condition_prefix,
            episode_loader=episode_loader, surfaces=surfaces):
        clean = raw_forward(net, base_tuple, device=device).logits
        edited = mediator_subspace_logits(
            net, base_tuple, donor_tuple, spec, basis,
            component=component, device=device)
        rows.extend(diagnostic_rows(
            edited, clean, base_tuple, metadata=metadata,
            condition=f"{condition_prefix}_{surface}", direction=direction,
            logit_sink=logit_sink))
    return rows


def factor_episode_batches(groups: tuple[dict, ...] | list[dict], *, factor_name: str,
                           condition_prefix: str, episode_loader,
                           surfaces: tuple[str, ...] | None = None):
    """Construct frozen bidirectional factor batches without neural inference."""
    if factor_name not in FACTOR_VARIANTS or not groups:
        raise ValueError("Unknown or empty factor confirmation panel")
    selected = [pair for pair in FACTOR_VARIANTS[factor_name]
                if surfaces is None or pair[0] in surfaces]
    if not selected:
        raise ValueError("No requested factor confirmation surfaces exist")
    batches = []
    for surface, left_name, right_name in selected:
        for direction, base_name, donor_name in (
                ("forward", left_name, right_name),
                ("reverse", right_name, left_name)):
            bases, donors, metadata = [], [], []
            for group in groups:
                base_rows = tuple(episode_loader(row) for row in group[base_name])
                donor_rows = tuple(episode_loader(row) for row in group[donor_name])
                for recipient, base in enumerate(base_rows):
                    donor = (donor_rows[0] if factor_name in ("content", "binding")
                             else donor_rows[recipient])
                    target = donor_rows[recipient].answer
                    condition = f"{condition_prefix}_{surface}"
                    bases.append(base)
                    donors.append(donor)
                    metadata.append({
                        "logical_group_id": group["group_id"], "recipient": recipient,
                        "target": target, "fixed_donor_answer": donor_rows[0].answer,
                        "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                        "item_id": (f"{factor_name}:{condition}:{direction}:"
                                    f"{group['group_id']}:{recipient}"),
                    })
            batches.append((surface, direction, tuple(bases), tuple(donors), tuple(metadata)))
    return tuple(batches)


def factor_equal_energy_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                             spec: MediatorSpec, basis: Tensor, *, factor_name: str,
                             condition_prefix: str, episode_loader, seed: int,
                             surfaces: tuple[str, ...] | None = None, device="cpu",
                             logit_sink=None) -> list[dict]:
    """Replace selected donor deltas by deterministic equal-norm complement edits."""
    rows = []
    for batch_index, (surface, direction, bases, donors, metadata) in enumerate(
            factor_episode_batches(
                groups, factor_name=factor_name, condition_prefix=condition_prefix,
                episode_loader=episode_loader, surfaces=surfaces)):
        clean = raw_forward(net, bases, device=device).logits
        pairs = mediator_state_pairs(net, bases, donors, spec, device=device)
        controlled = equal_energy_offspace_states(
            pairs.base, pairs.donor, basis, seed=seed + batch_index)
        edited = mediator_endpoint_logits(net, bases, spec, controlled, device=device)
        rows.extend(diagnostic_rows(
            edited, clean, bases, metadata=metadata,
            condition=f"{condition_prefix}_{surface}", direction=direction,
            logit_sink=logit_sink))
    return rows
