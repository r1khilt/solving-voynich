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
from .teacher13_intervene import position_patch, raw_forward
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


def factor_state_panel(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                       spec: MediatorSpec, *, factor_name: str, episode_loader,
                       device="cpu", batch_size: int = 32) -> FactorStatePanel:
    """Capture registered content, binding or order contrasts on discovery groups."""
    if not groups or batch_size <= 0:
        raise ValueError("Factor panel and batch size must be nonempty")
    if factor_name == "content":
        pairs = (
            ("marked", "base", "donor"),
            ("marker_free", "marker_free_base", "marker_free_donor"),
            ("reordered", "reordered_base", "reordered_donor"),
            ("format", "format_base", "format_donor"),
            ("distractor", "distractor_base", "distractor_donor"),
        )
    elif factor_name == "binding":
        pairs = (
            ("marked", "binding_base", "binding_donor"),
            ("marker_free", "binding_marker_free_base", "binding_marker_free_donor"),
            ("reordered", "binding_reordered_base", "binding_reordered_donor"),
            ("format", "binding_format_base", "binding_format_donor"),
            ("distractor", "binding_distractor_base", "binding_distractor_donor"),
        )
    elif factor_name == "order":
        pairs = (
            ("f0_marked", "base", "reordered_base"),
            ("f1_marked", "donor", "reordered_donor"),
            ("f0_marker_free", "marker_free_base", "marker_free_reordered_base"),
            ("f1_marker_free", "marker_free_donor", "marker_free_reordered_donor"),
            ("f0_format", "format_base", "format_reordered_base"),
            ("f1_format", "format_donor", "format_reordered_donor"),
            ("f0_distractor", "distractor_base", "distractor_reordered_base"),
            ("f1_distractor", "distractor_donor", "distractor_reordered_donor"),
        )
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
                bases.append(base)
                donors.append(donor)
                pair_metadata.append({"logical_group_id": group["group_id"],
                                      "nuisance": (nuisance_name, recipient),
                                      "recipient": recipient})
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
            metadata.append({**item, "factor": factor_name, "level": level})
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
    orthogonal = orthogonal_factor_geometry(*(geometry.basis for geometry in geometries))
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
                                     "fixed_donor_answer": source.answer})
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
                for recipient, (base, donor) in enumerate(zip(
                        base_rows, donor_rows, strict=True)):
                    bases.append(base)
                    donors.append(donor)
                    metadata.append({"logical_group_id": group["group_id"],
                                     "recipient": recipient, "target": donor.answer,
                                     "fixed_donor_answer": donor_rows[0].answer})
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
                                     "fixed_donor_answer": donor_rows[0].answer})
            batches.append((assignment, tuple(bases), tuple(donors), tuple(metadata)))
    else:
        raise ValueError(f"Unknown factor rank screen: {factor_name}")

    rows = []
    ranks = tuple(rank for rank in candidate_ranks if 0 < rank <= basis.shape[1])
    if not ranks:
        raise ValueError("No registered candidate rank fits the factor basis")
    for direction, bases, donors, metadata in batches:
        clean = raw_forward(net, bases, device=device).logits
        full = mediator_logits(net, bases, donors, spec, device=device)
        full_rows = diagnostic_rows(
            full, clean, bases, metadata=metadata,
            condition=f"{factor_name}_rank_full", direction=direction,
            logit_sink=logit_sink)
        for row in full_rows:
            row.update({"factor": factor_name, "rank": "full"})
        rows.extend(full_rows)
        for rank in ranks:
            logits = mediator_subspace_logits(
                net, bases, donors, spec, basis[:, :rank], device=device)
            rank_rows = diagnostic_rows(
                logits, clean, bases, metadata=metadata,
                condition=f"{factor_name}_rank_{rank}", direction=direction,
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
