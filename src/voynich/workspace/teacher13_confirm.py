"""Frozen-mediator confirmation primitives for TEACH-0013."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .teacher12_tasks import SYMBOL_START, Episode
from .teacher13_intervene import (
    cached_position_patch,
    ordered_two_site_patch,
    position_patch,
    raw_forward,
    record_materialized,
)
from .teacher13_tasks import semantic_layout


@dataclass(frozen=True)
class MediatorSpec:
    """One discovery-frozen single residual site or ordered two-site path."""

    kind: str
    site: str | None = None
    label: str | None = None
    early_site: str | None = None
    source_label: str | None = None
    late_site: str | None = None
    destination_label: str | None = None

    def validate(self) -> None:
        if self.kind == "single":
            if not self.site or not self.label or any((
                    self.early_site, self.source_label, self.late_site,
                    self.destination_label)):
                raise ValueError("Malformed single-site mediator")
        elif self.kind == "path":
            if self.site or self.label or not all((
                    self.early_site, self.source_label, self.late_site,
                    self.destination_label)):
                raise ValueError("Malformed two-site mediator")
        else:
            raise ValueError(f"Unknown mediator kind: {self.kind}")


def mediator_logits(net: nn.Module, base: tuple[Episode, ...], donor: tuple[Episode, ...],
                    spec: MediatorSpec, *, device="cpu") -> torch.Tensor:
    """Apply the frozen mediator protocol without adapting it to confirmation items."""
    spec.validate()
    if len(base) != len(donor) or not base:
        raise ValueError("Base and donor mediator batches must be nonempty and paired")
    if spec.kind == "single":
        return cached_position_patch(
            net, base, donor, site=spec.site, role=spec.label, device=device,
            semantic_label=True).logits
    return ordered_two_site_patch(
        net, base, donor, early_site=spec.early_site,
        early_label=spec.source_label, late_site=spec.late_site,
        late_label=spec.destination_label, device=device).path_logits


def _corresponding_position(reference: Episode, target: Episode, label: str) -> int:
    """Map one frozen composed occurrence to the same logical row/endpoint in a task."""
    reference_layout, target_layout = semantic_layout(reference), semantic_layout(target)
    reference_position = reference_layout.label_position(label)
    reference_row_index = reference_layout.row_indices[reference_position]
    if reference_row_index < 0:
        return target_layout.label_position(label)
    suffix = reference_layout.labels[reference_position].rsplit(".", 1)[-1]
    reference_row = reference.serialized_rows[reference_row_index]
    exact = [index for index, row in enumerate(target.serialized_rows)
             if row == reference_row]
    if not exact:
        # Copy uses the F0 graph while the composed reference uses F1. The queried
        # F row keeps its left/name endpoint and changes only its right/key endpoint.
        exact = [index for index, row in enumerate(target.serialized_rows)
                 if row[0] == reference_row[0]]
    if len(exact) != 1:
        raise ValueError("Frozen logical row has no unique cross-task correspondence")
    positions = [index for index, row_index in enumerate(target_layout.row_indices)
                 if row_index == exact[0]
                 and target_layout.labels[index].endswith(f".{suffix}")]
    if len(positions) != 1:
        raise ValueError("Frozen logical endpoint has no unique cross-task correspondence")
    return positions[0]


def _indexed_position_patch(donor: torch.Tensor, base_positions: tuple[int, ...],
                            donor_positions: tuple[int, ...]):
    if donor.ndim != 3 or donor.shape[0] != len(base_positions) \
            or len(base_positions) != len(donor_positions):
        raise ValueError("Indexed mediator positions and donor batch are incompatible")
    record_materialized(donor)

    def intervene(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[0] != donor.shape[0] \
                or value.shape[-1] != donor.shape[-1]:
            raise ValueError("Indexed mediator activation shapes are incompatible")
        changed = value.clone()
        local = donor.to(device=value.device, dtype=value.dtype)
        for index, (base_position, donor_position) in enumerate(zip(
                base_positions, donor_positions, strict=True)):
            changed[index, base_position] = local[index, donor_position]
        return changed

    return intervene


def cross_task_mediator_logits(net: nn.Module, base: tuple[Episode, ...],
                               donor: tuple[Episode, ...], references: tuple[Episode, ...],
                               spec: MediatorSpec, *, device="cpu") -> torch.Tensor:
    """Apply a frozen mediator through logical-row correspondence across task roles."""
    spec.validate()
    if not base or len(base) != len(donor) or len(base) != len(references):
        raise ValueError("Cross-task mediator batches must be nonempty and aligned")
    site = spec.site if spec.kind == "single" else spec.early_site
    label = spec.label if spec.kind == "single" else spec.source_label
    base_positions = tuple(_corresponding_position(reference, episode, label)
                           for reference, episode in zip(references, base, strict=True))
    donor_positions = tuple(_corresponding_position(reference, episode, label)
                            for reference, episode in zip(references, donor, strict=True))
    donor_output = raw_forward(net, donor, device=device, cache_names=(site,))
    early = _indexed_position_patch(
        donor_output.cache[site], base_positions, donor_positions)
    if spec.kind == "single":
        return raw_forward(net, base, device=device, interventions={site: early}).logits
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={site: early})
    late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    return raw_forward(net, base, device=device,
                       interventions={spec.late_site: late}).logits


def identity_error(net: nn.Module, episodes: tuple[Episode, ...], spec: MediatorSpec, *,
                   device="cpu") -> dict:
    """Separate fused/instrumented drift from a self-donor mediator identity edit."""
    spec.validate()
    fused = raw_forward(net, episodes, device=device).logits
    if spec.kind == "single":
        native = raw_forward(net, episodes, device=device, cache_names=(spec.site,)).logits
    else:
        native = raw_forward(
            net, episodes, device=device,
            cache_names=(spec.early_site, spec.late_site)).logits
    edited = mediator_logits(net, episodes, episodes, spec, device=device)
    fused_error = float((fused.float() - native.float()).abs().max().item())
    identity = float((edited.float() - native.float()).abs().max().item())
    finite = math.isfinite(fused_error) and math.isfinite(identity)
    return {"maximum_fused_instrumented_logit_error": fused_error,
            "maximum_identity_logit_error": identity, "finite": finite,
            "qualified": finite and fused_error < 1e-6 and identity < 1e-6}


def _scaled_noise_intervention(base_activation: torch.Tensor, donor_activation: torch.Tensor,
                               base_layouts, donor_layouts, *, base_label: str,
                               donor_label: str, seed: int):
    if base_activation.ndim != 3 or donor_activation.shape != base_activation.shape:
        raise ValueError("Random mediator control requires paired residual streams")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(base_activation.shape[0], base_activation.shape[-1],
                        generator=generator, dtype=torch.float32)
    base_positions = tuple(layout.label_position(base_label) for layout in base_layouts)
    donor_positions = tuple(layout.label_position(donor_label) for layout in donor_layouts)
    target_norms = torch.stack([
        (donor_activation[index, donor].float()
         - base_activation[index, base].float()).norm()
        for index, (base, donor) in enumerate(zip(
            base_positions, donor_positions, strict=True))])
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    noise = noise * target_norms[:, None].cpu()
    record_materialized(noise)

    def intervene(value: torch.Tensor) -> torch.Tensor:
        changed = value.clone()
        local_noise = noise.to(device=value.device, dtype=value.dtype)
        for index, position in enumerate(base_positions):
            changed[index, position] = value[index, position] + local_noise[index]
        return changed

    return intervene


def _scaled_donor_intervention(base_activation: torch.Tensor,
                               true_donor_activation: torch.Tensor,
                               control_activation: torch.Tensor, base_layouts,
                               true_layouts, control_layouts, *, base_label: str,
                               donor_label: str, seed_fallback: int,
                               control_positions: tuple[int, ...] | None = None):
    if any(value.ndim != 3 for value in (
            base_activation, true_donor_activation, control_activation)) \
            or any(value.shape[0] != base_activation.shape[0]
                   or value.shape[-1] != base_activation.shape[-1]
                   for value in (true_donor_activation, control_activation)):
        raise ValueError("Matched donor control requires paired residual streams")
    base_positions = tuple(layout.label_position(base_label) for layout in base_layouts)
    true_positions = tuple(layout.label_position(donor_label) for layout in true_layouts)
    if control_positions is None:
        control_positions = tuple(
            layout.label_position(donor_label) for layout in control_layouts)
    if len(control_positions) != base_activation.shape[0]:
        raise ValueError("Control positions must match the intervention batch")
    generator = torch.Generator(device="cpu").manual_seed(seed_fallback)
    fallback = torch.randn(base_activation.shape[0], base_activation.shape[-1],
                           generator=generator, dtype=torch.float32)
    deltas = []
    for index, (base_position, true_position, control_position) in enumerate(zip(
            base_positions, true_positions, control_positions, strict=True)):
        base_value = base_activation[index, base_position].float()
        true_norm = (true_donor_activation[index, true_position].float() - base_value).norm()
        control_delta = control_activation[index, control_position].float() - base_value
        control_norm = control_delta.norm()
        if control_norm <= 1e-12:
            control_delta = fallback[index]
            control_norm = control_delta.norm()
        deltas.append(control_delta / control_norm * true_norm)
    deltas = torch.stack(deltas)
    record_materialized(deltas)

    def intervene(value: torch.Tensor) -> torch.Tensor:
        changed = value.clone()
        local = deltas.to(device=value.device, dtype=value.dtype)
        for index, position in enumerate(base_positions):
            changed[index, position] = value[index, position] + local[index]
        return changed

    return intervene


def _nearest_wrong_positions(layouts, label: str) -> tuple[int, ...]:
    """Choose the nearest real token carrying any other semantic label."""
    positions = []
    for layout in layouts:
        source = layout.label_position(label)
        candidates = tuple(index for index, candidate in enumerate(layout.labels)
                           if candidate != label)
        if not candidates:
            raise ValueError("Wrong-position control requires another token")
        positions.append(min(candidates, key=lambda index: (abs(index - source), index)))
    return tuple(positions)


def norm_matched_random_logits(net: nn.Module, base: tuple[Episode, ...],
                               true_donor: tuple[Episode, ...], spec: MediatorSpec, *,
                               seed: int = 73211, device="cpu") -> torch.Tensor:
    """Run a seeded equal-norm Gaussian edit through the identical mediator protocol."""
    spec.validate()
    if len(base) != len(true_donor) or not base:
        raise ValueError("Random-control base and true donor batches must be paired")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in true_donor)
    if spec.kind == "single":
        base_output = raw_forward(net, base, device=device, cache_names=(spec.site,))
        donor_output = raw_forward(net, true_donor, device=device, cache_names=(spec.site,))
        random_patch = _scaled_noise_intervention(
            base_output.cache[spec.site], donor_output.cache[spec.site],
            base_layouts, donor_layouts, base_label=spec.label,
            donor_label=spec.label, seed=seed)
        return raw_forward(
            net, base, device=device, interventions={spec.site: random_patch}).logits

    base_output = raw_forward(net, base, device=device, cache_names=(spec.early_site,))
    donor_output = raw_forward(net, true_donor, device=device, cache_names=(spec.early_site,))
    random_early = _scaled_noise_intervention(
        base_output.cache[spec.early_site], donor_output.cache[spec.early_site],
        base_layouts, donor_layouts, base_label=spec.source_label,
        donor_label=spec.source_label, seed=seed)
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={spec.early_site: random_early})
    late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    return raw_forward(net, base, device=device, interventions={spec.late_site: late}).logits


def norm_matched_donor_logits(net: nn.Module, base: tuple[Episode, ...],
                              true_donor: tuple[Episode, ...],
                              control_donor: tuple[Episode, ...], spec: MediatorSpec, *,
                              fallback_seed: int = 73212, device="cpu") -> torch.Tensor:
    """Rescale a cyclic/wrong-key donor delta to each true donor delta norm."""
    spec.validate()
    if not base or len(base) != len(true_donor) or len(base) != len(control_donor):
        raise ValueError("Matched donor batches must be nonempty and equal length")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    true_layouts = tuple(semantic_layout(episode) for episode in true_donor)
    control_layouts = tuple(semantic_layout(episode) for episode in control_donor)
    site = spec.site if spec.kind == "single" else spec.early_site
    base_output = raw_forward(net, base, device=device, cache_names=(site,))
    true_output = raw_forward(net, true_donor, device=device, cache_names=(site,))
    control_output = raw_forward(net, control_donor, device=device, cache_names=(site,))
    label = spec.label if spec.kind == "single" else spec.source_label
    control = _scaled_donor_intervention(
        base_output.cache[site], true_output.cache[site], control_output.cache[site],
        base_layouts, true_layouts, control_layouts, base_label=label,
        donor_label=label, seed_fallback=fallback_seed)
    if spec.kind == "single":
        return raw_forward(net, base, device=device, interventions={site: control}).logits
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={site: control})
    late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    return raw_forward(net, base, device=device, interventions={spec.late_site: late}).logits


def wrong_position_logits(net: nn.Module, base: tuple[Episode, ...],
                          true_donor: tuple[Episode, ...], spec: MediatorSpec, *,
                          fallback_seed: int = 73213, device="cpu") -> torch.Tensor:
    """Use the nearest differently labelled donor token, rescaled to the true edit norm."""
    spec.validate()
    if not base or len(base) != len(true_donor):
        raise ValueError("Wrong-position batches must be nonempty and paired")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in true_donor)
    site = spec.site if spec.kind == "single" else spec.early_site
    label = spec.label if spec.kind == "single" else spec.source_label
    base_output = raw_forward(net, base, device=device, cache_names=(site,))
    donor_output = raw_forward(net, true_donor, device=device, cache_names=(site,))
    wrong_positions = _nearest_wrong_positions(donor_layouts, label)
    control = _scaled_donor_intervention(
        base_output.cache[site], donor_output.cache[site], donor_output.cache[site],
        base_layouts, donor_layouts, donor_layouts, base_label=label,
        donor_label=label, seed_fallback=fallback_seed,
        control_positions=wrong_positions)
    if spec.kind == "single":
        return raw_forward(net, base, device=device, interventions={site: control}).logits
    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={site: control})
    late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    return raw_forward(net, base, device=device, interventions={spec.late_site: late}).logits


def corruption_rescue_logits(net: nn.Module, base: tuple[Episode, ...],
                             true_donor: tuple[Episode, ...], spec: MediatorSpec, *,
                             seed: int = 73211, device="cpu"
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt the mediator by a matched Gaussian edit, then restore its native state."""
    spec.validate()
    if not base or len(base) != len(true_donor):
        raise ValueError("Corruption/rescue batches must be nonempty and paired")
    base_layouts = tuple(semantic_layout(episode) for episode in base)
    donor_layouts = tuple(semantic_layout(episode) for episode in true_donor)
    site = spec.site if spec.kind == "single" else spec.early_site
    label = spec.label if spec.kind == "single" else spec.source_label
    cache_names = (site,) if spec.kind == "single" else (site, spec.late_site)
    native = raw_forward(net, base, device=device, cache_names=cache_names)
    donor = raw_forward(net, true_donor, device=device, cache_names=(site,))
    corrupt = _scaled_noise_intervention(
        native.cache[site], donor.cache[site], base_layouts, donor_layouts,
        base_label=label, donor_label=label, seed=seed)
    if spec.kind == "single":
        corrupted = raw_forward(
            net, base, device=device, interventions={site: corrupt}).logits
        native_patch = position_patch(
            native.cache[site], base_layouts, base_layouts, base_role=label,
            semantic_label=True)

        def corrupt_then_restore(value: torch.Tensor) -> torch.Tensor:
            return native_patch(corrupt(value))

        rescued = raw_forward(
            net, base, device=device,
            interventions={site: corrupt_then_restore}).logits
        return corrupted, rescued

    propagated = raw_forward(
        net, base, device=device, cache_names=(spec.late_site,),
        interventions={site: corrupt})
    corrupted_late = position_patch(
        propagated.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    corrupted = raw_forward(
        net, base, device=device,
        interventions={spec.late_site: corrupted_late}).logits
    native_late = position_patch(
        native.cache[spec.late_site], base_layouts, base_layouts,
        base_role=spec.destination_label, semantic_label=True)
    rescued = raw_forward(
        net, base, device=device,
        interventions={spec.late_site: native_late}).logits
    return corrupted, rescued


def _diagnostic_rows(logits: torch.Tensor, clean_logits: torch.Tensor,
                     bases: tuple[Episode, ...], *, metadata: tuple[dict, ...],
                     condition: str, direction: str, logit_sink=None) -> list[dict]:
    logits = logits[:, SYMBOL_START:].float()
    clean_logits = clean_logits[:, SYMBOL_START:].float()
    if not torch.isfinite(logits).all() or not torch.isfinite(clean_logits).all():
        raise FloatingPointError("Nonfinite confirmation logits")
    probability, clean_probability = logits.softmax(-1), clean_logits.softmax(-1)
    if not torch.allclose(probability.sum(-1), torch.ones(len(bases), device=logits.device),
                          atol=1e-6, rtol=1e-6):
        raise FloatingPointError("Confirmation probabilities are not normalized")
    predictions = logits.argmax(-1) + SYMBOL_START
    clean_predictions = clean_logits.argmax(-1) + SYMBOL_START
    if logit_sink is not None:
        logit_sink(condition, direction, metadata, logits.detach().cpu(),
                   clean_logits.detach().cpu())
    rows = []
    for index, (episode, item) in enumerate(zip(bases, metadata, strict=True)):
        target, prediction = item["target"], int(predictions[index].item())
        target_index = target - SYMBOL_START
        candidates = {episode.query} | {right for _, right in episode.rows}
        other_targets = {row["target"] for row in metadata
                         if row["logical_group_id"] == item["logical_group_id"]
                         and row["recipient"] != item["recipient"]}
        if prediction == target:
            destination = "target"
        elif prediction == episode.answer:
            destination = "base_answer"
        elif prediction == item["fixed_donor_answer"]:
            destination = "fixed_donor_answer"
        elif prediction in other_targets:
            destination = "other_recipient_target"
        elif prediction in candidates:
            destination = "other_legal_candidate"
        else:
            destination = "outside_legal_candidates"
        rows.append({
            "condition": condition, "direction": direction,
            "logical_group_id": item["logical_group_id"],
            "group_id": f"{item['logical_group_id']}:{direction}",
            "recipient": item["recipient"], "target": target,
            "prediction": prediction, "base_answer": episode.answer,
            "clean_prediction": int(clean_predictions[index].item()),
            "fixed_donor_answer": item["fixed_donor_answer"],
            "candidate_member": prediction in candidates,
            "wrong_destination": destination,
            "target_probability": float(probability[index, target_index].item()),
            "base_target_probability": float(clean_probability[index, target_index].item()),
            "target_logit": float(logits[index, target_index].item()),
            "base_target_logit": float(clean_logits[index, target_index].item()),
            "prediction_logit": float(logits[index, prediction - SYMBOL_START].item()),
            **({"pair_id": item["pair_id"]} if "pair_id" in item else {}),
        })
    return rows


def diagnostic_rows(logits: torch.Tensor, clean_logits: torch.Tensor,
                    bases: tuple[Episode, ...], *, metadata: tuple[dict, ...],
                    condition: str, direction: str, logit_sink=None) -> list[dict]:
    """Public compact-row constructor shared by frozen post-discovery analyses."""
    return _diagnostic_rows(
        logits, clean_logits, bases, metadata=metadata, condition=condition,
        direction=direction, logit_sink=logit_sink)


def fresh_panel_confirmation_rows(net: nn.Module,
                                  groups: tuple[dict, ...] | list[dict], *,
                                  episode_loader, device="cpu", logit_sink=None,
                                  batch_size: int = 64) -> list[dict]:
    """Retain auditable per-item clean rows for every registered Stage-A cell."""
    if not groups or batch_size <= 0:
        raise ValueError("Fresh confirmation panel and batch size must be nonempty")
    cells: dict[str, tuple[list[Episode], list[dict]]] = {}

    def add(condition: str, episode: Episode, group_id: str, recipient: int,
            pair_id: str):
        episodes, metadata = cells.setdefault(condition, ([], []))
        episodes.append(episode)
        metadata.append({"logical_group_id": group_id, "recipient": recipient,
                         "target": episode.answer,
                         "fixed_donor_answer": episode.answer, "pair_id": pair_id})

    for group in groups:
        group_id = group["group_id"]
        for assignment, marked_name, marker_name, order_name in (
                ("f0", "base", "marker_free_base", "reordered_base"),
                ("f1", "donor", "marker_free_donor", "reordered_donor")):
            marked = tuple(episode_loader(row) for row in group[marked_name])
            marker = tuple(episode_loader(row) for row in group[marker_name])
            order = tuple(episode_loader(row) for row in group[order_name])
            for recipient, (original, marker_episode, order_episode) in enumerate(zip(
                    marked, marker, order, strict=True)):
                pair_id = f"{group_id}:{assignment}:{recipient}"
                add(f"stage_a_marked_{assignment}", original, group_id, recipient, pair_id)
                add(f"stage_a_marker_free_{assignment}", marker_episode,
                    group_id, recipient, pair_id)
                add(f"stage_a_reordered_{assignment}", order_episode,
                    group_id, recipient, pair_id)
        for assignment, name in (("f0", "distractor_base"),
                                 ("f1", "distractor_donor")):
            for recipient, row in enumerate(group[name]):
                add(f"stage_a_distractor_{assignment}", episode_loader(row),
                    group_id, recipient, f"{group_id}:{assignment}:{recipient}")
        for assignment, name in (("f0", "first_hop_base"),
                                 ("f1", "first_hop_donor")):
            add(f"stage_a_first_hop_{assignment}", episode_loader(group[name]),
                group_id, 0, f"{group_id}:{assignment}")
        for assignment, name in (("f0", "direct_base"), ("f1", "direct_donor")):
            add(f"stage_a_direct_{assignment}", episode_loader(group[name]),
                group_id, 0, f"{group_id}:{assignment}")
        add("stage_a_copy", episode_loader(group["copy_control"]), group_id, 0,
            f"{group_id}:copy")

    rows = []
    for condition in sorted(cells):
        episodes, metadata = cells[condition]
        for offset in range(0, len(episodes), batch_size):
            batch = tuple(episodes[offset:offset + batch_size])
            batch_metadata = tuple(metadata[offset:offset + batch_size])
            logits = raw_forward(net, batch, device=device).logits
            rows.extend(_diagnostic_rows(
                logits, logits, batch, metadata=batch_metadata,
                condition=condition, direction="stage_a", logit_sink=logit_sink))
    return rows


def summarize_fresh_panel_rows(rows: list[dict]) -> dict:
    """Recompute the exact registered Stage-A gate from retained item rows."""
    by_condition = {}
    for row in rows:
        if row["direction"] == "stage_a":
            by_condition.setdefault(row["condition"], []).append(row)
    required = {
        "stage_a_marked_f0", "stage_a_marked_f1",
        "stage_a_marker_free_f0", "stage_a_marker_free_f1",
        "stage_a_reordered_f0", "stage_a_reordered_f1",
        "stage_a_distractor_f0", "stage_a_distractor_f1",
        "stage_a_first_hop_f0", "stage_a_first_hop_f1",
        "stage_a_direct_f0", "stage_a_direct_f1", "stage_a_copy",
    }
    if set(by_condition) != required:
        raise ValueError("Fresh-panel retained conditions are incomplete")

    def correct(row):
        return row["prediction"] == row["target"]

    def accuracy(names):
        selected = [row for name in names for row in by_condition[name]]
        return sum(correct(row) for row in selected) / len(selected)

    def exact_recipient_groups(name):
        grouped = {}
        for row in by_condition[name]:
            grouped.setdefault(row["logical_group_id"], []).append(row)
        if any({row["recipient"] for row in group} != {0, 1, 2}
               for group in grouped.values()):
            raise ValueError("Fresh-panel recipient triples are incomplete")
        return sum(all(correct(row) for row in group) for group in grouped.values()) \
            / len(grouped)

    def paired(left_names, right_names):
        left = {row["pair_id"]: row for name in left_names for row in by_condition[name]}
        right = {row["pair_id"]: row for name in right_names for row in by_condition[name]}
        if left.keys() != right.keys() or len(left) != sum(
                len(by_condition[name]) for name in left_names):
            raise ValueError("Fresh-panel pair IDs are incomplete or duplicated")
        return sum(correct(left[key]) and correct(right[key]) for key in left) / len(left)

    marked = ("stage_a_marked_f0", "stage_a_marked_f1")
    return {
        "composed_items": accuracy(marked),
        "base_recipient_groups": exact_recipient_groups("stage_a_marked_f0"),
        "donor_recipient_groups": exact_recipient_groups("stage_a_marked_f1"),
        "marker_pairs": paired(marked, (
            "stage_a_marker_free_f0", "stage_a_marker_free_f1")),
        "order_pairs": paired(marked, (
            "stage_a_reordered_f0", "stage_a_reordered_f1")),
        "distractor_pairs": paired(
            marked, ("stage_a_distractor_f0", "stage_a_distractor_f1")),
        "first_hop": accuracy(("stage_a_first_hop_f0", "stage_a_first_hop_f1")),
        "direct": accuracy(("stage_a_direct_f0", "stage_a_direct_f1")),
        "copy": accuracy(("stage_a_copy",)),
    }


def bidirectional_sufficiency(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                              spec: MediatorSpec, *, episode_loader, device="cpu",
                              render="marked", logit_sink=None) -> list[dict]:
    """Reuse one G0 donor state across all three recipient tables in both directions."""
    if not groups:
        raise ValueError("Confirmation group list is empty")
    if render not in ("marked", "marker_free"):
        raise ValueError("Unknown confirmation render")
    base_name = "base" if render == "marked" else "marker_free_base"
    donor_name = "donor" if render == "marked" else "marker_free_donor"
    rows = []
    for direction, recipient_name, source_name, target_name in (
            ("forward", base_name, donor_name, "recipient_answers"),
            ("reverse", donor_name, base_name, "base_answers")):
        bases, donors, metadata = [], [], []
        for group in groups:
            recipient_episodes = tuple(episode_loader(row) for row in group[recipient_name])
            source_episodes = tuple(episode_loader(row) for row in group[source_name])
            if len(recipient_episodes) != 3 or len(source_episodes) != 3:
                raise ValueError("Confirmation groups require three recipients")
            targets = tuple(group[target_name])
            for recipient, episode in enumerate(recipient_episodes):
                bases.append(episode)
                donors.append(source_episodes[0])
                metadata.append({"logical_group_id": group["group_id"],
                                 "recipient": recipient, "target": targets[recipient],
                                 "fixed_donor_answer": source_episodes[0].answer})
        base_tuple, donor_tuple = tuple(bases), tuple(donors)
        clean = raw_forward(net, base_tuple, device=device).logits
        edited = mediator_logits(net, base_tuple, donor_tuple, spec, device=device)
        rows.extend(_diagnostic_rows(
            edited, clean, base_tuple, metadata=tuple(metadata),
            condition=f"sufficiency_{render}", direction=direction,
            logit_sink=logit_sink))
    return rows


def nuisance_preservation(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                          spec: MediatorSpec, *, family: str, episode_loader,
                          device="cpu", logit_sink=None) -> list[dict]:
    """Patch a same-logic format/order/distractor state and require answer preservation."""
    variants = {"format": ("donor", "format_donor"),
                "order": ("donor", "reordered_donor"),
                "distractor": ("donor", "distractor_donor")}
    if family not in variants:
        raise ValueError(f"Unknown nuisance family: {family}")
    base_name, source_name = variants[family]
    bases, donors, metadata = [], [], []
    for group in groups:
        base_rows = tuple(episode_loader(row) for row in group[base_name])
        source_rows = tuple(episode_loader(row) for row in group[source_name])
        if len(base_rows) != 3 or len(source_rows) != 3:
            raise ValueError("Nuisance confirmation requires three recipients")
        for recipient, (base, donor) in enumerate(zip(base_rows, source_rows, strict=True)):
            if base.answer != donor.answer:
                raise ValueError("Nuisance donor changed the registered answer")
            bases.append(base)
            donors.append(donor)
            metadata.append({"logical_group_id": group["group_id"],
                             "recipient": recipient, "target": base.answer,
                             "fixed_donor_answer": donor.answer})
    base_tuple, donor_tuple = tuple(bases), tuple(donors)
    clean = raw_forward(net, base_tuple, device=device).logits
    edited = mediator_logits(net, base_tuple, donor_tuple, spec, device=device)
    return _diagnostic_rows(
        edited, clean, base_tuple, metadata=tuple(metadata),
        condition=f"same_key_{family}", direction="preserve",
        logit_sink=logit_sink)


def matched_sufficiency_controls(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                                 spec: MediatorSpec, *, episode_loader,
                                 seed: int = 73211, device="cpu",
                                 logit_sink=None) -> list[dict]:
    """Evaluate all registered norm-matched negative controls in both directions."""
    if len(groups) < 2:
        raise ValueError("Cyclic confirmation control requires at least two groups")
    rows = []
    for direction, recipient_name, source_name, target_name in (
            ("forward", "base", "donor", "recipient_answers"),
            ("reverse", "donor", "base", "base_answers")):
        bases, true_donors, cyclic_donors, wrong_key_donors, metadata = [], [], [], [], []
        for group_index, group in enumerate(groups):
            cyclic_group = groups[(group_index + 1) % len(groups)]
            recipient_episodes = tuple(episode_loader(row) for row in group[recipient_name])
            source_episodes = tuple(episode_loader(row) for row in group[source_name])
            cyclic_source = episode_loader(cyclic_group[source_name][0])
            true_key_name = "key_donor" if direction == "forward" else "key_base"
            wrong_key_name = "key_base" if direction == "forward" else "key_donor"
            wrong_group = next((
                candidate for offset in range(1, len(groups))
                if (candidate := groups[(group_index + offset) % len(groups)])[wrong_key_name]
                != group[true_key_name]), None)
            if wrong_group is None:
                raise ValueError("No foreign next-group wrong-key control exists")
            wrong_key_source = episode_loader(wrong_group[recipient_name][0])
            targets = tuple(group[target_name])
            for recipient, episode in enumerate(recipient_episodes):
                bases.append(episode)
                true_donors.append(source_episodes[0])
                cyclic_donors.append(cyclic_source)
                wrong_key_donors.append(wrong_key_source)
                metadata.append({"logical_group_id": group["group_id"],
                                 "recipient": recipient, "target": targets[recipient],
                                 "fixed_donor_answer": source_episodes[0].answer})
        base_tuple, true_tuple = tuple(bases), tuple(true_donors)
        clean = raw_forward(net, base_tuple, device=device).logits
        random_logits = norm_matched_random_logits(
            net, base_tuple, true_tuple, spec, seed=seed, device=device)
        cyclic_logits = norm_matched_donor_logits(
            net, base_tuple, true_tuple, tuple(cyclic_donors), spec,
            fallback_seed=seed + 1, device=device)
        wrong_position = wrong_position_logits(
            net, base_tuple, true_tuple, spec, fallback_seed=seed + 2,
            device=device)
        wrong_key = norm_matched_donor_logits(
            net, base_tuple, true_tuple, tuple(wrong_key_donors), spec,
            fallback_seed=seed + 3, device=device)
        rows.extend(_diagnostic_rows(
            random_logits, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_gaussian", direction=direction,
            logit_sink=logit_sink))
        rows.extend(_diagnostic_rows(
            cyclic_logits, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_cyclic", direction=direction,
            logit_sink=logit_sink))
        rows.extend(_diagnostic_rows(
            wrong_position, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_wrong_position", direction=direction,
            logit_sink=logit_sink))
        rows.extend(_diagnostic_rows(
            wrong_key, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_wrong_key", direction=direction,
            logit_sink=logit_sink))
    return rows


def corruption_rescue_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                           spec: MediatorSpec, *, episode_loader,
                           seed: int = 73211, device="cpu",
                           logit_sink=None) -> list[dict]:
    """Pair registered corruption with native-state rescue on both F assignments."""
    rows = []
    for direction, recipient_name, source_name in (
            ("forward", "base", "donor"),
            ("reverse", "donor", "base")):
        bases, true_donors, metadata = [], [], []
        for group in groups:
            recipient_episodes = tuple(episode_loader(row) for row in group[recipient_name])
            source_episodes = tuple(episode_loader(row) for row in group[source_name])
            if len(recipient_episodes) != 3 or len(source_episodes) != 3:
                raise ValueError("Corruption/rescue requires three-recipient groups")
            for recipient, episode in enumerate(recipient_episodes):
                bases.append(episode)
                true_donors.append(source_episodes[0])
                metadata.append({"logical_group_id": group["group_id"],
                                 "recipient": recipient, "target": episode.answer,
                                 "fixed_donor_answer": source_episodes[0].answer})
        base_tuple = tuple(bases)
        clean = raw_forward(net, base_tuple, device=device).logits
        corrupted, rescued = corruption_rescue_logits(
            net, base_tuple, tuple(true_donors), spec, seed=seed,
            device=device)
        corrupt_rows = _diagnostic_rows(
            corrupted, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_corruption", direction=direction,
            logit_sink=logit_sink)
        rescue_rows = _diagnostic_rows(
            rescued, clean, base_tuple, metadata=tuple(metadata),
            condition="native_state_rescue", direction=direction,
            logit_sink=logit_sink)
        for corrupt_row, rescue_row in zip(corrupt_rows, rescue_rows, strict=True):
            changed = corrupt_row["prediction"] != corrupt_row["clean_prediction"]
            corrupt_row["corruption_changed_clean_prediction"] = changed
            rescue_row["corruption_changed_clean_prediction"] = changed
        rows.extend(corrupt_rows)
        rows.extend(rescue_rows)
    return rows


def summarize_rescue_rows(rows: list[dict]) -> dict:
    """Score restoration only where the paired corruption changed the clean prediction."""
    corruptions = {(row["group_id"], row["recipient"]): row for row in rows
                   if row["condition"] == "norm_matched_corruption"}
    rescues = {(row["group_id"], row["recipient"]): row for row in rows
               if row["condition"] == "native_state_rescue"}
    if not corruptions or corruptions.keys() != rescues.keys():
        raise ValueError("Corruption and rescue rows are not paired")
    changed = [key for key, row in corruptions.items()
               if row["prediction"] != row["clean_prediction"]]
    restored = sum(rescues[key]["prediction"] == rescues[key]["clean_prediction"]
                   for key in changed)
    return {"paired_items": len(corruptions), "changed_cases": len(changed),
            "restored_cases": restored,
            "rescue_rate": None if not changed else restored / len(changed)}


def positive_control_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict], *,
                          episode_loader, final_site: str, device="cpu",
                          logit_sink=None) -> list[dict]:
    """Clean/input-F gates and the registered fixed-answer final-state injection."""
    bases, donors, base_metadata, donor_metadata = [], [], [], []
    for group in groups:
        base_rows = tuple(episode_loader(row) for row in group["base"])
        donor_rows = tuple(episode_loader(row) for row in group["donor"])
        for recipient, (base, donor) in enumerate(zip(base_rows, donor_rows, strict=True)):
            bases.append(base)
            donors.append(donor)
            base_metadata.append({"logical_group_id": group["group_id"],
                                  "recipient": recipient, "target": base.answer,
                                  "fixed_donor_answer": donor_rows[0].answer})
            donor_metadata.append({"logical_group_id": group["group_id"],
                                   "recipient": recipient, "target": donor.answer,
                                   "fixed_donor_answer": donor_rows[0].answer})
    base_tuple, donor_tuple = tuple(bases), tuple(donors)
    base_logits = raw_forward(net, base_tuple, device=device).logits
    donor_logits = raw_forward(net, donor_tuple, device=device).logits
    rows = _diagnostic_rows(
        base_logits, base_logits, base_tuple, metadata=tuple(base_metadata),
        condition="clean_base", direction="clean", logit_sink=logit_sink)
    rows.extend(_diagnostic_rows(
        donor_logits, donor_logits, donor_tuple, metadata=tuple(donor_metadata),
        condition="clean_donor_input_f_replacement", direction="clean",
        logit_sink=logit_sink))
    fixed_donor_tuple = tuple(
        donor_tuple[offset - offset % 3] for offset in range(len(donor_tuple)))
    injected = cached_position_patch(
        net, base_tuple, fixed_donor_tuple, site=final_site, role="answer",
        device=device, semantic_label=True).logits
    injection_metadata = tuple({**row, "target": row["fixed_donor_answer"]}
                               for row in base_metadata)
    rows.extend(_diagnostic_rows(
        injected, base_logits, base_tuple, metadata=injection_metadata,
        condition="final_state_fixed_answer_injection", direction="positive_control",
        logit_sink=logit_sink))
    return rows


def task_specificity_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                          spec: MediatorSpec, *, episode_loader,
                          device="cpu", logit_sink=None) -> list[dict]:
    """Apply same-key nuisances to direct and copy tasks for the registered loss gates."""
    rows = []
    for task, base_name in (("direct", "direct_donor"), ("copy", "copy_control")):
        for family in ("format", "order", "distractor"):
            donor_name = f"{task}_{family}_donor"
            bases, donors, references, metadata = [], [], [], []
            for group in groups:
                base = episode_loader(group[base_name])
                donor = episode_loader(group[donor_name])
                reference = episode_loader(group["donor"][0])
                if base.answer != donor.answer:
                    raise ValueError("Task-specific nuisance changed the oracle answer")
                bases.append(base)
                donors.append(donor)
                references.append(reference)
                metadata.append({"logical_group_id": group["group_id"],
                                 "recipient": 0, "target": base.answer,
                                 "fixed_donor_answer": donor.answer})
            base_tuple = tuple(bases)
            clean = raw_forward(net, base_tuple, device=device).logits
            edited = cross_task_mediator_logits(
                net, base_tuple, tuple(donors), tuple(references), spec,
                device=device)
            rows.extend(_diagnostic_rows(
                edited, clean, base_tuple, metadata=tuple(metadata),
                condition=f"{task}_same_key_{family}", direction="specificity",
                logit_sink=logit_sink))
    return rows


def summarize_confirmation_rows(rows: list[dict], *, direction: str | None = None) -> dict:
    """Exact item/group/non-injection metrics for compact confirmation rows."""
    selected = [row for row in rows if direction is None or row["direction"] == direction]
    if not selected:
        raise ValueError("No confirmation rows for requested condition/direction")
    grouped = {}
    for row in selected:
        key = row["group_id"]
        grouped.setdefault(key, []).append(row)
    if any({row["recipient"] for row in group} != {0, 1, 2}
           for group in grouped.values()):
        raise ValueError("Confirmation rows do not form complete three-recipient groups")
    correct = sum(row["prediction"] == row["target"] for row in selected)
    exact = sum(all(row["prediction"] == row["target"] for row in group)
                for group in grouped.values())
    changed = [row for row in selected if row["recipient"] != 0]
    non_injected = sum(row["prediction"] != row["fixed_donor_answer"] for row in changed)
    gains = [row["target_probability"] - row["base_target_probability"] for row in selected]
    return {"items": len(selected), "correct": correct,
            "item_accuracy": correct / len(selected), "groups": len(grouped),
            "exact_groups": exact, "group_accuracy": exact / len(grouped),
            "changed_recipient_items": len(changed), "non_injected": non_injected,
            "non_injection": non_injected / len(changed),
            "mean_probability_gain": sum(gains) / len(gains)}
