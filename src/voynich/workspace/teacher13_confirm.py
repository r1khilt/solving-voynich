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
                               donor_label: str, seed_fallback: int):
    if any(value.ndim != 3 for value in (
            base_activation, true_donor_activation, control_activation)) \
            or any(value.shape[0] != base_activation.shape[0]
                   or value.shape[-1] != base_activation.shape[-1]
                   for value in (true_donor_activation, control_activation)):
        raise ValueError("Matched donor control requires paired residual streams")
    base_positions = tuple(layout.label_position(base_label) for layout in base_layouts)
    true_positions = tuple(layout.label_position(donor_label) for layout in true_layouts)
    control_positions = tuple(layout.label_position(donor_label) for layout in control_layouts)
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

    def intervene(value: torch.Tensor) -> torch.Tensor:
        changed = value.clone()
        local = deltas.to(device=value.device, dtype=value.dtype)
        for index, position in enumerate(base_positions):
            changed[index, position] = value[index, position] + local[index]
        return changed

    return intervene


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


def _diagnostic_rows(logits: torch.Tensor, clean_logits: torch.Tensor,
                     bases: tuple[Episode, ...], *, metadata: tuple[dict, ...],
                     condition: str, direction: str) -> list[dict]:
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
        })
    return rows


def bidirectional_sufficiency(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                              spec: MediatorSpec, *, episode_loader, device="cpu",
                              render="marked") -> list[dict]:
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
            condition=f"sufficiency_{render}", direction=direction))
    return rows


def nuisance_preservation(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                          spec: MediatorSpec, *, family: str, episode_loader,
                          device="cpu") -> list[dict]:
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
        condition=f"same_key_{family}", direction="preserve")


def matched_sufficiency_controls(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                                 spec: MediatorSpec, *, episode_loader,
                                 seed: int = 73211, device="cpu") -> list[dict]:
    """Evaluate Gaussian and cyclic-next-group controls in both transfer directions."""
    if len(groups) < 2:
        raise ValueError("Cyclic confirmation control requires at least two groups")
    rows = []
    for direction, recipient_name, source_name, target_name in (
            ("forward", "base", "donor", "recipient_answers"),
            ("reverse", "donor", "base", "base_answers")):
        bases, true_donors, cyclic_donors, metadata = [], [], [], []
        for group_index, group in enumerate(groups):
            cyclic_group = groups[(group_index + 1) % len(groups)]
            recipient_episodes = tuple(episode_loader(row) for row in group[recipient_name])
            source_episodes = tuple(episode_loader(row) for row in group[source_name])
            cyclic_source = episode_loader(cyclic_group[source_name][0])
            targets = tuple(group[target_name])
            for recipient, episode in enumerate(recipient_episodes):
                bases.append(episode)
                true_donors.append(source_episodes[0])
                cyclic_donors.append(cyclic_source)
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
        rows.extend(_diagnostic_rows(
            random_logits, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_gaussian", direction=direction))
        rows.extend(_diagnostic_rows(
            cyclic_logits, clean, base_tuple, metadata=tuple(metadata),
            condition="norm_matched_cyclic", direction=direction))
    return rows


def positive_control_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict], *,
                          episode_loader, final_site: str, device="cpu") -> list[dict]:
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
        condition="clean_base", direction="clean")
    rows.extend(_diagnostic_rows(
        donor_logits, donor_logits, donor_tuple, metadata=tuple(donor_metadata),
        condition="clean_donor_input_f_replacement", direction="clean"))
    fixed_donor_tuple = tuple(
        donor_tuple[offset - offset % 3] for offset in range(len(donor_tuple)))
    injected = cached_position_patch(
        net, base_tuple, fixed_donor_tuple, site=final_site, role="answer",
        device=device, semantic_label=True).logits
    injection_metadata = tuple({**row, "target": row["fixed_donor_answer"]}
                               for row in base_metadata)
    rows.extend(_diagnostic_rows(
        injected, base_logits, base_tuple, metadata=injection_metadata,
        condition="final_state_fixed_answer_injection", direction="positive_control"))
    return rows


def task_specificity_rows(net: nn.Module, groups: tuple[dict, ...] | list[dict],
                          spec: MediatorSpec, *, episode_loader,
                          device="cpu") -> list[dict]:
    """Apply same-key nuisances to direct and copy tasks for the registered loss gates."""
    rows = []
    for task, base_name in (("direct", "direct_donor"), ("copy", "copy_control")):
        for family in ("format", "order", "distractor"):
            donor_name = f"{task}_{family}_donor"
            bases, donors, metadata = [], [], []
            for group in groups:
                base = episode_loader(group[base_name])
                donor = episode_loader(group[donor_name])
                if base.answer != donor.answer:
                    raise ValueError("Task-specific nuisance changed the oracle answer")
                bases.append(base)
                donors.append(donor)
                metadata.append({"logical_group_id": group["group_id"],
                                 "recipient": 0, "target": base.answer,
                                 "fixed_donor_answer": donor.answer})
            base_tuple = tuple(bases)
            clean = raw_forward(net, base_tuple, device=device).logits
            edited = mediator_logits(
                net, base_tuple, tuple(donors), spec, device=device)
            rows.extend(_diagnostic_rows(
                edited, clean, base_tuple, metadata=tuple(metadata),
                condition=f"{task}_same_key_{family}", direction="specificity"))
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
