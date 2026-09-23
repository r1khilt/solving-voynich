"""Frozen-mediator confirmation primitives for TEACH-0013."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .teacher12_tasks import SYMBOL_START, Episode
from .teacher13_intervene import (
    cached_position_patch,
    ordered_two_site_patch,
    raw_forward,
)


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
