"""Prospective finite parser-gate rescue controls for trained TEACH-0014 models.

This module defines interventions only. It does not load a checkpoint, read a
neural outcome, or issue a bottleneck verdict. The public visible pair grammar
is used solely to construct oracle controls at inference time.
"""

import math

import torch

from .teacher14_models import CandidateEdgeWorkspace
from .teacher14_objectives import padded_tokens
from .teacher14_tasks import (
    SYMBOL_START, Episode, digest, strip_pair_rows,
)


GATE_MAGNITUDE = 20.0
CONDITIONS = ("native", "gold", "false_only", "reversed", "count_random")


def _ordinary_prediction(logits: torch.Tensor) -> int:
    return int(logits[0, SYMBOL_START:].argmax().item()) + SYMBOL_START


def _target_indices(episode: Episode) -> tuple[int, int]:
    rows = strip_pair_rows(episode.tokens)
    if episode.task != "composed" or episode.hops != 2:
        raise ValueError("Two-hop composed episode required")
    mapping = dict(rows)
    if len(mapping) != len(rows):
        raise ValueError("Visible rows are not a function")
    key = mapping[episode.query]
    answer = mapping[key]
    if answer != episode.answer:
        raise ValueError("Visible and recorded answers disagree")
    return 2 * rows.index((episode.query, key)), 2 * rows.index((key, answer))


def _path_indices(episode: Episode) -> tuple[int, ...]:
    """Locate requested edges in physical visible order, for any task length."""
    rows = strip_pair_rows(episode.tokens)
    mapping = dict(rows)
    if len(mapping) != len(rows):
        raise ValueError("Visible rows are not a function")
    value = episode.query
    indices = []
    for _ in range(episode.hops):
        if value not in mapping:
            raise ValueError("Requested visible path incomplete")
        next_value = mapping[value]
        indices.append(2 * rows.index((value, next_value)))
        value = next_value
    if value != episode.answer:
        raise ValueError("Visible and recorded answers disagree")
    return tuple(indices)


def _gate_conditions(native: torch.Tensor, mask: torch.Tensor,
                     render_id: str, random_rep: int) -> dict[str, torch.Tensor]:
    if native.ndim != 2 or native.shape[0] != 1 or mask.shape != native.shape or (
            mask.dtype != torch.bool):
        raise ValueError("Candidate gate/mask shape invalid")
    count = int(mask[0].sum().item())
    if count < 1 or count % 2 != 1 or not bool(mask[0, :count].all().item()) or (
            bool(mask[0, count:].any().item())):
        raise ValueError("Candidate mask not contiguous odd adjacent pairs")
    row_count = (count + 1) // 2
    index = torch.arange(native.shape[1], device=native.device)
    even = index.remainder(2).eq(0) & mask[0]
    odd = index.remainder(2).eq(1) & mask[0]
    gold = torch.full_like(native, -GATE_MAGNITUDE)
    gold[0, even] = GATE_MAGNITUDE
    reversed_gate = torch.full_like(native, -GATE_MAGNITUDE)
    reversed_gate[0, odd] = GATE_MAGNITUDE
    false_only = native.clone()
    false_only[0, odd] = -GATE_MAGNITUDE
    generator = torch.Generator(device="cpu")
    seed = int(digest(["TEACH-0014-gate-random", render_id,
                       random_rep])[:16], 16) % 2**63
    generator.manual_seed(seed)
    selected = torch.randperm(count, generator=generator)[:row_count].to(
        native.device)
    random_gate = torch.full_like(native, -GATE_MAGNITUDE)
    random_gate[0, selected] = GATE_MAGNITUDE
    return {"native": native.clone(), "gold": gold,
            "false_only": false_only, "reversed": reversed_gate,
            "count_random": random_gate}


@torch.no_grad()
def evaluate_episode(model: CandidateEdgeWorkspace, episode: Episode,
                     *, device: str, random_rep: int = 0) -> dict:
    if not isinstance(model, CandidateEdgeWorkspace) or model.oracle_rows:
        raise ValueError("Raw candidate-edge workspace required")
    if type(random_rep) is not int or random_rep < 0:
        raise ValueError("Nonnegative integer random replicate required")
    first, second = _target_indices(episode)
    ids = padded_tokens([episode], device)
    model.eval()
    native = model(ids, capture=True)
    gates = native.auxiliary["edge_gate_logits"]
    mask = native.auxiliary["candidate_mask"]
    rows = strip_pair_rows(episode.tokens)
    count = int(mask[0].sum().item())
    if count != 2 * len(rows) - 1:
        raise ValueError("Candidate count differs from visible rows")
    left = native.cache["candidate_left_ids"][0]
    right = native.cache["candidate_right_ids"][0]
    if tuple((int(left[2 * index]), int(right[2 * index]))
             for index in range(len(rows))) != rows:
        raise ValueError("Even candidates are not visible relation rows")
    assignments = _gate_conditions(gates, mask, episode.render_id, random_rep)
    conditions = {}
    for condition in CONDITIONS:
        if condition == "native":
            output = native
        else:
            output = model(ids, interventions={
                "edge_gate_logits": assignments[condition]}, capture=True)
        attention0 = output.cache["read.0.attention"][0]
        attention1 = output.cache["read.1.attention"][0]
        false_mask = torch.arange(mask.shape[1], device=mask.device).remainder(
            2).eq(1) & mask[0]
        logits = output.logits[0].float().cpu()
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Nonfinite diagnostic logits")
        conditions[condition] = {
            "prediction": _ordinary_prediction(output.logits),
            "logits": logits.tolist(),
            "first_target_attention": float(attention0[first].item()),
            "second_target_attention": float(attention1[second].item()),
            "first_false_attention": float(attention0[false_mask].sum().item()),
            "second_false_attention": float(attention1[false_mask].sum().item()),
        }
    identity = model(ids, interventions={"edge_gate_logits": assignments["native"]})
    identity_error = float((identity.logits - native.logits).abs().max().item())
    if not math.isfinite(identity_error):
        raise ValueError("Nonfinite diagnostic identity control")
    return {"render_id": episode.render_id, "answer": episode.answer,
            "row_count": len(rows), "candidate_count": count,
            "first_target_candidate_index": first,
            "second_target_candidate_index": second,
            "random_rep": random_rep,
            "random_selected_indices": torch.nonzero(
                assignments["count_random"][0, :count] > 0
            ).flatten().tolist(),
            "native_gate_logits": gates[0, :count].float().cpu().tolist(),
            "identity_max_abs_logit_error": identity_error,
            "conditions": conditions}


@torch.no_grad()
def evaluate_batch(model: CandidateEdgeWorkspace, episodes: list[Episode],
                   *, device: str, random_rep: int = 0) -> list[dict]:
    """Evaluate all five frozen gate conditions with one forward per batch.

    This includes copy/direct/long-hop panels. A missing read has a null
    attention value, rather than a fabricated zero-attention observation.
    Full logits are returned to the caller, which may archive a frozen sample.
    """
    if not isinstance(model, CandidateEdgeWorkspace) or model.oracle_rows:
        raise ValueError("Raw candidate-edge workspace required")
    if not episodes or type(random_rep) is not int or random_rep < 0:
        raise ValueError("Nonempty batch and nonnegative random replicate required")
    paths = [_path_indices(episode) for episode in episodes]
    rows = [strip_pair_rows(episode.tokens) for episode in episodes]
    ids = padded_tokens(episodes, device)
    model.eval()
    native = model(ids, capture=True)
    gates = native.auxiliary["edge_gate_logits"]
    masks = native.auxiliary["candidate_mask"]
    left = native.cache["candidate_left_ids"]
    right = native.cache["candidate_right_ids"]
    assignments = {name: [] for name in CONDITIONS}
    random_indices = []
    for index, episode in enumerate(episodes):
        count = int(masks[index].sum().item())
        if count != 2 * len(rows[index]) - 1 or tuple(
                (int(left[index, 2 * row]), int(right[index, 2 * row]))
                for row in range(len(rows[index]))) != rows[index]:
            raise ValueError("Candidate occurrences differ from visible rows")
        choices = _gate_conditions(
            gates[index:index + 1], masks[index:index + 1],
            episode.render_id, random_rep)
        for name in CONDITIONS:
            assignments[name].append(choices[name])
        random_indices.append(torch.nonzero(
            choices["count_random"][0, :count] > 0
        ).flatten().tolist())
    replacements = {name: torch.cat(parts, dim=0)
                    for name, parts in assignments.items()}
    identity = model(ids, interventions={"edge_gate_logits":
                                    replacements["native"]})
    identity_errors = (identity.logits - native.logits).abs().amax(dim=1)
    if not bool(torch.isfinite(identity_errors).all().item()):
        raise ValueError("Nonfinite diagnostic identity control")

    observations = [
        {"render_id": episode.render_id, "answer": episode.answer,
         "row_count": len(rows[index]),
         "candidate_count": int(masks[index].sum().item()),
         "path_candidate_indices": list(paths[index]),
         "native_gate_logits": gates[index, masks[index]].float().cpu().tolist(),
         "gate_assignment_logits": {
             name: replacements[name][index, masks[index]].float().cpu().tolist()
             for name in CONDITIONS},
         "random_rep": random_rep,
         "random_selected_indices": random_indices[index],
         "identity_max_abs_logit_error": float(identity_errors[index].item()),
         "conditions": {}}
        for index, episode in enumerate(episodes)
    ]
    for name in CONDITIONS:
        output = (native if name == "native" else model(
            ids, interventions={"edge_gate_logits": replacements[name]},
            capture=True))
        logits = output.logits.detach().float().cpu()
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Nonfinite diagnostic logits")
        for index, episode in enumerate(episodes):
            mask = masks[index]
            false_mask = (torch.arange(mask.numel(), device=mask.device)
                          .remainder(2).eq(1) & mask)
            target_attention = []
            false_attention = []
            target_rank = []
            for hop, target in enumerate(paths[index]):
                attention = output.cache[f"read.{hop}.attention"][index]
                target_attention.append(float(attention[target].item()))
                false_attention.append(float(attention[false_mask].sum().item()))
                target_rank.append(int((attention[mask] > attention[target]).sum(
                ).item()) + 1)
            observations[index]["conditions"][name] = {
                "prediction": _ordinary_prediction(output.logits[index:index + 1]),
                "logits": logits[index].tolist(),
                "target_attention": target_attention,
                "false_attention": false_attention,
                "target_rank": target_rank,
                "first_value_norm": (float(output.cache["read.0.value"][index]
                                           .norm().item()) if episode.hops else None),
                "first_state_norm": (float(output.cache["query.1"][index]
                                           .norm().item()) if episode.hops else None),
            }
    return observations
