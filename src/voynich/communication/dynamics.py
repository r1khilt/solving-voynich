"""A trainable, permutation-equivariant action world model for the synthetic lab.

The simulator is an independent label/execution oracle, not this network's
forward pass. This small relational model is an implementation of a research
hypothesis; it is not pretrained, a historical world model, or a decipherment.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .worlds import Action, Entity, EntityState, PROCEDURE_OPERATORS, execute_action, state_projection


@dataclass(frozen=True)
class DynamicsConfig:
    entities: int = 4
    width: int = 64
    layers: int = 2
    heads: int = 4

    def __post_init__(self) -> None:
        for name in ("entities", "width", "layers", "heads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.entities < 3 or self.width < 4 or self.width % self.heads:
            raise ValueError("need >=3 entities, width>=4, and width divisible by heads")


class ActionWorldModel(nn.Module):
    """Predict next entity state and action applicability from current state only.

    Entity IDs are pointers used to bind subject/object roles. There is no entity
    ID or absolute-position embedding, so renaming entities permutes predictions.
    State codes: 0 empty, 1 cold material, 2 hot material. Kind codes: 0 agent,
    1 vessel. Action: [operator, subject_index, object_index].

    ``hidden_patch`` replaces the final node representation before BOTH heads.
    It is either a full [B,N,width] tensor or a mapping with ``values`` of that
    shape and a boolean [B,N] ``mask``. Return_hidden yields the representation
    after this patch. This hook permits controlled interventions, not a claim
    that a hidden coordinate has an identified causal meaning.
    """

    def __init__(self, config: DynamicsConfig = DynamicsConfig()):
        super().__init__()
        self.config = config
        self.state_embedding = nn.Embedding(3, config.width)
        self.kind_embedding = nn.Embedding(2, config.width)
        self.operator_embedding = nn.Embedding(3, config.width)
        self.role_embedding = nn.Embedding(4, config.width)  # neither, subject, object, both
        layer = nn.TransformerEncoderLayer(
            d_model=config.width, nhead=config.heads, dim_feedforward=config.width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.width)
        self.state_head = nn.Linear(config.width, 3)
        self.validity_head = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(),
                                           nn.Linear(config.width, 1))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        kinds: torch.Tensor,
        *,
        return_hidden: bool = False,
        hidden_patch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ):
        self._validate_inputs(state, action, kinds)
        indices = torch.arange(self.config.entities, device=state.device).unsqueeze(0)
        roles = (indices == action[:, 1:2]).long() + 2 * (indices == action[:, 2:3]).long()
        hidden = (self.state_embedding(state) + self.kind_embedding(kinds) + self.role_embedding(roles) +
                  self.operator_embedding(action[:, 0]).unsqueeze(1)) * 0.5
        hidden = self.encoder(hidden)
        hidden = self.output_norm(hidden)
        if hidden_patch is not None:
            if isinstance(hidden_patch, torch.Tensor):
                values = hidden_patch
                mask = torch.ones(state.shape, dtype=torch.bool, device=state.device)
            elif isinstance(hidden_patch, Mapping) and set(hidden_patch) == {"values", "mask"}:
                values, mask = hidden_patch["values"], hidden_patch["mask"]
            else:
                raise ValueError("hidden_patch must be a tensor or a values/mask mapping")
            if (not isinstance(values, torch.Tensor) or values.shape != hidden.shape or
                    values.device != hidden.device or values.dtype != hidden.dtype or
                    not bool(torch.isfinite(values).all())):
                raise ValueError("patch values must be finite and match hidden shape, dtype and device")
            if (not isinstance(mask, torch.Tensor) or mask.shape != state.shape or
                    mask.dtype != torch.bool or mask.device != state.device):
                raise ValueError("patch mask must be boolean [batch, entities] on the input device")
            hidden = torch.where(mask.unsqueeze(-1), values, hidden)
        next_state_logits = self.state_head(hidden)
        validity_logit = self.validity_head(hidden.mean(dim=1)).squeeze(-1)
        if return_hidden:
            return next_state_logits, validity_logit, hidden
        return next_state_logits, validity_logit

    def _validate_inputs(self, state, action, kinds) -> None:
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.long for value in (state, action, kinds)):
            raise ValueError("state, action and kinds must be Long tensors")
        if state.ndim != 2 or state.shape[1] != self.config.entities or state.shape[0] < 1:
            raise ValueError("state must be [nonempty batch, configured entities]")
        if kinds.shape != state.shape or action.shape != (state.shape[0], 3):
            raise ValueError("kinds/actions have incompatible shapes")
        if any(value.device != state.device for value in (action, kinds)):
            raise ValueError("inputs must share a device")
        if (bool(((state < 0) | (state > 2)).any()) or bool(((kinds < 0) | (kinds > 1)).any()) or
                bool(((action[:, 0] < 0) | (action[:, 0] > 2)).any()) or
                bool(((action[:, 1:] < 0) | (action[:, 1:] >= self.config.entities)).any())):
            raise ValueError("state, action or kind value is outside its declared domain")


def transition_batch(seed: int, count: int, entities: int = 4) -> dict[str, torch.Tensor]:
    """Balanced executor-labeled valid/invalid transitions, with randomized identities.

    Exactly floor(count/2) rows are valid. Every state has one agent and one
    material unit in a vessel, at either temperature. The agent/holder identities
    vary independently. Invalid actions leave next_states identical to states by
    a STORAGE convention; that target must be masked out of the dynamics loss.
    No learned predictions enter labeling and no global RNG state is changed.
    """
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
        raise ValueError("seed/count must be nonnegative/positive integers")
    if type(entities) is not int or entities < 3:
        raise ValueError("need at least 3 entities")
    rng = random.Random(seed)
    rows = []
    requested_valid = [True] * (count // 2) + [False] * (count - count // 2)
    rng.shuffle(requested_valid)
    for want_valid in requested_valid:
        agent = rng.randrange(entities)
        holder = rng.choice([i for i in range(entities) if i != agent])
        hot = rng.choice((False, True))
        table = tuple(Entity(i, "agent" if i == agent else "vessel") for i in range(entities))
        state = tuple(EntityState(i, i == holder, hot and i == holder) for i in range(entities))
        candidates = []
        for operator, name in enumerate(PROCEDURE_OPERATORS):
            for subject in range(entities):
                for obj in range(entities):
                    action = Action(name, subject, obj)
                    try:
                        after = execute_action(table, state, action)
                        valid = True
                    except ValueError:
                        after, valid = state, False
                    if valid == want_valid:
                        candidates.append(((operator, subject, obj), after))
        action, after = rng.choice(candidates)
        rows.append((state_projection(state), action, tuple(int(x.kind == "vessel") for x in table),
                     state_projection(after), want_valid))
    return {
        "states": torch.tensor([row[0] for row in rows], dtype=torch.long),
        "actions": torch.tensor([row[1] for row in rows], dtype=torch.long),
        "kinds": torch.tensor([row[2] for row in rows], dtype=torch.long),
        "next_states": torch.tensor([row[3] for row in rows], dtype=torch.long),
        "valid": torch.tensor([row[4] for row in rows], dtype=torch.bool),
    }


def dynamics_loss(
    next_state_logits: torch.Tensor,
    validity_logit: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Applicability BCE on all examples; state CE only where execution is valid."""
    targets, valid = batch["next_states"], batch["valid"]
    if (targets.dtype != torch.long or targets.ndim != 2 or valid.dtype != torch.bool or
            valid.shape != targets.shape[:1] or next_state_logits.shape != (*targets.shape, 3) or
            validity_logit.shape != valid.shape):
        raise ValueError("invalid dynamics loss shapes/dtypes")
    if any(x.device != next_state_logits.device for x in (targets, valid, validity_logit)):
        raise ValueError("loss tensors must share a device")
    if bool(((targets < 0) | (targets > 2)).any()):
        raise ValueError("next-state target outside state domain")
    per_node = F.cross_entropy(next_state_logits.transpose(1, 2), targets, reduction="none")
    # Invalid storage labels contribute exactly zero gradient, even in all-invalid batches.
    state_loss = (per_node * valid.unsqueeze(1)).sum() / (valid.sum() * targets.shape[1]).clamp_min(1)
    validity_loss = F.binary_cross_entropy_with_logits(validity_logit, valid.to(validity_logit.dtype))
    return {"loss": state_loss + validity_loss, "state_loss": state_loss,
            "validity_loss": validity_loss, "valid_count": valid.sum()}
