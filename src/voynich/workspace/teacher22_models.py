"""Linked oracle-row reader for the independent TEACH-0022 campaign."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .teacher14_models import CandidateEdgeWorkspace, WorkspaceOutput, public_row_tensors
from .teacher14_objectives import padded_tokens
from .teacher14_tasks import Episode


class ScaledUpdate(nn.Module):
    """Learnable residual correction to the carried value embedding."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * self.base(x)


def make_reader(*, linked: bool, seed: int,
                device: str) -> CandidateEdgeWorkspace:
    torch.manual_seed(seed)
    model = CandidateEdgeWorkspace(oracle_rows=True)
    if linked:
        # A row's right-symbol embedding is carried into the next query.
        # The same map addresses that symbol when it occurs as a left operand.
        model.query_proj.weight = model.key_proj.weight
        model.value_proj = nn.Identity()
        model.update = ScaledUpdate(model.update)
    return model.to(device)


def route_loss(output: WorkspaceOutput, episodes: list[Episode],
               left: torch.Tensor, right: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Native read-address CE; targets reconstructed from visible rows."""
    if len(episodes) != left.shape[0] or left.shape != right.shape or (
            left.shape != mask.shape):
        raise ValueError("Malformed route batch")
    device = left.device
    batch = torch.arange(len(episodes), device=device)
    queries = torch.tensor([episode.query for episode in episodes],
                           dtype=torch.long, device=device)
    hops = torch.tensor([episode.hops for episode in episodes],
                        dtype=torch.long, device=device)
    if bool(((hops < 0) | (hops > 2)).any().item()):
        raise ValueError("TEACH-0022 train mixture supports 0-2 hops")
    targets = []
    value = queries
    for hop in range(2):
        active = hops > hop
        matching = left.eq(value[:, None]) & mask
        if bool((matching.sum(dim=1)[active] != 1).any().item()):
            raise ValueError("Visible target row is missing or ambiguous")
        target = matching.long().argmax(dim=1)
        targets.append(target)
        value = torch.where(active, right[batch, target], value)
    answers = torch.tensor([episode.answer for episode in episodes],
                           dtype=torch.long, device=device)
    if not torch.equal(value, answers):
        raise ValueError("Visible paths disagree with training answers")
    losses = []
    for hop in range(2):
        active = hops > hop
        if bool(active.any().item()):
            logits = output.cache[f"read.{hop}.address_logits"][active]
            candidate_mask = mask[active]
            losses.append(F.cross_entropy(
                logits.masked_fill(~candidate_mask, -torch.inf),
                targets[hop][active], reduction="sum"))
    count = hops.sum()
    if not losses or int(count.item()) == 0:
        raise ValueError("Route loss needs at least one active read")
    return sum(losses) / count


def reader_output(model: CandidateEdgeWorkspace,
                  episodes: list[Episode], device: str,
                  *, capture: bool = False
                  ) -> tuple[WorkspaceOutput, torch.Tensor, torch.Tensor,
                             torch.Tensor]:
    ids = padded_tokens(episodes, device)
    left, right, mask = public_row_tensors(ids)
    output = model(ids, row_left=left, row_right=right, row_mask=mask,
                   capture=capture)
    return output, left, right, mask
