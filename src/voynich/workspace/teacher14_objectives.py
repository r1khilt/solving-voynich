"""Prospective auxiliary objectives for the TEACH-0014 workspace."""

import torch
import torch.nn.functional as F

from .teacher14_models import CandidateEdgeWorkspace, WorkspaceOutput
from .teacher14_tasks import CausalPair, Episode, PAD, SYMBOL_START


def padded_tokens(episodes: list[Episode], device: torch.device | str) -> torch.Tensor:
    if not episodes:
        raise ValueError("At least one episode required")
    width = max(len(episode.tokens) for episode in episodes)
    rows = [episode.tokens + (PAD,) * (width - len(episode.tokens))
            for episode in episodes]
    return torch.tensor(rows, dtype=torch.long, device=device)


def answer_loss(output: WorkspaceOutput, answers: list[int] | torch.Tensor
                ) -> torch.Tensor:
    if output.logits.shape[0] != len(answers):
        raise ValueError("Answer count does not match logits")
    target = torch.as_tensor(answers, dtype=torch.long, device=output.logits.device)
    if not bool((target >= SYMBOL_START).all().item()):
        raise ValueError("Answer outside ordinary symbol range")
    return F.cross_entropy(output.logits[:, SYMBOL_START:], target - SYMBOL_START)


def deranged_interchange_targets(targets: torch.Tensor) -> torch.Tensor:
    """Preserve each recipient-cell label multiset but break group correspondence."""
    if targets.ndim != 2 or targets.shape[1] != 2 or targets.shape[0] < 2:
        raise ValueError("Expected at least two two-recipient groups")
    groups = targets.shape[0]
    for offset in range(1, groups):
        candidate = targets.roll(shifts=offset, dims=0)
        if bool((candidate != targets).all().item()):
            return candidate
    raise ValueError("No within-cell derangement without a fixed point")


def interchange_loss(model: CandidateEdgeWorkspace, pairs: list[CausalPair],
                     device: torch.device | str, *, wrong_targets: bool = False
                     ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Patch one donor first-read state into two different recipient G tables."""
    if not pairs or model.oracle_rows:
        raise ValueError("Raw workspace and at least one causal pair required")
    donors = [pair.donor for pair in pairs]
    bases = [base for pair in pairs for base in pair.bases]
    targets = torch.tensor([pair.targets for pair in pairs],
                           dtype=torch.long, device=device)
    if wrong_targets:
        targets = deranged_interchange_targets(targets)
    donor_first = model(padded_tokens(donors, device)).auxiliary["first_state"]
    repeated_first = donor_first.repeat_interleave(2, dim=0)
    base_output = model(
        padded_tokens(bases, device),
        interventions={"query.1": repeated_first})
    loss = answer_loss(base_output, targets.reshape(-1))
    return loss, {
        "targets": targets,
        "patched_logits": base_output.logits,
        "donor_first_state": donor_first,
    }
