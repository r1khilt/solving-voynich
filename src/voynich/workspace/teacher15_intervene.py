"""Finite first-read state patch primitives for the conditional TEACH-0015 assay.

These functions do not load checkpoints or make a mechanism decision. They
operate on frozen TEACH-0014 workspace models and generated episode objects.
"""

from dataclasses import dataclass

import torch

from .teacher14_models import CandidateEdgeWorkspace
from .teacher14_objectives import padded_tokens
from .teacher14_tasks import SYMBOL_START, Episode
from .teacher15_tasks import TransferGroup


@dataclass(frozen=True)
class Captured:
    logits: torch.Tensor
    first_state: torch.Tensor
    final_state: torch.Tensor


def _predict(logits: torch.Tensor) -> int:
    return int(logits[0, SYMBOL_START:].argmax().item()) + SYMBOL_START


@torch.no_grad()
def capture(model: CandidateEdgeWorkspace, episode: Episode,
            device: str) -> Captured:
    if not isinstance(model, CandidateEdgeWorkspace) or model.oracle_rows:
        raise ValueError("TEACH-0015 requires a raw candidate-edge model")
    if episode.task != "composed" or episode.hops != 2:
        raise ValueError("Two-hop composed episode required for key state")
    model.eval()
    output = model(padded_tokens([episode], device), capture=True)
    if "query.1" not in output.cache or "query.2" not in output.cache:
        raise RuntimeError("First/second read states were not captured")
    return Captured(output.logits.detach().clone(),
                    output.cache["query.1"].detach().clone(),
                    output.cache["query.2"].detach().clone())


@torch.no_grad()
def patch(model: CandidateEdgeWorkspace, episode: Episode,
          state: torch.Tensor, *, site: str, device: str) -> torch.Tensor:
    if site not in ("query.1", "query.2"):
        raise ValueError("Unregistered TEACH-0015 intervention site")
    model.eval()
    return model(padded_tokens([episode], device),
                 interventions={site: state}).logits.detach().clone()


def recipient_vjp(model: CandidateEdgeWorkspace, episode: Episode,
                  *, target_symbol: int, base_symbol: int,
                  device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Differentiate a recipient answer contrast through only post-read work.

    A detached native `query.1` state is reinserted as a leaf. This avoids
    confusing a parameter/input derivative with the downstream state VJP.
    The result is not Anthropic's averaged Jacobian-lens construction.
    """
    if target_symbol == base_symbol or min(target_symbol, base_symbol) < SYMBOL_START:
        raise ValueError("Distinct ordinary answer symbols required")
    native = capture(model, episode, device)
    parameters = tuple(model.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            leaf = native.first_state.detach().requires_grad_(True)
            output = model(padded_tokens([episode], device),
                           interventions={"query.1": leaf})
            contrast = (output.logits[0, target_symbol] -
                        output.logits[0, base_symbol])
            gradient, = torch.autograd.grad(contrast, leaf)
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("Nonfinite recipient VJP")
        return native.first_state, gradient.detach(), float(contrast.detach().item())
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)


def _cell(group: TransferGroup, f: int, g: int, distractor: int,
          marked: bool, order: int) -> Episode:
    found = [cell.episode for cell in group.cells if (
        cell.f, cell.g, cell.distractor, cell.marked, cell.order, cell.task)
        == (f, g, distractor, marked, order, "composed")]
    if len(found) != 1:
        raise ValueError("TEACH-0015 composed cell missing or repeated")
    return found[0]


@torch.no_grad()
def evaluate_surface(model: CandidateEdgeWorkspace, group: TransferGroup,
                     *, distractor: int, marked: bool, order: int,
                     device: str,
                     wrong_donor: Episode | None = None,
                     deranged_donor: Episode | None = None) -> list[dict]:
    """One donor used unchanged against all three G recipients, both directions."""
    if distractor not in (0, 1) or type(marked) is not bool or order not in (0, 1):
        raise ValueError("Invalid TEACH-0015 surface coordinates")
    donor_episode = _cell(group, 1, 0, distractor, marked, order)
    donor = capture(model, donor_episode, device)
    nuisance_episode = _cell(group, 1, 0, distractor, not marked, order)
    nuisance = capture(model, nuisance_episode, device)
    wrong = capture(model, wrong_donor, device) if wrong_donor is not None else None
    deranged = (capture(model, deranged_donor, device)
                if deranged_donor is not None else None)
    rows = []
    for g in (0, 1, 2):
        base_episode = _cell(group, 0, g, distractor, marked, order)
        target_episode = _cell(group, 1, g, distractor, marked, order)
        base = capture(model, base_episode, device)
        target = capture(model, target_episode, device)
        identity = patch(model, base_episode, base.first_state,
                         site="query.1", device=device)
        transferred = patch(model, base_episode, donor.first_state,
                            site="query.1", device=device)
        same_key = patch(model, base_episode, nuisance.first_state,
                         site="query.1", device=device)
        reverse = patch(model, target_episode, base.first_state,
                        site="query.1", device=device)
        final_donor = patch(model, base_episode, donor.final_state,
                            site="query.2", device=device)
        delta = donor.first_state - base.first_state
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(group.group_id[:16], 16) ^
                              (g << 16) ^ (distractor << 8) ^
                              (int(marked) << 4) ^ order)
        random_delta = torch.randn(delta.shape, generator=generator).to(
            device=device, dtype=delta.dtype)
        random_delta *= delta.norm() / random_delta.norm().clamp_min(1e-12)
        random_logits = patch(
            model, base_episode, base.first_state + random_delta,
            site="query.1", device=device)
        wrong_logits = None
        wrong_replacement = None
        if wrong is not None:
            wrong_delta = wrong.first_state - base.first_state
            wrong_delta *= delta.norm() / wrong_delta.norm().clamp_min(1e-12)
            wrong_replacement = base.first_state + wrong_delta
            wrong_logits = patch(
                model, base_episode, wrong_replacement,
                site="query.1", device=device)
        deranged_logits = None
        deranged_replacement = None
        if deranged is not None:
            deranged_delta = deranged.first_state - base.first_state
            deranged_delta *= delta.norm() / deranged_delta.norm().clamp_min(1e-12)
            deranged_replacement = base.first_state + deranged_delta
            deranged_logits = patch(
                model, base_episode, deranged_replacement,
                site="query.1", device=device)
        rows.append({
            "group_id": group.group_id, "g": g, "distractor": distractor,
            "marked": marked, "order": order,
            "base_render_id": base_episode.render_id,
            "donor_render_id": donor_episode.render_id,
            "same_key_donor_render_id": nuisance_episode.render_id,
            "wrong_donor_render_id": (wrong_donor.render_id
                                      if wrong_donor is not None else None),
            "deranged_donor_render_id": (deranged_donor.render_id
                                         if deranged_donor is not None else None),
            "target_render_id": target_episode.render_id,
            "base_answer": base_episode.answer,
            "target_answer": target_episode.answer,
            "fixed_donor_answer": donor_episode.answer,
            "base_prediction": _predict(base.logits),
            "target_prediction": _predict(target.logits),
            "donor_prediction": _predict(donor.logits),
            "transfer_prediction": _predict(transferred),
            "same_key_prediction": _predict(same_key),
            "reverse_prediction": _predict(reverse),
            "final_donor_prediction": _predict(final_donor),
            "random_prediction": _predict(random_logits),
            "wrong_key_prediction": (_predict(wrong_logits)
                                     if wrong_logits is not None else None),
            "deranged_prediction": (_predict(deranged_logits)
                                    if deranged_logits is not None else None),
            "identity_max_abs_logit_error": float((
                identity - base.logits).abs().max().item()),
            "final_donor_max_abs_logit_error": float((
                final_donor - donor.logits).abs().max().item()),
            "donor_delta_norm": float(delta.norm().item()),
            "replacement_vectors": {
                "base_native": base.first_state[0].float().cpu().tolist(),
                "target_native": target.first_state[0].float().cpu().tolist(),
                "donor_native": donor.first_state[0].float().cpu().tolist(),
                "same_key_native": nuisance.first_state[0].float().cpu().tolist(),
                "reverse_base": base.first_state[0].float().cpu().tolist(),
                "final_donor": donor.final_state[0].float().cpu().tolist(),
                "random": (base.first_state + random_delta)[0]
                    .float().cpu().tolist(),
                "wrong_key": (wrong_replacement[0].float().cpu().tolist()
                              if wrong_replacement is not None else None),
                "wrong_source_native": (
                    wrong.first_state[0].float().cpu().tolist()
                    if wrong is not None else None),
                "deranged": (deranged_replacement[0].float().cpu().tolist()
                             if deranged_replacement is not None else None),
                "deranged_source_native": (
                    deranged.first_state[0].float().cpu().tolist()
                    if deranged is not None else None),
            },
            "base_logits": base.logits[0].float().cpu().tolist(),
            "target_logits": target.logits[0].float().cpu().tolist(),
            "donor_logits": donor.logits[0].float().cpu().tolist(),
            "transfer_logits": transferred[0].float().cpu().tolist(),
            "same_key_logits": same_key[0].float().cpu().tolist(),
            "reverse_logits": reverse[0].float().cpu().tolist(),
            "final_donor_logits": final_donor[0].float().cpu().tolist(),
            "random_logits": random_logits[0].float().cpu().tolist(),
            "wrong_key_logits": (wrong_logits[0].float().cpu().tolist()
                                 if wrong_logits is not None else None),
            "deranged_logits": (deranged_logits[0].float().cpu().tolist()
                                if deranged_logits is not None else None),
        })
    return rows


@torch.no_grad()
def evaluate_surface_batched(
        model: CandidateEdgeWorkspace, group: TransferGroup, *,
        distractor: int, marked: bool, order: int, device: str,
        wrong_donor: Episode | None = None,
        deranged_donor: Episode | None = None,
        full_logits: bool = True) -> list[dict]:
    """The same finite grid as ``evaluate_surface`` with grouped forwards.

    Source captures, six clean F/G cells, and each three-recipient patch are
    batched separately. Each output retains the same identities and tensors as
    the single-example reference, including actual replacement vectors.
    """
    if not isinstance(model, CandidateEdgeWorkspace) or model.oracle_rows:
        raise ValueError("TEACH-0015 requires a raw candidate-edge model")
    if distractor not in (0, 1) or type(marked) is not bool or order not in (0, 1):
        raise ValueError("Invalid TEACH-0015 surface coordinates")
    donor_episode = _cell(group, 1, 0, distractor, marked, order)
    nuisance_episode = _cell(group, 1, 0, distractor, not marked, order)
    sources = [donor_episode, nuisance_episode]
    if wrong_donor is not None:
        sources.append(wrong_donor)
    if deranged_donor is not None:
        sources.append(deranged_donor)
    if any(episode.task != "composed" or episode.hops != 2
           for episode in sources):
        raise ValueError("Two-hop composed donor episodes required")
    model.eval()
    source_out = model(padded_tokens(sources, device), capture=True)
    source_states = source_out.cache["query.1"].detach()
    source_final = source_out.cache["query.2"].detach()
    donor_state = source_states[0:1]
    nuisance_state = source_states[1:2]
    wrong_index = 2 if wrong_donor is not None else None
    deranged_index = (2 + int(wrong_donor is not None)
                      if deranged_donor is not None else None)
    bases = [_cell(group, 0, g, distractor, marked, order)
             for g in (0, 1, 2)]
    targets = [_cell(group, 1, g, distractor, marked, order)
               for g in (0, 1, 2)]
    clean = model(padded_tokens(bases + targets, device), capture=True)
    base_states = clean.cache["query.1"][:3].detach()
    target_states = clean.cache["query.1"][3:].detach()
    donor_repeat = donor_state.expand(3, -1)
    nuisance_repeat = nuisance_state.expand(3, -1)
    final_repeat = source_final[0:1].expand(3, -1)
    delta = donor_repeat - base_states

    def apply(episodes: list[Episode], site: str,
              replacement: torch.Tensor) -> torch.Tensor:
        return model(padded_tokens(episodes, device),
                     interventions={site: replacement}).logits.detach()

    identity = apply(bases, "query.1", base_states)
    transferred = apply(bases, "query.1", donor_repeat)
    same_key = apply(bases, "query.1", nuisance_repeat)
    reverse = apply(targets, "query.1", base_states)
    final_donor = apply(bases, "query.2", final_repeat)
    random_rows = []
    for g in (0, 1, 2):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(group.group_id[:16], 16) ^
                              (g << 16) ^ (distractor << 8) ^
                              (int(marked) << 4) ^ order)
        random_delta = torch.randn((1, delta.shape[1]),
                                   generator=generator).to(
                                       device=device, dtype=delta.dtype)
        random_delta *= delta[g:g + 1].norm() / random_delta.norm().clamp_min(
            1e-12)
        random_rows.append(base_states[g:g + 1] + random_delta)
    random_states = torch.cat(random_rows)
    random_logits = apply(bases, "query.1", random_states)

    def matched_source(index: int | None) -> tuple[torch.Tensor | None,
                                                   torch.Tensor | None]:
        if index is None:
            return None, None
        source = source_states[index:index + 1]
        raw_delta = source.expand(3, -1) - base_states
        sizes = raw_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        states = base_states + raw_delta * delta.norm(
            dim=1, keepdim=True) / sizes
        return source, states

    wrong_source, wrong_states = matched_source(wrong_index)
    deranged_source, deranged_states = matched_source(deranged_index)
    wrong_logits = (apply(bases, "query.1", wrong_states)
                    if wrong_states is not None else None)
    deranged_logits = (apply(bases, "query.1", deranged_states)
                       if deranged_states is not None else None)

    def prediction(logits: torch.Tensor, index: int) -> int:
        return int(logits[index, SYMBOL_START:].argmax().item()) + SYMBOL_START

    def vector(value: torch.Tensor, index: int) -> list[float]:
        return value[index].float().cpu().tolist()

    def logit(value: torch.Tensor, index: int) -> list[float] | None:
        return value[index].float().cpu().tolist() if full_logits else None

    rows = []
    for g in (0, 1, 2):
        base_logits = clean.logits[g:g + 1]
        target_logits = clean.logits[g + 3:g + 4]
        donor_logits = source_out.logits[0:1]
        identity_error = (identity[g] - clean.logits[g]).abs().max()
        final_error = (final_donor[g] - source_out.logits[0]).abs().max()
        row = {
            "group_id": group.group_id, "g": g, "distractor": distractor,
            "marked": marked, "order": order,
            "base_render_id": bases[g].render_id,
            "target_render_id": targets[g].render_id,
            "donor_render_id": donor_episode.render_id,
            "same_key_donor_render_id": nuisance_episode.render_id,
            "wrong_donor_render_id": (wrong_donor.render_id
                                      if wrong_donor is not None else None),
            "deranged_donor_render_id": (deranged_donor.render_id
                                         if deranged_donor is not None else None),
            "base_answer": bases[g].answer,
            "target_answer": targets[g].answer,
            "fixed_donor_answer": donor_episode.answer,
            "base_prediction": prediction(base_logits, 0),
            "target_prediction": prediction(target_logits, 0),
            "donor_prediction": prediction(donor_logits, 0),
            "transfer_prediction": prediction(transferred, g),
            "same_key_prediction": prediction(same_key, g),
            "reverse_prediction": prediction(reverse, g),
            "final_donor_prediction": prediction(final_donor, g),
            "random_prediction": prediction(random_logits, g),
            "wrong_key_prediction": (prediction(wrong_logits, g)
                                     if wrong_logits is not None else None),
            "deranged_prediction": (prediction(deranged_logits, g)
                                    if deranged_logits is not None else None),
            "identity_max_abs_logit_error": float(identity_error.item()),
            "final_donor_max_abs_logit_error": float(final_error.item()),
            "donor_delta_norm": float(delta[g].norm().item()),
            "replacement_vectors": {
                "base_native": vector(base_states, g),
                "target_native": vector(target_states, g),
                "donor_native": vector(donor_state, 0),
                "same_key_native": vector(nuisance_state, 0),
                "reverse_base": vector(base_states, g),
                "final_donor": vector(source_final, 0),
                "random": vector(random_states, g),
                "wrong_key": (vector(wrong_states, g)
                              if wrong_states is not None else None),
                "wrong_source_native": (vector(wrong_source, 0)
                                        if wrong_source is not None else None),
                "deranged": (vector(deranged_states, g)
                             if deranged_states is not None else None),
                "deranged_source_native": (vector(deranged_source, 0)
                                           if deranged_source is not None else None),
            },
            "base_logits": logit(clean.logits, g),
            "target_logits": logit(clean.logits, g + 3),
            "donor_logits": logit(source_out.logits, 0),
            "transfer_logits": logit(transferred, g),
            "same_key_logits": logit(same_key, g),
            "reverse_logits": logit(reverse, g),
            "final_donor_logits": logit(final_donor, g),
            "random_logits": logit(random_logits, g),
            "wrong_key_logits": (logit(wrong_logits, g)
                                 if wrong_logits is not None else None),
            "deranged_logits": (logit(deranged_logits, g)
                                if deranged_logits is not None else None),
        }
        if not full_logits:
            row = {key: value for key, value in row.items()
                   if not key.endswith("_logits")}
        rows.append(row)
    return rows
