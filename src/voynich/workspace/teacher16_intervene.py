"""Cross-order full-state interchange primitives for the conditional TEACH-0016 assay.

No trained checkpoint is loaded and no mechanistic decision is made here.
Every source/recipient G pair is evaluated, including the diagonal controls.
"""

import hashlib

import torch

from .teacher14_models import CandidateEdgeWorkspace
from .teacher14_objectives import padded_tokens
from .teacher14_tasks import SYMBOL_START, Episode
from .teacher15_tasks import TransferGroup


VECTOR_NAMES = (
    "base_native", "target_native", "source_native", "same_key_native",
    "wrong_source_native", "distinct_source_native", "random_replacement",
    "wrong_replacement", "distinct_replacement", "source_final_native",
)


def _cell(group: TransferGroup, f: int, g: int, d: int,
          marked: bool, order: int) -> Episode:
    found = [cell.episode for cell in group.cells if (
        cell.f, cell.g, cell.distractor, cell.marked, cell.order, cell.task
    ) == (f, g, d, marked, order, "composed")]
    if len(found) != 1 or found[0].hops != 2:
        raise ValueError("Cross-order composed cell absent or repeated")
    return found[0]


def _physical_slot(episode: Episode, key: int, value: int) -> int:
    hits = [index for index, row in enumerate(episode.serialized_rows)
            if tuple(row) == (key, value)]
    if len(hits) != 1:
        raise ValueError("Cross-order target G row absent or repeated")
    return hits[0]


def _random_seed(group_id: str, a: int, b: int, d: int,
                 marked: bool, order: int) -> int:
    message = f"TEACH-0016-random|{group_id}|{a}|{b}|{d}|{int(marked)}|{order}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],
                          "big") % (2**63 - 1)


def _predict(logits: torch.Tensor, index: int) -> int:
    return int(logits[index, SYMBOL_START:].argmax().item()) + SYMBOL_START


@torch.no_grad()
def evaluate_cross_surface(
        model: CandidateEdgeWorkspace, group: TransferGroup,
        distinct_group: TransferGroup, *, distractor: int, marked: bool,
        source_order: int, device: str, full_logits: bool = False
        ) -> list[dict]:
    """Capture one crossed-order surface and batch all nine recipient attempts.

    The distinct-group donor is a split-wide derangement selected outside this
    function. Matched controls use the true source-to-recipient delta norm.
    """
    if not isinstance(model, CandidateEdgeWorkspace) or model.oracle_rows:
        raise ValueError("TEACH-0016 requires a raw candidate-edge model")
    if (distractor not in (0, 1) or type(marked) is not bool or
            source_order not in (0, 1) or group.group_id == distinct_group.group_id
            or group.key1 == distinct_group.key1):
        raise ValueError("Invalid TEACH-0016 surface/control")
    recipient_order = 1 - source_order
    source = [_cell(group, 1, g, distractor, marked, source_order)
              for g in range(3)]
    wrong = [_cell(group, 0, g, distractor, marked, source_order)
             for g in range(3)]
    distinct = [_cell(distinct_group, 1, g, distractor, marked, source_order)
                for g in range(3)]
    base = [_cell(group, 0, g, distractor, marked, recipient_order)
            for g in range(3)]
    target = [_cell(group, 1, g, distractor, marked, recipient_order)
              for g in range(3)]
    episodes = source + wrong + distinct + base + target
    model.eval()
    clean = model(padded_tokens(episodes, device), capture=True)
    first = clean.cache["query.1"].detach()
    final = clean.cache["query.2"].detach()
    source_state, wrong_state = first[:3], first[3:6]
    distinct_state, base_state, target_state = (
        first[6:9], first[9:12], first[12:15])
    indices = [(a, b) for a in range(3) for b in range(3)]
    recipients = [base[b] for _, b in indices]
    targets = [target[b] for _, b in indices]
    base_vectors = torch.stack([base_state[b] for _, b in indices])
    target_vectors = torch.stack([target_state[b] for _, b in indices])
    donor_vectors = torch.stack([source_state[a] for a, _ in indices])
    same_key_vectors = torch.stack([source_state[b] for _, b in indices])
    wrong_source_vectors = torch.stack([wrong_state[a] for a, _ in indices])
    distinct_source_vectors = torch.stack([
        distinct_state[a] for a, _ in indices])
    final_vectors = torch.stack([final[a] for a, _ in indices])
    delta_norm = (donor_vectors - base_vectors).norm(dim=1, keepdim=True)

    def matched(source_vectors: torch.Tensor) -> torch.Tensor:
        raw = source_vectors - base_vectors
        return base_vectors + raw * delta_norm / raw.norm(
            dim=1, keepdim=True).clamp_min(1e-12)

    wrong_replacement = matched(wrong_source_vectors)
    distinct_replacement = matched(distinct_source_vectors)
    random_rows = []
    for a, b in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_random_seed(group.group_id, a, b, distractor,
                                           marked, source_order))
        sample = torch.randn((1, first.shape[1]), generator=generator).to(
            device=device, dtype=first.dtype)
        offset = a * 3 + b
        sample *= delta_norm[offset:offset + 1] / sample.norm().clamp_min(1e-12)
        random_rows.append(base_vectors[offset:offset + 1] + sample)
    random_replacement = torch.cat(random_rows)

    def apply(cells: list[Episode], site: str,
              states: torch.Tensor) -> torch.Tensor:
        return model(padded_tokens(cells, device),
                     interventions={site: states}).logits.detach()

    logits = {
        "identity": apply(recipients, "query.1", base_vectors),
        "transfer": apply(recipients, "query.1", donor_vectors),
        "same_key": apply(targets, "query.1", same_key_vectors),
        "wrong_key": apply(recipients, "query.1", wrong_replacement),
        "distinct": apply(recipients, "query.1", distinct_replacement),
        "random": apply(recipients, "query.1", random_replacement),
        "reverse": apply(targets, "query.1", wrong_source_vectors),
        "final_donor": apply(recipients, "query.2", final_vectors),
    }
    vector_map = {
        "base_native": base_vectors, "target_native": target_vectors,
        "source_native": donor_vectors, "same_key_native": same_key_vectors,
        "wrong_source_native": wrong_source_vectors,
        "distinct_source_native": distinct_source_vectors,
        "random_replacement": random_replacement,
        "wrong_replacement": wrong_replacement,
        "distinct_replacement": distinct_replacement,
        "source_final_native": final_vectors,
    }
    rows = []
    for index, (a, b) in enumerate(indices):
        source_slot = _physical_slot(
            source[a], group.key1, group.recipient_outputs[a][1])
        recipient_slot = _physical_slot(
            base[b], group.key1, group.recipient_outputs[b][1])
        row = {
            "group_id": group.group_id, "source_g": a, "recipient_g": b,
            "distractor": distractor, "marked": marked,
            "source_order": source_order, "recipient_order": recipient_order,
            "source_slot": source_slot, "recipient_slot": recipient_slot,
            "shifted_slot": source_slot != recipient_slot,
            "base_render_id": base[b].render_id,
            "target_render_id": target[b].render_id,
            "source_render_id": source[a].render_id,
            "same_key_render_id": source[b].render_id,
            "wrong_render_id": wrong[a].render_id,
            "distinct_render_id": distinct[a].render_id,
            "base_answer": base[b].answer,
            "target_answer": target[b].answer,
            "source_answer": source[a].answer,
            "base_prediction": _predict(clean.logits, 9 + b),
            "target_prediction": _predict(clean.logits, 12 + b),
            "source_prediction": _predict(clean.logits, a),
            "identity_prediction": _predict(logits["identity"], index),
            "transfer_prediction": _predict(logits["transfer"], index),
            "same_key_prediction": _predict(logits["same_key"], index),
            "wrong_key_prediction": _predict(logits["wrong_key"], index),
            "distinct_prediction": _predict(logits["distinct"], index),
            "random_prediction": _predict(logits["random"], index),
            "reverse_prediction": _predict(logits["reverse"], index),
            "final_donor_prediction": _predict(logits["final_donor"], index),
            "identity_max_abs_logit_error": float((
                logits["identity"][index] - clean.logits[9 + b]).abs().max()),
            "final_donor_max_abs_logit_error": float((
                logits["final_donor"][index] - clean.logits[a]).abs().max()),
            "source_delta_norm": float(delta_norm[index]),
            "replacement_vectors": {
                name: value[index].float().cpu().tolist()
                for name, value in vector_map.items()},
        }
        if full_logits:
            row.update({
                "base_logits": clean.logits[9 + b].float().cpu().tolist(),
                "target_logits": clean.logits[12 + b].float().cpu().tolist(),
                "source_logits": clean.logits[a].float().cpu().tolist(),
                **{f"{name}_logits": value[index].float().cpu().tolist()
                   for name, value in logits.items()},
            })
        rows.append(row)
    return rows
