"""Intervention-ready candidate-edge workspace for the TEACH-0014 design phase.

Only oracle, answer-only, one-read and mean-address variants exist here. This
module is not a frozen training campaign or a successful mechanism result.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .teacher14_tasks import (
    ANSWER, COMPOSE, COPY, DIRECT, FIRST_HOP, HOP3, HOP4, PAD,
    SYMBOL_START, VOCAB_SIZE,
)


WIDTH = 512
HEADS = 8
FF_WIDTH = 1536
ENCODER_LAYERS = 6
PARSER_LAYERS = 2
MAX_CANDIDATES = 63


@dataclass
class WorkspaceOutput:
    logits: torch.Tensor
    cache: dict[str, torch.Tensor]


class BidirectionalBlock(nn.Module):
    """Full-context block with inspectable Q/K/V and attention output."""

    def __init__(self, width: int = WIDTH, heads: int = HEADS,
                 ff_width: int = FF_WIDTH):
        super().__init__()
        if width % heads:
            raise ValueError("Width must divide head count")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width)
        self.up = nn.Linear(width, ff_width, bias=False)
        self.down = nn.Linear(ff_width, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, *,
                cache: dict[str, torch.Tensor] | None = None,
                prefix: str = "") -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, self.head_width)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        heads_out = torch.matmul(attention, v)
        attention_out = self.out(heads_out.transpose(1, 2).reshape(
            batch, length, self.width))
        x = x + attention_out
        x = x + self.down(F.gelu(self.up(self.norm2(x))))
        x = x * valid[:, :, None]
        if cache is not None:
            cache[f"{prefix}.q"] = q
            cache[f"{prefix}.k"] = k
            cache[f"{prefix}.v"] = v
            cache[f"{prefix}.attention"] = attention
            cache[f"{prefix}.attention_out"] = attention_out
            cache[f"{prefix}.residual"] = x
        return x


def candidate_indices(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
    """All adjacent ordinary-symbol pairs, including wrong cross-row pairs."""
    if ids.ndim != 2:
        raise ValueError("Expected a padded batch of raw token IDs")
    batch, length = ids.shape
    valid = ids.ne(PAD)
    if not valid[:, 0].all() or bool((valid[:, 1:] & ~valid[:, :-1]).any().item()):
        raise ValueError("Only contiguous right padding is supported")
    lengths = valid.sum(dim=1)
    if bool((lengths < 4).any().item()):
        raise ValueError("Malformed episode length")
    rows: list[torch.Tensor] = []
    for item in range(batch):
        size = int(lengths[item].item())
        if int(ids[item, 0].item()) != 1 or int(ids[item, size - 1].item()) != ANSWER:
            raise ValueError("Malformed raw episode envelope")
        body = ids[item, 1:size - 3]
        position = torch.nonzero(body.ge(SYMBOL_START), as_tuple=False).flatten() + 1
        if position.numel() < 2 or position.numel() % 2:
            raise ValueError("Expected complete two-operand graph rows")
        rows.append(position)
    count = max(row.numel() - 1 for row in rows)
    if count > MAX_CANDIDATES:
        raise ValueError("Candidate count exceeds registered maximum")
    left = torch.zeros((batch, count), dtype=torch.long, device=ids.device)
    right = torch.zeros_like(left)
    mask = torch.zeros_like(left, dtype=torch.bool)
    for item, positions in enumerate(rows):
        n = positions.numel() - 1
        left[item, :n] = positions[:-1]
        right[item, :n] = positions[1:]
        mask[item, :n] = True
    return left, right, mask


def public_row_tensors(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
    """The visible strip-and-pair ceiling in physical serialization order."""
    left_pos, right_pos, candidate_mask = candidate_indices(ids)
    batch = torch.arange(ids.shape[0], device=ids.device)[:, None]
    left_pos = left_pos[:, ::2]
    right_pos = right_pos[:, ::2]
    row_mask = candidate_mask[:, ::2]
    return ids[batch, left_pos], ids[batch, right_pos], row_mask


class CandidateEdgeWorkspace(nn.Module):
    """Raw occurrence encoder, learned candidate gate, K/V memory and hop reader."""

    def __init__(self, *, oracle_rows: bool = False, one_read: bool = False,
                 mean_address: bool = False):
        super().__init__()
        self.oracle_rows = oracle_rows
        self.one_read = one_read
        self.mean_address = mean_address
        self.symbol = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.position = nn.Embedding(192, WIDTH)
        self.encoder = nn.ModuleList(
            BidirectionalBlock() for _ in range(ENCODER_LAYERS))
        self.pair_proj = nn.Linear(4 * WIDTH, WIDTH, bias=False)
        self.pair_distance = nn.Embedding(4, WIDTH)
        self.parser = nn.ModuleList(
            BidirectionalBlock() for _ in range(PARSER_LAYERS))
        self.edge_gate = nn.Linear(WIDTH, 1)
        self.key_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.value_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.query_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.task = nn.Embedding(11, WIDTH)
        self.update = nn.Sequential(
            nn.Linear(2 * WIDTH, WIDTH), nn.GELU(),
            nn.Linear(WIDTH, WIDTH))
        self.state_norm = nn.RMSNorm(WIDTH)
        self.final_norm = nn.RMSNorm(WIDTH)
        self.unembedding = nn.Linear(WIDTH, VOCAB_SIZE, bias=False)
        self.unembedding.weight = self.symbol.weight
        self._initialize()

    def _initialize(self) -> None:
        nn.init.normal_(self.symbol.weight, std=.02)
        nn.init.normal_(self.position.weight, std=.02)
        nn.init.normal_(self.task.weight, std=.02)
        nn.init.normal_(self.pair_distance.weight, std=.02)
        nn.init.constant_(self.edge_gate.bias, 0.0)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _hook(name: str, value: torch.Tensor,
              interventions: dict[str, torch.Tensor] | None,
              cache: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if interventions and name in interventions:
            replacement = interventions[name]
            if replacement.shape != value.shape or replacement.device != value.device:
                raise ValueError(f"Intervention shape/device mismatch at {name}")
            value = replacement
        if cache is not None:
            cache[name] = value
        return value

    def _raw_slots(self, ids: torch.Tensor, body_valid: torch.Tensor,
                   interventions: dict[str, torch.Tensor] | None,
                   cache: dict[str, torch.Tensor] | None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = ids.shape
        position = torch.arange(length, device=ids.device)[None, :]
        x = (self.symbol(ids) + self.position(position)) * body_valid[:, :, None]
        for index, block in enumerate(self.encoder):
            x = block(x, body_valid, cache=cache, prefix=f"encoder.{index}")
        left_pos, right_pos, candidate_mask = candidate_indices(ids)
        batch_index = torch.arange(batch, device=ids.device)[:, None]
        left_ids = ids[batch_index, left_pos]
        right_ids = ids[batch_index, right_pos]
        left_context = x[batch_index, left_pos]
        right_context = x[batch_index, right_pos]
        left_symbol = self.symbol(left_ids)
        right_symbol = self.symbol(right_ids)
        feature = self.pair_proj(torch.cat(
            (left_context, right_context, left_symbol, right_symbol), dim=-1))
        distance = (right_pos - left_pos).clamp(0, 3)
        feature = (feature + self.pair_distance(distance)) * candidate_mask[:, :, None]
        for index, block in enumerate(self.parser):
            feature = block(feature, candidate_mask, cache=cache,
                            prefix=f"parser.{index}")
        gate_logits = self.edge_gate(feature).squeeze(-1)
        gate_logits = self._hook("edge_gate_logits", gate_logits,
                                 interventions, cache)
        keys = self.key_proj(left_symbol)
        values = self.value_proj(right_symbol)
        if cache is not None:
            cache["candidate_left_positions"] = left_pos
            cache["candidate_right_positions"] = right_pos
            cache["candidate_left_ids"] = left_ids
            cache["candidate_right_ids"] = right_ids
            cache["candidate_mask"] = candidate_mask
        return keys, values, gate_logits, candidate_mask

    def _oracle_slots(self, row_left: torch.Tensor, row_right: torch.Tensor,
                      row_mask: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if row_left.shape != row_right.shape or row_left.shape != row_mask.shape:
            raise ValueError("Oracle row shapes differ")
        if row_left.dtype != torch.long or row_right.dtype != torch.long:
            raise ValueError("Oracle rows must contain symbol IDs")
        if row_mask.dtype != torch.bool or not bool(row_mask.any(dim=1).all().item()):
            raise ValueError("Oracle row mask invalid")
        return (self.key_proj(self.symbol(row_left)),
                self.value_proj(self.symbol(row_right)),
                torch.zeros_like(row_left, dtype=self.symbol.weight.dtype), row_mask)

    @staticmethod
    def read(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
             gate_logits: torch.Tensor, mask: torch.Tensor, *,
             query_proj: nn.Module, mean_address: bool = False
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mean_address:
            address_logits = torch.zeros_like(gate_logits)
        else:
            address_logits = torch.einsum(
                "bd,bnd->bn", query_proj(query), keys) / math.sqrt(query.shape[-1])
        scores = address_logits + F.logsigmoid(gate_logits)
        attention = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        result = torch.einsum("bn,bnd->bd", attention, values)
        return result, address_logits, attention

    def forward(self, ids: torch.Tensor, *,
                row_left: torch.Tensor | None = None,
                row_right: torch.Tensor | None = None,
                row_mask: torch.Tensor | None = None,
                interventions: dict[str, torch.Tensor] | None = None,
                capture: bool = False) -> WorkspaceOutput:
        if ids.ndim != 2 or ids.shape[1] > 192:
            raise ValueError("Expected B×T episodes at most 192 tokens long")
        valid = ids.ne(PAD)
        lengths = valid.sum(dim=1)
        batch_index = torch.arange(ids.shape[0], device=ids.device)
        marker = ids[batch_index, lengths - 3]
        query_ids = ids[batch_index, lengths - 2]
        if not bool((ids[batch_index, lengths - 1] == ANSWER).all().item()):
            raise ValueError("Every episode must end at ANSWER")
        hop_map = {COPY: 0, FIRST_HOP: 1, DIRECT: 1,
                   COMPOSE: 2, HOP3: 3, HOP4: 4}
        markers = marker.tolist()
        if any(item not in hop_map for item in markers):
            raise ValueError("Unknown task marker")
        hops = torch.tensor([hop_map[item] for item in markers], device=ids.device)
        cache: dict[str, torch.Tensor] | None = {} if capture else None
        if self.oracle_rows:
            if row_left is None or row_right is None or row_mask is None:
                raise ValueError("Oracle model requires visible row parser inputs")
            expected_left, expected_right, expected_mask = public_row_tensors(ids)
            if (row_left.shape != expected_mask.shape or
                    row_right.shape != expected_mask.shape or
                    row_mask.shape != expected_mask.shape or not torch.equal(
                    row_mask, expected_mask) or not torch.equal(
                    row_left[row_mask], expected_left[row_mask]) or not torch.equal(
                    row_right[row_mask], expected_right[row_mask])):
                raise ValueError("Oracle rows must equal public visible parsing in physical order")
            keys, values, gate_logits, mask = self._oracle_slots(
                row_left, row_right, row_mask)
        else:
            if any(item is not None for item in (row_left, row_right, row_mask)):
                raise ValueError("Raw model must not receive oracle rows")
            positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
            body_valid = valid & (positions < (lengths - 3)[:, None])
            keys, values, gate_logits, mask = self._raw_slots(
                ids, body_valid, interventions, cache)
        keys = self._hook("memory.keys", keys, interventions, cache)
        values = self._hook("memory.values", values, interventions, cache)
        state = self.symbol(query_ids) + self.task(marker)
        initial_query = state
        state = self._hook("query.0", state, interventions, cache)
        for index in range(4):
            active = hops > index
            if not bool(active.any().item()):
                break
            read_query = initial_query if self.one_read and index > 0 else state
            result, address_logits, attention = self.read(
                read_query, keys, values, gate_logits, mask,
                query_proj=self.query_proj, mean_address=self.mean_address)
            result = self._hook(f"read.{index}.value", result,
                                interventions, cache)
            updated = self.state_norm(
                result + self.update(torch.cat((read_query, result), dim=-1)))
            state = torch.where(active[:, None], updated, state)
            state = self._hook(f"query.{index + 1}", state,
                               interventions, cache)
            if cache is not None:
                cache[f"read.{index}.address_logits"] = address_logits
                cache[f"read.{index}.attention"] = attention
        logits = self.unembedding(self.final_norm(state))
        logits = self._hook("answer.logits", logits, interventions, cache)
        return WorkspaceOutput(logits, {} if cache is None else cache)
