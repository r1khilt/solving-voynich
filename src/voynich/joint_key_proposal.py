"""Conditional whole-key proposals with explicit autoregressive row uncertainty.

Cipher records reset encoder positions and share the same contextual encoder.
Each decoder choice conditions on earlier complete row assignments. No
injectivity, source-length target, plaintext segmentation or key-count aid is
imposed. This network proposes keys; its probability is not a fitting weight.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def unit_pool(glyph_count):
    if type(glyph_count) is not int or glyph_count < 1:
        raise ValueError("Positive declared glyph count required")
    return tuple((i,) for i in range(glyph_count))+tuple(itertools.product(range(glyph_count), repeat=2))


@dataclass(frozen=True)
class CanonicalGlyphs:
    records: tuple
    observed: tuple
    unseen: tuple

    @property
    def symbols(self):
        return self.observed+self.unseen


def canonicalize_records(records, glyph_alphabet):
    """Observed-symbol first occurrence; retain declared unseen uncertainty."""
    records, alphabet = tuple(records), tuple(glyph_alphabet)
    if (not records or any(not isinstance(r, str) or not r for r in records)
            or not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)
            or any(set(r)-set(alphabet) for r in records)):
        raise ValueError("Nonempty records within a unique declared glyph alphabet required")
    seen = tuple(dict.fromkeys("".join(records)))
    unseen = tuple(c for c in alphabet if c not in seen)
    table = {c: i for i, c in enumerate(seen)}
    return CanonicalGlyphs(tuple(tuple(table[c] for c in r) for r in records), seen, unseen)


def canonical_key_indices(units, canonical):
    pool = unit_pool(len(canonical.symbols))
    table = {c: i for i, c in enumerate(canonical.symbols)}
    units = tuple(units)
    if not units or any(not isinstance(u, str) or not 1 <= len(u) <= 2 or set(u)-set(table) for u in units):
        raise ValueError("Whole key must use declared one/two-glyph units")
    return tuple(pool.index(tuple(table[c] for c in u)) for u in units)


def rebind_key_orbit(key_indices, canonical, *, max_permutations=720):
    """Set-valued residual equivariance; never assign multiplicity as fit prior."""
    pool = unit_pool(len(canonical.symbols))
    key_indices = tuple(key_indices)
    if (not key_indices or any(type(i) is not int or not 0 <= i < len(pool) for i in key_indices)
            or type(max_permutations) is not int or max_permutations < 1):
        raise ValueError("Legal whole-key indices and positive orbit budget required")
    if math.factorial(len(canonical.unseen)) > max_permutations:
        raise RuntimeError("Residual glyph-permutation budget exceeded; no partial orbit returned")
    keys = set()
    for permutation in itertools.permutations(canonical.unseen):
        symbols = canonical.observed+permutation
        keys.add(tuple("".join(symbols[c] for c in pool[i]) for i in key_indices))
    return tuple(sorted(keys))


@dataclass(frozen=True)
class KeyProposalConfig:
    source_rows: int = 23
    glyphs: int = 6
    width: int = 768
    heads: int = 12
    encoder_layers: int = 8
    decoder_layers: int = 4
    ff_multiplier: int = 4
    max_record_glyphs: int = 448
    max_records: int = 4

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in asdict(self).values())
                or self.width % self.heads):
            raise ValueError("Positive model dimensions and head-divisible width required")


class JointKeyProposal(nn.Module):
    def __init__(self, config=KeyProposalConfig()):
        super().__init__()
        self.config = asdict(config)
        self.rows, self.glyphs, self.units = config.source_rows, config.glyphs, len(unit_pool(config.glyphs))
        self.glyph_embedding = nn.Embedding(self.glyphs+1, config.width, padding_idx=self.glyphs)
        self.cipher_position = nn.Embedding(config.max_record_glyphs, config.width)
        # Instantiate every layer independently. A cloned template would start
        # all blocks with identical parameters and obscure the intended scale.
        self.encoder = nn.ModuleList(nn.TransformerEncoderLayer(config.width, config.heads,
            config.width*config.ff_multiplier, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(config.encoder_layers))
        self.encoder_norm = nn.LayerNorm(config.width)
        self.query_row = nn.Embedding(self.rows, config.width)
        self.previous_row = nn.Embedding(self.rows+1, config.width)
        self.previous_unit = nn.Embedding(self.units+1, config.width)
        self.choice_position = nn.Embedding(self.rows, config.width)
        self.decoder = nn.ModuleList(nn.TransformerDecoderLayer(config.width, config.heads,
            config.width*config.ff_multiplier, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(config.decoder_layers))
        self.decoder_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, self.units)

    def _indices(self, value, shape, upper, name):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or tuple(value.shape) != shape
                or value.device != next(self.parameters()).device
                or torch.any(value < 0).item() or torch.any(value >= upper).item()):
            raise ValueError("Invalid "+name+" inventory/device/indices")

    def encode_records(self, records):
        if not isinstance(records, torch.Tensor) or records.ndim != 3:
            raise ValueError("Cipher records need [batch, records, glyphs] dimensions")
        batch, count, length = records.shape
        if (batch < 1 or not 1 <= count <= self.config["max_records"]
                or not 1 <= length <= self.config["max_record_glyphs"]):
            raise ValueError("Fixed cipher record budget exceeded; no truncation")
        self._indices(records, (batch, count, length), self.glyphs+1, "cipher glyph")
        padding = records == self.glyphs
        if (padding.all(dim=-1).any().item()
                or ((padding.cumsum(dim=-1) > 0) & ~padding).any().item()):
            raise ValueError("Every record must be nonempty with suffix-only padding")
        values = self.glyph_embedding(records.reshape(batch*count, length))
        values = values+self.cipher_position(torch.arange(length, device=records.device))[None]
        mask = padding.reshape(batch*count, length)
        for layer in self.encoder:
            values = layer(values, src_key_padding_mask=mask)
        values = self.encoder_norm(values)
        return values.reshape(batch, count*length, -1), padding.reshape(batch, count*length)

    def row_order(self, batch, device, order=None):
        if order is None:
            return torch.arange(self.rows, device=device)[None].expand(batch, -1)
        self._indices(order, (batch, self.rows), self.rows, "source row order")
        if not torch.equal(order.sort(dim=-1).values, torch.arange(self.rows, device=device)[None].expand(batch, -1)):
            raise ValueError("Row order must visit every source row exactly once")
        return order

    def decode_partial(self, memory, padding, previous_units, order=None):
        """Return choices including next row; no current/future label is input."""
        if (not isinstance(memory, torch.Tensor) or memory.ndim != 3 or memory.shape[0] < 1
                or not 1 <= memory.shape[1] <= self.config["max_records"]*self.config["max_record_glyphs"]
                or memory.dtype != next(self.parameters()).dtype
                or memory.shape[2] != self.config["width"] or memory.device != next(self.parameters()).device
                or not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool
                or tuple(padding.shape) != tuple(memory.shape[:2]) or padding.device != memory.device
                or padding.all(dim=-1).any().item()):
            raise ValueError("Valid encoder memory and support mask required")
        batch = memory.shape[0]
        if not isinstance(previous_units, torch.Tensor) or previous_units.ndim != 2:
            raise ValueError("Partial assignments need [batch, previous choices] dimensions")
        steps = previous_units.shape[1]
        if not 0 <= steps < self.rows:
            raise ValueError("Partial whole-key assignment budget exceeded")
        self._indices(previous_units, (batch, steps), self.units, "previous unit")
        order = self.row_order(batch, memory.device, order)
        unit_inputs = torch.cat((torch.full((batch, 1), self.units, device=memory.device, dtype=torch.long), previous_units), dim=1)
        row_inputs = torch.cat((torch.full((batch, 1), self.rows, device=memory.device, dtype=torch.long), order[:, :steps]), dim=1)
        positions = torch.arange(steps+1, device=memory.device)
        values = (self.query_row(order[:, :steps+1])+self.previous_row(row_inputs)
                  +self.previous_unit(unit_inputs)+self.choice_position(positions)[None])
        causal = torch.ones((steps+1, steps+1), dtype=torch.bool, device=memory.device).triu(1)
        for layer in self.decoder:
            values = layer(values, memory, tgt_mask=causal, memory_key_padding_mask=padding)
        return self.output(self.decoder_norm(values))

    def forward(self, records, keys, order=None):
        memory, padding = self.encode_records(records)
        self._indices(keys, (records.shape[0], self.rows), self.units, "complete target key")
        order = self.row_order(records.shape[0], records.device, order)
        return self.decode_partial(memory, padding, keys.gather(1, order)[:, :-1], order)

    def joint_log_probability(self, memory, padding, keys, order=None):
        self._indices(keys, (memory.shape[0], self.rows), self.units, "complete key")
        order = self.row_order(memory.shape[0], memory.device, order)
        targets = keys.gather(1, order)
        logits = self.decode_partial(memory, padding, targets[:, :-1], order)
        return F.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1).sum(dim=-1)

    @torch.inference_mode()
    def propose(self, memory, padding, *, samples=1, greedy=False, temperature=1., generator=None,
                order=None, after_step=None):
        if (self.training or type(samples) is not int or samples < 1
                or type(greedy) is not bool or (greedy and samples != 1)
                or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Eval mode and finite positive proposal configuration required")
        batch = memory.shape[0]
        order = self.row_order(batch, memory.device, order).repeat_interleave(samples, dim=0)
        memory, padding = memory.repeat_interleave(samples, dim=0), padding.repeat_interleave(samples, dim=0)
        partial = torch.empty((batch*samples, 0), dtype=torch.long, device=memory.device)
        logq = torch.zeros(batch*samples, dtype=torch.float64)
        for step in range(self.rows):
            logits = self.decode_partial(memory, padding, partial, order)[:, -1].cpu().double()/temperature
            logp = F.log_softmax(logits, dim=-1)
            chosen = logits.argmax(dim=-1) if greedy else torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            logq += logp.gather(1, chosen[:, None]).squeeze(-1)
            partial = torch.cat((partial, chosen.to(memory.device)[:, None]), dim=1)
            if after_step is not None:
                after_step({"choice": step+1, "keys": batch*samples})
        keys = torch.empty_like(partial).scatter(1, order, partial)
        return keys.reshape(batch, samples, self.rows), logq.reshape(batch, samples)
