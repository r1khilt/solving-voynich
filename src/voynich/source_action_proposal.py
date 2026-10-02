"""Reading-supervised proposals in a literal shared-dictionary environment.

This is an action-path proposal, not a whole-key posterior density. Rollouts
may die; neither successful-path conditioning nor key marginalization is
implemented. Training snapshots contain only bindings made by past actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from voynich.joint_key_proposal import unit_pool


@dataclass(frozen=True)
class ReadingState:
    key: tuple[int, ...]
    offsets: tuple[int, ...]
    texts: tuple[tuple[int, ...], ...]


class ReadingEnvironment:
    """Observation-only, non-erasing length1/2 channel; duplicate units allowed."""

    def __init__(self, records, *, rows=23, glyphs=6, max_glyphs=4096, max_records=4):
        self.records = tuple(tuple(r) for r in records)
        if (any(type(n) is not int or n < 1 for n in (rows, glyphs, max_glyphs, max_records))
                or rows > 64 or glyphs > 64 or max_glyphs > 4096 or max_records > 16
                or not 1 <= len(self.records) <= max_records
                or any(not r or len(r) > max_glyphs
                       or any(type(g) is not int or not 0 <= g < glyphs for g in r)
                       for r in self.records)):
            raise ValueError("Bounded nonempty integer cipher records required")
        self.rows, self.glyphs = rows, glyphs
        self.pool = unit_pool(glyphs)
        self.initial = ReadingState((-1,)*rows, (0,)*len(self.records), ((),)*len(self.records))

    def selected_record(self, state):
        self.validate(state)
        chosen = None
        for record, offset in enumerate(state.offsets):
            if offset == len(self.records[record]):
                continue
            if (chosen is None or offset*len(self.records[chosen])
                    < state.offsets[chosen]*len(self.records[record])):
                chosen = record
        return chosen

    def validate(self, state):
        if (not isinstance(state, ReadingState) or type(state.key) is not tuple
                or type(state.offsets) is not tuple or type(state.texts) is not tuple
                or any(type(t) is not tuple for t in state.texts) or len(state.key) != self.rows
                or len(state.offsets) != len(self.records) or len(state.texts) != len(self.records)
                or any(type(k) is not int or not -1 <= k < len(self.pool) for k in state.key)
                or any(type(o) is not int or not 0 <= o <= len(r)
                       for o, r in zip(state.offsets, self.records, strict=True))):
            raise ValueError("Invalid reading state")
        used = set()
        for text, offset, record in zip(state.texts, state.offsets, self.records, strict=True):
            if any(type(a) is not int or not 0 <= a < self.rows or state.key[a] < 0 for a in text):
                raise ValueError("Every past letter must already have a legal binding")
            encoded = tuple(g for a in text for g in self.pool[state.key[a]])
            if encoded != record[:offset]:
                raise ValueError("Past letters must literally encode the consumed prefix")
            used.update(text)
        if any((k >= 0) != (a in used) for a, k in enumerate(state.key)):
            raise ValueError("No future or unused binding may enter a prefix state")

    def legal_actions(self, state):
        record = self.selected_record(state)
        if record is None:
            return ()
        offset = state.offsets[record]
        choices = []
        for row, bound in enumerate(state.key):
            for length in (1, 2):
                unit = self.records[record][offset:offset+length]
                if len(unit) == length and (bound < 0 or self.pool[bound] == unit):
                    choices.append(2*row+length-1)
        return tuple(choices)

    def advance(self, state, action):
        if type(action) is not int or action not in self.legal_actions(state):
            raise ValueError("Action must be legal at the observed prefix")
        record = self.selected_record(state)
        row, length = action//2, action%2+1
        offset = state.offsets[record]
        unit = self.records[record][offset:offset+length]
        key, offsets, texts = list(state.key), list(state.offsets), list(state.texts)
        if key[row] < 0:
            key[row] = self.pool.index(unit)
        offsets[record] += length
        texts[record] += (row,)
        result = ReadingState(tuple(key), tuple(offsets), tuple(texts))
        self.validate(result)
        return result

    def teaching_trace(self, texts, key):
        """Gold is used only to select targets; saved states precede each target."""
        texts, key = tuple(tuple(t) for t in texts), tuple(key)
        if (len(texts) != len(self.records) or len(key) != self.rows
                or any(type(k) is not int or not 0 <= k < len(self.pool) for k in key)
                or any(not t or any(type(a) is not int or not 0 <= a < self.rows for a in t)
                       for t in texts)
                or tuple(tuple(g for a in t for g in self.pool[key[a]]) for t in texts) != self.records):
            raise ValueError("Training text/key must literally encode every whole record")
        state, snapshots, actions = self.initial, [], []
        while (record := self.selected_record(state)) is not None:
            row = texts[record][len(state.texts[record])]
            action = 2*row+len(self.pool[key[row]])-1
            snapshots.append(state)
            actions.append(action)
            state = self.advance(state, action)
        if state.texts != texts:
            raise AssertionError("Deterministic schedule must reconstruct the complete target")
        return tuple(snapshots), tuple(actions), state


@dataclass(frozen=True)
class SourceActionConfig:
    rows: int = 23
    glyphs: int = 6
    width: int = 768
    heads: int = 12
    encoder_layers: int = 8
    decoder_layers: int = 4
    ff_multiplier: int = 4
    max_glyphs: int = 448
    max_records: int = 4
    binding_input: bool = True

    def __post_init__(self):
        if (type(self.binding_input) is not bool
                or any(type(v) is not int or v < 1 for k, v in vars(self).items() if k != 'binding_input')
                or self.width % self.heads or self.rows > 64 or self.glyphs > 64
                or self.max_glyphs > 4096 or self.max_records > 16):
            raise ValueError("Bounded positive dimensions and head-divisible width required")


class SourceActionProposal(nn.Module):
    """Full observed cipher encoder + causal action decoder + explicit bindings.

    Row/unit *pair* embeddings retain assignments; summing independent row and
    unit embeddings would erase the pairing. Record IDs break permutation
    invariance deliberately; reset glyph positions and one shared key are kept.
    """

    def __init__(self, config=SourceActionConfig()):
        super().__init__()
        self.config = config
        d, units = config.width, len(unit_pool(config.glyphs))
        self.glyph = nn.Embedding(config.glyphs+1, d, padding_idx=config.glyphs)
        self.position = nn.Embedding(config.max_glyphs, d)
        self.record = nn.Embedding(config.max_records, d)
        self.binding = nn.Embedding(config.rows*(units+1), d)
        self.previous = nn.Embedding(2*config.rows+1, d)
        self.offset = nn.Embedding(config.max_glyphs+1, d)
        self.encoder = nn.ModuleList(nn.TransformerEncoderLayer(d, config.heads,
            d*config.ff_multiplier, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(config.encoder_layers))
        self.encoder_norm = nn.LayerNorm(d)
        self.decoder = nn.ModuleList(nn.TransformerDecoderLayer(d, config.heads,
            d*config.ff_multiplier, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(config.decoder_layers))
        self.decoder_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 2*config.rows)

    def pack(self, environments, traces, *, require_complete=True):
        """Build observations from verified prefix states, never full target keys."""
        if not environments or len(environments) != len(traces):
            raise ValueError("Nonempty matching observation/trace batches required")
        c = self.config
        steps = max(len(trace[1]) for trace in traces)
        count = len(environments[0].records)
        if (not steps or any(len(e.records) != count or e.rows != c.rows or e.glyphs != c.glyphs
                             or max(map(len, e.records)) > c.max_glyphs for e in environments)
                or count > c.max_records):
            raise ValueError("Consistent declared record/letter/glyph budgets required")
        target_device = next(self.parameters()).device
        # Compile the verified trace on CPU, then transfer eight tensors once.
        # Thousands of tiny MPS assignments would obscure real training cost.
        device = torch.device('cpu')
        batch, length, units = len(environments), max(len(r) for e in environments for r in e.records), len(unit_pool(c.glyphs))
        records = torch.full((batch, count, length), c.glyphs, dtype=torch.long, device=device)
        keys = torch.zeros((batch, steps, c.rows), dtype=torch.long, device=device)
        offsets = torch.zeros((batch, steps), dtype=torch.long, device=device)
        selected = torch.zeros((batch, steps), dtype=torch.long, device=device)
        previous = torch.full((batch, steps), 2*c.rows, dtype=torch.long, device=device)
        targets = torch.zeros((batch, steps), dtype=torch.long, device=device)
        active = torch.zeros((batch, steps), dtype=torch.bool, device=device)
        legal = torch.zeros((batch, steps, 2*c.rows), dtype=torch.bool, device=device)
        legal[..., 0] = True  # Dummy padded query avoids all-negative-infinity softmax.
        for i, (env, trace) in enumerate(zip(environments, traces, strict=True)):
            snapshots, actions, terminal = trace
            if len(snapshots) != len(actions) or not actions:
                raise ValueError("A complete, nonempty teaching trace is required")
            state = env.initial
            for j, (snapshot, action) in enumerate(zip(snapshots, actions, strict=True)):
                if snapshot != state:
                    raise ValueError("Trace state is not the actual preceding-action state")
                record = env.selected_record(state)
                keys[i, j] = torch.tensor([row*(units+1)+k+1 for row, k in enumerate(state.key)], device=device)
                offsets[i, j], selected[i, j] = state.offsets[record], record
                legal[i, j] = False
                legal[i, j, list(env.legal_actions(state))] = True
                targets[i, j], active[i, j] = action, True
                if j:
                    previous[i, j] = actions[j-1]
                state = env.advance(state, action)
            if state != terminal or (require_complete and env.selected_record(state) is not None):
                raise ValueError("A partial/dead trace cannot be passed as a successful target")
            for record, values in enumerate(env.records):
                records[i, record, :len(values)] = torch.tensor(values, device=device)
        return tuple(value.to(target_device) for value in
                     (records, keys, offsets, selected, previous, legal, active, targets))

    def forward(self, packed):
        records, keys, offsets, selected, previous, legal, active, _ = packed
        batch, count, length = records.shape
        padding = records == self.config.glyphs
        positions = torch.arange(length, device=records.device)
        record_ids = torch.arange(count, device=records.device)
        memory = self.glyph(records)+self.position(positions)[None, None]+self.record(record_ids)[None, :, None]
        memory = memory.reshape(batch*count, length, -1)
        for layer in self.encoder:
            memory = layer(memory, src_key_padding_mask=padding.reshape(batch*count, length))
        memory = self.encoder_norm(memory).reshape(batch, count*length, -1)
        bindings = self.binding(keys).mean(dim=-2)*int(self.config.binding_input)
        values = self.previous(previous)+self.offset(offsets)+self.record(selected)+bindings
        # Strictly exclude future actions at every decoder layer. Cipher memory
        # is observed in full; it contains no source targets or full dictionary.
        causal = torch.triu(torch.ones(values.shape[1], values.shape[1], dtype=torch.bool,
                                      device=records.device), diagonal=1)
        for layer in self.decoder:
            values = layer(values, memory, tgt_mask=causal, tgt_key_padding_mask=~active,
                           memory_key_padding_mask=padding.reshape(batch, count*length))
        return self.output(self.decoder_norm(values)).masked_fill(~legal, -torch.inf)

    def loss(self, packed):
        """Mean WHOLE-path NLL; no gold-length normalization or used-row loss."""
        logits = self(packed)
        active, targets = packed[-2:]
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        return losses.masked_fill(~active, 0.).sum(dim=1).mean()

    @torch.no_grad()
    def propose(self, environment, *, seed=0, temperature=1.):
        """One finite path, never restart/refill a dead end; unused rows stay -1.

        Reference implementation re-encodes the observation at every step and
        has no KV cache. Its cost must be measured before long-path deployment.
        Returned density is for this action path, including any failed prefix;
        it is NOT the marginal density of its terminal partial/full dictionary.
        """
        import math
        import numpy as np

        if (type(seed) is not int or seed < 0 or isinstance(temperature, bool)
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("A fixed seed and positive finite temperature are required")
        rng, state = np.random.default_rng(seed), environment.initial
        snapshots, actions, log_probability = [], [], 0.
        was_training = self.training
        self.eval()
        try:
            for _ in range(sum(map(len, environment.records))):
                if environment.selected_record(state) is None:
                    break
                legal = environment.legal_actions(state)
                if not legal:
                    return {"status": "dead_end", "state": state, "actions": tuple(actions),
                            "path_log_probability": log_probability}
                # The dummy current target never enters its own query. Only its
                # preceding snapshot/previous actions reach the network.
                dummy = legal[0]
                trace = (tuple(snapshots+[state]), tuple(actions+[dummy]), environment.advance(state, dummy))
                logits = self(self.pack([environment], [trace], require_complete=False))[0, -1]
                values = logits.detach().cpu().double().numpy()[list(legal)]/temperature
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError("Nonfinite legal-action scores; no repaired proposal")
                values -= values.max()
                probabilities = np.exp(values)
                probabilities /= probabilities.sum()
                index = int(rng.choice(len(legal), p=probabilities))
                action = legal[index]
                log_probability += math.log(float(probabilities[index]))
                snapshots.append(state)
                actions.append(action)
                state = environment.advance(state, action)
            if environment.selected_record(state) is not None:
                raise AssertionError("Every legal action must consume at least one glyph")
            return {"status": "complete_path", "state": state, "actions": tuple(actions),
                    "path_log_probability": log_probability}
        finally:
            self.train(was_training)
