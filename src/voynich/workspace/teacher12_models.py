"""Large raw-token and parsed-memory arms for TEACH-0012."""

import math

import torch
from torch import nn

from voynich.model import ActivationContext, ModelConfig, ModelOutput, VoynichTransformer

from .teacher12_tasks import (
    ANSWER, COMPOSE, CONTEXT_LENGTH, COPY, PAD, VOCAB_SIZE,
)


WIDTH = 512
RAW_FF = 1536
RAW_HEADS = 8


def raw_config(layers: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        d_model=WIDTH,
        n_layers=layers,
        n_heads=RAW_HEADS,
        d_ff=RAW_FF,
        context_length=CONTEXT_LENGTH,
        dropout=0.0,
        qk_norm=True,
        tie_embeddings=True,
    )


class RawClassifier(nn.Module):
    """Causal token transformer; the final ANSWER token predicts the answer symbol."""

    def __init__(self, layers: int):
        super().__init__()
        self.core = VoynichTransformer(raw_config(layers))

    def forward(self, ids, *, cache_names=None, interventions=None):
        output = self.core(ids, cache_names=cache_names, interventions=interventions)
        lengths = ids.ne(PAD).sum(-1)
        if not torch.equal(ids[torch.arange(ids.shape[0], device=ids.device), lengths - 1],
                           torch.full_like(lengths, ANSWER)):
            raise ValueError("Every raw sequence must end at ANSWER before right padding")
        batch = torch.arange(ids.shape[0], device=ids.device)
        logits = output.logits[batch, lengths - 1]
        return ModelOutput(logits, output.auxiliary_logits, output.cache)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class LoopedRawClassifier(nn.Module):
    """Four shared blocks applied three times, with a learned pass embedding."""

    unique_layers = 4
    passes = 3

    def __init__(self):
        super().__init__()
        self.core = VoynichTransformer(raw_config(self.unique_layers))
        self.pass_embedding = nn.Parameter(torch.empty(self.passes, WIDTH))
        nn.init.normal_(self.pass_embedding, std=.02)
        # The core initializes residual writes for depth four. Repeating it three times
        # requires the same depth-12 residual scale used by the untied primary arm.
        factor = math.sqrt(self.unique_layers / (self.unique_layers * self.passes))
        with torch.no_grad():
            for block in self.core.blocks:
                block.attn.out.weight.mul_(factor)
                block.down.weight.mul_(factor)

    def forward(self, ids, *, cache_names=None, interventions=None):
        if ids.ndim != 2 or ids.shape[1] > CONTEXT_LENGTH:
            raise ValueError("Expected a padded B×T raw program batch")
        valid = ids.ne(PAD)
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Only contiguous right padding is supported")
        ctx = ActivationContext(cache_names, interventions)
        x = ctx("embed", self.core.embedding(ids))
        for pass_index in range(self.passes):
            x = ctx(f"passes.{pass_index}.input", x + self.pass_embedding[pass_index])
            for layer_index, block in enumerate(self.core.blocks):
                x = block(x, valid, ctx,
                          f"passes.{pass_index}.blocks.{layer_index}")
        x = ctx("final_norm", self.core.final_norm(x))
        all_logits = ctx("logits", self.core.unembedding(x))
        ctx.validate()
        lengths = valid.sum(-1)
        batch = torch.arange(ids.shape[0], device=ids.device)
        if not torch.equal(ids[batch, lengths - 1], torch.full_like(lengths, ANSWER)):
            raise ValueError("Every raw sequence must end at ANSWER")
        return ModelOutput(all_logits[batch, lengths - 1], {}, ctx.cache)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class ParsedMemory(nn.Module):
    """Oracle row-boundary upper bound; row relevance and equality remain learned."""

    def __init__(self):
        super().__init__()
        self.symbol = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.task = nn.Embedding(9, WIDTH)
        self.q1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.k1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.v1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.q2 = nn.Linear(2 * WIDTH, WIDTH, bias=False)
        self.k2 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.v2 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.norm1 = nn.RMSNorm(WIDTH)
        self.norm2 = nn.RMSNorm(WIDTH)
        self.copy_state = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(),
                                        nn.RMSNorm(WIDTH))
        self.unembed = nn.Linear(WIDTH, VOCAB_SIZE, bias=False)
        self.unembed.weight = self.symbol.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    @staticmethod
    def _read(query, left, right, mask, q_proj, k_proj, v_proj):
        scores = (q_proj(query)[:, None] * k_proj(left)).sum(-1) / math.sqrt(query.shape[-1])
        attention = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        state = (attention[:, :, None] * v_proj(right)).sum(1)
        return state, attention

    def forward(self, ids, row_left, row_right, row_mask):
        valid = ids.ne(PAD)
        lengths = valid.sum(-1)
        batch = torch.arange(ids.shape[0], device=ids.device)
        marker = ids[batch, lengths - 3]
        query_id = ids[batch, lengths - 2]
        if not torch.equal(ids[batch, lengths - 1], torch.full_like(lengths, ANSWER)):
            raise ValueError("Every parsed batch sequence must end at ANSWER")
        query = self.symbol(query_id) + self.task(marker)
        left, right = self.symbol(row_left), self.symbol(row_right)
        first, first_attention = self._read(
            query, left, right, row_mask, self.q1, self.k1, self.v1)
        first = self.norm1(first)
        second_query = self.q2(torch.cat((query, first), -1))
        second, second_attention = self._read(
            second_query, left, right, row_mask, nn.Identity(), self.k2, self.v2)
        second = self.norm2(second)
        state = torch.where((marker == COMPOSE)[:, None], second,
                            torch.where((marker == COPY)[:, None],
                                        self.copy_state(query), first))
        return ModelOutput(self.unembed(state), {}, {
            "first_attention": first_attention.detach(),
            "second_attention": second_attention.detach(),
            "first_state": first.detach(),
            "second_state": second.detach(),
        })

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


ARMS = ("parsed_memory", "raw_shallow", "raw_looped", "raw_deep", "raw_null")


def model_for_arm(arm: str) -> nn.Module:
    if arm == "parsed_memory":
        return ParsedMemory()
    if arm in ("raw_shallow", "raw_null"):
        return RawClassifier(4)
    if arm == "raw_looped":
        return LoopedRawClassifier()
    if arm == "raw_deep":
        return RawClassifier(12)
    raise ValueError(f"Unknown TEACH-0012 arm: {arm}")


def collate_rows(episodes, device):
    width = max(len(ep.rows) for ep in episodes)
    left = torch.zeros((len(episodes), width), dtype=torch.long, device=device)
    right = torch.zeros_like(left)
    mask = torch.zeros_like(left, dtype=torch.bool)
    for index, episode in enumerate(episodes):
        count = len(episode.rows)
        left[index, :count] = torch.tensor([a for a, _ in episode.rows], device=device)
        right[index, :count] = torch.tensor([b for _, b in episode.rows], device=device)
        mask[index, :count] = True
    return left, right, mask
