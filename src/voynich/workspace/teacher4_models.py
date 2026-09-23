"""Shared parsed-row interface and learnable TEACH-0004 architecture arms."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .teacher4_tasks import COPY, FIRST_HOP


WIDTH = 128


class RowInterface(nn.Module):
    """All arms see the same typed row records, task marker and query token."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.symbol = nn.Embedding(45, width)
        self.role = nn.Embedding(6, width)
        self.f_row = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.g_row = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[1] != 14:
            raise ValueError("Expected B×14 TEACH-0004 episodes")
        role = self.role.weight
        f_keys = self.symbol(ids[:, (2, 4)]) + role[0]
        f_values = self.symbol(ids[:, (3, 5)]) + role[1]
        g_keys = self.symbol(ids[:, (7, 9)]) + role[2]
        g_values = self.symbol(ids[:, (8, 10)]) + role[3]
        f_rows = self.f_row(torch.cat((f_keys, f_values), dim=-1))
        g_rows = self.g_row(torch.cat((g_keys, g_values), dim=-1))
        marker = self.symbol(ids[:, 11]) + role[4]
        query = self.symbol(ids[:, 12]) + role[5]
        return f_rows, g_rows, marker, query


class AnswerHeads(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.key = nn.Linear(width, 45)
        self.object = nn.Linear(width, 45)
        self.copy = nn.Linear(width, 45)

    def forward(self, ids, first, second, copied):
        marker = ids[:, 11]
        return torch.where((marker == FIRST_HOP)[:, None], self.key(first),
                           torch.where((marker == COPY)[:, None], self.copy(copied),
                                       self.object(second)))


class LearnedMemory(nn.Module):
    """Two learned row reads, or a one-read composed-path ablation."""

    def __init__(self, two_read: bool, width=WIDTH):
        super().__init__()
        self.two_read = two_read
        self.interface = RowInterface(width)
        self.f_query = nn.Linear(width, width)
        self.f_key = nn.Linear(width, width)
        self.f_value = nn.Linear(width, width)
        self.g_query = nn.Linear(2*width, width)
        self.g_key = nn.Linear(width, width)
        self.g_value = nn.Linear(width, width)
        self.copy_state = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.heads = AnswerHeads(width)

    def forward(self, ids):
        f_rows, g_rows, marker, query = self.interface(ids)
        q = query + marker
        f_scores = (self.f_key(f_rows)*self.f_query(q)[:, None]).sum(-1)/math.sqrt(q.shape[-1])
        f_attention = F.softmax(f_scores, dim=-1)
        first = (f_attention[:, :, None]*self.f_value(f_rows)).sum(1)
        context = first if self.two_read else self.f_value(f_rows).mean(1)
        g_query = self.g_query(torch.cat((q, context), dim=-1))
        g_scores = (self.g_key(g_rows)*g_query[:, None]).sum(-1)/math.sqrt(q.shape[-1])
        g_attention = F.softmax(g_scores, dim=-1)
        second = (g_attention[:, :, None]*self.g_value(g_rows)).sum(1)
        return self.heads(ids, first, second, self.copy_state(q)), f_attention, g_attention


class DenseRows(nn.Module):
    """Dense attention over exactly the same parsed row-record interface."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.interface = RowInterface(width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4,
                                           dim_feedforward=4*width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4,
                                                  enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(width)
        self.heads = AnswerHeads(width)

    def forward(self, ids):
        f_rows, g_rows, marker, query = self.interface(ids)
        slots = torch.cat((f_rows, g_rows, marker[:, None], query[:, None]), dim=1)
        state = self.final_norm(self.transformer(slots)[:, -1])
        return self.heads(ids, state, state, state), None, None


def model_for_arm(arm):
    if arm == "two_read":
        return LearnedMemory(True)
    if arm == "one_read":
        return LearnedMemory(False)
    if arm == "dense_row":
        return DenseRows()
    raise ValueError("Unknown TEACH-0004 arm")
