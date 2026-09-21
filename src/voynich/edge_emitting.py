"""EXP-0020: edge-emitting belief channel vs transformer on joint futures.

Fits and scores use visible symbols only. Sealed generator parameters never enter
selection. Not a Voynich decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ModelConfig, VoynichTransformer
from .runtime import digest, environment, resolve_device, write_json

EXPERIMENT_ID = "EXP-0020"
ALPHABET = 4
STATE_GRID = (1, 2, 4, 8, 16)
FAMILIAR_KEYS = 6
FRESH_KEYS = 4
PREFIX = 32
HORIZON = 3
SEQ_LEN = 40
DATA_SEED = 4020
MODEL_SEED = 42
STRUCTURED_FAMILIES = ("cycle_null", "delayed_parity", "copy_lag")
ALL_FAMILIES = ("iid", "cycle_null", "delayed_parity", "copy_lag", "equivalent")

# Frozen gates (must match docs/experiments/EXP-0020.md).
JOINT_MARGIN = 0.05
FIRST_SYMBOL_EXTRA = 0.02
K_NEAR_TIE_BITS = 0.01
EQUIV_KL_GAP_MAX = 0.05
UPDATE_CLOSURE_TOL = 1e-5
PROB_TOL = 1e-6


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return _sha_bytes(str(arr.shape).encode() + str(arr.dtype).encode() + arr.tobytes())


# ---------------------------------------------------------------------------
# Sealed generators
# ---------------------------------------------------------------------------


def _normalize_edges(edge: np.ndarray) -> np.ndarray:
    """edge[a, i, j] = P(next=j, symbol=a | i); renormalize rows over (a, j)."""
    totals = edge.sum(axis=(0, 2), keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("zero outgoing mass")
    return edge / totals


def keyed_cycle_null(permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-state ambiguous cycle with homophones (alphabet 4)."""
    states = 2
    edge = np.zeros((ALPHABET, states, states), dtype=np.float64)
    for state in range(states):
        edge[:, state, state] += 0.3 / ALPHABET
        nxt = (state + 1) % states
        tokens = permutation[nxt * 2 : (nxt + 1) * 2]
        edge[tokens, state, nxt] += 0.7 / len(tokens)
    prior = np.full(states, 1 / states)
    return _normalize_edges(edge), prior


def keyed_delayed_parity(permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Miniature RRXOR: phase0 -> bit0, phase1 -> bit1, phase2 -> xor; 5 states."""
    # states: 0 wait, 1 after first=0, 2 after first=1, 3 emit-xor for 0, 4 emit-xor for 1
    # Homophones: symbols perm[0:2] -> bit0, perm[2:4] -> bit1
    edge = np.zeros((ALPHABET, 5, 5), dtype=np.float64)
    bit_of = np.empty(ALPHABET, dtype=np.int64)
    for b in range(2):
        for tok in permutation[b * 2 : (b + 1) * 2]:
            bit_of[int(tok)] = b
    for tok in range(ALPHABET):
        b = int(bit_of[tok])
        # from wait: emit first bit -> state 1+b
        edge[tok, 0, 1 + b] += 0.5
        # from after-first: emit second bit -> xor state 3+(first^b)
        for first in range(2):
            edge[tok, 1 + first, 3 + (first ^ b)] += 0.5
        # from xor-state: emit matching bit, return to wait
        edge[tok, 3 + b, 0] += 1.0
    # Each of the half-weights above is over 2 homophones already via loop; fix:
    # Actually we added 0.5 once per token; two tokens per bit => 1.0 total per transition class.
    prior = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
    return _normalize_edges(edge), prior


def equivalent_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimal 2-state unique-symbol cycle and redundant 4-state unfolding."""
    # Minimal: 0 -a-> 1 -b-> 0 with symbols 0 then 1 deterministically.
    edge_min = np.zeros((ALPHABET, 2, 2), dtype=np.float64)
    edge_min[0, 0, 1] = 1.0
    edge_min[1, 1, 0] = 1.0
    # leftover symbols unused (mass 0); renormalize would break determinism — keep only used.
    # Use exclusive symbols 0,1 only; pad unused rows with tiny mass on self-loops of unused.
    edge_min[2, 0, 0] = 0.0
    edge_min[3, 0, 0] = 0.0
    # Ensure row sums: state0 only emits 0->1; state1 only emits 1->0.
    prior_min = np.array([0.5, 0.5], dtype=np.float64)

    # Redundant: duplicate each state (0~2, 1~3); same observations.
    edge_red = np.zeros((ALPHABET, 4, 4), dtype=np.float64)
    # from {0,2} emit 0 go to {1,3} equally
    for s in (0, 2):
        edge_red[0, s, 1] = 0.5
        edge_red[0, s, 3] = 0.5
    for s in (1, 3):
        edge_red[1, s, 0] = 0.5
        edge_red[1, s, 2] = 0.5
    prior_red = np.full(4, 0.25, dtype=np.float64)
    return edge_min, prior_min, edge_red, prior_red


def sample_edge_process(
    edge: np.ndarray, prior: np.ndarray, n: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    visible = np.empty((n, length), dtype=np.uint8)
    states = rng.choice(len(prior), size=n, p=prior)
    # cumulative over (a,j) for each i
    flat = edge.transpose(1, 0, 2).reshape(len(prior), -1)
    table = np.cumsum(flat, axis=1)
    for t in range(length):
        u = rng.random(n)
        chosen = (u[:, None] > table[states]).sum(axis=1)
        a = chosen // len(prior)
        j = chosen % len(prior)
        visible[:, t] = a.astype(np.uint8)
        states = j
    return visible


def sample_copy_lag(n: int, length: int, rng: np.random.Generator, lag: int = 3, p_copy: float = 0.75) -> np.ndarray:
    visible = np.empty((n, length), dtype=np.uint8)
    for t in range(length):
        visible[:, t] = rng.integers(ALPHABET, size=n)
        if t >= lag:
            copied = rng.random(n) < p_copy
            visible[copied, t] = visible[copied, t - lag]
    return visible


def sample_iid(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(ALPHABET, size=(n, length), dtype=np.uint8)


def sample_family(family: str, permutation: np.ndarray, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if family == "iid":
        return sample_iid(n, length, rng)
    if family == "copy_lag":
        return sample_copy_lag(n, length, rng)
    if family == "cycle_null":
        edge, prior = keyed_cycle_null(permutation)
        return sample_edge_process(edge, prior, n, length, rng)
    if family == "delayed_parity":
        edge, prior = keyed_delayed_parity(permutation)
        return sample_edge_process(edge, prior, n, length, rng)
    if family == "equivalent":
        edge_min, prior_min, _, _ = equivalent_pair()
        return sample_edge_process(edge_min, prior_min, n, length, rng)
    raise ValueError(family)


def oracle_next_marginal(edge: np.ndarray, prior: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    """P(next symbol | prefix) under sealed edge model. prefix shape (T,)."""
    belief = prior.astype(np.float64).copy()
    for a in prefix:
        mass = belief @ edge[int(a)]
        total = mass.sum()
        if total <= 0:
            belief = np.full_like(belief, 1 / len(belief))
        else:
            belief = mass / total
    # next-symbol marginal
    return np.array([(belief @ edge[a]).sum() for a in range(ALPHABET)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Edge-emitting model
# ---------------------------------------------------------------------------


class EdgeEmittingChannel(nn.Module):
    """Explicit edge-emitting belief channel with softmax row constraints."""

    def __init__(self, n_states: int, n_symbols: int = ALPHABET):
        super().__init__()
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.initial_logits = nn.Parameter(torch.zeros(n_states))
        # per state: logits over (symbol, next_state)
        self.edge_logits = nn.Parameter(torch.zeros(n_states, n_symbols * n_states))

    def edge_matrices(self) -> torch.Tensor:
        """Return M with shape (A, K, K), float."""
        probs = F.softmax(self.edge_logits, dim=-1)  # (K, A*K)
        k, a = self.n_states, self.n_symbols
        return probs.view(k, a, k).permute(1, 0, 2).contiguous()

    def initial_belief(self, batch: int = 1, device=None, dtype=None) -> torch.Tensor:
        b = F.softmax(self.initial_logits, dim=-1)
        if device is not None or dtype is not None:
            b = b.to(device=device, dtype=dtype)
        return b.unsqueeze(0).expand(batch, -1).clone()

    def step(self, symbol: torch.Tensor, belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """symbol (B,) int; belief (B, K) -> next_symbol_dist (B, A), new_belief (B, K)."""
        m = self.edge_matrices().to(device=belief.device, dtype=belief.dtype)  # (A,K,K)
        # P(a|b) = sum_j belief_i M_a[i,j]
        # belief_m[a,b] = sum_i belief[b,i] * M[a,i,:].sum(-1) wait
        # joint next mass per symbol: (belief @ M_a).sum(-1)
        masses = torch.einsum("bk,akj->ba", belief, m)  # (B, A) = sum_{i,j} b_i M_a[i,j]
        dist = masses / masses.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # update with observed symbol
        a = symbol.long()
        m_a = m[a]  # (B, K, K) via advanced indexing? m is (A,K,K); need per-batch
        m_sel = m[a]  # (B, K, K)
        new_mass = torch.einsum("bk,bkj->bj", belief, m_sel)
        new_belief = new_mass / new_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return dist, new_belief

    def score_continuation(self, belief: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
        """Joint log-prob of symbols (B, H) given belief (B, K). Returns (B,)."""
        logp = torch.zeros(belief.shape[0], device=belief.device, dtype=belief.dtype)
        state = belief
        for t in range(symbols.shape[1]):
            dist, state = self.step(symbols[:, t], state)
            logp = logp + torch.log(dist.gather(1, symbols[:, t : t + 1].long()).squeeze(1).clamp_min(1e-12))
        return logp

    def snapshot(self) -> dict:
        return {
            "initial_logits": self.initial_logits.detach().cpu().clone(),
            "edge_logits": self.edge_logits.detach().cpu().clone(),
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
        }

    def restore(self, snap: dict) -> None:
        with torch.no_grad():
            self.initial_logits.copy_(snap["initial_logits"])
            self.edge_logits.copy_(snap["edge_logits"])

    @torch.no_grad()
    def probability_validity(self, tol: float = PROB_TOL) -> dict:
        m = self.edge_matrices().detach().cpu().double().numpy()
        row_sums = m.sum(axis=(0, 2))
        ok = bool(np.all(np.isfinite(m)) and np.all(m >= -tol) and np.allclose(row_sums, 1.0, atol=tol))
        init = F.softmax(self.initial_logits.detach().cpu().double(), dim=-1).numpy()
        ok = ok and bool(np.all(np.isfinite(init)) and abs(init.sum() - 1.0) <= tol and np.all(init >= -tol))
        return {"ok": ok, "row_sums": row_sums.tolist(), "init_sum": float(init.sum())}

    @torch.no_grad()
    def update_closure_error(self, n_beliefs: int = 32, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        cpu_model = EdgeEmittingChannel(self.n_states, self.n_symbols)
        cpu_model.load_state_dict({k: v.detach().cpu() for k, v in self.state_dict().items()})
        m = cpu_model.edge_matrices().double().numpy()
        k = self.n_states
        max_err = 0.0
        for _ in range(n_beliefs):
            raw = rng.random(k)
            b = raw / raw.sum()
            for a in range(self.n_symbols):
                mass = b @ m[a]
                total = mass.sum()
                if total <= 0:
                    continue
                b_after = mass / total
                belief = torch.tensor(b, dtype=torch.float64).unsqueeze(0)
                sym = torch.tensor([a], dtype=torch.long)
                dist_t, new_t = cpu_model.step(sym, belief)
                err = float(np.max(np.abs(new_t.numpy()[0] - b_after)))
                max_err = max(max_err, err)
                pred = np.array([(b @ m[aa]).sum() for aa in range(self.n_symbols)])
                pred = pred / pred.sum()
                err_d = float(np.max(np.abs(dist_t.numpy()[0] - pred)))
                max_err = max(max_err, err_d)
        return max_err


def nll_bits_next_symbol(model: EdgeEmittingChannel, sequences: np.ndarray, device: str) -> float:
    """Mean next-symbol NLL in bits over all positions with online belief updates."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(sequences), 64):
            batch = sequences[start : start + 64]
            x = torch.as_tensor(batch, device=device, dtype=torch.long)
            belief = model.initial_belief(len(batch), device=device, dtype=torch.float32)
            m = model.edge_matrices()
            for t in range(x.shape[1]):
                masses = torch.einsum("bk,akj->ba", belief, m)
                dist = masses / masses.sum(-1, keepdim=True).clamp_min(1e-12)
                total += float((-torch.log(dist.gather(1, x[:, t : t + 1]).squeeze(1).clamp_min(1e-12))).sum())
                count += len(batch)
                m_sel = m[x[:, t]]
                mass = torch.einsum("bk,bkj->bj", belief, m_sel)
                belief = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
    return (total / max(count, 1)) / math.log(2)


def fit_edge_model(
    sequences: np.ndarray,
    val_sequences: np.ndarray,
    n_states: int,
    device: str,
    seed: int,
    updates: int = 400,
    lr: float = 0.05,
) -> tuple[EdgeEmittingChannel, float]:
    torch.manual_seed(seed)
    model = EdgeEmittingChannel(n_states).to(device)
    # small random init
    with torch.no_grad():
        model.initial_logits.normal_(0, 0.1)
        model.edge_logits.normal_(0, 0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.as_tensor(sequences, device=device, dtype=torch.long)
    best_state = None
    best_val = float("inf")
    for step_i in range(updates):
        model.train()
        idx = torch.randint(0, len(data), (min(64, len(data)),), device=device)
        batch = data[idx]
        belief = model.initial_belief(len(batch), device=device, dtype=torch.float32)
        loss = batch.new_zeros((), dtype=torch.float32)
        m = model.edge_matrices()
        for t in range(batch.shape[1]):
            masses = torch.einsum("bk,akj->ba", belief, m)
            dist = masses / masses.sum(-1, keepdim=True).clamp_min(1e-12)
            loss = loss + F.nll_loss(torch.log(dist.clamp_min(1e-12)), batch[:, t], reduction="mean")
            m_sel = m[batch[:, t]]
            mass = torch.einsum("bk,bkj->bj", belief, m_sel)
            belief = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
        loss = loss / batch.shape[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step_i + 1) % 50 == 0 or step_i == updates - 1:
            val = nll_bits_next_symbol(model, val_sequences, device)
            if val < best_val:
                best_val = val
                best_state = model.snapshot()
    if best_state is not None:
        model.restore(best_state)
        model.to(device)
    return model, float(best_val)


def select_edge_model(fit_seq: np.ndarray, val_seq: np.ndarray, device: str, seed: int) -> dict:
    candidates = []
    for k in STATE_GRID:
        best_for_k = None
        for restart in range(3):
            model, val = fit_edge_model(fit_seq, val_seq, k, device, seed + 1000 * k + restart)
            if best_for_k is None or val < best_for_k["val_bits"]:
                best_for_k = {
                    "k": k,
                    "val_bits": val,
                    "snapshot": model.snapshot(),
                    "validity": model.probability_validity(),
                    "closure_err": model.update_closure_error(seed=seed + k),
                }
        candidates.append(best_for_k)
    best_val = min(c["val_bits"] for c in candidates)
    near = [c for c in candidates if c["val_bits"] <= best_val + K_NEAR_TIE_BITS]
    chosen = min(near, key=lambda c: c["k"])
    return {"chosen": chosen, "candidates": [{"k": c["k"], "val_bits": c["val_bits"]} for c in candidates]}


# ---------------------------------------------------------------------------
# Transformer reference
# ---------------------------------------------------------------------------


def make_transformer() -> VoynichTransformer:
    cfg = ModelConfig(
        vocab_size=ALPHABET + 1,
        pad_id=0,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        context_length=64,
        dropout=0.0,
    )
    return VoynichTransformer(cfg)


def train_transformer(train: np.ndarray, val: np.ndarray, device: str, seed: int = MODEL_SEED) -> tuple[VoynichTransformer, dict]:
    torch.manual_seed(seed)
    model = make_transformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    train_t = torch.as_tensor(train + 1, device=device, dtype=torch.long)  # shift for pad=0
    val_t = torch.as_tensor(val + 1, device=device, dtype=torch.long)
    best = None
    best_val = float("inf")
    history = []
    for step_i in range(800):
        model.train()
        idx = torch.randint(0, len(train_t), (64,), device=device)
        batch = train_t[idx]
        logits = model(batch[:, :-1]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step_i + 1) % 100 == 0 or step_i == 799:
            model.eval()
            with torch.no_grad():
                vlog = model(val_t[:, :-1]).logits
                vloss = float(F.cross_entropy(vlog.reshape(-1, vlog.size(-1)), val_t[:, 1:].reshape(-1)))
            vbits = vloss / math.log(2)
            history.append({"step": step_i + 1, "val_bits": vbits})
            if vbits < best_val:
                best_val = vbits
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return model, {"best_val_bits": best_val, "history": history}


class LogitAdapter(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.log_temp.exp().clamp(0.05, 20.0)
        return logits / temp + self.bias


def maybe_adapt_transformer(
    model: VoynichTransformer, fit_seq: np.ndarray, val_seq: np.ndarray, device: str, seed: int
) -> tuple[LogitAdapter | None, str]:
    """Optional last-logit bias/temperature; keep frozen if val worsens."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = LogitAdapter(ALPHABET + 1).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.05)
    fit_t = torch.as_tensor(fit_seq + 1, device=device, dtype=torch.long)
    val_t = torch.as_tensor(val_seq + 1, device=device, dtype=torch.long)

    def bits_with(ad: LogitAdapter | None) -> float:
        with torch.no_grad():
            logits = model(val_t[:, :-1]).logits
            if ad is not None:
                logits = ad(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), val_t[:, 1:].reshape(-1))
        return float(loss) / math.log(2)

    frozen_bits = bits_with(None)
    torch.manual_seed(seed)
    for _ in range(200):
        idx = torch.randint(0, len(fit_t), (min(64, len(fit_t)),), device=device)
        batch = fit_t[idx]
        with torch.no_grad():
            base = model(batch[:, :-1]).logits
        logits = adapter(base)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    adapted_bits = bits_with(adapter)
    if adapted_bits < frozen_bits:
        return adapter, "adapted"
    return None, "frozen"


@torch.no_grad()
def transformer_joint_bits(
    model: VoynichTransformer, adapter: LogitAdapter | None, sequences: np.ndarray, device: str
) -> tuple[float, float]:
    """Mean joint bits/symbol over H and mean first-symbol bits on test prefixes."""
    model.eval()
    x = torch.as_tensor(sequences + 1, device=device, dtype=torch.long)
    joint_total = 0.0
    first_total = 0.0
    n = len(sequences)
    for start in range(0, n, 64):
        batch = x[start : start + 64]
        # condition on prefix; score PREFIX .. PREFIX+H-1 (0-index: PREFIX is first future)
        # Provide full context up through each prediction.
        logp_joint = torch.zeros(len(batch), device=device)
        for h in range(HORIZON):
            ctx = batch[:, : PREFIX + h]
            logits = model(ctx).logits[:, -1, :]
            if adapter is not None:
                logits = adapter(logits)
            log_probs = F.log_softmax(logits, dim=-1)
            target = batch[:, PREFIX + h]
            lp = log_probs.gather(1, target[:, None]).squeeze(1)
            logp_joint = logp_joint + lp
            if h == 0:
                first_total += float((-lp).sum())
        joint_total += float((-logp_joint).sum())
    joint_bits = (joint_total / n / HORIZON) / math.log(2)
    first_bits = (first_total / n) / math.log(2)
    return joint_bits, first_bits


@torch.no_grad()
def edge_joint_bits(model: EdgeEmittingChannel, sequences: np.ndarray, device: str) -> tuple[float, float]:
    model.eval()
    joint_total = 0.0
    first_total = 0.0
    n = len(sequences)
    for start in range(0, n, 64):
        batch_np = sequences[start : start + 64]
        batch = torch.as_tensor(batch_np, device=device, dtype=torch.long)
        belief = model.initial_belief(len(batch), device=device, dtype=torch.float32)
        m = model.edge_matrices()
        for t in range(PREFIX):
            m_sel = m[batch[:, t]]
            mass = torch.einsum("bk,bkj->bj", belief, m_sel)
            belief = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
        # score continuation
        logp = torch.zeros(len(batch), device=device)
        state = belief
        for h in range(HORIZON):
            masses = torch.einsum("bk,akj->ba", state, m)
            dist = masses / masses.sum(-1, keepdim=True).clamp_min(1e-12)
            target = batch[:, PREFIX + h]
            lp = torch.log(dist.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12))
            logp = logp + lp
            if h == 0:
                first_total += float((-lp).sum())
            m_sel = m[target]
            mass = torch.einsum("bk,bkj->bj", state, m_sel)
            state = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
        joint_total += float((-logp).sum())
    joint_bits = (joint_total / n / HORIZON) / math.log(2)
    first_bits = (first_total / n) / math.log(2)
    return joint_bits, first_bits


def equivalent_kl_gap(model: EdgeEmittingChannel, test_seq: np.ndarray, device: str) -> dict:
    edge_min, prior_min, edge_red, prior_red = equivalent_pair()
    gaps = []
    oracle_agree = []
    model.eval()
    with torch.no_grad():
        for row in test_seq:
            prefix = row[:PREFIX]
            p_min = oracle_next_marginal(edge_min, prior_min, prefix)
            p_red = oracle_next_marginal(edge_red, prior_red, prefix)
            oracle_agree.append(float(np.max(np.abs(p_min - p_red))))
            # model predictive
            belief = model.initial_belief(1, device=device, dtype=torch.float32)
            m = model.edge_matrices()
            x = torch.as_tensor(prefix, device=device, dtype=torch.long).unsqueeze(0)
            for t in range(PREFIX):
                mass = torch.einsum("bk,bkj->bj", belief, m[x[:, t]])
                belief = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
            masses = torch.einsum("bk,akj->ba", belief, m)
            p_m = (masses / masses.sum(-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()[0]
            p_m = np.clip(p_m, 1e-12, 1)
            p_m = p_m / p_m.sum()

            def kl(p, q):
                q = np.clip(q, 1e-12, 1)
                q = q / q.sum()
                return float(np.sum(p * (np.log(p) - np.log(q))) / math.log(2))

            gaps.append(abs(kl(p_m, p_min) - kl(p_m, p_red)))
    return {
        "mean_abs_kl_gap_bits": float(np.mean(gaps)),
        "max_oracle_marginal_diff": float(np.max(oracle_agree)),
        "oracle_agree_ok": bool(np.max(oracle_agree) <= 1e-8),
    }


# ---------------------------------------------------------------------------
# Data preparation + experiment
# ---------------------------------------------------------------------------


def prepare_data(root: Path) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    master = np.random.SeedSequence(DATA_SEED)
    children = iter(master.spawn(500))
    inventory = {}
    lm_train_parts = []
    lm_val_parts = []
    for family in ALL_FAMILIES:
        inventory[family] = {"familiar": [], "fresh": []}
        for key_i in range(FAMILIAR_KEYS + FRESH_KEYS):
            perm = np.random.default_rng(next(children)).permutation(ALPHABET)
            pools = {}
            for pool_name, count in (("lm_train", 256), ("fit", 128), ("val", 64), ("test", 128)):
                rng = np.random.default_rng(next(children))
                if family in {"iid", "copy_lag"} and pool_name == "lm_train" and key_i >= FAMILIAR_KEYS:
                    continue
                if pool_name == "lm_train" and key_i >= FAMILIAR_KEYS:
                    continue
                data = sample_family(family, perm, count, SEQ_LEN, rng)
                path = root / f"{family}-key{key_i}-{pool_name}.npy"
                np.save(path, data)
                pools[pool_name] = {"file": path.name, "sha256": digest(path), "n": count, "digest": _array_digest(data)}
                if pool_name == "lm_train" and key_i < FAMILIAR_KEYS:
                    lm_train_parts.append(data)
                if pool_name == "val" and key_i < FAMILIAR_KEYS and family in STRUCTURED_FAMILIES:
                    lm_val_parts.append(data)
            bucket = "familiar" if key_i < FAMILIAR_KEYS else "fresh"
            inventory[family][bucket].append(
                {"key": key_i, "permutation": perm.tolist(), "pools": pools}
            )
    lm_train = np.concatenate(lm_train_parts)
    lm_val = np.concatenate(lm_val_parts) if lm_val_parts else lm_train[:512]
    np.save(root / "lm_train.npy", lm_train)
    np.save(root / "lm_val.npy", lm_val)
    # seal permutations separately
    sealed = {
        family: {
            "familiar": [{"key": e["key"], "permutation": e["permutation"]} for e in inventory[family]["familiar"]],
            "fresh": [{"key": e["key"], "permutation": e["permutation"]} for e in inventory[family]["fresh"]],
        }
        for family in ALL_FAMILIES
    }
    write_json(root / "sealed_keys.json", sealed)
    manifest = {
        "experiment": EXPERIMENT_ID,
        "data_seed": DATA_SEED,
        "alphabet": ALPHABET,
        "prefix": PREFIX,
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
        "families": ALL_FAMILIES,
        "lm_train_sha256": digest(root / "lm_train.npy"),
        "lm_val_sha256": digest(root / "lm_val.npy"),
        "inventory": {
            family: {
                bucket: [{"key": e["key"], "pools": e["pools"]} for e in inventory[family][bucket]]
                for bucket in ("familiar", "fresh")
            }
            for family in ALL_FAMILIES
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def run_experiment(repo: Path, device: str) -> dict:
    t0 = time.perf_counter()
    data_root = repo / "data" / "processed" / "exp0020"
    out_root = repo / "outputs" / EXPERIMENT_ID
    out_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "manifest.json").exists():
        prepare_data(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())

    lm_train = np.load(data_root / "lm_train.npy")
    lm_val = np.load(data_root / "lm_val.npy")
    tf, tf_info = train_transformer(lm_train, lm_val, device, MODEL_SEED)
    torch.save(tf.state_dict(), out_root / "transformer.pt")

    per_family = {}
    validity_ok = True
    closure_ok = True
    equiv_report = None

    for family in ALL_FAMILIES:
        fresh_rows = []
        for entry in manifest["inventory"][family]["fresh"]:
            key = entry["key"]
            fit = np.load(data_root / entry["pools"]["fit"]["file"])
            val = np.load(data_root / entry["pools"]["val"]["file"])
            test = np.load(data_root / entry["pools"]["test"]["file"])
            fam_salt = int(_sha_bytes(family.encode())[:8], 16) % 10007
            selected = select_edge_model(fit, val, device, seed=DATA_SEED + key * 17 + fam_salt)
            chosen = selected["chosen"]
            edge = EdgeEmittingChannel(chosen["k"]).to(device)
            edge.restore(chosen["snapshot"])
            edge.to(device)
            if not chosen["validity"]["ok"]:
                validity_ok = False
            if chosen["closure_err"] >= UPDATE_CLOSURE_TOL:
                closure_ok = False
            e_joint, e_first = edge_joint_bits(edge, test, device)
            adapter, adapt_mode = maybe_adapt_transformer(tf, fit, val, device, seed=MODEL_SEED + key)
            t_joint, t_first = transformer_joint_bits(tf, adapter, test, device)
            row = {
                "key": key,
                "selected_k": chosen["k"],
                "val_bits": chosen["val_bits"],
                "k_candidates": selected["candidates"],
                "validity": chosen["validity"],
                "closure_err": chosen["closure_err"],
                "edge_joint_bits": e_joint,
                "edge_first_bits": e_first,
                "tf_joint_bits": t_joint,
                "tf_first_bits": t_first,
                "tf_adapt_mode": adapt_mode,
                "delta_joint_tf_minus_edge": t_joint - e_joint,
                "delta_first_tf_minus_edge": t_first - e_first,
            }
            if family == "equivalent":
                row["equiv"] = equivalent_kl_gap(edge, test, device)
            fresh_rows.append(row)
            # free adapter
            del adapter
        per_family[family] = {
            "keys": fresh_rows,
            "macro_edge_joint": float(np.mean([r["edge_joint_bits"] for r in fresh_rows])),
            "macro_tf_joint": float(np.mean([r["tf_joint_bits"] for r in fresh_rows])),
            "macro_edge_first": float(np.mean([r["edge_first_bits"] for r in fresh_rows])),
            "macro_tf_first": float(np.mean([r["tf_first_bits"] for r in fresh_rows])),
            "macro_delta_joint": float(np.mean([r["delta_joint_tf_minus_edge"] for r in fresh_rows])),
            "macro_delta_first": float(np.mean([r["delta_first_tf_minus_edge"] for r in fresh_rows])),
            "mean_selected_k": float(np.mean([r["selected_k"] for r in fresh_rows])),
        }
        if family == "equivalent":
            gaps = [r["equiv"]["mean_abs_kl_gap_bits"] for r in fresh_rows]
            oracle_ok = all(r["equiv"]["oracle_agree_ok"] for r in fresh_rows)
            equiv_report = {
                "mean_abs_kl_gap_bits": float(np.mean(gaps)),
                "oracle_agree_ok": oracle_ok,
                "max_oracle_marginal_diff": float(max(r["equiv"]["max_oracle_marginal_diff"] for r in fresh_rows)),
            }

    # Structured macro
    d_joint = float(np.mean([per_family[f]["macro_delta_joint"] for f in STRUCTURED_FAMILIES]))
    d_first = float(np.mean([per_family[f]["macro_delta_first"] for f in STRUCTURED_FAMILIES]))

    mode = None
    reasons = []
    if not validity_ok:
        mode = "probability_invalid"
        reasons.append("selected edge model failed probability validity")
    elif not closure_ok:
        mode = "no_update_closure"
        reasons.append(f"update closure error >= {UPDATE_CLOSURE_TOL}")
    elif equiv_report is None or not equiv_report["oracle_agree_ok"]:
        mode = "false_unique_recovery"
        reasons.append("equivalent-generator oracle marginals disagree (generator bug)")
    elif equiv_report["mean_abs_kl_gap_bits"] > EQUIV_KL_GAP_MAX:
        mode = "false_unique_recovery"
        reasons.append(
            f"equiv KL gap {equiv_report['mean_abs_kl_gap_bits']:.4f} > {EQUIV_KL_GAP_MAX}"
        )
    elif d_joint < JOINT_MARGIN:
        mode = "joint_not_better"
        reasons.append(f"structured macro Δ_joint={d_joint:.4f} < {JOINT_MARGIN}")
    elif d_joint < d_first + FIRST_SYMBOL_EXTRA:
        mode = "first_symbol_only"
        reasons.append(
            f"Δ_joint={d_joint:.4f} < Δ_1+{FIRST_SYMBOL_EXTRA} ({d_first + FIRST_SYMBOL_EXTRA:.4f})"
        )
    else:
        reasons.append("all EXP-0020 gates cleared")

    passed = mode is None
    decision = {
        "passed": passed,
        "mode": mode,
        "reasons": reasons,
        "rule": EXPERIMENT_ID,
        "gates": {
            "joint_margin": JOINT_MARGIN,
            "first_symbol_extra": FIRST_SYMBOL_EXTRA,
            "equiv_kl_gap_max": EQUIV_KL_GAP_MAX,
            "update_closure_tol": UPDATE_CLOSURE_TOL,
            "k_near_tie_bits": K_NEAR_TIE_BITS,
            "delta_joint_structured": d_joint,
            "delta_first_structured": d_first,
            "probability_validity_ok": validity_ok,
            "update_closure_ok": closure_ok,
            "equivalent": equiv_report,
        },
        "thresholds": {
            "JOINT_MARGIN": JOINT_MARGIN,
            "FIRST_SYMBOL_EXTRA": FIRST_SYMBOL_EXTRA,
            "EQUIV_KL_GAP_MAX": EQUIV_KL_GAP_MAX,
            "UPDATE_CLOSURE_TOL": UPDATE_CLOSURE_TOL,
            "K_NEAR_TIE_BITS": K_NEAR_TIE_BITS,
            "PROB_TOL": PROB_TOL,
            "STATE_GRID": list(STATE_GRID),
            "DATA_SEED": DATA_SEED,
            "MODEL_SEED": MODEL_SEED,
            "PREFIX": PREFIX,
            "HORIZON": HORIZON,
        },
    }

    results = {
        "experiment": EXPERIMENT_ID,
        "question": (
            "Does an explicit edge-emitting channel predict joint continuation strings on fresh keys "
            "better than a transformer reference under P2 controls?"
        ),
        "device": device,
        "elapsed_s": time.perf_counter() - t0,
        "transformer": tf_info,
        "data_manifest_sha256": digest(data_root / "manifest.json"),
        "per_family": per_family,
        "structured_macro": {
            "families": list(STRUCTURED_FAMILIES),
            "delta_joint": d_joint,
            "delta_first": d_first,
        },
        "environment": environment(),
        "decision": decision,
    }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP-0020 edge-emitting vs transformer joint futures")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    repo = args.root.resolve()
    results = run_experiment(repo, device)
    out = repo / "results" / EXPERIMENT_ID
    write_json(out / "results.json", results)
    write_json(out / "decision.json", results["decision"])
    # compact data manifest for git
    data_root = repo / "data" / "processed" / "exp0020"
    write_json(
        repo / "data" / "manifests" / "exp0020_data.json",
        {
            "experiment": EXPERIMENT_ID,
            "processed_root": "data/processed/exp0020",
            "manifest_sha256": digest(data_root / "manifest.json"),
            "lm_train_sha256": digest(data_root / "lm_train.npy"),
            "data_seed": DATA_SEED,
        },
    )
    print(json.dumps({"passed": results["decision"]["passed"], "mode": results["decision"]["mode"],
                      "delta_joint": results["structured_macro"]["delta_joint"],
                      "elapsed_s": results["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
