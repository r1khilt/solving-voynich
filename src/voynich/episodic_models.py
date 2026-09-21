"""Small observable-stream models for EXP-0012; no latent labels are accepted.

The signed recurrence is our CPU/MPS adaptation, not a DeltaProduct replication.
Conventions, equations, and source limitations: docs/research/EPISODIC_MODELS.md.
"""

from dataclasses import asdict, dataclass
import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer


@dataclass(frozen=True)
class ModelSpec:
    kind: str = "transformer"
    alphabet: int = 4
    canonical: bool = False
    width: int = 64
    layers: int = 2
    heads: int = 4
    ff_width: int = 128
    context: int = 256
    dropout: float = 0.0

    def __post_init__(self):
        if self.kind not in ("transformer", "gru", "signed_delta"):
            raise ValueError("kind must be transformer, gru, or signed_delta")
        for name in ("alphabet", "width", "layers", "heads", "ff_width", "context"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.alphabet < 2:
            raise ValueError("alphabet must be at least two")
        if not isinstance(self.canonical, bool):
            raise ValueError("canonical must be a boolean")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0,1)")
        if self.kind == "transformer" and (self.width % self.heads or (self.width // self.heads) % 2):
            raise ValueError("transformer width must divide heads with an even head dimension")

    @property
    def output_dim(self) -> int:
        return self.alphabet + int(self.canonical)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(tokens: Tensor, alphabet: int, *, allow_empty: bool = False) -> None:
    if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError("tokens must have [nonempty batch,time] shape")
    if not allow_empty and tokens.shape[1] == 0:
        raise ValueError("tokens must have nonempty time")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must have integer dtype")
    if tokens.numel() and ((tokens < 0).any() or (tokens >= alphabet).any()):
        raise ValueError("token outside observed alphabet")


def raw_log_probs(logits: Tensor, counts: Tensor, inverse: Tensor) -> Tensor:
    """Convert rank/NEW logits to exact raw-symbol log probabilities.

    ``counts[...]`` is the number of distinct symbols seen through the current
    input; ``inverse[...,rank]`` is that rank's raw ID, with -1 in unseen slots.
    ``logits[...,A]`` predicts NEW. Unsupported rank events are masked before
    normalization, and NEW is shared uniformly among the remaining raw IDs.
    Maps may be supplied at arbitrary leading shape, including [batch,time].
    """
    if not isinstance(logits, Tensor) or logits.ndim < 1 or logits.shape[-1] < 3:
        raise ValueError("logits must have [...,alphabet+1] shape")
    alphabet = logits.shape[-1] - 1
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if not isinstance(counts, Tensor) or not isinstance(inverse, Tensor):
        raise ValueError("counts and inverse must be tensors")
    if counts.shape != logits.shape[:-1] or inverse.shape != (*counts.shape, alphabet):
        raise ValueError("counts/inverse do not match logits shape")
    if counts.device != logits.device or inverse.device != logits.device:
        raise ValueError("counts, inverse and logits must share a device")
    if counts.dtype not in (torch.int32, torch.int64) or inverse.dtype not in (torch.int32, torch.int64):
        raise ValueError("counts and inverse must have integer dtype")
    if ((counts < 0) | (counts > alphabet)).any():
        raise ValueError("counts outside [0,alphabet]")
    ranks = torch.arange(alphabet, device=logits.device)
    assigned = ranks < counts[..., None]
    malformed = (assigned & ((inverse < 0) | (inverse >= alphabet))) | (~assigned & inverse.ne(-1))
    if malformed.any():
        raise ValueError("inverse must contain valid assigned ranks followed by -1")
    # A is small (four in EXP-0012); equality avoids duplicate padding scatter.
    rank_to_raw = inverse[..., :, None].eq(ranks)
    if (rank_to_raw.sum(-2) > 1).any():
        raise ValueError("inverse contains duplicate raw symbols")
    allowed = torch.cat((assigned, (counts < alphabet)[..., None]), dim=-1)
    canonical_logp = F.log_softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
    raw_to_rank = rank_to_raw.to(torch.int64).argmax(-2)
    seen_logp = canonical_logp.gather(-1, raw_to_rank)
    new_logp = canonical_logp[..., alphabet] - (alphabet - counts).clamp_min(1).to(logits.dtype).log()
    return torch.where(rank_to_raw.any(-2), seen_logp, new_logp[..., None])


class EpisodicTransformer(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.backbone = VoynichTransformer(ModelConfig(
            vocab_size=spec.alphabet + 1, pad_id=spec.alphabet,
            d_model=spec.width, n_layers=spec.layers, n_heads=spec.heads,
            d_ff=spec.ff_width, context_length=spec.context, dropout=spec.dropout,
        ))

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.spec.alphabet)
        return self.backbone(tokens).logits[..., :self.spec.output_dim]

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _RecurrentBase(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.embedding = nn.Embedding(spec.alphabet, spec.width)
        self.norm = nn.LayerNorm(spec.width)
        self.head = nn.Linear(spec.width, spec.output_dim)

    def initial_state(self, batch: int, device=None) -> Tensor:
        if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
            raise ValueError("batch must be a positive integer")
        return torch.zeros(batch, self.spec.layers, self.spec.width,
                           device=self.embedding.weight.device if device is None else device,
                           dtype=self.embedding.weight.dtype)

    def _state(self, state: Tensor) -> None:
        if (not isinstance(state, Tensor) or state.ndim != 3 or state.shape[0] == 0
                or state.shape[1:] != (self.spec.layers, self.spec.width)):
            raise ValueError("state must have [batch,layers,width] shape")
        if state.dtype != self.embedding.weight.dtype or state.device != self.embedding.weight.device:
            raise ValueError("state must match model device and dtype")

    def readout(self, state: Tensor) -> Tensor:
        self._state(state)
        return self.head(self.norm(state[:, -1]))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EpisodicGRU(_RecurrentBase):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.gru = nn.GRU(spec.width, spec.width, spec.layers, batch_first=True,
                          dropout=spec.dropout if spec.layers > 1 else 0.0)

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.spec.alphabet)
        outputs, _ = self.gru(self.embedding(tokens))
        return self.head(self.norm(outputs))

    def encode(self, prefix: Tensor) -> Tensor:
        _tokens(prefix, self.spec.alphabet, allow_empty=True)
        if prefix.shape[1] == 0:
            return self.initial_state(prefix.shape[0], prefix.device)
        _, state = self.gru(self.embedding(prefix))
        return state.transpose(0, 1).contiguous()

    def step(self, symbol: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        self._state(state)
        if not isinstance(symbol, Tensor) or symbol.shape != (state.shape[0],):
            raise ValueError("symbol must have [batch] shape matching state")
        _tokens(symbol[:, None], self.spec.alphabet)
        _, updated = self.gru(self.embedding(symbol[:, None]), state.transpose(0, 1).contiguous())
        updated = updated.transpose(0, 1).contiguous()
        return self.readout(updated), updated


class SignedDeltaLayer(nn.Module):
    """Two independently parameterized generalized Householder factors.

    For each input x, H_i = I - beta_i k_i k_i^T with unit k_i, beta_i in
    (0,2).  h' = retain * H_2 H_1 h + (1-retain) * tanh(write(x)).
    H_i has operator norm <= 1; the convex write bounds state norms. The two
    key functions have separate weights; learned directions can still coincide.
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.keys = nn.ModuleList([nn.Linear(width, width), nn.Linear(width, width)])
        self.beta = nn.Linear(width, 2)
        self.retain = nn.Linear(width, 1)
        self.write = nn.Linear(width, width)
        nn.init.constant_(self.retain.bias, 1.0)

    def factors(self, inputs: Tensor) -> dict[str, Tensor]:
        x = self.norm(inputs)
        return {
            "keys": torch.stack([F.normalize(key(x), dim=-1, eps=1e-8) for key in self.keys], dim=-2),
            "beta": 2.0 * self.beta(x).sigmoid(),
            "retain": self.retain(x).sigmoid(),
            "write": self.write(x).tanh(),
        }

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        factors = self.factors(inputs)
        moved = state
        for factor in range(2):
            key = factors["keys"][..., factor, :]
            beta = factors["beta"][..., factor, None]
            moved = moved - beta * key * (key * moved).sum(-1, keepdim=True)
        retain = factors["retain"]
        return retain * moved + (1.0 - retain) * factors["write"]


class SignedDeltaRNN(_RecurrentBase):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.cells = nn.ModuleList([SignedDeltaLayer(spec.width) for _ in range(spec.layers)])
        self.dropout = nn.Dropout(spec.dropout)

    def _advance(self, symbol: Tensor, state: Tensor) -> Tensor:
        inputs = self.embedding(symbol)
        layers = []
        for index, cell in enumerate(self.cells):
            value = cell(inputs, state[:, index])
            layers.append(value)
            inputs = self.dropout(value.tanh())
        return torch.stack(layers, dim=1)

    def step(self, symbol: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        self._state(state)
        if not isinstance(symbol, Tensor) or symbol.shape != (state.shape[0],):
            raise ValueError("symbol must have [batch] shape matching state")
        _tokens(symbol[:, None], self.spec.alphabet)
        updated = self._advance(symbol, state)
        return self.readout(updated), updated

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.spec.alphabet)
        state = self.initial_state(tokens.shape[0], tokens.device)
        states = []
        for symbol in tokens.unbind(1):
            state = self._advance(symbol, state)
            states.append(state[:, -1])
        return self.head(self.norm(torch.stack(states, dim=1)))

    def encode(self, prefix: Tensor) -> Tensor:
        _tokens(prefix, self.spec.alphabet, allow_empty=True)
        state = self.initial_state(prefix.shape[0], prefix.device)
        for symbol in prefix.unbind(1):
            state = self._advance(symbol, state)
        return state


def build_model(spec: ModelSpec) -> nn.Module:
    if not isinstance(spec, ModelSpec):
        raise ValueError("spec must be a ModelSpec")
    return {"transformer": EpisodicTransformer, "gru": EpisodicGRU,
            "signed_delta": SignedDeltaRNN}[spec.kind](spec)


def _numpy_sequence(tokens, alphabet: int, *, allow_empty=True) -> np.ndarray:
    sequence = np.asarray(tokens)
    if sequence.ndim != 1 or (not allow_empty and sequence.size == 0):
        raise ValueError("tokens must be a one-dimensional sequence")
    if sequence.size and (not np.issubdtype(sequence.dtype, np.integer)
                          or np.any(sequence < 0) or np.any(sequence >= alphabet)):
        raise ValueError("tokens must be integers in the observed alphabet")
    return sequence.astype(np.int64, copy=False)


class EdgeHMM:
    """Positive edge-emitting HMM: edge[a,i,j] = P(a,s'=j | s=i)."""

    def __init__(self, edge, prior):
        edge = np.asarray(edge, dtype=np.float64)
        prior = np.asarray(prior, dtype=np.float64)
        if edge.ndim != 3 or edge.shape[0] < 2 or edge.shape[1] == 0 or edge.shape[1] != edge.shape[2]:
            raise ValueError("edge must have [alphabet,states,states] shape")
        if prior.shape != (edge.shape[1],):
            raise ValueError("prior must have [states] shape")
        if not np.isfinite(edge).all() or np.any(edge <= 0):
            raise ValueError("edge weights must be finite and strictly positive")
        if not np.isfinite(prior).all() or np.any(prior <= 0):
            raise ValueError("prior weights must be finite and strictly positive")
        self.edge = edge.copy() / edge.sum(axis=(0, 2))[None, :, None]
        self.prior = prior.copy() / prior.sum()
        self.alphabet, self.states = edge.shape[:2]

    def _belief(self, belief) -> np.ndarray:
        belief = np.asarray(belief, dtype=np.float64)
        if (belief.shape != (self.states,) or not np.isfinite(belief).all()
                or np.any(belief < 0) or not np.isclose(belief.sum(), 1.0, rtol=1e-9, atol=1e-12)):
            raise ValueError("belief must be a finite normalized nonnegative state vector")
        return belief

    def update(self, belief, symbol: int) -> np.ndarray:
        belief = self._belief(belief)
        if not isinstance(symbol, (int, np.integer)) or isinstance(symbol, (bool, np.bool_)):
            raise ValueError("symbol must be an integer")
        if not 0 <= symbol < self.alphabet:
            raise ValueError("symbol outside observed alphabet")
        updated = belief @ self.edge[int(symbol)]
        return updated / updated.sum()

    def filter(self, tokens, initial=None) -> np.ndarray:
        belief = self.prior.copy() if initial is None else self._belief(initial).copy()
        for symbol in _numpy_sequence(tokens, self.alphabet):
            belief = self.update(belief, symbol)
        return belief

    def next_probs(self, belief=None) -> np.ndarray:
        belief = self.prior if belief is None else self._belief(belief)
        return np.einsum("i,aij->a", belief, self.edge)

    def log_likelihood(self, tokens, initial=None) -> float:
        belief = self.prior.copy() if initial is None else self._belief(initial).copy()
        log_likelihood = 0.0
        for symbol in _numpy_sequence(tokens, self.alphabet):
            unnormalized = belief @ self.edge[symbol]
            probability = unnormalized.sum()
            log_likelihood += math.log(probability)
            belief = unnormalized / probability
        return log_likelihood

    def score_continuation(self, belief, tokens) -> float:
        return self.log_likelihood(tokens, initial=belief)

    def joint_probs(self, belief=None, horizon: int = 1) -> np.ndarray:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if self.alphabet ** horizon > 1_048_576:
            raise ValueError("joint enumeration exceeds one million outcomes")
        # Keep unnormalized state masses along the complete continuation tree.
        masses = (self.prior if belief is None else self._belief(belief)).copy()
        for _ in range(horizon):
            masses = np.einsum("...i,aij->...aj", masses, self.edge)
        return masses.sum(axis=-1)


def _sequences(values, alphabet: int, name: str) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = [values]
    if not isinstance(values, (Sequence, np.ndarray)) or len(values) == 0:
        raise ValueError(f"{name} must contain at least one sequence")
    return [_numpy_sequence(sequence, alphabet, allow_empty=False) for sequence in values]


def _em_expectation(model: EdgeHMM, sequences: list[np.ndarray]):
    edge_counts = np.zeros_like(model.edge)
    prior_counts = np.zeros_like(model.prior)
    total_log_likelihood = 0.0
    for sequence in sequences:
        length = len(sequence)
        transitions = model.edge[sequence]
        alpha = np.empty((length + 1, model.states), dtype=np.float64)
        beta = np.empty_like(alpha)
        scales = np.empty(length, dtype=np.float64)
        alpha[0] = model.prior
        for time in range(length):
            alpha[time + 1] = alpha[time] @ transitions[time]
            scales[time] = alpha[time + 1].sum()
            alpha[time + 1] /= scales[time]
        beta[-1] = 1.0
        for time in range(length - 1, -1, -1):
            beta[time] = (transitions[time] @ beta[time + 1]) / scales[time]
        prior_counts += alpha[0] * beta[0]
        expected = (alpha[:-1, :, None] * transitions * beta[1:, None, :]
                    / scales[:, None, None])
        np.add.at(edge_counts, sequence, expected)
        total_log_likelihood += np.log(scales).sum()
    return edge_counts, prior_counts, float(total_log_likelihood)


def fit_edge_hmm(train_sequences, validation_sequences, *, alphabet: int = 4,
                 states=(1, 2, 4, 8), restarts: int = 2, iterations: int = 30,
                 seed: int = 0, pseudocount: float = 0.01, penalty: float = 0.5,
                 condition_validation: bool = True) -> tuple[EdgeHMM, dict]:
    """Bounded visible-only NumPy EM; select with held-out suffix loss + penalty.

    By default each validation sequence continues its paired training sequence.
    Selection minimizes validation nats/token + penalty*d*log(Ntrain)/Ntrain,
    d=K*(A*K-1)+(K-1). No refit on development suffixes is performed. Set
    condition_validation=False only for independent validation episodes.
    """
    for name, value in (("alphabet", alphabet), ("restarts", restarts), ("iterations", iterations)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if alphabet < 2:
        raise ValueError("alphabet must be at least two")
    states = tuple(states)
    if not states or len(set(states)) != len(states) or any(
            not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in states):
        raise ValueError("states must be unique positive integers")
    if (not isinstance(pseudocount, (int, float)) or isinstance(pseudocount, bool)
            or not math.isfinite(pseudocount) or pseudocount <= 0):
        raise ValueError("pseudocount must be finite and positive")
    if (not isinstance(penalty, (int, float)) or isinstance(penalty, bool)
            or not math.isfinite(penalty) or penalty < 0):
        raise ValueError("penalty must be finite and nonnegative")
    if not isinstance(condition_validation, bool):
        raise ValueError("condition_validation must be boolean")
    train = _sequences(train_sequences, alphabet, "train_sequences")
    validation = _sequences(validation_sequences, alphabet, "validation_sequences")
    if condition_validation and len(train) != len(validation):
        raise ValueError("conditional validation suffixes must pair with training sequences")
    n_train = sum(map(len, train))
    n_validation = sum(map(len, validation))
    rng = np.random.default_rng(seed)
    best_model, best_key = None, None
    candidates = []
    for state_count in states:
        for restart in range(restarts):
            model = EdgeHMM(rng.gamma(1.0, size=(alphabet, state_count, state_count)) + 1e-8,
                            rng.gamma(1.0, size=state_count) + 1e-8)
            train_trace = []
            for _ in range(iterations):
                edge_counts, prior_counts, train_ll = _em_expectation(model, train)
                train_trace.append(train_ll)
                model = EdgeHMM(edge_counts + pseudocount, prior_counts + pseudocount)
            train_ll = sum(model.log_likelihood(sequence) for sequence in train)
            train_trace.append(train_ll)
            validation_ll = 0.0
            for index, sequence in enumerate(validation):
                initial = model.filter(train[index]) if condition_validation else model.prior
                validation_ll += model.log_likelihood(sequence, initial=initial)
            degrees = state_count * (alphabet * state_count - 1) + state_count - 1
            complexity = penalty * degrees * math.log(max(2, n_train)) / n_train
            validation_nll = -validation_ll / n_validation
            score = validation_nll + complexity
            record = {
                "states": state_count, "restart": restart, "parameters": degrees,
                "train_log_likelihood": train_ll, "validation_log_likelihood": validation_ll,
                "validation_nats_per_token": validation_nll, "complexity_penalty": complexity,
                "selection_score": score, "train_log_likelihood_trace": train_trace,
            }
            candidates.append(record)
            key = (score, state_count, restart)
            if best_key is None or key < best_key:
                best_model, best_key = model, key
    report = {
        "method": "positive_edge_hmm_numpy_em", "seed": seed, "alphabet": alphabet,
        "state_grid": list(states), "restarts": restarts, "iterations": iterations,
        "pseudocount": pseudocount, "penalty_coefficient": penalty,
        "selection_rule": "dev_nats_per_token + penalty*d*log(Ntrain)/Ntrain",
        "condition_validation_on_paired_train_prefix": condition_validation,
        "refit_on_validation": False, "train_tokens": n_train, "validation_tokens": n_validation,
        "selected_states": best_model.states, "selected_restart": best_key[2],
        "candidates": candidates,
    }
    return best_model, report
