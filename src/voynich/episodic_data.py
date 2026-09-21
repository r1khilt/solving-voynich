"""Fresh task-seeded edge processes and strictly prefix-causal symbol encoding.

Synthetic ground truth lives in task dictionaries; predictors receive only sampled
symbols. These deliberately small processes are controls, not Voynich models.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

SCHEMA_VERSION = 1
TRAIN_FAMILIES = ("cycle", "branch", "pair_parity", "iid")
HELDOUT_FAMILIES = ("rr_xor", "switching")
FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _stamp(task):
    """Hash model parameters, independent of optional diagnostic provenance."""
    fields = ("schema_version", "family", "alphabet", "edge", "prior", "params")
    encoded = json.dumps({key: task[key] for key in fields}, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    task["task_id"] = hashlib.sha256(encoded).hexdigest()
    return task


def _stationary(transition):
    """Solve for a stationary prior; all generated families have a unique one."""
    states = len(transition)
    equations = transition.T - np.eye(states)
    equations[-1] = 1
    target = np.zeros(states)
    target[-1] = 1
    prior = np.linalg.solve(equations, target)
    if np.any(prior < -1e-12):
        raise ValueError("invalid stationary solution")
    prior = np.maximum(prior, 0)
    return prior / prior.sum()


def make_task(family: str, seed: int, alphabet: int = 4) -> dict:
    """Create one independently parameterized JSONable task.

    ``edge[a,i,j]`` is P(emission=a,next state=j | current state=i).
    Seeds determine both a symbol permutation and continuous nuisance parameters.
    Four symbols are deliberate: joint futures through horizon four are enumerable.
    Priors are stationary, preventing fixed episode position from revealing phase.
    """
    seed = _integer(seed, "seed")
    alphabet = _integer(alphabet, "alphabet", 2)
    if alphabet != 4:
        raise ValueError("this registered task suite requires alphabet=4")
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(alphabet)
    params = {"emission_permutation": permutation.tolist()}

    if family == "iid":
        probabilities = rng.dirichlet(np.full(alphabet, 2.0))
        edge = probabilities[:, None, None]
        params["symbol_probabilities"] = probabilities.tolist()
    elif family in {"cycle", "branch"}:
        transition = np.zeros((4, 4))
        if family == "cycle":
            progress = rng.uniform(0.65, 0.95, 4)
            for state in range(4):
                transition[state, state] = 1 - progress[state]
                transition[state, (state + 1) % 4] = progress[state]
            params["progress"] = progress.tolist()
        else:
            branch = float(rng.uniform(0.25, 0.75))
            dwell = rng.uniform(0.03, 0.20, 4)
            transition[0, 1:3] = [branch, 1 - branch]
            transition[1:3, 3] = 1
            transition[3, 0] = 1
            transition *= (1 - dwell[:, None])
            transition[np.arange(4), np.arange(4)] = dwell
            params.update(branch_probability=branch, dwell=dwell.tolist())
        noise = float(rng.uniform(0.04, 0.20))
        background = rng.dirichlet(np.full(alphabet, 2.0))
        emission = (1 - noise) * np.eye(alphabet) + noise * background[None]
        edge = np.einsum("ia,ij->aij", emission, transition)
        params.update(emission_noise=noise, background=background.tolist())
    elif family in {"pair_parity", "rr_xor"}:
        # Each canonical symbol encodes (binary signal, independent binary tag).
        tag_probability = float(rng.uniform(0.20, 0.80))
        noise = float(rng.uniform(0.01, 0.09))
        tags = [1 - tag_probability, tag_probability]
        params.update(tag_probability=tag_probability, parity_noise=noise)
        if family == "pair_parity":
            # 0/1: ready with parity c; 2+2*c+x: remembered first bit x.
            edge = np.zeros((4, 6, 6))
            switch = float(rng.uniform(0.025, 0.16))
            params["mode_switch_probability"] = switch
            params["state_layout"] = "ready(c)=c; remember(c,x)=2+2*c+x"
            for parity in range(2):
                for bit in range(2):
                    memory = 2 + 2 * parity + bit
                    for tag, tag_weight in enumerate(tags):
                        edge[2 * bit + tag, parity, memory] += 0.5 * tag_weight
                        for flip in range(2):
                            symbol = 2 * (bit ^ parity ^ flip) + tag
                            weight = tag_weight * (noise if flip else 1 - noise)
                            edge[symbol, memory, parity] += weight * (1 - switch)
                            edge[symbol, memory, 1 - parity] += weight * switch
        else:
            # 0: ready; 1+x: one bit; 3+2*x+y: two remembered bits.
            edge = np.zeros((4, 7, 7))
            params["state_layout"] = "ready=0; first(x)=1+x; pair(x,y)=3+2*x+y"
            for bit in range(2):
                for tag, tag_weight in enumerate(tags):
                    edge[2 * bit + tag, 0, 1 + bit] += 0.5 * tag_weight
                    for first in range(2):
                        edge[2 * bit + tag, 1 + first, 3 + 2 * first + bit] += 0.5 * tag_weight
                        memory = 3 + 2 * first + bit
                        for flip in range(2):
                            symbol = 2 * (first ^ bit ^ flip) + tag
                            edge[symbol, memory, 0] += tag_weight * (noise if flip else 1 - noise)
    else:  # switching: slowly changing hidden regime, with full emission support.
        switches = rng.uniform(0.025, 0.13, 2)
        transition = np.array([[1 - switches[0], switches[0]],
                               [switches[1], 1 - switches[1]]])
        tag_probability = rng.uniform(0.20, 0.80, 2)
        noise = float(rng.uniform(0.04, 0.18))
        emission = np.zeros((2, 4))
        for state in range(2):
            for bit in range(2):
                for tag in range(2):
                    emission[state, 2 * bit + tag] = (1 - noise if bit == state else noise) * (
                        tag_probability[state] if tag else 1 - tag_probability[state])
        edge = np.einsum("ia,ij->aij", emission, transition)
        params.update(switch_probabilities=switches.tolist(),
                      tag_probabilities=tag_probability.tolist(), emission_noise=noise)

    renamed = np.empty_like(edge)
    renamed[permutation] = edge
    prior = _stationary(renamed.sum(axis=0))
    task = {"schema_version": SCHEMA_VERSION, "family": family, "seed": seed,
            "alphabet": alphabet, "edge": renamed.tolist(), "prior": prior.tolist(), "params": params}
    _task_arrays(task)
    return _stamp(task)


def _task_arrays(task):
    alphabet = _integer(task["alphabet"], "alphabet", 2)
    edge = np.asarray(task["edge"], dtype=np.float64)
    prior = np.asarray(task["prior"], dtype=np.float64)
    if prior.ndim != 1 or len(prior) == 0 or edge.shape != (alphabet, len(prior), len(prior)):
        raise ValueError("expected edge[A,K,K] and prior[K]")
    if not np.isfinite(edge).all() or not np.isfinite(prior).all() or (edge < 0).any() or (prior < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(edge.sum(axis=(0, 2)), 1, rtol=0, atol=1e-10) or not np.isclose(
            prior.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("process probabilities must be normalized")
    return edge, prior


def sample_tasks(tasks: list[dict], length: int, seed: int) -> np.ndarray:
    """Draw one stream per task; draws are vectorized over a padded batch."""
    length = _integer(length, "length")
    seed = _integer(seed, "seed")
    if not tasks:
        return np.empty((0, length), dtype=np.int64)
    arrays = [_task_arrays(task) for task in tasks]
    alphabet = arrays[0][0].shape[0]
    if any(edge.shape[0] != alphabet for edge, _ in arrays):
        raise ValueError("all tasks in a batch must share the alphabet")
    batch = len(tasks)
    states = max(len(prior) for _, prior in arrays)
    initial = np.zeros((batch, states))
    tables = np.zeros((batch, states, alphabet * states))
    for row, (edge, prior) in enumerate(arrays):
        count = len(prior)
        padded = np.zeros((states, alphabet, states))
        padded[:count, :, :count] = edge.transpose(1, 0, 2)
        # Unreachable padded states are valid absorbing rows for a total CDF.
        padded[count:, 0, count:] = np.eye(states - count)
        tables[row] = padded.reshape(states, -1).cumsum(axis=-1)
        initial[row, :count] = prior
    # Avoid a rounding gap at the final CDF entry.
    tables[..., -1] = 1
    initial = initial.cumsum(axis=-1)
    initial[:, -1] = 1
    rng = np.random.default_rng(seed)
    current = (rng.random(batch)[:, None] >= initial).sum(axis=1)
    visible = np.empty((batch, length), dtype=np.int64)
    rows = np.arange(batch)
    for t in range(length):
        chosen = (rng.random(batch)[:, None] >= tables[rows, current]).sum(axis=1)
        visible[:, t], current = chosen // states, chosen % states
    return visible


def sample_task(task: dict, n: int, length: int, seed: int) -> np.ndarray:
    """Draw n independent streams from one fixed task."""
    n = _integer(n, "n")
    length = _integer(length, "length")
    seed = _integer(seed, "seed")
    edge, prior = _task_arrays(task)
    states = len(prior)
    initial = prior.cumsum()
    initial[-1] = 1
    table = edge.transpose(1, 0, 2).reshape(states, -1).cumsum(axis=-1)
    table[:, -1] = 1
    rng = np.random.default_rng(seed)
    current = (rng.random(n)[:, None] >= initial).sum(axis=1)
    visible = np.empty((n, length), dtype=np.int64)
    for t in range(length):
        chosen = (rng.random(n)[:, None] >= table[current]).sum(axis=1)
        visible[:, t], current = chosen // states, chosen % states
    return visible


def _tokens(tokens, alphabet, ndim):
    value = np.asarray(tokens)
    if value.ndim != ndim:
        raise ValueError(f"tokens must have {ndim} dimensions")
    if value.size and (value.dtype.kind not in "iu" or value.min() < 0 or value.max() >= alphabet):
        raise ValueError("token IDs must be integers in [0, alphabet)")
    return value.astype(np.int64, copy=False)


def _filter(edge, prior, prefix):
    belief = prior.copy()
    for symbol in prefix:
        belief = belief @ edge[symbol]
        probability = belief.sum()
        if probability <= 0:
            raise ValueError("prefix has zero probability under this task")
        belief /= probability
    return belief


def oracle_belief(task: dict, prefix) -> np.ndarray:
    """Return hidden-state belief after all prefix emissions, using scaled updates."""
    edge, prior = _task_arrays(task)
    return _filter(edge, prior, _tokens(prefix, edge.shape[0], 1))


def oracle_next(task: dict, prefix) -> np.ndarray:
    """Return the normalized next-symbol probabilities after a prefix."""
    return oracle_joint(task, prefix, 1)


def oracle_joint(task: dict, prefix, horizon: int) -> np.ndarray:
    """Enumerate future strings lexicographically: index=sum(a_t*A**(h-1-t)).

    Horizon zero is the unit mass on the empty string. Enumeration is intentionally
    bounded to one million strings to reject accidental exponential allocations.
    """
    horizon = _integer(horizon, "horizon")
    edge, prior = _task_arrays(task)
    alphabet = edge.shape[0]
    if alphabet ** horizon > 1_000_000:
        raise ValueError("joint enumeration exceeds one million continuations")
    weights = _filter(edge, prior, _tokens(prefix, alphabet, 1))[None]
    for _ in range(horizon):
        weights = np.einsum("ni,aij->naj", weights, edge).reshape(-1, len(prior))
    probabilities = weights.sum(axis=-1)
    return probabilities / probabilities.sum()


def rename_task(task: dict, permutation) -> dict:
    """Return the same hidden process under raw-symbol renaming old->new."""
    edge, _ = _task_arrays(task)
    alphabet = edge.shape[0]
    permutation = _tokens(permutation, alphabet, 1)
    if len(permutation) != alphabet or len(np.unique(permutation)) != alphabet:
        raise ValueError("permutation must contain every raw symbol exactly once")
    result = copy.deepcopy(task)
    renamed = np.empty_like(edge)
    renamed[permutation] = edge
    result["edge"] = renamed.tolist()
    if "emission_permutation" in result["params"]:
        old = np.asarray(result["params"]["emission_permutation"], dtype=np.int64)
        result["params"]["emission_permutation"] = permutation[old].tolist()
    else:
        result["params"]["renaming"] = permutation.tolist()
    return _stamp(result)


def equivalent_state_split(task: dict, state: int = 0, weight: float = 0.5) -> dict:
    """Duplicate a hidden state without changing any observable probability.

    Incoming probability and prior mass split by weight. Both copies have the
    original outgoing law, with every arrival to the split state split again.
    """
    edge, prior = _task_arrays(task)
    state = _integer(state, "state")
    if state >= len(prior) or not np.isfinite(weight) or not 0 < weight < 1:
        raise ValueError("split requires a valid state and weight strictly between zero and one")
    old_index = np.r_[np.arange(len(prior)), state]
    factors = np.ones(len(prior) + 1)
    factors[state], factors[-1] = weight, 1 - weight
    result = copy.deepcopy(task)
    result["edge"] = (edge[:, old_index][:, :, old_index] * factors[None, None]).tolist()
    result["prior"] = (prior[old_index] * factors).tolist()
    result["params"]["equivalent_split"] = {"source_task_id": task.get("task_id"),
                                              "state": state, "weight": float(weight)}
    _task_arrays(result)
    return _stamp(result)


def canonicalize(tokens: np.ndarray, alphabet: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign symbol ranks by first occurrence, separately and causally per row.

    Outputs: canonical[B,T], counts[B,T], inverse[B,T,A], all int64. At t the
    snapshot includes token[t]. inverse[...,rank] maps rank->raw ID; -1 is unseen.
    Neither the mapping nor NEW eligibility consults a later token.
    """
    alphabet = _integer(alphabet, "alphabet", 2)
    tokens = _tokens(tokens, alphabet, 2)
    batch, length = tokens.shape
    canonical = np.empty_like(tokens)
    counts = np.empty_like(tokens)
    inverse = np.empty((batch, length, alphabet), dtype=np.int64)
    raw_to_rank = np.full((batch, alphabet), -1, dtype=np.int64)
    rank_to_raw = np.full((batch, alphabet), -1, dtype=np.int64)
    count = np.zeros(batch, dtype=np.int64)
    rows = np.arange(batch)
    for t in range(length):
        raw = tokens[:, t]
        rank = raw_to_rank[rows, raw]
        fresh = rank < 0
        rank[fresh] = count[fresh]
        raw_to_rank[rows[fresh], raw[fresh]] = rank[fresh]
        rank_to_raw[rows[fresh], rank[fresh]] = raw[fresh]
        count += fresh
        canonical[:, t], counts[:, t], inverse[:, t] = rank, count, rank_to_raw
    return canonical, counts, inverse


def canonical_targets(tokens: np.ndarray, alphabet: int = 4) -> np.ndarray:
    """Next-symbol event targets [B,T-1]; NEW=A for symbols unseen at t."""
    canonical, counts, _ = canonicalize(tokens, alphabet)
    return np.where(canonical[:, 1:] < counts[:, :-1], canonical[:, 1:], alphabet).astype(np.int64)


def canonical_raw_probs(logits, counts, inverse) -> np.ndarray:
    """Reference conversion from A seen-rank events + NEW to raw probabilities.

    Invalid ranks and NEW when all symbols are seen are masked BEFORE softmax.
    NEW mass is shared equally over remaining raw IDs. Any leading shape is valid.
    """
    logits = np.asarray(logits, dtype=np.float64)
    counts = np.asarray(counts)
    inverse = np.asarray(inverse)
    if logits.ndim < 1 or logits.shape[-1] < 3:
        raise ValueError("logits must end in A+1 event classes")
    alphabet = logits.shape[-1] - 1
    shape = logits.shape[:-1]
    if counts.shape != shape or inverse.shape != (*shape, alphabet):
        raise ValueError("counts/inverse shapes must match logits")
    if counts.dtype.kind not in "iu" or inverse.dtype.kind not in "iu":
        raise ValueError("counts and inverse must contain integers")
    if (counts < 0).any() or (counts > alphabet).any():
        raise ValueError("counts must be between zero and alphabet")
    ranks = np.arange(alphabet)
    valid_ranks = ranks < counts[..., None]
    if ((inverse < 0) & valid_ranks).any() or ((inverse >= alphabet) & valid_ranks).any() or (
            (inverse != -1) & ~valid_ranks).any():
        raise ValueError("inverse must hold seen raw IDs then -1 padding")
    sorted_inverse = np.sort(np.where(valid_ranks, inverse, alphabet), axis=-1)
    if ((sorted_inverse[..., 1:] == sorted_inverse[..., :-1]) & (
            sorted_inverse[..., 1:] < alphabet)).any():
        raise ValueError("inverse raw IDs must be unique")
    valid_events = np.concatenate([valid_ranks, (counts < alphabet)[..., None]], axis=-1)
    if not np.isfinite(logits[valid_events]).all():
        raise ValueError("valid event logits must be finite")
    masked = np.where(valid_events, logits, -np.inf)
    weights = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
    events = weights / weights.sum(axis=-1, keepdims=True)
    mapping = (inverse[..., :, None] == ranks) & valid_ranks[..., :, None]
    seen = mapping.any(axis=-2)
    raw = np.einsum("...r,...ra->...a", events[..., :alphabet], mapping)
    raw += (~seen) * (events[..., -1] / np.maximum(alphabet - counts, 1))[..., None]
    return raw
