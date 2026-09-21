# Episodic model reference for EXP-0012

Implementation date: 2026-09-21. This document describes code and engineering tests,
not scientific results. Training, held-out evaluation, resource limits and promotion
criteria belong to the experiment registration. No manuscript decoding is implied.

## Reviewed rationale and limits

The reviewed sources and reading depth are recorded in
[ARCHITECTURES.md](deep-review-2026-09-21/ARCHITECTURES.md),
[THEORY.md](deep-review-2026-09-21/THEORY.md),
[NEXT_DESIGN.md](deep-review-2026-09-21/NEXT_DESIGN.md), and their source ledgers.
The present implementation follows that comparison: an existing compact transformer,
a nonlinear recurrent control, a signed product recurrence, and a separate explicit
edge-emitting probabilistic model. Their assumptions can fail independently.

- Prior Voynich work includes a two-state HMM approximately recovering the Currier
  vocabulary division in [Reddy and Knight (2011)](https://aclanthology.org/W11-1511.pdf),
  as reviewed in [PRIOR_WORK.md](PRIOR_WORK.md). A useful predictive state or familiar
  vocabulary partition is therefore neither a new semantic discovery nor decipherment.
- [DeltaProduct, source A12](https://arxiv.org/html/2502.10297v3) motivates testing
  several different generalized Householder factors per symbol. Expressivity and its
  synthetic task results do not establish learnability from unknown ciphertext. Our
  vector recurrence below is a **simplified local adaptation**, not a replication of
  its full architecture, parameterization, training recipe or optimized kernels.
- The explicit symbol matrices implement the elementary edge-emitting equations in
  THEORY.md. This does not import the ordinary node-emitting HMM identifiability
  theorem reviewed as [T04](https://alice.cleynen.fr/wp-content/uploads/2015/03/GCR_2015.pdf).
  State permutations and observationally equivalent processes remain indistinguishable.
- First-occurrence canonicalization removes arbitrary bijective symbol names; its
  NEW event assumes a fixed known alphabet and an exchangeable prior over unseen
  names. It does not remove ambiguity from homophony, omissions or segmentation.

## Neural interface

`src/voynich/episodic_models.py` exports immutable `ModelSpec` and `build_model`.
Defaults are `kind="transformer"`, `alphabet=4`, `canonical=False`, `width=64`,
`layers=2`, `heads=4`, `ff_width=128`, `context=256`, and `dropout=0.0`.
The experiment must record the actual configuration and parameter count.

Every model accepts integer `tokens[B,T]`, with observed/raw or canonical ranks in
`0..A-1`, and returns logits `[B,T,O]`. Position t predicts the symbol following
input t. `O=A` for raw models and `O=A+1` for canonical models. No latent labels,
true state counts or hidden generator parameters are accepted by this interface.

| Model | Full sequence | Exposed persistent state | Notes |
| --- | --- | --- | --- |
| `transformer` | Existing causal `VoynichTransformer` | Token prefix through existing backbone | Vocabulary A+1, padding ID A; A is never an input here. Raw readout discards output A. Maximum context is enforced. |
| `gru` | Native `torch.nn.GRU` | `[B,layers,width]` | Uses the same native GRU weights in sequence and single-step paths. |
| `signed_delta` | Sequential native PyTorch operations | `[B,layers,width]` | Two independent input-conditioned key functions per layer, convex write and nonlinear interlayer/readout transformations. |

Recurrent models expose:

- `initial_state(batch, device=None)`: zero state with the model's floating dtype.
- `encode(prefix[B,T])`: state after the prefix, including an empty prefix.
- `readout(state)`: next-symbol logits from the final layer via `head(norm(state))`.
- `step(symbol[B], state)`: `(next_logits, updated_state)` after observing the symbol.
- `spec`, `parameter_count`, `embedding`, `norm`, and linear `head` for inspection.

Methods preserve autograd and do not detach or mutate the supplied state. The state
must match model device and dtype. Copying it with `clone()` supports exact replay.
Recurrent sequence length is not limited by `context`; this enables declared length
extrapolation tests. `context` limits only the transformer. Single-step/full-sequence
identity is guaranteed in evaluation mode or with zero dropout, as registered here;
stochastic interlayer dropout need not draw identical masks in both paths.

Canonical recurrent state alone is insufficient for decoding probabilities: its
symbol map and count must travel with it. Raw recurrent models are used for the
current selective state intervention interface to avoid silently omitting that state.

## Signed product update

For layer input x, each independently parameterized key function yields a normalized
vector `k_i(x)`, and a gate gives `beta_i(x) = 2 sigmoid(b_i(x))`. Define
`H_i(x) = I - beta_i(x) k_i(x) k_i(x)^T`. The implementation computes the two
rank-one operations without constructing dense matrices:

`h_after = r(x) H_2(x) H_1(x) h_before + (1-r(x)) tanh(W_write x + b_write)`.

The input is layer-normalized, `r(x)` is a scalar sigmoid gate, and a tanh of each
new layer state supplies the next layer. The final readout uses layer normalization
and a linear projection. Each H has singular values bounded by one for beta in
[0,2], including possible negative eigenvalues; the convex write bounds the new
state norm by the larger of the old-state and write norms. This is a numerical
stability property, not a proof of useful state tracking.

`model.cells[layer].factors(inputs)` returns `keys[...,2,width]`, `beta[...,2]`,
`retain[...,1]`, and `write[...,width]`. The two key networks have separate weights;
they are not repeated applications of one shared key. Training can nevertheless
make their directions coincide, which an analysis must measure rather than rule
out by assumption. There are no CUDA, Triton, custom scan, or paper-specific kernels.
This simple reference may be slower on MPS than the native GRU or parallel transformer.

## Canonical probabilities

`raw_log_probs(logits[...,A+1], counts[...], inverse[...,A])` is a differentiable
PyTorch adapter. `counts` records the distinct symbols seen **through the current
input**. `inverse[...,rank]` gives that rank's raw symbol ID; unassigned ranks contain
`-1`. Counts and maps are supplied by the causal data encoder; the adapter does not
inspect future tokens or recanonicalize any sequence.

Ranks at or above the count are masked before log-softmax. Output A means NEW and
is disabled after all symbols have appeared. An observed raw symbol receives its
rank probability. Each unobserved raw symbol receives `P(NEW)/(A-count)`.
At count zero the raw distribution is uniform. At count A it is a permutation of
the rank distribution. Invalid counts, duplicate raw IDs, holes, wrong shapes,
mixed devices, noninteger maps, and nonfinite logits raise errors. No post-hoc
renormalization hides malformed maps. The implementation preserves exact renaming
equivariance up to floating-point operations and finite input-logit gradients.

## Explicit edge HMM and visible-only fitting

`EdgeHMM(edge[A,K,K], prior[K])` accepts finite strictly positive weights and
normalizes `sum_(a,j) edge[a,i,j] = 1` and the prior to unit mass. Its float64 NumPy
reference exposes `filter(prefix, initial=None)`, `update(belief, symbol)`,
`next_probs(belief=None)`, `log_likelihood(sequence, initial=None)`,
`score_continuation(belief, symbols)`, and `joint_probs(belief=None, horizon=1)`.
Log likelihood is a natural-log scalar for a single one-dimensional sequence.
Empty sequences preserve the prior/belief and have log likelihood zero. Beliefs
must already be normalized and nonnegative. Complete joint trees are bounded to
1,048,576 outcomes to prevent accidental exponential memory growth.

For observed symbol a, the posterior update is `b M_a / (b M_a 1)`; joint sequence
probability is `b M_a1 ... M_ah 1`. Joint enumeration retains unnormalized state
masses along every branch. The forward likelihood and EM recursion rescale after
every observation to avoid multiplying the full sequence probability directly.

`fit_edge_hmm(train_sequences, validation_sequences, *, alphabet=4,
states=(1,2,4,8), restarts=2, iterations=30, seed=0, pseudocount=0.01,
penalty=0.5, condition_validation=True)` returns `(model, report)`.

The default protocol treats each validation sequence as a **continuation suffix of
its paired training sequence**; the lists must have equal length. EM sees only
training observations. Each validation suffix starts from the candidate's filtered
training-prefix belief. Explicit `condition_validation=False` instead scores
independent validation episodes from the fitted prior. This choice is recorded.

Each restart uses a seeded positive gamma initialization. The fixed number of
Baum–Welch iterations updates expected edge and initial-state counts with the
declared additive pseudocount. No hidden state labels, oracle counts or generator
parameters are read. Candidates minimize

`validation_nats_per_token + penalty * d * log(max(2,Ntrain)) / Ntrain`,

where `d = K*(A*K-1) + (K-1)` and Ntrain counts visible training symbols. This is
our frozen **BIC-shaped selection regularizer**, not an asserted Bayesian evidence
calculation or an identified true state count. Lower K then lower restart index
break exact ties. The returned model is not refitted on validation. The report
records all candidates, train likelihood traces, suffix scores, parameter counts,
penalty, seeds and selection settings. Pseudocount MAP-style updates need not
monotonically improve unregularized training likelihood.

The one-state candidate is the smoothed IID comparator. A short prefix and this
penalty may favor it even when the generating process is more complex. Report that
as finite-data/selection behavior; do not interpret selected K as historical truth.
Rectangular arrays and lists of unequal-length sequences are supported. No paid
services or external fitting libraries are required.

## Validation scope

`tests/test_episodic_models.py` checks output shapes, causal suffix invariance,
finite gradients, step/sequence equivalence, empty-prefix replay, rank/NEW masking,
map rejection, symbol-renaming equivariance, canonical autograd finite differences,
explicit Householder multiplication and norm bounds, manual HMM path summation,
normalization of complete futures, state relabeling, reproducible EM, K=1 smoothed
counts, and prevention of validation-to-EM leakage. CPU/MPS comparison tests are
included and skip explicitly when Metal is unavailable to the current process.

These tests validate implementation contracts. They do not validate fresh-key
inference, causal-variable recovery, manuscript relevance or decoding.
