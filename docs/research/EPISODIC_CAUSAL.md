# EXP-0012 predictive-state causal battery

Implementation specification, 2026-09-21. This file describes an implemented
analysis and its limits, not a scientific result. Campaign registration and
results determine whether any trained predictor qualifies for analysis.

## Question and evidence level

Can a small, fixed subspace transfer a frozen recurrent predictor's **joint
future distribution**, preserve its immediate prediction when donor and
recipient agree there, and continue behaving coherently after a new symbol?

The implemented target is a donor predictor distribution. It does **not** define
an intervention on a historically meaningful latent variable. A good score
would establish a useful predictive-state transplant under the specified tests;
it would not recover the simulator's unique algorithm, an identifiable hidden
state, or a Voynich decoding. The report always sets `causal_selectivity_claim`
to false. A stronger claim needs a separately registered latent counterfactual
and an independent nuisance variable that this fitting objective never uses.

This distinction follows the failures of supervised probe steering in EXP-0004,
the output-span control in EXP-0006, the unfamiliar-key failures of EXP-0008,
and the broad-versus-selective distinction in EXP-0009. The rationale and source
reading records are in [INTERPRETABILITY.md](deep-review-2026-09-21/INTERPRETABILITY.md),
[THEORY.md](deep-review-2026-09-21/THEORY.md),
[NEXT_DESIGN.md](deep-review-2026-09-21/NEXT_DESIGN.md), and the earlier
[Voynich/decipherment review](PRIOR_WORK.md). The current implementation is a
small synthetic analysis; it is not a replication of frontier-model circuit
tracing or a direct Voynich decipherment method.

The orthogonal interchange parameterization is motivated by the DAS/causal
abstraction literature recorded as M01–M04 in the interpretation ledger. Exact
joint tests and independent update closure follow the predictive-state concerns
in T07 and the explicit state equations in the theory review. Concrete
disagreement strings follow the counterexample emphasis of M20. These are
inherited ideas, not novelty claims. The nullspace critique and reply remain
contested; neither one steering metric nor a neat projection decides faithfulness.

## Complete state contract

`src/voynich/episodic_causal.py` supports the raw-symbol recurrent models:

```python
model.encode(prefix_batch)       # [B,T] integer IDs -> full state [B,L,H]
model.readout(state)             # [B,A] next-symbol logits
model.step(symbols, state)       # consume [B] symbols -> (next_logits, next_state)
```

No future is supplied to `encode`. Exact future enumeration records each
symbol's probability before consuming that symbol. The returned distribution is
in lexicographic continuation order and has shape `[B,A**h]`; it is normalized
by the autoregressive construction, with normalization error measured rather
than hidden by a final renormalization. Enumeration remains differentiable.

The audit accepts horizons 2–4. With A=4, H=3 and batch size 32, the largest
frontier during one enumeration has 512 recurrent states. H=4 raises this to
2,048. Basis QR, PCA/SVD, and ridge solves use CPU paths for MPS compatibility;
the backbone stays on its existing device. Raw recurrent state is complete for
future computation. A canonical model also needs its symbol-map side state:
passing only its neural tensor is invalid and rejected by the output contract.

## Isolation and pairing

Three prefix pools are required: fit, validation, confirmation. Exact duplicate
prefixes within or across pools are rejected. Explicit generator/key group labels must be
disjoint across pools, and each key must have at least two contexts. Pairing is
always within a key. If labels are absent, the caller asserts one key per pool;
the report clearly marks key independence as unverified. Task identifiers must
refer to actual generator/key identity rather than row indexes or invented
split prefixes for reused keys. The caller is responsible for truthful labels
and fresh source keys. Hashes of the exact prefixes and paired indexes are saved.

Pair selection uses only the observations and frozen model predictions:

1. Candidate donors share the recipient's key and are distinct contexts.
2. Strictly matched donors also share the final observed symbol and have
   immediate-distribution total variation at most 0.05 by default.
3. Among strict matches, select the largest joint KL minus first-symbol KL.
   This targets a disagreement in conditional future predictions.
4. If no strict match exists, select the closest immediate prediction, preferring
   the same final symbol, and mark this row **unmatched**. Report it in the broad
   aggregate, never silently in the matched aggregate.
5. A wrong/near-equivalent donor is the same-key context closest in joint KL to
   the recipient. Report when it coincides with the intended donor; with tiny or
   behaviorally uniform pools this control can be uninformative.

Matching on model predictions establishes a teacher-defined comparison. It does
not establish matched true generator beliefs. Very weak or untrained teachers
can have nearly uniform distributions, so all donors can satisfy the immediate
match while exhibiting no meaningful later difference. The unchanged-recipient
conditional KL and the campaign's teacher-accuracy gate must expose this case.

## Fitting and controls

Flatten the complete state to dimension D. A trainable D-by-r matrix is converted
to an orthonormal basis Q by differentiable QR. Interchange is

`recipient + ((donor - recipient) Q) Q.T`.

Adam fits Q against donor joint KL using only fit contexts and frozen teacher
probabilities. It never sees generator state labels, beliefs, nuisance labels,
oracle probabilities, or confirmation data. Initial and five fixed intermediate
checkpoints are scored on validation; validation selects the lowest donor KL.
The rank, horizon, fit budget and matching threshold are caller-specified and
must be frozen by registration. The confirmation set is scored only after
selection. There is no automatic pass/fail rule in this analysis library.

All confirmation comparisons use the same recipient/donor pairs:

| Comparison | Meaning and limitation |
| --- | --- |
| Unchanged recipient | Available teacher-distribution gap; zero commuting error here is trivial. |
| Full donor state | Positive reproduction control, not selective evidence. |
| Learned Q | Candidate predictive-state transplant. |
| Shuffled target fit | Same optimizer/steps/initial basis; donor distribution assignments are shuffled within key independently in fit and validation. Confirmation uses the intended donor. Actual changed-assignment fractions are reported. |
| PCA | Same dimension, fit-state covariance directions only. |
| Output Jacobian | Same dimension, leading right-singular directions of centered readout Jacobians on at most 16 fit states. LayerNorm is included. Effective rank and null-space completion are disclosed. |
| Future Jacobian | Same dimension, leading directions of a bounded fit-only Fisher sensitivity sketch of joint futures through horizon 3. |
| Delayed Jacobian | Same dimension, future sensitivity inside the complement of the sampled rank-r output-control span. This is not the complete nullspace of immediate behavior. |
| Three random subspaces | Same dimension, independent frozen seeds, each displacement norm matched per pair to the learned intervention. This scaling is a perturbation control, not exact coordinate interchange. |
| Untrained backbone | A caller-supplied independently initialized model repeats the entire fit/control/confirmation pipeline on the same pools. Missing control is explicitly recorded. |

Random projections with exactly zero displacement have no direction to rescale;
the implementation retains zero and reports norm-matching error. It does not
manufacture a direction outside the declared random span.

The untrained model must have the same architecture and interface. The caller
controls initialization and records its seed. The library does not guess which
architecture-specific parameters a generic reset method would reinitialize.
Untrained models select their own behavior-matched pairs; their pair hashes and
available donor gaps are reported, so their raw KL magnitude is not directly
comparable to a trained teacher's without its baseline gap.

## Bounded future sensitivity candidates

These two additional candidates are chosen without validation or confirmation
scores. The construction uses at most eight fit states at evenly spaced indexes
through the supplied fit pool, eight independent deterministic Rademacher sign
vectors per state, and exact joint futures at horizon 3. The fit indexes and seed
(`audit seed + 201`) are recorded. This costs at most 64 input-gradient queries
per backbone and uses no hidden labels, oracle probabilities, or final data.

For each selected state h and independent sign vector r, form

`g = gradient_h sum_w r[w] * stop_gradient(sqrt(p(w|h))) * log p(w|h)`.

The square-root weights are detached: conditional on h, the expected outer
product of g is the Fisher matrix `sum_w p(w|h) grad logp(w|h) grad logp(w|h).T`.
Differentiating the probability weight too would yield another operator. Rows
are scaled by the square root of their count, then SVD gives the rank-r
`future_jacobian` basis. This is a finite randomized approximation; it need not
find all informative directions or be stable with eight states.

For `delayed_jacobian`, compute an explicit orthonormal complement of the same
rank-r `output_jacobian` control basis, project the sketch into that complement,
and take its leading rank-r directions. Computing SVD inside the complement
prevents numerical residuals from reintroducing excluded directions. If fewer
than r complementary dimensions exist, the delayed candidate is skipped with
an explicit reason. No extra model gradients are needed for this candidate.

The excluded span includes only the previously sampled output-control directions
and may include arbitrary null-space completions when that control has lower
effective rank. It does **not** remove all immediate sensitivity at all states,
and finite nonlinear patches can change immediate outputs even along a locally
orthogonal direction. An infinitesimal gradient is **not** a causal mechanism.
The same finite interchange/update/preservation battery evaluates both candidates.

The report records effective sketch rank, any arbitrary null completion,
captured sketch energy, leading singular values, delayed/output span overlap,
and subspace overlap between the first and second halves of the sketch rows.
That split-half value is a descriptive sampling-stability diagnostic, not a
confidence interval or independent model-seed stability. Rank deficiency can
make null completions look artificially stable, so effective rank accompanies it.

## Confirmation metrics

For every mode, report all-pair and strictly matched-pair results separately:

- Donor KL in bits for each prefix horizon 1 through H.
- Joint KL(H) minus KL(1), the donor-weighted conditional divergence beyond the
  first symbol. This prevents first-symbol improvement from hiding later failure.
- Total variation from the recipient's immediate prediction. On matched pairs
  this is a limited observable preservation test, not an independent semantic
  nuisance test.
- State displacement norm and error against the requested matched norm.
- Donor KL after each possible new symbol, weighted by the donor's original
  next-symbol probabilities, over a further horizon H−1.
- Symmetric KL between **patch then consume a** and **consume a in recipient and
  donor then patch**. The original backbone performs both updates, with no patch
  to future positions. This is a finite fixed-subspace commuting test.
- Joint KL from the recipient after the near-equivalent/wrong donor transplant.
  It is interpreted against the full-wrong-donor control and its available gap,
  since the selected wrong donor need not be truly equivalent.

Summary rows include context-weighted means, equal-key means, key counts,
per-key means and row counts keyed by actual group ID, and key minima/maxima.
They are descriptive; no confidence interval is fabricated
from correlated contexts. Independent-key uncertainty and cross-model/seed
stability belong to campaign aggregation. A maximum-discrepancy continuation
witness records the paired indexes, exact future symbols, and both probabilities
for replay. It is a counterexample to exact donor matching, not a selected
representative success.

The commuting criterion is deliberately stringent and narrow. A correct causal
variable can move between coordinates as time advances; that representation can
fail a *fixed* Q test. Failure rejects this fixed-subspace account, not every
possible changing or nonlinear state abstraction. Conversely, unchanged and
full-donor controls commute for trivial reasons. Commuting error must be read
together with donor fidelity, rank, preservation, and the comparison controls.

## Independent descriptive update closure

For each fixed basis, train separate per-symbol affine maps

`z_after_a ≈ [z_before, 1] @ W_a`, where `z = flattened_state @ Q`.

The maps use only fit states, actual backbone updates after every possible
symbol, and ridge regularization 0.001 (unpenalized bias). They do not participate
in learning Q and no identity is imposed on them by construction. On confirmation
states, score coordinate MSE and MSE relative to a fit-mean coordinate baseline.
Also reconstruct states using either current observed coordinates or predicted
updated coordinates and the **fixed fit mean**
in the orthogonal complement, then score its joint future KL. No confirmation
complement is leaked into this reconstruction.

These are held-out approximation tests. Failure may mean a nonlinear update,
an inadequate summary, an off-manifold reconstruction, or poor fit. Success is
not causal identification. Unit calibration checks both a fully observed shift
register whose affine update succeeds and an incomplete coordinate that fails.

## Numerical restoration and use

The original parameter/buffer state, per-module training flags and parameter
gradient flags are saved. Fitting changes only the external basis. Exact
unchanged-weight status is recorded, and the original state is restored in a
`finally` block, including on failure. Existing backbone gradients are not
cleared or overwritten. All random selection/bases use local RNG objects.
Identity state/joint patches, encoding replay, orthogonality and probability
normalization are recorded. Orthogonality and identity are implementation checks,
not scientific evidence for the mechanism.

```python
from voynich.episodic_causal import audit

report, tensors = audit(
    frozen_raw_recurrent_model, fit_prefixes, dev_prefixes, fresh_prefixes,
    train_groups=fit_key_ids, validation_groups=dev_key_ids,
    test_groups=fresh_key_ids, alphabet=4, rank=4, horizon=3,
    steps=150, batch_size=32, seed=12091,
    untrained_model=same_architecture_fresh_initialization,
)
```

`report` is JSON-safe. `tensors` contains CPU bases, affine maps, fit mean states,
paired indexes and matched masks. Save tensor artifacts under ignored outputs
and compact report/provenance under results. The caller supplies immutable
checkpoint hashes, generator manifests, software versions and resource records.
No paid API, backbone training or manuscript final-set scoring occurs here.

Tests live in `tests/test_episodic_causal.py`. They compare exact enumeration with
brute force, expose information invisible in immediate outputs, detect a
noncommuting fixed coordinate, verify full-donor restoration, check independent
closure success/failure, reject split leakage, and run a small complete audit
with a separate untrained model. Passing software tests does not establish a
scientific recovery result.

The Fisher tests additionally compare the score gradient to central finite
differences with fixed probability weights, verify that a delayed register is
absent before its causal horizon, check finite/deterministic gradients and
orthonormal spans, and verify delayed exclusion of the declared output span.
