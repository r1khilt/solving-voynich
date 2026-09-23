# TEACH-0006: causal key-subspace geometry, neuron coordinates and cross-seed alignment

**Preregistration before generating the new suite or loading checkpoints for this study.** TEACH-0005 found perfect cross-G transfer for the full 128-dimensional post-F `first` state, but failed a task-specificity gate because direct lookup also followed the transplanted key. This study asks how the key effect is organized geometrically. It does not change TEACH-0005's verdict and does not use its exposed groups to fit or score anything.

## Fresh discovery and confirmation design

Generate 256 new held-out cross-G groups from suite seed `66111` using the frozen TEACH-0005 generator. The first 128 groups are discovery; the remaining 128 are untouched confirmation. Both TEACH-0004 two-read checkpoints are analyzed separately. A group's base and donor share a query and map it to different keys; three recipient G tables use six distinct outputs. Model weights remain frozen. Discovery may fit representations; all causal claims use confirmation only.

For each model, collect one base and one donor `first` state per discovery group. Center states globally, compute the mean state for each of 12 known synthetic keys, center the 12 centroids, and take the nonzero left singular vectors of the 128×12 centroid matrix. The resulting orthonormal key-centroid span has rank at most 11. No dimension is selected on confirmation. Report singular values, explained centroid energy, key counts and confirmation nearest-centroid accuracy using discovery centroids. Cosine and Euclidean within/between-key summaries are descriptive because invertible reparameterizations can change them without changing the model.

## Confirmation interventions

For each confirmation base/donor pair and all three recipient G tables, let `d = donor_first - base_first`, key-span projector `P`, and complement `I-P`.

- Full patch: `base + d`, the TEACH-0005 positive control.
- Key-span patch: `base + P d`.
- Complement patch: `base + (I-P)d`.
- Rank curve: repeat with the first 1, 2, 4, 8 and all fitted centroid directions.
- Random controls: 32 deterministic Haar-random subspaces with the same fitted rank, seed `66211`; report the full distribution and its 95th percentile.
- Native-coordinate patches: rank neurons by projector leverage `diag(P)` and patch the top `r`, top `2r`, top 32 and all 128 coordinates.
- Rotated-coordinate controls: for 16 deterministic Haar rotations, rotate states/projector together, rank rotated coordinates by leverage, patch the top `r`, and map back. A rotation preserves the represented subspace and model function but changes which coordinates are called neurons.

The primary `CAUSAL-SUBSPACE-SUPPORTED` verdict requires both seeds to have at least 95% clean/full-patch confirmation competence; fitted rank≤11; full-rank key-span donor-target item accuracy≥80% and exact three-G group accuracy≥65%; complement patch preserves base answers≥90% and reaches donor targets≤10%; key-span donor accuracy exceeds the 95th percentile of matched random subspaces by≥40 points; and confirmation nearest-centroid key accuracy≥90%. Failure does not show that no nonlinear or distributed key manifold exists.

## Cross-seed alignment

Fit an affine orthogonal Procrustes map between matched discovery states of seed 0 and seed 1, including separately fitted means. On confirmation, map donor states from one model into the other's coordinates, patch them into the recipient model, and score recipient-specific `Gj(k1)`. Test both directions. A shuffled-pair control uses one frozen permutation from seed `66311` when fitting the map. `CROSS-SEED-CAUSAL-ALIGNMENT` requires ≥80% item accuracy in both directions and ≥40-point advantage over the corresponding shuffled map. This is separate from the within-model subspace verdict.

Report principal angles between key-centroid spans before and after Procrustes alignment, relative mapping residuals on discovery and confirmation, and nearest-centroid transfer. Raw unaligned angles between independently trained models are not treated as semantic disagreement because their coordinate gauges are unconstrained.

## Locality and trajectory diagnostics

On the first 128 confirmation recipient items per seed, compute the gradient at the base state of the donor-answer logit minus base-answer logit. Compare `grad dot d` with the actual full-patch margin change and report Pearson/Spearman correlations, sign agreement and relative error. This is descriptive: a poor first-order approximation can coexist with a valid finite intervention. Along the full donor direction evaluate interpolation fractions `0,.25,.5,.75,1`; report donor/base probabilities and the first fraction at which the prediction changes. No locality threshold enters either verdict.

## Neuron interpretation boundary

Report leverage concentration, participation ratio, top coordinates and their confirmation causal accuracy in each model. Define a descriptive `NATIVE-AXIS-CONCENTRATED` label only if top-r native coordinates achieve≥80% donor accuracy and exceed the 95th percentile of rotated top-r coordinates by≥30 points in both seeds. Even a pass would mean concentration in the trained parameterization, not an invariant semantic neuron. A failure with a successful subspace would mean the key is causally real but distributed across coordinates.

All exact inputs, states-derived summaries, intervention predictions, source/checkpoint hashes and deterministic seeds are retained. Independent audit regenerates the suite and decision from compact rows but does not rerun inference. CPU-only cap 600 seconds; no training, paid API, download, manuscript data or final-test scoring. A pass qualifies a synthetic mechanistic instrument, not a Voynich interpretation.
