# EPISODIC-0013 — Joint-future causal abstractions and temporal update tests

Publication note (2026-09-21): this registration was frozen under the original identifier EXP-0013 at scientific source `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`. Its publication alias is **EPISODIC-0013** to avoid collision with the separate latent-recovery series. The registration below is historical; execution is now complete. See [results](EPISODIC-0013-results.md). Original command paths and JSON identifiers remain unchanged; rules were not revised after results.

Status: registered before backbone training/causal confirmation. Date2026-09-21. Depends on frozen recurrent models from [EXP-0012](EPISODIC-0012.md). This is an expansive but bounded account of trained-model computation, not a discovered manuscript variable.

## Motivation and reviewed methods

Earlier broad synthetic patches affected future answers; they did not isolate a continuing state algorithm. DAS, patching methodology, circuit-faithfulness critiques, output-span alternatives, Jacobian methods and cross-seed stability motivate stronger controls. See the [interpretation review](../research/deep-review-2026-09-21/INTERPRETABILITY.md) and [implementation details](../research/EPISODIC_CAUSAL.md). No new sparse-feature name or persuasive description is treated as semantic evidence.

## Frozen design

Backbones: both raw recurrent families from EXP-0012, all three seeds. Analyze ranks4 (primary) and16 (secondary). Each run includes an independently initialized same-architecture untrained control. Model weights/buffers/gradient flags/modes must restore exactly, including on exceptions.

New pair-parity processes only: fit8keys×24contexts, development4×24, confirmation8×24; every prefix length64. Parameter seed = data_seed+50000000+split_index*10000+key_index; sequence seed adds100000. Splits have disjoint tasks and prefixes. Exact sampler/source hashes and pair indices are saved. Neither true state labels nor oracle distributions fit the explanatory basis.

Select donor and recipient within the same key. Prefer same-final-symbol pairs with immediate teacher-distribution TV≤.05 but differing joint futures. Record unmatched rows separately. Wrong donors preserve the recipient's joint behavior as closely as available within key. These are teacher-prediction matches, not known historical counterfactual equivalences.

Fit a rank-r orthogonal subspace by swapping donor coordinates and matching the donor's complete three-symbol distribution. Use120 steps, fixed seed=model_seed+700000, batch32; choose a checkpoint using development teacher discrepancy. Fit a shuffled-target counterpart under the same schedule. Never select on confirmation.

Comparisons: unchanged recipient; whole donor state; learned subspace; shuffled targets; PCA; sampled immediate-output Jacobian span; three rank- and displacement-matched random spans; joint-future Fisher-Jacobian span; delayed Fisher span projected off the sampled immediate-output span. The last two are **secondary diagnostic methods**, not independently preregistered successes selected after seeing results.

Future Fisher basis uses≤8fit states×8 seeded Rademacher sketches of the exact H3 distribution. Detach sqrt(probability) weights before differentiating weighted log probabilities. Record effective rank, null completion, captured sensitivity, limited output-span overlap and split-half subspace overlap. These gradients describe local sensitivity; finite interventions test whether that description survives real changes.

## Measurements

For all pairs and the immediate-matched subset, record per-key means/counts and aggregate:
- donor joint KL and KL beyond the first symbol;
- recipient immediate-output TV and displacement/norm-match error;
- subsequent prediction after each possible observed symbol;
- patch→update versus update→patch symmetric KL;
- wrong-donor preservation and a largest-disagreement continuation witness.

Fit independent affine coordinate update maps on fit contexts only. Confirm their coordinate errors and reconstructed-model behavior on new keys, with a fixed-complement current-state comparator. This separates inadequate coordinates from inadequate linear updates. A nonlinear true variable can fail affine closure or rotate between coordinates; failure rejects the tested fixed-subspace account, not every possible state explanation.

Teacher oracle qualification is measured after all choices freeze: average exact H3 KL≤.30bits on the confirmation contexts. Failure does not cancel the diagnostic battery, but disallows a generator-recovery interpretation even if teacher transplantation succeeds.

## Primary claim gate

A **qualified predictive-state transplantation result**, still not latent identification, requires all three trained seeds of a recurrent family to satisfy:
1. Teacher oracle qualification, exact restoration, identity errors≤1e-6, joint normalization error≤1e-5.
2. At least32 immediate-matched confirmation pairs spanning≥4 independent keys, with unchanged beyond-first KL≥.01bits.
3. Rank4 learned intervention reduces that KL by≥20% and≥.005bits; recipient immediate TV≤.02 and commuting symmetric KL≤.02bits.
4. After averaging seeds within key, paired key-bootstrap upper95% for learned minus **each** shuffled/PCA/output/random control is below0, with mean gain≥.005bits. Retain all controls; no post-hoc weaker comparison.
5. The untrained backbone is reported with its own available donor gap and relative reduction. A qualifying untrained comparison has unchanged beyond-first KL≥.01bits and≥32matched pairs across≥4keys. If it matches/exceeds trained relative reduction, trained-specific interpretation fails. A smaller gap or insufficient matches makes this control inconclusive and the full primary claim unestablished, rather than evidence of specificity. No post-hoc gap-ratio threshold is used.

The Fisher candidates and rank16 remain descriptive. Interpreting their best observed result as a new positive claim needs a new final pool. No selective semantic-variable claim follows: the nuisance measure is immediate output, hybrid states may be off-manifold, and teacher behavior may approximate a different process.

## Resources, artifacts and scope

At most12trained-model/rank audits, each with its untrained counterpart; one-hour total causal cap checked between audits. Fit steps and pool sizes bound individual audits. CPU execution uses four threads; no paid API. Run only after all EXP-0012 neural checkpoints freeze; preserve raw bases/coefficients locally with hashes and track compact per-key reports.

Architecture reuse deliberately excludes canonical recurrent interventions until symbol-map state is included in the full intervention contract. No handwritten manuscript, plaintext, image semantics or latent labels enter this study. No adaptive retries on these confirmation pools.
