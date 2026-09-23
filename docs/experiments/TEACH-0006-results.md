# TEACH-0006 results: a distributed 11D causal key code aligns across models

**Registered outcomes after corrected audit:** `CAUSAL-SUBSPACE-SUPPORTED`; `CROSS-SEED-CAUSAL-ALIGNMENT`; native-axis concentration `NOT SUPPORTED`. The valid corrected run passed independent audit. A first execution was invalid because float32 rank estimation reported the mathematically impossible rank 12 for twelve centered centroids; its artifacts and the pre-rerun repair are preserved and documented in [TEACH-0006-rank-amendment.md](TEACH-0006-rank-amendment.md). Because that invalid report exposed aggregate behavior, the corrected run is a transparent non-blind pipeline repair rather than a pristine confirmation.

## What ran

The study used 256 new held-out cross-G groups from seed `66111`: 128 discovery groups fit geometry, and 128 confirmation groups were reserved for interventions. Both TEACH-0004 two-read checkpoints stayed frozen. Discovery key centroids defined an 11-dimensional span inside the 128-dimensional post-F state. Confirmation applied the full donor change, projections into ranks 1/2/4/8/11, the 117-dimensional orthogonal complement, 32 deterministic random rank-11 subspaces, native-coordinate masks, 16 random coordinate rotations and bidirectional cross-model Procrustes maps. The valid CPU run completed in 0.357 seconds; exact compact-artifact audit passed.

## A low-rank subspace is causally sufficient and necessary

Clean base and full donor-state positive controls were perfect on 384/384 items and 128/128 three-G groups in both seeds. The discovery-fitted **11D key span alone** was also perfect on every confirmation item and group. The orthogonal 117D complement preserved the base answer on 384/384 items and produced the donor answer on 0/384 in both seeds. Thus, for these counterfactuals, the fitted subspace is sufficient while the removed complement contains no detectable independent donor-key route.

The prespecified dimension curve was reproducible:

| Dimensions | Seed 0 items / groups | Seed 1 items / groups |
| ---: | ---: | ---: |
| 1 | 13.28% / 10.16% | 10.16% / 7.81% |
| 2 | 29.69% / 24.22% | 28.13% / 20.31% |
| 4 | 68.23% / 57.03% | 55.21% / 45.31% |
| 8 | 98.70% / 97.66% | 100% / 100% |
| 11 | 100% / 100% | 100% / 100% |

Matched random rank-11 subspaces reached at most 1/384 donor items in seed 0 and 1/384 across the seed-1 set of controls; their registered 95th percentiles were 1/384 and 0/384. Confirmation nearest-discovery-centroid key decoding was 100% in both seeds. Descriptive centered cosine similarity was 0.930/0.911 within keys and −0.082/−0.077 between keys, but the causal projection and controls—not cosine—support the mechanism claim.

## The code is distributed across neurons

The top 11 native coordinates by key-projector leverage transferred 0/384 answers in both seeds. The top 22 reached 20/384 and 1/384; top 32 reached 47/384 and 16/384. Only all 128 coordinates recovered the perfect full effect. Projector leverage participation ratios were 120.10 and 119.64 out of 128, meaning the 11D code is oriented across almost the entire neuron basis rather than concentrated in a few coordinates.

Randomly rotating the coordinate system while preserving the subspace/model function did not reveal a privileged sparse basis: top-11 rotated-coordinate transfer ranged 0–8/384 and 0–3/384. Both native-axis registered clauses failed. The implication is precise: there is a compact **subspace-level** key variable, but statements such as “neuron 65 stores the key” are unsupported. Native coordinates 8, 9, 43, 76, 113, 125 and 127 ranked highly in both seeds, yet patching the high-ranked set was not sufficient.

## The two models learned the same geometry up to a rotation

Before alignment, corresponding 11D subspace principal angles ranged from 60.0° to 88.6°, which would make raw coordinate comparison look like disagreement. An affine orthogonal Procrustes map fitted only on matched discovery states reduced all confirmation-relevant subspace angles to 0.09°–7.35°.

More importantly, the alignment was causal. Mapping seed-0 donor states into seed-1 coordinates and patching seed 1 gave 384/384 recipient-specific answers. The reverse direction gave 382/384. Frozen shuffled-pair maps scored 138/384 and 152/384, so matched alignment advantages were 64.06 and 59.90 points. Discovery residuals were not tiny (0.474 and 0.428), indicating that the full 128D state is not simply identical after one rotation; nevertheless, the functionally relevant key geometry transfers almost perfectly.

## Finite movement is nonlinear

Moving continuously from base to donor state produced a sharp but smooth answer transition. At fraction 0.25, donor accuracy was 4.43% and 0.52%; at 0.5 it was 47.14% and 44.79%; at 0.75 it was 95.05% and 99.22%; at 1 it was 100%. Mean donor probability followed the same curve.

The local gradient at the base predicted the *sign* of all 128 tested donor-margin changes in each seed, but its magnitude was poor: Pearson correlations were −0.022/−0.033, Spearman −0.025/−0.057 and median absolute relative errors approximately 1.0. The causal path therefore cannot be summarized reliably by a first-order derivative at the saturated base endpoint. Finite interventions and trajectories reveal behavior that a local saliency map would miss.

## Verification and implications

`scripts/teacher0006_analyze.py` regenerated the 256-group suite, checked frozen source/checkpoint/row hashes, independently reconstructed all primary confirmation scores and every registered decision clause, and validated spectra, leverage, control counts and interpolation structure. It reports `audit: pass`. It does not rerun inference or independently recompute the unarchived hidden states behind secondary summaries.

This is the strongest synthetic mechanistic identification in the project: a fresh-data, causally sufficient and necessary low-rank variable, with matched random controls and cross-model transfer. It also demonstrates why token/neuron cosine and coordinate-by-coordinate comparisons can fail: independently trained models place the same functional geometry in different gauges, and the useful 11D code is dense across native neurons.

The limitations remain fundamental. The architecture was trained on explicit typed rows with known synthetic key identities and an architecture-defined intermediate site. The 12 keys naturally admit an 11-dimensional centered simplex-like code; discovering one here does not imply language concepts generally use 11 dimensions. No Voynich token, latent, language or historical mechanism was tested. The next mechanistic step is to search for an equivalent key state inside TEACH-0004's successful dense-row transformer, where no explicit intermediate memory bottleneck was supplied.
