# TEACH-0007 results: dense attention spontaneously forms a mid-layer causal key

**Registered outcomes: `EMERGENT-KEY-SITE-SUPPORTED` and `DENSE-CAUSAL-SUBSPACE-SUPPORTED`. Independent audit: PASS.** A discovery-only screen of all 30 layer-cut×slot sites selected the query slot after two transformer layers. On untouched confirmation groups, that single state transferred the counterfactual key perfectly across recipient G tables in both independently trained dense models. Its discovery-fitted 11D key subspace was sufficient; its orthogonal complement was inert.

## What ran

The study generated 256 new held-out cross-G groups from seed `67111`, split 128 discovery/128 confirmation. It loaded the two frozen TEACH-0004 dense-row checkpoints, manually reproduced their parsed six-slot input and four transformer layers, and screened cuts 0–4 across F0/F1/G0/G1/marker/query slots. Exact manual/native logits matched with maximum error 0.0 in both seeds. The CPU run completed in 1.408 seconds; compact audit independently regenerated the suite, site selection, primary scores and decisions.

## Information moves through a clear sequence

Discovery donor item/group accuracies for the causally relevant sites were:

| Cut | F row 0 | F row 1 | Query slot | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| Parsed input | 55.5% / 55.5% | 44.5% / 44.5% | 0% / 0% | Replacing the queried relation row changes the answer; one row alone covers its query share. |
| After layer 1 | 54–56% / 54% | 43–45% / 42% | ~0% / 0% | Relation remains in row slots. |
| **After layer 2** | **0% / 0%** | **0% / 0%** | **100% / 100%** | Selected key has moved into the query slot. |
| After layer 3 | 0% / 0% | 0% / 0% | 33.3% / 0% | Query state carries a fixed donor-G answer, correct only for `G0`. |
| After layer 4 | 0% / 0% | 0% / 0% | 33.3% / 0% | Answer-level state persists to output. |

The same qualitative map appeared in both seeds. G-row single-slot patches never produced exact three-G groups; marker patches were inert. Discovery selection therefore froze cut 2, slot 5 (the query slot), with minimum-both-seed item, group and cross-G non-injection rates all 100%.

This temporal sequence is stronger than a static probe: early F-row interventions control the output; two layers later those same sites lose control while the query gains it; one layer later the query no longer adapts across recipient tables and behaves like a downstream answer state. It identifies a moving causal control point, not merely information readable everywhere.

## Fresh confirmation and specificity controls

On each seed's 128 untouched groups:

- selected layer-2 query state: 384/384 recipient-specific items and 128/128 exact groups;
- cut-0 replacement of both F rows: 384/384 and 128/128, confirming the supplied-relation positive control;
- same-key/different-distractor selected state: 384/384 original base answers;
- adjacent marker-slot donor patch: 384/384 original base answers;
- norm-matched random query state: 188/384 donor items and 39/128 groups in seed 0; 180/384 and 48/128 in seed 1;
- all six donor slots at cut 2: 128/384 donor items and 0/128 groups, with zero cross-G non-injection.

The broad all-slot patch is revealing. It carries donor G-table information and gives the fixed donor answer, which is correct only under one of three recipient mappings. Patching only the query slot strips away that contamination and transfers a reusable key. Thus successful interchange depends on *where and how narrowly* the state is transplanted.

## The dense query state contains the same kind of distributed key subspace

Discovery key centroids again had centered rank 11. On confirmation, the 11D query-slot key span alone scored 384/384 items and 128/128 groups in both seeds. The 117D complement preserved 384/384 base answers and produced 0 donor answers. Every one of 32 matched random rank-11 subspaces scored 0/384 in both seeds; confirmation nearest-centroid key decoding was 100%.

The rank curve was:

| Dimensions | Seed 0 items / groups | Seed 1 items / groups |
| ---: | ---: | ---: |
| 1 | 7.81% / 7.81% | 16.67% / 14.06% |
| 2 | 34.11% / 30.47% | 49.74% / 46.09% |
| 4 | 83.07% / 78.91% | 84.64% / 82.03% |
| 8 | 100% / 100% | 100% / 100% |
| 11 | 100% / 100% | 100% / 100% |

Native neuron axes were again insufficient. Top-11 coordinates transferred 0/384 in both seeds; top-32 reached 44/384 and 18/384. Projector leverage participation ratios were 113.31 and 114.07 of 128. The dense transformer and explicit memory architecture therefore converge on the same qualitative representational solution: a compact class subspace distributed across most native coordinates.

## Verification, meaning and next mechanism

`scripts/teacher0007_analyze.py` independently reconstructed the deterministic suite and global earliest-site rule, verified frozen source/checkpoint/row hashes and every confirmation label, rescored all ten archived intervention conditions, checked geometry-control counts and reproduced both registered decisions. It does not rerun neural inference or secondary subspace fits from hidden states.

This result removes an important alternative explanation for TEACH-0005/0006: the causal key was not only present because the memory model hardwired a named intermediate. The dense transformer learned an emergent mid-layer query representation with the same cross-G counterfactual semantics and distributed low-rank geometry.

The remaining synthetic gifts are substantial: explicit row records, type roles, a known query token, task markers and supervised answers. This is not evidence for a Voynich key or semantic latent. The next narrow mechanistic question is which components of the second transformer layer write the key into the query slot—attention heads, the layer MLP, or a distributed combination—using discovery-only component screening and fresh confirmation.
