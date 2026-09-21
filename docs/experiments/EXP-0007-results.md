# EXP-0007 results — Restricted word-group order has little measured effect

Completed 2026-09-20 PDT / 2026-09-21 UTC. [Registration](EXP-0007.md), [complete measurements](../../results/EXP-0007/report.json).

## Outcome

Moving complete, equal-length transcription groups in distant context barely changes prediction loss. Destroying those groups' forms with an exactly matched changed-position mask is only slightly worse: **0.005090 bits/unit** on average. All three seeds have positive differences, but each seed's descriptive leaf-bootstrap interval includes zero. This experiment does not clearly distinguish distant group-form information from ordering of groups.

“Groups” means text between definite transcription spaces/newlines. We do not assume these are plaintext words. The last 16 units are always preserved; none of these results says nearby word forms or general syntax are unimportant.

## Measurements

Three frozen models, 768 fixed validation targets across 24 pages / 10 physical leaf groups, four corruption repetitions per target. Target sampling is exactly the EXP-0005 sample. Each number below averages repetitions per target, then targets, then model seeds. Lower loss is better; positive change means damage relative to the original 128-unit context.

| Distant-context intervention | Bits/unit | Change from original |
| --- | ---: | ---: |
| Original | 1.916253 | 0 |
| Permute individual units (block size 1) | 1.956290 | +0.040038 |
| Permute blocks of 2 | 1.950347 | +0.034094 |
| Permute blocks of 4 | 1.947597 | +0.031344 |
| Permute blocks of 8 | 1.952940 | +0.036687 |
| Permute blocks of 16 | 1.948963 | +0.032710 |
| Permute blocks of 28 | 1.950095 | +0.033842 |
| Swap blocks of 56 | 1.961448 | +0.045195 |
| Permute complete equal-length groups | 1.913687 | −0.002566 |
| Matched character scramble | 1.918777 | +0.002524 |

Block results are not monotonic: preserving longer local fragments does not steadily eliminate damage. A 56-unit swap also changes recency while moving separators. This spectrum does not isolate a natural code-unit size.

The primary matched contrast is character scramble minus group permutation:

| Model seed | Target-weighted difference | Equal-leaf difference | Descriptive equal-leaf bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| 42 | +0.008229 | +0.006710 | [−0.003273, +0.017688] |
| 43 | +0.001626 | +0.000569 | [−0.005990, +0.008695] |
| 44 | +0.005416 | +0.004795 | [−0.000136, +0.010609] |

Target-weighted and equal-leaf averages weight the data differently and must not be conflated. With only ten leaf groups, repeated validation inspection and multiple comparisons, these are descriptive uncertainty summaries, not a confirmatory significance result.

## Strength and limits of the controls

Both matched conditions preserve the exact remote symbol inventory, separator positions, group lengths, edge fragments and local suffix. Their changed-position masks relative to original are identical. On average **43.14% of distant positions change**, and the character control differs from the group permutation at **38.64% of distant positions**. All 3,072 target/repetition combinations change some positions. The contrast is not explained by both interventions being almost identity operations.

However, the allowed group movement is restricted to equal lengths and interior groups of length at least two. The character control uses a constrained random walk, not uniform sampling over all possible scrambles. No-effect results here cannot rule out arbitrary between-group dependence. Corruptions can also take the model outside its training distribution.

**New hypothesis, not an established result:** separator/line-boundary placement or distance-weighted symbol statistics could explain part of the gap between broad scrambling and these matched interventions. Both matched conditions keep separators fixed and score close to original; broad permutations move separators and alter recency. Those factors are confounded in this experiment. Separately perturbing spaces, line boundaries and recency would be a new registered test, not a conclusion from this one.

## Provenance and execution

- Clean source `389a12c` was committed and pushed before the run; exact revision, corpus and model digests are in the report.
- MPS runtime **11.36 seconds**, with no new neural fitting or paid service. Manuscript final test remains unscored.
- Every original-context score matches EXP-0005 within 1e-5; exact histogram, local-suffix and matched-mask invariants were checked. Tests cover the permutation constructions and difficult/no-op group cases.
- Per-target losses, coordinates, repetition exposure, all seeds and descriptive intervals are retained in the tracked JSON; no favorable-example selection. Bulk corpus/weights remain ignored.

Reproduce on the recorded revision with the prior selected checkpoints and EXP-0005 output present, using a fresh output destination:

```sh
.venv/bin/python -m voynich.order_scale --output-root outputs/EXP-0007 --device mps
```

The earlier EXP-0003 hint of negligible distant-order sensitivity was revised by EXP-0005's larger sample. This experiment narrows the next question but does not restore that broad no-order interpretation or provide a decipherment.
