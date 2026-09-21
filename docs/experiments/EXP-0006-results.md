# EXP-0006 results — Effective output steering, with limits on the mechanism claim

Completed 2026-09-20 PDT / 2026-09-21 UTC. [Registration](EXP-0006.md), [machine-readable summary](../../results/EXP-0006/summary.json), [run manifest](../../results/EXP-0006/manifest.json).

## Outcome

The learned rank-three intervention passes the registered practical steering criterion at the late residual site in all three trained models. However, a simple subspace constructed from the output weights achieves nearly identical performance. At the early site, the learned intervention fails to outperform training with shuffled supervision consistently. This establishes useful supervised output control on a known toy process; it does not establish recovery of a unique internal decoding algorithm.

The language models were frozen throughout. Only a 128-by-3 orthonormal intervention basis was fitted, using the known generator's Bayesian next-symbol probabilities. Thus these are **5,600 alignment updates**, not another 5,600 language-model training updates and not an unsupervised cipher solver.

## Fixed experiment and provenance

- Clean source `2a367274e0ec75c8b264467441f6fe200722f69b`, published before execution. Three EXP-0004 trained checkpoints and seed42's initial checkpoint; all model weights verified unchanged afterward.
- 7,168 fresh synthetic prefixes from independent fit, validation and test pools; none duplicates another full prefix or occurs as an exact substring of the previous structured synthetic corpus. Dataset hashes, pair counts and generator settings: [dataset manifest](../../results/EXP-0006/dataset_manifest.json).
- Fourteen fits of 400 steps each: true/shuffled supervision at two sites in three trained models, plus true supervision at both sites in the untrained control. Checkpoint selection uses the corresponding validation objective; test results did not revise any fit.
- Test128: 512 fixed recipient/donor pairs per model. Test96 and test192: 256 pairs each. A further 256 pairs with similar generator beliefs test preservation. Pairs share the final observed symbol; primary pairs differ in inferred state. Selection uses the generator, never model performance.
- Runtime 42.17 seconds on MPS; peak process resident memory **1,320,108,032 bytes (1.32 GB / 1.23 GiB)**. This is process RSS, not an independent measurement of all GPU/unified-memory allocations. Caching avoids repeatedly running frozen layers. No paid service or manuscript test scoring.

## Counterfactual prediction fidelity

These numbers measure distance between the desired donor's Bayesian next-symbol distribution and the model's prediction after intervention. The distance is KL divergence in bits; **lower is better**, with zero meaning the distributions agree. It is not ordinary next-token loss on Voynich.

Three-seed means on the fixed 128-symbol test:

| Intervention | Early residual | Late residual |
| --- | ---: | ---: |
| Unchanged recipient | 1.386457 | 1.386457 |
| Whole final-position residual from donor | 1.118261 | 0.025283 |
| Learned rank-three basis | 1.084289 | 0.034799 |
| Output-weight span | 1.209958 | 0.036336 |
| Basis learned with shuffled supervision | 1.080681 | 0.353068 |
| Norm-matched random bases, mean of eight | 1.344032 | 0.853539 |

The late learned intervention selects the donor oracle's most likely next category on **97.07%** of pairs, averaged across seeds. This is agreement about four artificial next-symbol categories, not 97% recovery of actual hidden states or manuscript characters. Full-residual donor replacement is a reference, not a learned algorithm. At the early site, replacing only the final position leaves other prefix activations from the recipient, so exact donor reproduction is not expected.

The registered decision requires at least 0.10 bits improvement over unchanged, and 0.05 bits over both shuffled supervision and mean random intervention, in every trained seed at a site. Late passes; early fails. These are engineering thresholds, not statistical significance tests. Donors can be reused and pairs share pools, so treating every pair as independent would be misleading.

## Length and preservation checks

No fitting used the two alternate test lengths. Means across seeds:

| Prefix length | Early learned | Early shuffled | Late learned | Late shuffled | Late output-weight span |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 | 1.128155 | 1.128861 | 0.04207 | 0.36875 | 0.04303 |
| 128 | 1.084289 | 1.080681 | 0.03480 | 0.35307 | 0.03634 |
| 192 | 1.14657 | 1.14021 | 0.01783 | 0.32969 | 0.02451 |

The qualitative pattern persists across these lengths. All data still come from one generator and alphabet; this is no evidence of transfer across unknown cipher families, languages or keys.

For pairs with closely matched generator beliefs, learned interventions change the original model distribution by mean KL **0.000981 bits early / 0.021854 late**. Preservation is approximate, not exact. The late output-weight control changes it by 0.025122 bits. Against the generator oracle these preservation pairs score 0.01444 bits unchanged and 0.01163 after late learned intervention; moving closer to the oracle does not erase the measured change to the model's own distribution.

The untrained model improves only about 0.002 bits at either site. Its unchanged counterfactual KL (~0.515) is lower than the trained model's (~1.386) because an untrained near-uniform prediction is less committed to the recipient's state. That comparison does **not** make it a better ordinary predictor.

## Interpretation and validation

The output-weight span requires no alignment fitting and almost matches late learned steering. Near the output, directly adjusting category evidence can therefore explain the apparent success. The early negative result is also consequential: real supervision does not reliably beat shuffled supervision, and improvements often soften the wrong prediction without selecting the intended category. Neither result licenses labeling an internal component a recovered state machine.

Selected true-supervision steps were 200/300/350 early and 400/400/400 late for seeds42/43/44. We did not extend training after seeing these outcomes. Raw per-pair values, all controls, fit histories, selected steps and basis/checkpoint digests are archived beside the summary. Basis tensors and bulk activations remain local and ignored.

Maximum cached/full forward error was 7.63e-6; late full-donor conditional-log-probability error was 9.54e-7, within the registered 1e-5 tolerance. CPU tests also verify intervention-gradient equivalence, orthogonality, norm controls, pairing rules and readout construction. All saved bases passed the archival orthogonality audit. Tiny negative KL values around numerical zero in identity diagnostics are floating-point roundoff.

This fresh synthetic test has now been exposed. Adaptive successor methods need fresh evaluation pools. A useful next challenge is to recover and causally test predictive states without generator labels, then test transfer to new keys and generator families; it remains unexecuted.

## Reproduction

On the recorded source revision, with the preceding synthetic checkpoints present and empty output/data destinations:

```sh
.venv/bin/python -m voynich.causal_alignment prepare
.venv/bin/python -m voynich.causal_alignment run --device mps
```

Dataset generation and fitting refuse accidental overwrite. The recorded source, environment and hashes define the run; bitwise equivalence across hardware/library versions is not promised.
