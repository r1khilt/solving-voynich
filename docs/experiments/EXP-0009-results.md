# EXP-0009 results — Earlier synthetic computations affect later predictions

The registered synthetic test passed across all three trained models and failed on the untrained control. The manuscript test failed across all three models. This is progress in causal calibration, not a recovered Voynich algorithm.

## What changed relative to earlier steering

EXP-0006 changed the immediate answer using a small learned direction, but an output-weight control was equally effective. EXP-0009 intervened only in the original prefix, then let the model process the same additional observed symbols. A final-layer output change cannot influence these later predictions; an earlier-layer change can propagate into later attention computations.

The intervention replaced an entire component's output vectors at selected prefix positions. It did **not** isolate a state-only variable, learn an explicit transition rule, or generate an independently decoded continuation. The later symbols were supplied as shared observations. The Bayesian reference specifies what the synthetic model should predict after each observation.

## Fixed source, data and execution

- Published source `72f632804a162009b488e6de5d8550493670f937`, clean run manifest, MPS, two CPU threads, no language-model updates or paid APIs.
- Seven frozen models: synthetic trained seeds42/43/44 plus untrained seed42; manuscript compact seeds42/43/44. All checkpoint digests and unchanged weights verified.
- Synthetic:512 discovery and1,024 confirmation pairs from independent4,096/8,192-prefix pools. No prefix overlap across splits or with registered old corpus/pool substrings. Same known generator and alphabet; not a mechanism-family/key holdout.400/809 distinct correct donors respectively, so pairs are not fully independent.
- Manuscript:768 discovery and768 confirmation positions; disjoint groups of five physical leaves each, drawn from previously explored validation pages. No final manuscript test scoring. This is within-experiment selection isolation, not a new historically untouched holdout.
- Every model received the complete48-site discovery and confirmation maps. One first-layer head/MLP site was selected solely on discovery horizons2/4 and saved before confirmation. Every other confirmation comparison is exploratory.
- Total wall time240.839545seconds (about4.01minutes) while the other two campaign jobs also ran. Peak process resident memory529,235,968bytes, about0.493GiB. This is not a measurement of total GPU allocation or utilization.

## Synthetic confirmation

Here, horizon1 means the immediate next prediction; horizons2/4/8 follow one/three/seven shared observed symbols. All values below are mean KL bits to the correct donor's Bayesian next-symbol distribution, averaged across the three trained seeds. Lower is better. The denominator is1,024 fixed confirmation pairs per model; these are counterfactual prediction errors, not decipherment accuracy.

| Intervention | Horizon1 | Horizon2 | Horizon4 | Horizon8 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged recipient |1.394853|1.056515|0.339824|0.018858|
| Discovery-selected earlier component |0.516327|0.398010|0.177210|0.018559|
| Mean norm-matched random controls |1.269413|0.927533|0.277279|0.018777|
| Wrong donor at selected component |1.372229|1.029500|0.337125|0.018851|
| Late readout-weight projection |0.033344|1.056515|0.339824|0.018858|
| Clean donor model, reference floor |0.021622|0.018499|0.022580|0.013286|

The late readout intervention almost fixes the immediate answer while leaving every later prediction exactly unchanged. In contrast, the earlier intervention improves horizons2 and4, beating both registered controls.

| Seed | Discovery-selected site | Gain at horizon2 | Gain at horizon4 | Random-control margin at2/4 | Wrong-donor margin at2/4 | Registered result |
| --- | --- | ---: | ---: | --- | --- | --- |
|42|First-layer MLP, all128 prefix positions|0.573860|0.165486|0.434804 /0.099167|0.551618 /0.163099|Pass|
|43|First-layer MLP, all128 prefix positions|0.678097|0.170456|0.570387 /0.112055|0.645681 /0.166896|Pass|
|44|First-layer MLP, recent8 prefix positions|0.723557|0.151900|0.583377 /0.088984|0.697170 /0.149749|Pass|

In plain English: we changed some earlier internal calculations, then showed both versions of the model the same next symbols. The change still helped after one and three new symbols. After seven new symbols, the changed and unchanged versions predicted almost equally well. The unchanged recipient's counterfactual error had already fallen near the clean donor floor by horizon8; mean intervention gain there is only0.000298bits. One plausible explanation is that the shared observations let the two paths resynchronize, making their earlier disagreement less relevant. That is an interpretation of these measurements, not a discovered transition algorithm. We do not claim durable memory transfer over arbitrary continuations.

The untrained model selected first-layer head1 across all128 positions but failed: gains at horizons2/4 are−0.000530/+0.000107bits. Its small counterfactual KL should not be compared with the trained recipient KL as ordinary model quality: an untrained nearly uniform model is less committed to the recipient's conflicting belief.

Zeroing the selected component on **clean donor inputs** increases mean oracle KL by0.024675 at horizon2 and0.016520 at horizon4. This is evidence that the component contributes under that ablation; zeroing is distribution-shifting and does not establish a necessary, unique or complete algorithm.

### Exploratory localization, not an extra confirmation result

Across the synthetic maps, first-layer MLP `recent8` and `all128` replacements yield nearly identical future gains. Replacing only the remote120 MLP positions has effects near zero. This suggests that the useful changes are concentrated in the recent prefix representations under this particular task/intervention.

Final-position-only MLP patches also yield smaller positive horizon2/4 gains: roughly0.111–0.135 and0.058–0.085bits. Some individual head patches have smaller positive effects as well; the largest confirmation-map head average gain over horizons2/4 is about0.116–0.124bits. Those alternatives did not receive independently selected control suites, so they are exploratory observations, not additional registered circuit discoveries.

The selected `all128` condition is especially broad: it replaces the entire128-dimensional MLP write at every prefix position. Even `recent8` replaces full vectors at eight positions. Passing demonstrates useful propagation through an earlier component, without proving that only hidden-state information moved or extracting the underlying transition computation.

## Manuscript confirmation failed

The manuscript primary metric is actual-target surprise in bits. The averages below weight the five confirmation physical leaves equally and then average the three model seeds. They cannot be compared directly with previous experiments that sampled different targets or used different weighting.

| Condition | Horizon1 | Horizon2 | Horizon4 | Horizon8 |
| --- | ---: | ---: | ---: | ---: |
| Remotely shuffled recipient |1.929332|2.061543|1.867771|2.046867|
| Discovery-selected component restored |1.897825|2.052791|1.863909|2.027448|
| Mean norm-matched random controls |1.929985|2.056932|1.867935|2.037757|
| Clean original prefix |1.893300|2.049578|1.853326|2.025186|

Seed42 selected first-layer MLP/all128, with gains0.022278/0.002548bits at horizons2/4: it fails the horizon4 threshold/control margin. Seed43 chose the same site but gains only0.005041/0.008499, and its horizon2 random-control margin is negative. Seed44 selected first-layer head0/all128, with gains−0.001063/+0.000540. All three fail.

Wrong donors are much more damaging than norm-matched random changes. They come from other training pages and are matched only on the final token, so that comparison is not a clean isolation of a specific state. The stronger random comparison and unchanged-recipient thresholds prevent treating this donor mismatch alone as success.

These negative results do not show that the manuscript has no structure or that the model lacks useful computations. This specific remote-shuffle recovery question, intervention family, selected sites and future-horizon thresholds did not establish stable causal mediation. There is no evidence here that the manuscript shares the toy generator's mechanism.

## Integrity and retained artifacts

All measured identity, full-prefix embedding restoration, full final-prefix residual restoration and late-prefix future-invariance logit errors are exactly0.0. Every model stayed frozen. The archival audit recomputes all condition means from the compressed per-example arrays, redoes the discovery selection and registered decisions, and verifies committed-source, dataset, checkpoint and report hashes. Persisted selections precede their confirmation reports.

Compact manifests, all48-site maps, selections, controls and completion reports are in `results/EXP-0009/`. `archive_audit.json` retains paths/digests for ignored raw inputs and per-example NPZ arrays; no corpus text, activation tensor or checkpoint is copied into Git. Source registration: [EXP-0009](EXP-0009.md).

The new synthetic confirmation pool is now exposed. Adaptive follow-ups need fresh pools. A useful next step would test smaller, state-specific interventions against these broader component replacements and unfamiliar generator settings, keeping future prediction consequences and output/random controls. That is an unexecuted proposal, not a result or background job.
