# EXP-0010 results — Longer context did not improve the registered predictor comparison

All 21 runs completed from clean, published source `72f632804a162009b488e6de5d8550493670f937`: 25,200 optimizer updates, 88,088,063 repeatedly sampled scorable training targets and 3,273.27 seconds (54.55 minutes) for the track. No manuscript final-test scoring or paid API use. [Registration](EXP-0010.md).

The practical result is negative for increasing maximum context under this training schedule. The 512-unit main model is essentially tied with its 256-unit control, while the 1,024/2,048 conditions are worse on average. Models trained at 2,048 nevertheless use some additional history: restricting those same models to 256 units hurts their predictions. Learning to use more history did not make them better predictors than the separately trained shorter-context controls.

## Fixed-endpoint results and registered decisions

Scores below use the registered final checkpoint at update 1,200. All conditions score the same 384 primary development targets across 24 pages / 10 leaf groups. Lower bits per unit means less surprise. These are EVA transcription units, including certain separators, not deciphered words. Different target sampling and training-window weighting mean these scores are not directly comparable with earlier experiments' whole-validation or 768-target scores.

| Condition | Seed 10111 | Seed 10112 | Seed 10113 | Mean primary bits |
| --- | ---: | ---: | ---: | ---: |
| Main, 256 | 1.917336 | 1.915791 | 1.908578 | **1.913902** |
| Main, 512 | 1.918009 | 1.893434 | 1.928943 | 1.913462 |
| Main, 1,024 | 1.956392 | 1.937200 | 1.883439 | 1.925677 |
| Main, 2,048 | 1.924639 | 1.961550 | 1.960097 | 1.948762 |
| Compact, 256 | 1.884807 | 1.919659 | 1.890344 | **1.898270** |
| Compact, 2,048 | 1.932413 | 1.937624 | 1.916837 | 1.928958 |
| Main, 2,048, locus identity merged into spaces | 1.980722 | 1.962653 | 2.009570 | 1.984315 |

Positive gain below means longer-context training improved over the same-capacity 256-unit model. The registered follow-up criterion requires a mean gain of at least 0.02 bits, positive gain in all three seeds, and positive benefit of normal versus cap256 context within each longer-context model.

| Contrast | Seed gains | Mean gain | Registered criterion |
| --- | --- | ---: | --- |
| Main512 versus main256 | −0.000673, +0.022357, −0.020365 | +0.000440 | Fails |
| Main1024 versus main256 | −0.039056, −0.021410, +0.025139 | −0.011776 | Fails |
| Main2048 versus main256 | −0.007303, −0.045760, −0.051519 | −0.034861 | Fails |
| Compact2048 versus compact256 | −0.047606, −0.017965, −0.026494 | −0.030688 | Fails |

No longer-context condition met the criterion. The descriptive equal-leaf bootstrap intervals are broad: all nine main-context contrasts include zero; only compact2048 seed10111 has an interval wholly below zero (approximately −0.0676 to −0.0117). These repeatedly consulted development data and multiple comparisons do not warrant a final significance claim. Thirteen of the 21 runs reached their best observed primary validation score before update 1,200; endpoint conclusions remain those registered, rather than being replaced with favorable checkpoint selections. The result is specific to this data, sampling scheme, optimizer and finite schedule.

## Does a trained longer-context model actually use its extra input?

The following interventions act on frozen final checkpoints. Positive damage means an intervention made prediction worse; means are across the three seeds on the primary targets.

| Model | Cap preceding history at 256 | Shuffle history older than the latest 256 |
| --- | ---: | ---: |
| Main512 | +0.007312 | −0.000551 |
| Main1024 | +0.015229 | +0.006923 |
| Main2048 | **+0.033039** | **+0.007686** |
| Compact2048 | +0.015469 | +0.000263 |
| Main2048 with locus identity merged | +0.028722 | +0.006771 |

For main2048, cap256 damage is +0.038916 / +0.030330 / +0.029872 across the seeds; distant-shuffle damage is +0.007616 / +0.014361 / +0.001080. Both directions are consistent. Compact2048 cap damage is also positive in each seed, while its shuffle effects vary. Thus some additional history is used, but broad ordering beyond 256 has a much smaller measured effect than the truncation intervention. These perturbations change input distributions; they do not identify semantics or a specific historical mechanism.

## What the locus-boundary ablation says

Merging previous `<line>` inputs into `<space>` keeps input lengths, target positions and the original output alphabet fixed. It worsens main2048 primary loss by **+0.035553** bits on average: +0.056083 / +0.001103 / +0.049473 across the three seeds. The experiment tests the information in previous **transcription locus boundaries**. Loci are not uniformly physical manuscript lines.

| Primary target type | Target count | Original main2048 | Boundary-merged main2048 | Mean damage |
| --- | ---: | ---: | ---: | ---: |
| Glyph/transcription character | 331 | 2.118272 | 2.144553 | +0.026281 |
| Certain space | 47 | 0.701636 | 0.755122 | +0.053485 |
| Locus separator | 6 | 2.366603 | 2.773213 | +0.406610 |

The overall average is not solely due to separator targets, but the glyph-only effect is not unanimous: +0.058398, **−0.002913**, +0.023357. There are only six primary locus-separator targets. Report these subgroup counts and signs rather than treating an apparent large separator effect as precise evidence. Two of three overall descriptive leaf-bootstrap intervals include zero. No physical-line reset, cipher state, plaintext-word rule or scribe identity follows from this ablation.

## Full 2,048-unit prefixes are a narrow secondary sample

Only f76r, f76v, f111r and f111v offer eligible full 2,048-unit histories: **four pages, two physical leaves**. The 128 secondary coordinates contain 101 glyph targets, 23 spaces and four locus separators. Even the primary set has only 24/384 targets with full 2,048-unit history; its average available history at that maximum is 542.61 units.

| Condition | Mean loss on the 128 full-prefix targets |
| --- | ---: |
| Main256 | 1.502038 |
| Main512 | 1.497987 |
| Main1024 | 1.501300 |
| Main2048 | 1.495444 |
| Compact256 | 1.534376 |
| Compact2048 | 1.581364 |
| Main2048, boundary merged | 1.561351 |

The main2048 mean lead over main256 is only 0.006595 bits here. Its seed gains are +0.028528, +0.009269 and −0.018014. Averaged across seeds, page gains are −0.002684 (f76r), +0.007566 (f76v), +0.076522 (f111r), and −0.055025 (f111v). Nearby pages can disagree. Restricting main2048 itself to 256 slightly improves this subgroup on average (damage −0.001544); shuffling its older history also improves it slightly (−0.007704). Do not read these small, heterogeneous results as manuscript-wide benefit from 2,048-unit context. No interval-based generalization claim is made from two leaves.

## Fixed simple controls

All baselines fit training pages only and score identical registered coordinates. No mixture coefficient was tuned on validation.

| Baseline | Primary bits | Full-prefix subgroup bits |
| --- | ---: | ---: |
| Unigram | 3.980395 | 3.918408 |
| Five-gram | 2.281540 | 1.632804 |
| Five-gram + uniform prefix histogram, fixed 50/50 mixture | 2.589876 | 2.236202 |
| Five-gram + recency τ64, fixed 50/50 mixture | 2.599362 | 2.241382 |
| Five-gram + recency τ256, fixed 50/50 mixture | 2.592701 | 2.240129 |
| Five-gram + recency τ1024, fixed 50/50 mixture | 2.590343 | 2.237343 |

The fixed frequency mixtures are worse than the five-gram on both samples. This is a negative result for these particular mixtures; it does not eliminate all recency/frequency explanations. They differ from EXP-0005's multiplicative histogram adjustment. Every neural condition still predicts these samples better than the unchanged five-gram.

## Integrity, resource accounting and remaining uncertainty

Independent post-run auditing reproduced the 512-coordinate union, all 209 source-block coordinates, target masks and coverage. It reconstructed the sampler stream from each seed and matched the stored stream hashes and exact 1,200-update target totals: **4,160,465 / 4,188,664 / 4,234,880**, each repeated across that seed's seven conditions. Same-capacity initial weight digests match across contexts/representations. All 21 clean-source manifests agree with the campaign source/corpus. All 63 initial/best/last checkpoint payloads were inspected and their file hashes recorded; the 21 last-file digests and 21 initial-tensor digests match their saved run records. Corpus/tokenizer identity and final-test guards remain intact. Auditing normalized tuple/list serialization of the empty auxiliary-horizon setting before comparing JSON and checkpoint configs; no model change was required.

The complete track stayed within its three-hour budget. Peak process RSS was 819,118,080 bytes (0.763 GiB); the largest sampled MPS driver allocation in evaluation history was 6,181,486,592 bytes (5.757 GiB). These counters measure different things, are not additive, and the sampled driver value is not a measured allocation peak. The run executed 151,760,384 padded input positions and 215,845,699,584 unweighted attention-pair slots; these are accounting proxies, not measured FLOPs. All checkpoint files remain local/ignored; compact reports retain their identities and the outcome vectors. Concurrent GPU use makes per-run timing unsuitable for an isolated architecture-speed claim.

The next useful work should investigate why context reset and locus information affect optimization and prediction, with fresh registered comparisons and more discriminating synthetic tasks. These results do not justify scaling the same Voynich-only predictor merely to consume more memory. They also do not show that the manuscript lacks long-range structure: the current data coverage and models may fail to expose it. No decoded text or historical encoding rule was recovered.

Artifacts: [compact summary](../../results/EXP-0010/summary.json), [run and target manifest](../../results/EXP-0010/manifest.json), [fixed baselines](../../results/EXP-0010/baselines.json), and per-run score/history files beneath `results/EXP-0010/runs/`.
