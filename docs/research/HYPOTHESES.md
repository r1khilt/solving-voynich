# Hypothesis register

The historical explanations below are **unresolved**. They are derived from the charter or introduced as methodological controls. The method-status sections record bounded experiments; no historical explanation has been established. Mechanisms can coexist; a model family is not necessarily a historical explanation.

| ID | Candidate explanation | Proposed discriminating evidence | Main failure mode / control |
| --- | --- | --- | --- |
| HYP-001 | Information-bearing material is mixed with structured null/filler material. | A fixed inferred selection/decoding rule improves preregistered recovery or dependence metrics on untouched data, beyond matched deletion baselines. | Arbitrary deletion can manufacture patterns. Penalize complexity and deletion rate; compare position-, frequency-, length-, and amount-matched controls. |
| HYP-002 | Multiple observed forms represent shared latent units: homophony, verbose coding, or abbreviations. | Stable equivalence classes predict independent contexts and held-out observations under a compact encoding rule. | Surface similarity alone creates misleading clusters. Compare morphology-only, frequency-matched, and copy/mutate baselines. |
| HYP-003 | Encoding or interpretation depends on state, position, or manuscript context. | A limited state model makes advance predictions on held-out blocks that a simpler model misses. | Per-page rules can memorize data. Bound states and parameters and separate layout effects from semantic claims. |
| HYP-004 | Surface forms reflect linguistic structure without removable filler. | A constrained linguistic/generative account generalizes across held-out material and explains observations as well as or better than filler models. | Language-like statistics are insufficient to demonstrate language or identify a language. |
| HYP-005 | A copy/mutate or other structured production process explains observations without a recoverable plaintext. | Explicit nonsemantic generators reproduce multiple independent preregistered properties on fresh samples. | Matching selected statistics does not prove absence of meaning. Include withheld diagnostics and competing semantic generators. |
| HYP-006 | Apparent words or glyphs are not the relevant coding units. | Alternative segmentation improves held-out predictive/compression or synthetic recovery metrics after accounting for model complexity. | Flexible segmentation overfits; do not choose boundaries using the test set. |

## Method proposals, not manuscript hypotheses

- **M-001 — Synthetic recovery:** test whether signal/filler or latent-unit recovery transfers to independently held-out languages, alphabets, generator families, and keys. Distinguish successful recovery within a known family from transfer to an unseen mechanism.
- **M-002 — Tiny models:** train small models on explicitly defined Voynich-only text inputs after data policy and splits exist. Use multiple seeds and causal interventions; generation fluency alone is not a research result about meaning.
- **M-003 — Comparative mechanisms:** apply the same architectures and analyses to known synthetic generators and manuscript text. Similar learned computations can motivate follow-up tests but do not establish that the historical processes are identical.
- **M-004 — Latent families and repeated sequences:** test whether learned equivalence classes or filtering rules expose reproducible dependencies under frozen rules and matched controls.

## Identifiability constraint

As a methodological deduction, if two proposed processes produce the same distribution of observable evidence, those observations alone cannot distinguish them. A synthetic benchmark must state what makes recovery possible: a distributional asymmetry, encoding constraints, side information, or other explicit assumptions. A model cannot be credited with recovering an arbitrary hidden distinction for which its inputs supply no evidence.

## Updating an entry

Keep the ID stable. Add operational definitions, supporting and conflicting source IDs, experiment IDs, status changes, scope, and next discriminating tests. Preserve failed versions and reasons for revision. Do not recast an exploratory finding as a prediction made in advance.

## Method status after EXP-0002/0003/0004

Historical HYP-001 through HYP-006 remain unresolved; the synthetic generator was invented as a calibration and is not evidence that Voynich uses its rules.

- **M-001:** first within-family synthetic calibration complete. Trained-model activations support supervised state/role readouts; IID labels remain unrecoverable. Cross-key/language/family and label-free recovery are not tested. [EXP-0004](../experiments/EXP-0004-results.md).
- **M-002:** 18 controlled Voynich runs complete, compact model selected, paired context tests and one seed's head interventions performed. Prediction beyond five-gram is established on repeatedly consulted validation, not final test. [EXP-0002](../experiments/EXP-0002-results.md), [EXP-0003](../experiments/EXP-0003-results.md).
- **M-003:** causal assay implemented; synthetic probe-direction steering failed to outperform a norm-matched random control meaningfully. No shared historical/synthetic circuit established.
- **New exploratory implication for HYP-003:** distant symbol mixture/page context could explain some prediction benefit; preserved-local-context distant shuffling caused inconsistent damage. Requires length/content/section-matched replacement controls before interpretation.

## Updated status after EXP-0005/0006/0007

- **HYP-003 / M-002:** EXP-0005 revises the earlier small-sample hint: distant shuffling consistently harms prediction on a larger fixed sample. Both content and some order information affect these models. Category donor differences partly shrink with histogram matching, but residual histogram imbalance remains; no distinct category-specific cipher is established.
- **HYP-006:** EXP-0007 finds a small, uncertain contrast between moving complete groups and destroying forms at exactly matched positions. This does not identify plaintext word boundaries or a natural coding-unit size. Broad block-permutation effects are not monotonic in block length.
- **M-001 / M-003:** EXP-0006 supplies supervised causal calibration on fresh within-family synthetic contexts. Late steering passes the fixed practical threshold, but an output-weight span nearly matches it. Early steering fails against shuffled supervision. Label-free state discovery, family/key transfer and faithful algorithm recovery remain untested.
- **Exploratory candidate for HYP-003:** layout boundaries or recency-weighted symbol statistics may account for part of the distant-order effect. Current interventions confound these factors. A future test should independently perturb definite spaces, line boundaries and recency while matching changed positions/content. This is a new proposal, not supporting evidence for a historical mechanism.

Results: [EXP-0005](../experiments/EXP-0005-results.md), [EXP-0006](../experiments/EXP-0006-results.md), [EXP-0007](../experiments/EXP-0007-results.md). All historical HYP-001 through HYP-006 remain unresolved; no signal/filler assignment or translation has been inferred for Voynich.

## Updated status after CAMPAIGN-0001

- **M-001 / M-004:** EXP-0008 establishes label-free fitting/selection before truth diagnostics. Some familiar-key synthetic partitions improve; no unfamiliar cycle/branch key passes the stronger criterion, and omitted RRXOR fails. Copy-label success concerns mainly the next source symbol. Prediction-only clustering generally matches or beats residual clustering. No general state/transition recovery or manuscript application is justified yet.
- **M-003:** EXP-0009's discovery-selected earlier interventions pass synthetic future horizons2/4 in all trained seeds; untrained and manuscript cases fail. This is stronger known-process causal calibration. The patches replace broad component vectors, not state-only variables, and do not show a shared historical mechanism.
- **M-002 / HYP-003:** EXP-0010 finds no registered longer-context improvement under matched exposure. Extra context helps trained2048models relative to their own truncation, but not relative to separately trained256controls. Removing locus-boundary identity hurts overall with mixed glyph-only effects. This concerns transcription information, not established physical-line resets or historical syntax.
- **Limits:** the longer-prefix subgroup spans only4pages/2leaves; development pages have been repeatedly consulted. All new synthetic diagnostic pools are now exposed. Future adaptive work needs fresh keys/families/contexts. Historical HYP-001 through HYP-006 remain unresolved.

Results: [EXP-0008](../experiments/EXP-0008-results.md), [EXP-0009](../experiments/EXP-0009-results.md), [EXP-0010](../experiments/EXP-0010-results.md).

## Later status after EXP-0039–0040

- **HYP-005 remains open.** The known end-of-token to next-start dependency survives the independent Claston v101 glyph reading on the exposed nine-leaf validation panel ([EXP-0039](../experiments/EXP-0039-results.md)). A source-pinned, train-only transfer of the published `stock_flow` generator misses that held-out predictive edge contrast while roughly matching singleton-type abundance; `stroke` comes closer on edge contrast but misses singleton abundance and absolute predictive gain ([EXP-0040](../experiments/EXP-0040-results.md)). This rejects these particular frozen implementations under one assay, not every structured nonsemantic account. The first EXP-0040 run failed exact Timm replay because of an unrecorded Python hash seed; the corrected whole-panel rerun passed an independent 80-stream/16,000-shuffle audit after transparent source amendment. The repeatedly exposed validation and source-informed generator defaults bar a historical-process claim.
- **HYP-004 and HYP-003 also remain open.** The same observed edge dependency could arise from language, stateful encoding, or procedural generation. A stronger contextual no-copy predictor, explicit language/cipher controls and evidence external to these reused leaves are needed before ranking historical explanations. No translation or semantic anchor was produced.
