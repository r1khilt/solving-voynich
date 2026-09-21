# Hypothesis register

All entries below are **untested**. They are derived from the charter or introduced as methodological controls. No probabilities, historical plausibility rankings, or experimental support are assigned. Mechanisms can coexist; a model family is not necessarily a historical explanation.

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
