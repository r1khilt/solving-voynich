# Research protocol

This protocol governs active research. The user has authorized literature/methods review, corpus acquisition, and model/pipeline implementation; bounded validation runs are recorded individually.

## Evidence and claims

Treat the manuscript's text, transcription, layout, and imagery as distinct evidence layers. Preserve uncertainty rather than forcing disputed glyphs, word boundaries, or image identifications into ground truth. Keep model suggestions distinct from observations.

A decipherment proposal should specify a constrained decoding procedure, explain its degrees of freedom, make predictions on material excluded from development, and withstand independent application. Plausible translations, low prediction loss, visual resemblance, and recovered patterns alone do not meet that standard. Do not force readings from guessed plant identifications or permit unexplained material to become unlimited filler.

## Before each experiment

Review previous applications to Voynich and relevant work on the method in decipherment, cryptography, and machine learning. Record what supervision, data size, assumptions, and evaluation differ from our proposed use. New architecture features must have a stated rationale and a control configuration.

Create a record with:

- Stable experiment ID, question, linked hypotheses, and exploratory or confirmatory status.
- Input source IDs, immutable versions/checksums, units of analysis, uncertainty handling, and preprocessing.
- Explicit generator/model/decoder assumptions and what would make the hidden variables identifiable.
- Train/development/test assignment fixed before fitting or adaptive selection. Account for adjacency, repeated material, transcription variants, sections, scribes, and other possible leakage paths.
- Baselines and controls, primary metrics, decision criteria, uncertainty estimates, and failure criteria.
- Seeds, architecture, optimization settings, software environment, commands, expected runtime/cost, and stop conditions.

Use held-out pages or larger blocks when the question requires it, rather than assuming random token splits are independent. Choose the split unit to fit the claim; a test within one section does not establish transfer across sections. Reserve final holdouts from repeated adaptive exploration.

## Minimum controls by proposal

| Proposal | Required comparison or limitation |
| --- | --- |
| Filler removal | Matched deletion rates and positions, frequency/length controls, frozen selection rules, remaining-text length effects, model complexity, and shuffled/nonsemantic controls. |
| Synthetic recovery | Known latent ground truth, leakage audits, disjoint languages/alphabets/keys where relevant, mechanism-family holdouts, easy and hard null cases, and unrecoverable cases. |
| Tiny predictive model | Simple statistical/compression and copy/mutate baselines, split isolation, multiple seeds, and separation of prediction from historical interpretation. |
| Mechanistic interpretation | Causal interventions with matched controls, seed/model stability, and measured behavioral effects; attention visualization alone is insufficient. |
| Cross-generator circuit similarity | Alternative generators, architectural bias controls, matched fit/data size where feasible, and a clear statement that similarity does not uniquely identify a source process. |
| Semantic decoding | Frozen mapping/rules, accounted exceptions, independent text, alternative readings, and external constraints not selected after seeing the output. |

## After each experiment

Record exact commands, input/output manifests, compact metrics, failures, uncertainty, cost when available, and which criteria were or were not met. Separate observation from interpretation. Log researcher model/provider/version and relevant prompts when available; do not claim an exact model identity that tooling does not expose. Retain candidate and negative-result history so repeated searches and multiple comparisons are visible.

Save lightweight reproducible results in Git. Store large assets/checkpoints outside tracked paths with checksums and documented retrieval or regeneration instructions. Revisit dependencies and artifact retention as actual requirements become known; no framework is selected at setup.

Update the notebook and hypothesis status, perform relevant validation, and commit and push a coherent checkpoint. Record blocked runs rather than silently substituting a different experiment.
