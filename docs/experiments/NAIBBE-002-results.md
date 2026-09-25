# NAIBBE-002 — near-exact plaintext, incomplete rare-letter key recovery

**Registered decision FAIL; independent audit PASS.** After removing the supplied cross-table same-letter links, the Latin decoder recovered130/138 key entries and made **11 edits in12,326 fresh characters**, CER **0.08924%**. The same frozen key produced8,182/8,192 exact token chunks. This is substantial text recovery on a known-answer control, but the unweighted key requirement was≥95%;130/138 is **94.20%**, so the four-part gate fails. That threshold is unchanged.

Every incorrect Latin mapping is a **y/z swap in four tables**. All126 assignments for the other21 letters are correct. A conditional exhaustive diagnosis shows the frozen source-scoring model actually prefers these rare mistakes on the fit block. More restarts under the same objective would not make the correct key its preferred answer. No Voynich text, language, glyph reading or meaning has been recovered.

## Assistance removed and assistance retained

[NAIBBE-001](NAIBBE-001-results.md) supplied the true same-letter group across all tables. This experiment supplied138 separate anonymous local classes and required inference of six independent23-letter permutations. It did **not** merge identical-looking glyph strings from different tables. Correctly connecting most of those local classes to shared plaintext letters is the new recovery achievement.

However, six-table membership, within-table unigram/prefix/suffix same-letter linkage, role grammar, ciphertext-token boundaries, alphabet size/letters and a Latin language candidate remain supplied. The legal parse lattice is constructed from the known codebook roles. Thus this is restricted homophone-group inference, not blind discovery of the entire Naibbe mechanism or a historical Voynich cipher. The source objective omits card-deck probabilities and is not marginal ciphertext evidence.

## Freeze, resource use and validation

The [registration](NAIBBE-002.md), source and immutable inputs were committed/pushed/remotely verified at `2aee093e2db46e41ce610bf50760327188ff0430` before any target fit. Fit uses exposed Pliny tokens[0,8192). New transfer tokens[18432,26624) lie beyond both001 evaluation blocks. The independent source priors are unchanged317,326-character Caesar Latin and English texts; Pliny plaintext does not train them. Data preparation checks gold grammar support mechanically but does not optimize or print it.

All three arms completed8 starts×7 cycles under their20-minute deadlines. They ran concurrently on the Mac: Latin local+coordinated462.621s, Latin local-only420.523s, English local+coordinated518.024s. Sum of arm wall runtimes1,401.168s; concurrent elapsed approximately518s. No paid compute or neural training. Both Latin neighborhoods selected exactly the same key at their first cycle; further registered restarts did not change it. This establishes tested stability, not a proof of global optimum.

Selected keys/traces and the independently written auditor were pushed/remotely verified at `8b27c21f397d7c0e3eb5fa8a03f6fae5aa2372ba` before answer access and fresh transfer evaluation. Evaluator and auditor each ran once without a result-dependent repair. All data/checkpoint/prediction bulk remains ignored; compact results and provenance are tracked.

The auditor independently reconstructs exact local-class supports from original table rows and evaluator-only original table labels, checks raw cipher/plaintext slices, and rescores all source strings/lattices using dictionary counts and a separate dynamic program. It verifies all **16,384 token lattices**, **65,536 emitted chunks**, keys, hash links, key metrics, edit distances and four criteria. Largest source-score difference1.46×10⁻¹⁰; audit4.323s. It does not independently rerun optimization or establish the actual historical card/table draws.

Full suite: **1,181 passed,8 skipped,23 subtests passed**,264.27s. Fifteen relevant new/previous recovery tests and changed-file Ruff/compilation pass. Five pre-existing full-tree Ruff issues remain in unchanged legacy files, as recorded in NB-190. No unrelated working changes were committed.

## Fresh transfer observations

| Decoder | Correct key entries /138 | Edits /12,326 | CER | Exact chunks /8,192 | Correct ambiguous chunks /201 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latin local+coordinated, primary |130|11|0.08924%|8,182|199|
| Latin local-only |130|11|0.08924%|8,182|199|
| English local+coordinated |128|136|1.10336%|8,061|193|
| Latin known-key oracle |138 supplied|3|0.02434%|8,190|199|

Primary criteria: CER≤2% **PASS**; within0.5percentage points of oracle **PASS** (difference0.06490points); macro key≥95% **FAIL** (94.20%); unique-class-weighted key≥99% **PASS** (99.93366%). Formal overall **FAIL**. Reporting near-perfect plaintext must not quietly substitute a different success criterion.

The frequency metric uses only12,059 transfer character positions whose candidate paths uniquely determine the local class. **267 positions are excluded** because their generating local class is unresolved. Eight local classes have zero such unique evidence in this block. This does not prove they are absent from all candidate paths or impossible to infer from context. Published aligned plaintext does not give the original table-draw trace, so no such trace was invented for scoring.

On fit, Latin has9/12,483 edits and unique-class weighted key99.97544%; English has133 edits and99.05869%. The English control now makes materially more transfer errors, but one control and uncalibrated source scores do not establish general language identification. Coordinated global moves give no benefit over local-only on this sample; both recover the same key.

## What caused the remaining Latin key errors

This section is **post-hoc diagnosis of the exposed fit result**, not a new confirmatory experiment. Keep all other126 letter assignments fixed at gold and enumerate every one of the64 possible orientations of six y/z pairs. The learned key is the **unique best-scoring orientation in that restricted family**;15 orientations score above the true key. Learned fit score−28,282.512515 exceeds gold−28,282.874942 by0.362427 source-score units. The correct key scores better on exposed transfer by2.116379, but that cannot be used to refit or claim a new transfer success.

The source corpus contains only **8 y characters and zero z characters**. Its observed y followers are r4/i1/n3. Inspecting all16 changed4gram positions between learned-key and known-key fit parses exactly explains the score difference: each of four y→z changes slightly lowers the score at the replacement itself, while the immediately following character gains more. Later affected positions have zero difference. Both corresponding three-character contexts are absent at those following positions; lower-order smoothing drives the preference. This identifies a concrete rare-context weakness of the fixed source prior, rather than an unexplained failure of the search to find gold.

An independent reviewer replayed all64 fit scores and16 contributions exactly using direct corpus counts and scalar conditional probabilities. At the bigram level, an unseen follower after z receives1/23≈.04348; the same unseen follower after y receives.1/(8+2.3)≈.00971. That discrepancy rewards the wrong preceding letter; one case also gains from the trigram component. This validates the sparse-source/smoothing diagnosis, not a claim that every alternative smoothing method will fix it.

The uniform additive fallback and sparse, different-genre source data are therefore specific targets for improvement. No one claimed that unseen letters or zero uniquely assigned counts alone prove absolute nonidentifiability: every table's y/z pair appears somewhere in the candidate support, and the restricted objective has a unique maximum. The challenge is trustworthy inference under weak source evidence, not retroactively excluding inconvenient errors from the metric.

- [Evaluation](../../results/NAIBBE-002/evaluation.json), SHA-256 `81afa0fe7467ca1123081258dab6e40ba668a65a74b7c4b288c7b808bc79daa2`.
- [Independent audit](../../results/NAIBBE-002/audit.json), SHA-256 `c98f9082426b21419b1246defddf2e4e64cc76cd70136eed28411f175711b868`.
- [Conditional rare-key diagnosis](../../results/NAIBBE-002/posthoc_rare_key_diagnostic.json), SHA-256 `f25bba9bb640869d1eeaad3945f5a13d7b305d435b247edd04eaf1c2038c9d2e`.

## Next work

Improve the source prior using source-only validation of hierarchical/adaptive backoff and better rare-symbol uncertainty, then test a separately frozen solver on genuinely unscored text. Do not tune on the newly exposed transfer and relabel it fresh. Remaining unscored published tokens start at26624. Simultaneously prepare a genuine historical ciphertext control such as Borg, whose data/contamination issues are recorded in [the feasibility review](../research/historical-controls-2026-09-25.md). Removing manufactured codebook assistance on a historical document matters more than polishing an impressive score on this one control.

Any future neural or mechanistic branch should target these observed inference weaknesses and compare against this inexpensive explicit solver. Direct-ink measurement remains deferred until a candidate decoder makes a distinguishable segmentation prediction. No outcome here establishes Voynich semantics.
