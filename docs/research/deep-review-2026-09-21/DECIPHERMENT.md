# Decipherment: what prior successes actually assume

Review date: 2026-09-21. Research only: no new training, paid research API calls or decipherment claim. The [source ledger](decipherment.sources.json) records 25 primary papers, original reports and source-maintainer resources: 23 selected-section/documentation reads and two abstract-only entries. These are not 25 fully read or independently replicated papers. Final metadata audit corrected incomplete author lists and upgraded ALICE to v2.

The most consequential conclusion is that **generalizing across unfamiliar encodings is a separate objective from predicting one manuscript well**. Our EXP-0008 failure on new keys makes representation, task diversity and explicit channel assumptions high priorities.

## What different kinds of prior work provide

| Evidence | What it supports | What remains missing |
| --- | --- | --- |
| Yale custody and Zandbergen transcription documentation, D01–D02 | Manuscript access, transcription conventions and uncertainty | A ground-truth plaintext or settled reading unit |
| Voynich linguistic/entropy/topic/co-occurrence studies, D03–D06 | Quantifiable structure and distributional comparisons | A unique language, cipher or meaning |
| Self-citation generator and Naibbe cipher resources, D07–D08 | Competing structured generators and a reversible verbose-cipher positive control | Evidence that either historical process produced the manuscript |
| Neural substitution methods, D12–D15 | Learnable inference under supplied language/channel distributions | General unknown-script, unknown-language decipherment |
| Ancient-script methods, D16–D18 | Constrained lexical, phonetic or mapping search with external priors | Automatic elimination of those priors |
| Image methods, D19–D20 | Joint treatment of visual and textual uncertainty | Unsupervised discovery of arbitrary image-to-meaning mappings |

The ledger has exact URLs and method-specific reading limits. “Unsupervised” must name what is absent: a method may lack paired examples while still receiving a plaintext-language corpus or a strong phonetic prior.

## Key transfer: the most relevant distinction

Recurrence encoding indexes symbols by first appearance. It removes arbitrary names while retaining repetition structure. It does not make changes in homophone grouping or cipher family disappear. The English-trained Latin example reports 45.14% symbol error before human corrections; this was not fully automatic unknown-language recovery. [D13](https://aclanthology.org/2023.findings-eacl.160/)

ALICE v2 trains on plaintext/ciphertext pairs with random one-to-one substitutions. It includes symbol-consistent outputs and an optional inspectable permutation head. Its dynamic-embedding ablation does not improve the reported comparison; its layerwise analyses do not establish causal algorithm recovery. This motivates structured outputs and diverse-key training, not imposing bijectivity on Voynich. [D14](https://arxiv.org/html/2509.07282v2)

The 2026 homophonic LSTM study uses a shared code-symbol pool with invariant plaintext meanings; keys select subsets. Its strong accuracy therefore does not demonstrate recovery under arbitrary reassignment of symbol meanings. [D21](https://arxiv.org/html/2606.05078v1)

Our next benchmark must distinguish pure relabeling, new homophone partitions, new transition parameters, new languages and new mechanisms. Calling all of these “new keys” would conceal where generalization succeeds or fails.

## Image evidence and historical decoder structure

The Copiale work shows that a real decoder can include multiple symbol roles and multiletter operations. A small typed channel is a more informative hypothesis space than an unrestricted language model or a mandatory glyph-to-letter permutation. [D11](https://aclanthology.org/W11-1202/)

Direct image-to-German Copiale work in 2026 uses the known cipher to generate supervised training material. It informs an image-noise benchmark, not unknown-key discovery. Earlier joint image decipherment also simplified its evaluation documents. These assumptions belong in any replication. [D20](https://arxiv.org/html/2606.27700v1), [D19](https://arxiv.org/abs/1810.04297v3)

For Voynich, preserve alternate glyph readings and alignment to manuscript coordinates before fitting a language model. A model must not silently “correct” awkward glyphs toward a preferred translation. Record scan source terms and local manifests when this branch starts.

## Competing explanations to keep alive

The following are **our test proposals**, not conclusions from the cited sources:

| Candidate family | Observable prediction worth testing | Strong confound/control |
| --- | --- | --- |
| Substitution or verbose encoding | Constrained mappings and expansion rules transfer across passages | Known-language controls; scrambled inputs; cost for exceptions |
| Stateful encoding | Compact state improves joint prediction and updates consistently | Copying and local-template generators with matched short-range statistics |
| Abbreviation or omitted material | One bounded channel explains systematic missing information | Matched deletion rates; language-prior hallucinations; alternative segmentation |
| Copy/mutate production | Predictable relations to earlier forms and layout-dependent source selection | Ordinary language repetition and stateful-cipher controls |
| Mixed production or scribal conventions | Changes track independently established metadata | Section/topic/layout and transcription confounding |
| Nonsemantic structured production | Simple generator explains held-out diagnostic statistics | Meaningful verbose-cipher controls with similar surface entropy |

Fit distributions and inspect rules under each family; the easiest-to-fit family is not automatically the historical winner. Some pairs may be indistinguishable from the available text.

The 2026 LLM pastiche report is a source hypothesis, not established decoding. Its primary page pairs identifier `2609.20835` with a July 28 submission date; that unresolved conflict is retained. The shorthand proposal is abstract-level in this review. Neither supplies a validated reading here. [D23](https://arxiv.org/abs/2609.20835), [D24](https://mattruckman.com/papers/voice-but-not-the-song/voice-but-not-the-song.pdf)

## Semantic anchors and information still needed

A text-only model can estimate repeatable relations without knowing what they mean. Independent evidence could restrict meanings, but an anchor must be chosen without seeing which guessed translation it favors.

Candidate acquisition work: versioned transcription alternatives; manuscript-coordinate alignment; defensible section/scribe annotations; comparison corpora with language, period and genre metadata; and blinded image motif annotations. Availability, rights, reliability and coverage require separate verification. This review does not claim those resources have been acquired.

For image associations, hold out physical leaf groups and control section, layout and text length. For a candidate language, compare alternatives under the same channel complexity and selection budget. For abbreviations, test known historical positives first.

A plant guess plus a flexible decoder can manufacture agreement. A frozen mapping that makes multiple previously unseen, independently checked predictions is much stronger.

## Decision

Prioritize fresh-key inference and explicit observation channels. Keep bounded program search as a later candidate: the grammar and evidence score must be defined first. DreamCoder supplies inspiration for learned search and reusable primitives, not missing manuscript supervision; it is also T13 and counted once. [D22](https://arxiv.org/abs/2006.08381v1)

The next [architecture and experiment design](NEXT_DESIGN.md) treats known-language decoding, unknown observable-process inference and historical semantics as separate tests. None is solved by a fluent output alone.
