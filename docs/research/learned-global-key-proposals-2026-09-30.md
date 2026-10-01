# Learned global dictionary proposals: prospective next search branch

2026-09-30. Source-based design note, **not implemented or trained**. The
GLOBAL-KEY-SEARCH-001 fit result and separate reading comparison remain fixed.
No future training time, accuracy or latent mechanism is claimed here.

## Why dictionary proposals and text proposals are separate

The new shared-prefix neural core can supply readings missing from a
statistical candidate list, but only within a supplied dictionary bank. It
cannot restore an absent key. The existing source-only LSTMs are likewise
readers rather than key-inference networks. A dictionary proposal model must
learn an inverse problem: infer a consistent mapping from several ciphertext
records under a learned language prior. Its candidates still receive the
existing exact fitting likelihood and literal model code; a neural guess never
becomes a translation without independent recovery evidence.

## Relevant primary results and assistance

[Kambhatla, Born and Sarkar2023](https://aclanthology.org/2023.findings-eacl.160.pdf)
introduce a recurrence encoding and train a Transformer on2million synthetic
homophonic cipher/plaintext pairs. This is direct precedent for learning an
inverse from synthetic examples and removing arbitrary cipher-symbol names.
Their supplied single-symbol substitution and retained word boundaries differ
from our unsegmented one/two-glyph emissions. The earlier project PRIOR_WORK
review records human correction in some historical examples; do not equate
those examples with automatic unknown-script decipherment.

[Shen and Smith2025, ALICEv1](https://arxiv.org/html/2509.07282v1), sections2–3
and AppendixC, use a Transformer encoder and pooled cipher-symbol decoding.
Training examples encrypt quote text on the fly while preserving spaces and
punctuation. The main85M model trains100ksteps/batch96 in about6hours on anH100;
reported27M/308M runs take about2.5/16.5hours. Those are their measurements,
not estimates for this Mac. The1,000–1,500transition counts distinct training
keys, not training examples. Dynamic embeddings did not improve the main
ablation; the explicit bijective head trades accuracy for mapping structure.
None of these results validates a variable-emission Latin solver or a Voynich
language/cipher assumption.

These papers make a larger supervised proposal network reasonable to
investigate. They do not justify merely multiplying parameters in a text-only
predictor, importing bijection constraints into every historical family, or
choosing a full training budget before a local throughput/memory measurement.
The project's failed episodic/latent teacher controls and positive source-only
reader results remain relevant: supervision, latent slot identity and actual
unseen-key recovery must be separated.

## Proposed adapted architecture, not a claimed reproduction

Input only the original fitting ciphertext records and public coding context.
A per-record contextual encoder processes glyph sequences with explicit record
boundaries; global latent dictionary-row queries attend all fitting records.
Propose a categorical23-by42table over literal eligible units, with a joint
row-refinement module to represent correlations. Emit whole legal dictionaries
and fit-score them with the unchanged exact native verifier. Preserve unknown
unit boundaries, source lengths, duplicate-allowing learner family and fit-only
selection. Hard injectivity or known6singleton/17digram counts would be extra
assistance and require a separately named comparison, not a hidden repair.

Canonicalize observed glyphs by first occurrence across the fixed record order
and rebind output units to their original names. This can make rename
invariance exact for observed glyphs. Unobserved declared glyphs remain an
exchangeability problem: report it and handle the residual permutation group
explicitly or abstain on the proposal, preserving the ordinary solver's
outcome. Do not quietly remove such cases. Test all renamings on small alphabets.

Create synthetic training episodes only from permitted Latin training segments,
with new independent random dictionary keys and recorded canonicalization.
Key-row supervision is declared additional training information; fitting-time
plaintext is absent. Mask uninferable unused-row targets or model their
uncertainty explicitly; do not reward guesses of nonexistent evidence.
Validation uses reserved source-selection material with disjoint keys, while
Nepos/Apuleius remain excluded from training. Actual recovery needs a new
unused qualification after development. Structured/null input must not become
confident apparent plaintext; exact verifier score is also not semantic truth.

An initial architecture-scale study should compare a modest conditional model
with a larger roughly80–100Mclass model, matched data and bounded exposures.
These are proposed scales, not instantiated parameter counts. Benchmark random
weights/optimizer steps on the actual machine before choosing finite training
steps, memory and wall budgets. No paid/cloud resources, implicit credit
balance or H100-to-Mac timing extrapolation is assumed.

## What would count as an advance

All-case equal-budget proposal-plus-search versus ordinary global search,
source/model/code unchanged. Measure how often a newly proposed key beats the
old best, whether actual true-text support expands, recovery errors, iid and
structured-null flags, full failures and total compute. Do not select only the
previously bad cases or publish a training loss as a solver result.

For causal analysis, treat query/row representations as intentionally supervised
variables. Renaming and query-order controls distinguish symbol binding from
physical-slot shortcuts. Intervene on a proposed mapping while keeping other
key rows/content fixed; use matched incorrect/random row controls, downstream
literal decoding effects and both seeds. A global memory module is an
architectural choice, not evidence of Anthropic-like global workspace behavior.
Cosine or an easily decoded key label does not establish that the inferred
mapping is causally used. No trained or historical mechanism is claimed.
