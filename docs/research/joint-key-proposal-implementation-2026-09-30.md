# Whole-key conditional proposals: implementation and evidence boundary

Project design, 2026-09-30. This implements the prospective conclusions in
[learned-global-key-proposals](learned-global-key-proposals-2026-09-30.md) and
[key-proposal-uncertainty](key-proposal-uncertainty-2026-09-30.md). Neither note
established neural key recovery. GLOBAL-KEY-READ-001's exposed development result
motivates a different candidate generator: wider statistical fitting rescued one
key but worsened another, with true text still outside the bank in both cases.

## Prior work and choices

[ALICE v1, Shen and Smith, 2025](https://arxiv.org/html/2509.07282v1), sections
1.1, 2 and 4, studies one-to-one substitutions with preserved spaces/punctuation.
Its pooled symbol representations enforce consistency, and its bijective variant
uses a learned-query cross-attention head. It reports generalization for an 85M
model on a large quote corpus. This supports testing contextual representations
and an explicit key output at substantial scale. It does not establish our
variable-length, duplicate-allowing family or any Voynich interpretation.

[Kambhatla, Born and Sarkar, 2023](https://aclanthology.org/2023.findings-eacl.160/)
provides the previously reviewed recurrence-representation/historical decipherment
comparison. Prior Voynich applications and transfer caveats remain in
PRIOR_WORK.md and ARCHITECTURE_REVIEW.md. This implementation is a project
proposal, not a reproduction of either architecture or a claimed new method.

The project uses a fixed **23-row source alphabet and six observed glyph types**,
each source row emitting one or two glyphs. There are 42 legal units, repeated
units are allowed, and source spaces/segmentation are not supplied. These are
restrictive synthetic assumptions; they are not findings about the manuscript.
No Sinkhorn/bijective head, fixed number of singletons, fixed plaintext length
input, or known-key prior is imposed. A known source alphabet/language remains
an important future training assumption.

## Implemented model

One contextual encoder processes each of up to four records independently, with
positions reset per record, then concatenates their memories. The decoder
cross-attends to all records while choosing each dictionary row in sequence.
Its inputs are current source-row identity, choice position, and the previous
chosen row/unit; causal self-attention excludes later choices. The model defines
`q(K | C) = product_j q(K_j | C, K_<j)`. Full-row cross-entropy is proper log loss
for the declared synthetic joint distribution. It is not automatically the
posterior under a separately fitted language model or a historical cipher.

| Configuration | Width / heads | Encoder / decoder blocks | Actual parameters |
| --- | --- | --- | --- |
| Control | 256 / 8 | 4 / 2 | 5,423,146 |
| Large | 768 / 12 | 8 / 4 | 94,981,674 |

Both use pre-LayerNorm, GELU, fourfold feedforward width, zero dropout, learned
positions, and independently initialized blocks. The large implementation is
about 220 times the early 430,720-parameter predictive model, but has a different
purpose and cannot be compared by parameter count as evidence of better reading.
[PyTorch 2.14 TransformerDecoder documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoder.html)
warns that template-cloned layers start identically; explicit ModuleLists avoid
that initialization issue. Autoregressive generation currently recomputes decoder
prefixes and has **no KV cache**; measured costs must include that implementation.

## Renaming, order and uncertainty

Observed glyphs receive IDs by first occurrence, using only ciphertext. Declared
unseen glyphs remain uncertain: literal key rebinding returns the complete
residual permutation orbit, deduplicated, and fails before returning a partial
orbit if the permutation cap is exceeded. An orbit is a set of proposals, not
extra prior mass. This permits exact input glyph-renaming invariance and
set-valued rebound equivariance. It does not identify absent glyph assignments.

The encoder/decoder are invariant to permuting **already canonicalized** records
in real arithmetic. Recanonicalizing after record reordering can change glyph
IDs, so the whole preprocessing pipeline is not claimed invariant to record
ordering. Use a fixed record order unless a separately registered averaging
scheme is introduced. Likewise, changing source-row decoding order defines a
different normalized model law; no order-invariance claim is made.

The earlier analytic counterexamples remain relevant: independent correct row
marginals can produce impossible combinations, and plaintext-dependent masking
of unused-row targets changes the objective. This implementation supervises all
23 rows, including unused rows, rather than masking them. Under truly absent
evidence, irreducible uncertainty must remain. A network can still learn spurious
teacher priors; iid duplicate-allowing teachers and separate prior-shift controls
are required before interpreting competence.

## Checks and causal analysis boundary

Tiny exact enumeration normalizes all 36 legal two-row keys; sequential and
teacher-forced probabilities agree. Perturbing a current/future teacher label
leaves earlier logits exactly unchanged. Tests cover padding and record reset,
batch isolation, deterministic sampling and teacher replay, duplicate assignments,
all 24 four-glyph renamings, complete unseen orbits, independent block
initialization, full parameter gradients, and a genuine finite-difference check.

KEY-PROPOSAL-SYSTEMS-001 measures actual optimizer and whole-key proposal cost
before selecting a finite language training budget. Its random artificial data
does not evaluate language learning or key generalization. Further work needs
disjoint training/development/qualification source blocks and dictionaries,
key-bank coverage and reading accuracy, seed stability, shuffled/structured nulls,
and prior-shift testing. Proposed keys require exact shared-key fitting; their
neural probability must not become a reader's key weight without a new model.

Explicit key choices make interventions meaningful: test whether changes to
passages or selective internal states cause appropriate changes to later key
assignments, compared with identity/restoration, random/norm-matched and output
controls. Query-row identities are supplied by design; a probe that recovers
them is trivial. Attention/early-exit patterns alone would not establish a
latent key-update circuit. No causal mechanism or decipherment is claimed here.
