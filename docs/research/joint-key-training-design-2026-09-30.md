# Learning a conditional dictionary inverse: source, symmetry and evidence

Project design, 2026-09-30. This follows the implemented architecture and
completed local KEY-PROPOSAL-SYSTEMS-001 measurement. It does not establish
unseen-key recovery or a historical encoding rule.

[ALICEv1](https://arxiv.org/html/2509.07282v1), section4, distinguishes fitting a
few known substitution tasks from generalizing to new keys. Its large-scale
synthetic training supports testing many independent dictionaries, but the
single-symbol bijective, spaced-text task differs from this proposal. It cannot
supply a Mac timing estimate or a Voynich language assumption.
[Kambhatla, Born and Sarkar2023](https://aclanthology.org/2023.findings-eacl.160/)
is the previously reviewed synthetic decipherment/recurrence representation
precedent. PRIOR_WORK.md/ARCHITECTURE_REVIEW.md retain prior Voynich attempts and
assistance caveats. The project's earlier episodic failures, good supplied-key
reader result and failed unknown-key qualification remain evidence, not erased
by this larger model. See joint-key-proposal-implementation-2026-09-30.md and
key-proposal-uncertainty-2026-09-30.md for architecture and proper-loss choices.

## What is learned and what is supplied

Training creates a random dictionary, encodes two sampled Latin passages, and
supervises its full23-row key. Inference sees ciphertext only. Source alphabet,
Latin training language and deterministic one/two-glyph family are supplied
assumptions. Source lengths vary independently from64through224letters and are
not network inputs; spaces or unit boundaries are absent. All42units and
duplicate assignments are permitted. No bijection or fixed singleton count.

IID row draws are conditioned on rejection of the finite validation literal
and canonical key sets. This makes train/validation dictionary isolation exact;
it is technically a conditioned teacher distribution, not an unchanged unlimited
IID prior. The exclusions are negligible relative to42^23but are declared. The
network learns this finite teacher source/window/key population, not necessarily
the posterior under a separately fitted geometric-length language model. Its
probability is a proposal score, not an exact fitting weight.

Complete proper joint log loss is primary; no target row is masked because the
training plaintext lacks its source letter. That preserves legitimate uncertainty
instead of optimizing a plaintext-presence conditional. Used-row accuracy is
reported as a diagnostic with a ground-truth mask, which is unavailable to the
network. True complete-key accuracy can remain poor because unseen or ambiguous
assignments are not identifiable. Better validation density alone does not
establish better reading or calibrated rejection of nonsense.

## Fixed-shape duplication adds no information

The measured resource shape has fourrecords. Train/validate on two distinct
records duplicated in orderA,B,A,B, canonicalizing the original pair once.
The per-record contextual encoder has shared blocks and reset positions; there
are no record-ID embeddings or cross-record encoder interactions. Its output
memory is exactlyM,M in real arithmetic.

For any decoder queryqand keys/valuesk_i,v_i, repeating every memory entry twice
changes its cross-attention numerator and denominator by the same factor2:

`sum_i 2 exp(q*k_i) v_i / sum_i 2 exp(q*k_i)`

equals the original. Each decoder block therefore remains identical; induction
over blocks and causal choices proves the same full-key law. Shared parameters
and differentiability also give the same gradients in real arithmetic. No
BatchNorm, dropout, record bias or memory-relative positional terms break that
argument. Tests check both output and every parameter gradient on a real tiny
model; actual checkpoints compare2vs4on MPS and CPUdouble. This is a numerical
engineering choice, not four independent passages, extra evidence, an extra
language target or a distinct teacher prior. Do not advertise doubled source
exposure. Small numerical device differences remain bounded checks.

## Source and key isolation

Reuse the pinned, already-audited Perseus corpus and its conservative normalization,
source-role guards, segment boundaries and64-letter overlap exclusions. Both
capacities use the same6,497,939training letters in eligible>=512letter segments
from12catalogue authors. Select windows uniformly over eligible starts **after**
drawing their length; never cross a retained segment boundary. The remaining
319,711training letters are excluded consistently, not silently made available
to one arm. Corpus author/quotation/short-overlap limitations remain.

Pliny is the validation author; eligible>=224letter segments support every
declared source length. The original selection corpus has359,276letters; retain
the actual eligible total/segment list in the preparation manifest.64fixed
two-record episodes use fresh validation key seed72221. Literal and canonical
validation dictionaries are excluded from all training draws. Pliny was used for
prior source selection and is explicitly development/selection material here.
It is not an unopened qualification author. Nepos/Apuleius and prior cipher panels
are not opened; future fresh qualification must exclude their exposed spans and
register priors, structured nulls and full reading comparisons separately.

## Ambition with finite evidence

Two seeds at both actual5.4M/95Mcapacities receive identical20,000update exposures
per matched seed, batch4:80,000generated keys/fit and1,840,000supervised key-row
targets/fit. With variable source lengths, record actual original source letters
rather than pretending every sample has224letters. Seed matches are not identical
initial weights across incompatible model dimensions. Full generated episodes,
rejected draws, parameter digests and every update are retained for replay.

Measured update rates suggest roughly8.85hours for four fits before new data,
checkpoint/validation and sustained thermal overhead. Finite caps and all-arm
failure retention govern the run, not open-ended optimization. Validation selects
the lowest full-row NLL, with earlier checkpoint ties; both trained and initial
checkpoints remain candidates. Any eventual fit/reading comparison uses exact
global fitting and all positive/null cases, not selected rescued keys.

Mechanistic analysis follows actual competence: compare both seeds, renamings,
input perturbations and matched causal memory interventions against behavioral
key changes. A supplied row-ID probe, attention picture or manual dictionary
change would not identify an internal inference mechanism. No source language,
semantic feature, global workspace or Voynich decipherment is established.
