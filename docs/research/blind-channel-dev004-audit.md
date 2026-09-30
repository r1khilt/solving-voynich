# Independent DEV004 replay audit

Implementation prepared 2026-09-29 before empirical artifact access. This note
describes the auditor and its artificial validation; it does **not** report an
empirical pass. Root owns registration, ordered freezes, execution and results.

`scripts/audit_blind_channel_dev004.py` imports no production estimator,
inference, edit-distance, coding, runner, or other auditor algorithms. Imports
perform no artifact reads. The pure `audit_payloads(...)` entry point accepts
in-memory dictionaries for testing. The empirical command requires both the
prediction-freeze revision and an explicit access gate:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/audit_blind_channel_dev004.py \
  --freeze PREDICTION_FREEZE --after-evaluation
```

Default output is `results/BLIND-CHANNEL-DEV-004/independent_audit.json`.
Existing output is never overwritten. The command sets a 900-second CPU limit
and a 1200-second wall alarm, within the registered combined block budget.
Resource failure produces no claimed completed audit or approximate replay.

## Byte identity and stage boundaries

The CLI requires its own source to match the supplied Git revision. It checks
the case/corpus manifests, source-selection manifest, prediction manifest, and
four DEV003 learned-key freezes against that same revision. The case manifest
must also have the exact registered SHA-256. It verifies source-selection and
prediction provenance, evaluation's prediction-freeze identity, and every
loaded artifact's SHA-256 and declared byte count before JSON/gzip decoding.
Relative paths cannot escape the repository. The final report hashes the
auditor, evaluation, tracked manifests and prediction archive, and records the
verified input identities.

Only Caesar and Virgil source-text entries are opened for source estimation.
They must retain their P1/P2 roles, 50,000 letters each, selected-text hashes,
alphabet and strictly increasing body boundaries. The registered fit,
transfer, answer and fixed-key artifacts are accessed only by the gated
post-evaluation replay. No Sallust/Tacitus source, new case, or final author is
loaded. The source estimator never receives ciphertext or answer text.

## Source-only replay

The independent estimator counts complete ngrams separately by length and
record, rather than invoking the production per-position estimator. It
reconstructs the smoothed unigram and every context through order three using
the fixed recursive suffix prior. A separate count-based scorer includes the
short prefix of each record and never joins body boundaries.

The audit recomputes all 19 Caesar-fit/Virgil-validation configurations in
their registered order, every validation bits-per-character value, stable
per-order choices and the global source-only winner. It independently refits
the selected four models on Caesar+Virgil and checks every serialized context
and probability, full normalization and alphabet order. It separately rebuilds
order-one/tau64 to verify the original-source reduction. Probability and
source-selection tolerances are 1e-12 and 1e-10 bits/character, respectively.

## Fixed-channel inference and evaluation

The auditor validates each deterministic one-state channel and preserves all
source-row aliases. Its inference first discovers reachable offset/history
nodes, then works **backward** from the final stop, independently summing and
maximizing suffix probabilities. This differs from the production higher-order
forward recurrence. Full finite histories are retained even when two current
probability rows coincide. Matching units consume at least one glyph, so the
graph is acyclic. It raises above three million nodes; it never prunes a beam.

All 180 archived source-record predictions are replayed: five priors across
six fixed channel arms, four fit records and two transfer records each. The
audit checks marginal and MAP joint scores, reencodes every archived MAP
reading, and separately evaluates that reading's full source/continue/stop
probability. A distinct equally scoring MAP representative is allowed and
counted explicitly; it is not silently called a literal agreement. Inference
and aggregate score comparisons use absolute tolerance 1e-7.

Literal model cost is derived independently from the bounded field widths,
including unused source rows and each emitted glyph. The unchanged additional
one bit is the legacy channel-versus-iid-family selector, not an assumed
five-source selector. Source comparisons remain the registered conditional
sensitivity panel. The audit recomputes every reported edit count, decoded and
gold length, exact/supported record count, aggregate marginal/data/total bits,
and per-record MAP surprisal. Shuffle controls must retain null accuracy fields.

## Decision-bank and dictionary-floor checks

For each of the 36 old-source decision records, the audit verifies the fixed
32/256 draw counts, record/arm seed formula, domain-separated stream seeds,
bank hashes, MAP inclusion and candidate first-occurrence order. Every distinct
sample must reencode exactly and have positive source support. Multiplicities
in the 256-reference bank are retained.

It recomputes **every** candidate-reference edit loss and candidate mean/total,
then verifies the selected minimum and deterministic tie rule. The independent
distance implementation uses ordinary integer dynamic programming, vectorized
over reference strings. Insertion relaxation is a cumulative minimum across
prefix columns. It does not use the production Myers bit-vector routine; a
separate scalar Wagner–Fischer recurrence validates the vectorized result on
fixtures and computes correctness metrics. Counters distinguish all weighted
draw pairs from distinct string pairs actually calculated.

This audit does **not** numerically regenerate the posterior sample banks from
their PRNG streams. It verifies their identities, support, dimensions and all
downstream decisions; it cannot establish from one archived bank that the
sampling distribution is correct. Separate production/independent fixture
tests of the sampler supply that qualification. The final report explicitly
sets `posterior_banks_numerically_resampled` to false.

The 24 positive record/arm dictionary edit floors receive an independent
zero/one shortest-path traversal over glyph-offset × truth-offset states.
Edges represent a source insertion, truth deletion, or match/substitution;
zero-cost matches go to the front of a deque and unit-cost edits to its back.
This independently checks the production topological edit-product recurrence.
The five-million-cell cap fails loudly. No source prior enters this diagnostic;
with zero source probabilities it would be a relaxed structural lower bound.
Null cases must not contain truth-based floor reports.

## Artificial validation and resource indication

The owned test file is `tests/test_blind_channel_dev004_audit.py`. Its fixtures
include exhaustive Fraction path sums/MAP checks for orders 0–3, every pair of
four small units and binary strings through length four; independent rational
smoothing and record-reset checks; exhaustive tiny dictionary floors;
Unicode/empty/long-string integer-distance comparisons; decision-bank
multiplicity/ties; and a complete artificial 180-prediction/36-decision panel.
Adversarial mutations alter the source grid and scores, selected source,
probability tables, body boundaries, key, case/record inventories, MAP strings,
likelihood, literal cost, metrics, conditional surprisal, edit floors, risk
banks, seeds and decisions. They must be rejected. Import isolation, hash-before-
decode, path containment, access gating and no-overwrite checks are also tested.

One bounded synthetic timing used seed 20260929, alphabet `abcd`, one
224-character candidate and 256 independent random references of lengths
220–228. The integer batched distance call took **0.04269 seconds**; the first
eight distances matched the separate scalar recurrence. The disposable
process had a 30-second CPU cap and single-thread numerical environment.
Multiplying that one call by 33 candidates and 36 records gives roughly
51 seconds for that hypothetical distance workload; this is only an
extrapolation, not a measured empirical-audit runtime. Source reconstruction,
exact inference, edit floors, archive I/O and duplicate-bank savings are
separate costs. No empirical predictions or answers were opened for this timing.

Passing this implementation audit would establish arithmetic/artifact
consistency for the fixed diagnostic. It would not independently establish
fresh-key generalization, a uniquely correct dictionary, global minimum edit
risk, a source language, or a Voynich decipherment.
