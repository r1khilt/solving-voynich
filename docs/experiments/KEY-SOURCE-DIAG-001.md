# KEY-SOURCE-DIAG-001: frozen-source replay of exposed key-search failures

2026-09-30. Exploratory diagnostic, not fresh qualification or a repair of
BLIND-CHANNEL-CONFIRM-001. No key is refit, repaired, reranked for deployment,
or replaced. NEURAL-READER-001 remains unchanged.

## Question and source review

The new supplied-key reader passed, but the previous unknown-key experiment
failed. Does the improved source resolve the old objective's preference for
wrong keys, and how much error remains with the *unchanged learned keys*?
This distinguishes a source improvement from a search improvement before
spending a larger budget on a new unknown-key campaign.

[Berg-Kirkpatrick and Klein (2013)](https://aclanthology.org/D13-1087.pdf)
show strong restart dependence and cases where better fitted likelihood gives
worse reading accuracy in supplied-symbol homophonic decipherment. Their GPU
restart counts are not a resource prescription for our variable-unit model.
[Nuhn et al. (2013)](https://aclanthology.org/P13-1154.pdf) motivate stronger
source context while distinguishing it from exact optimization of a weaker
source. [Kambhatla et al. (2018)](https://aclanthology.org/D18-1102.pdf) support
neural source scoring in key search but do not justify treating a beam result
as an exact marginal. [Reddy and Knight (2011)](https://aclanthology.org/W11-1511.pdf)
describe manuscript/transcription uncertainties; this diagnostic tests no
Voynich language or encoding hypothesis. Historical paper statements are
interpreted in their publication context, not as current unsolved-cipher status.

The project's [conditional geometry analysis](../research/cipher-conditional-geometry-2026-09-30.md)
explains why source loss and within-cipher choices can disagree. All sources
below were selected without this diagnostic's readings; these old ciphertexts
and failures are already exposed to the researcher.

## Fixed inputs and arms

Use all eight positive keys and all eight matched glyph-shuffle nulls from the
original `data/manifests/blind_channel_confirm001.json`, the original selected
key freezes and original evaluation. Four fitting and two transfer records per
case; 224 true letters per positive record. Never filter, redraw, or repair a
record/key. Keep old unsupported readings and all old FAIL statuses intact.

Four fixed source arms, all previously completed and audited:

- LATIN-SOURCE-COMPACT-001 small, order12/tau64.
- LATIN-SOURCE-COMPACT-001 large, order12/tau64.
- LATIN-SOURCE-MODEL-001 large neural31103, selected step4000, beam128.
- LATIN-SOURCE-MODEL-001 large neural31109, selected step4000, beam128.

For positives, decode with the original learned key and, separately, the true
generating key. For nulls, use only the original learned key; they have no true
plaintext. This is144readings per source,576total. The old order3 results are
retained as a labeled historical comparator, not recomputed or pooled with new
measurements. Literal key description lengths use the unchanged old coding
context and separate literal-code reference. No language-model parameter is fit.

Sallust/Tacitus are excluded from the broad source training by the recorded
corpus policy. Nevertheless, this is an answer-informed, adaptively selected
diagnostic on an exposed campaign; it is not a fresh author/key test.

## Exact versus approximate objectives

Statistical key comparisons use the *exact path-marginal* fit score plus the
unchanged literal key code. Report `learned_total_bits - true_total_bits` for
each positive key: positive means the true key is a known better objective
candidate. This does not prove it is globally best or uniquely identifiable.

Neural comparisons instead use the sum of **returned-path joint scores** minus
the same literal code cost. These are lower bounds on the joint MAP key/text
score, not cipher marginal likelihoods and not directly comparable to the
statistical totals. For each record an upper bound on the best path is the
maximum of its returned score and its discarded-prefix completion bound.
Sum lower/upper bounds across independent records, then subtract key code cost.
Only disjoint intervals establish ordering of the two *joint MAP* objectives
under ideal real arithmetic. Numerical versions are labeled floating-bound
diagnostics, not interval-arithmetic proofs. A simple difference of beam scores
cannot certify a key preference. Missing support yields null scores and a
separate support flag, never infinity in JSON or removal from denominators.

No pass gate: report all per-key fit/transfer errors, exact records, support,
source/code scores, replay deltas, neural score-bound intervals, and null
likelihoods/search summaries. Nulls are old glyph shuffles with no accuracy
target, not the newer plaintext-permutation controls. No new language-detection
threshold or claim is introduced.

## Validation, freeze, resources

Freeze/push/verify this protocol, runner, tests, original inputs and all four
source/audit identities before the sole run. The runner checks every source
prerequisite and all original bindings. Separate backward statistical inference
replays marginal/MAP/node counts and returned path scores within1e-7nats.
Neural returned paths replay full-forward within1e-4nats per letter. Each positive
record's edit count is compared with the independent full-grid calculation.
Every supported returned reading re-encodes exactly; all true keys re-encode
their stored positive answers. Case partial archives are retained.

One GPUprocess, two PyTorchhostthreads, BLASthreads1. Wall900seconds, CPU900seconds;
500,000statistical lattice nodes/record,200,000neural expanded prefixes/record,
8GiB GPUdriver limit checked per neural record and8GiB host planning. Estimate
under six local minutes from the previous384reading/199second run, but this is
not a promised duration. Zero paid API/cloud use. Exclusive attempt/output
markers; a failure is retained, with no restart, case replacement or extension.

Command after publication, `PYTHONPATH=.:src`:

```
.venv/bin/python scripts/run_key_source_diag001.py --freeze REV
```

This campaign changes the evidence informing the next *fresh unknown-key*
pipeline; it establishes neither blind recovery nor a historical reading.
