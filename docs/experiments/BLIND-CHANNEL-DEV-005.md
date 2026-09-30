# BLIND-CHANNEL-DEV-005: refine the dictionary under the selected source

2026-09-29. Registered before any refinement fit. **Adaptive exposed development.**
No new author, key, source language, historical cipher or manuscript is tested.

## Question and fixed intervention

[DEV004](BLIND-CHANNEL-DEV-004-results.md) independently established a concrete
direction: its source-only selected order3/tau256 makes the generating B2
dictionary beat the existing learned dictionary, and the learned dictionary
excludes exact truth. Can a ciphertext-only search now find better reusable
units without hand-fixing exposed rows? Prior-method rationale is the
[source-context review](../research/fixed-channel-source-context.md),
[search review](../research/unknown-unit-search-alternatives.md) and their
primary decipherment references. This is an objective intervention plus
disclosed warm-start refinement, not an independent discovery from scratch.

Use the unchanged DEV001 manifest and all four B/paired-shuffle cases. Use
only the frozen DEV004 primary source (order3/tau256), selected without cipher
scores from Caesar/Virgil. The immutable source-selection manifest is
`1a0136627f04e3b68b7e2b3d801e5d8cdebc617fc1e5f3f0fa7b7e344ab74c60`;
its ignored source archive is
`a314c7120a14f6636a7b0a129e803fc517c0488016422a3fd84764b092179615`.
Warm-start each case from its own DEV003 learned dictionary, already frozen
at `8e6e5c8d17bf4bdd568f22a2e1df22d12972b1ac`. No oracle/gold row,
previous gold-assisted witness, transfer string or answer enters fitting.

## Search and criteria

One deterministic best-improving search per case. At each sweep, score all
nonidentity pair swaps and all single-row replacements from the complete42-unit
one/two-glyph pool. All23source-letter rows are eligible. Exact full marginal
likelihood under the selected source plus the unchanged literal channel code
determines the winner; strict first-encountered ties. No beam, channel grammar
change, random restart, gold repair or post-result budget extension.

Maximum20sweeps,300wall seconds per core fit, improvement tolerance1e-8bits.
Keep the best completed candidate if interrupted; a completed full neighborhood
without improvement certifies only its returned parent as local, never global.
On numeric, support-initialization, hash, state-cap or audit failure stop and
preserve the failed run. Do not silently swap in a different case.

Primary observations: transfer edit count versus each fixed DEV004/order3
starting dictionary (B1=0,B2=22per448letters); fit/transfer marginal and exact
record count; generating-oracle comparison; literal key-independent minimum
edit floor. Improvement means fewer transfer edits on these exposed cases,
not a confirmatory pass threshold. Report regressions and nulls. No new null
rejection threshold or source/language choice. Source-only training remained
isolated, but the decision to run this intervention follows exposed results.
An unsupported transfer record counts as deleting all its true letters, with
null likelihood and a distinct missing support floor; do not abort merely
because a learned dictionary cannot read new ciphertext. Inconsistent support
between production and independent inference is a hard audit failure.

## Execution, checks and limits

At most2single-thread disposableCPUworkers,8GiB aggregate planning memory,
360CPU/420parent-wall seconds per case,200MiB compressed trace per case.
Total block45CPUminutes including validation/replay, no paidcompute/APIuse.
Each atomic inference can overrun the cooperative deadline; the disposable
process cap remains. Exact lattices retain the existing3million-node cap.
The complete trace records every candidate, objective and selected move.

New refiner tests compare every tiny neighbor score with complete plaintext
enumeration and independently authored literal-code arithmetic; test partial
deadline best-retention, unsupported models and certificate ownership. Reuse
the independently qualified exact higher-order source, unit enumeration and
original finite code. Subsequent numerical replay uses the previously
independently authored backward inference, ordinary edit-distance and0/1
support-floor references fromDEV004. Check every trace's moves, full-neighborhood
coverage, literal-cost/objective arithmetic, retention and certificates;
independently recompute final-model scores/readings. Do not describe trace
accounting as numerical replay of every neighbor likelihood.

The new refiner/runner is root-authored; independent agents hit usage limits
after completingDEV004. Independent reference code and exhaustive controls
remain usable, but this is not a new independent review of the optimizer.

Publish code/protocol, run the four fit jobs without answer access, freeze all
selected dictionaries before evaluation, then save predictions/metrics and
independent replay results. Old baselines/failures and all input hashes remain
preserved. Stop exposed-case tuning after this bounded intervention; fresh
qualification and broader-family/historical work are the next unresolved steps.
