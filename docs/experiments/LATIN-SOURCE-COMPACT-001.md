# LATIN-SOURCE-COMPACT-001: explicit storage revision after a baseline cap failure

2026-09-30. Adaptive engineering follow-up, **not** a successful rerun of
LATIN-SOURCE-MODEL-001's failed large statistical arm. The original 1,200,000
context limit was exceeded before that arm produced any scores. Its failure
record remains immutable; the original full architecture screen is unavailable.
Four original neural fits continue under their unchanged source and budgets.
The small statistical source completed, selecting tau64 at3.03741466899Pliny
bits/character. That score is already known and this is not a new blind trial.

## Change, invariant model and rationale

Use sparse arrays of integer context IDs, nonzero context-successor IDs and
counts. Query the one observed next character when scoring text; do not build
a Python dictionary and full23-way probability table for every context.
This is an engineering derivation from the same recurrence in
[LATIN-SOURCE-MODEL-001](LATIN-SOURCE-MODEL-001.md), whose Kambhatla2018,
Chen/Goodman1996 and Hauer/Kondrak2016 source review and claim limitations
apply unchanged. No new learned representation or empirical source claim is
borrowed from those papers.

Keep order12, context-total cutoff4, all successor counts, root add-.5,
recursive interpolation and the16/64/256/1024mass grid. Keep the exact same
97,850/6,497,939eligible letters, boundaries, loader and Pliny359,276selection
letters, split into at-most512letter chunks. Preserve all corpus checksums and
reserved-author restrictions. The model distribution is intended to equal an
uncapped version of the old estimator; prove exact count equality and numerical
score equality where the old implementation completed. No cutoff, order,
training text, selection author or metric changes in response to performance.

The **new, explicit** representation limit is8,000,000contexts and2GiBretained
array storage, checked after each order, with8GiBhost-memory planning. Old
limits remain intact in original files and experiment results. An8million-row
limit without sparse storage would be a materially different memory proposal;
here each context stores a uint64ID/total pair and each nonzero successor
another uint64ID/count pair. Temporary sorting buffers are additional memory.
No assertion that the old run exhausted physical RAM; it hit a preset context cap.

## Registration and decision use

One small fit, one large fit, then one audit each. Freeze source/tests/this plan,
all original source bindings, and old success/failure records; push/verify before
execution. Each fit and audit has1,800CPU/1,800wallseconds, compressed archive
≤512MiB, no paiduse. Only one CPU fit/audit at a time; it may run alongside the
single existing GPU fit. Attempt markers and outputs are exclusive. No retry,
changed limits or changed source rule within this revision after failure.

Commands with `PYTHONPATH=.:src` and three BLASthread variables set to1:

```
.venv/bin/python scripts/run_latin_source_compact001.py fit --dataset small --freeze REV
.venv/bin/python scripts/run_latin_source_compact001.py fit --dataset large --freeze REV
.venv/bin/python scripts/run_latin_source_compact001.py audit --dataset small --freeze REV
.venv/bin/python scripts/run_latin_source_compact001.py audit --dataset large --freeze REV
```

Select lowest PlinyBPC and smaller mass on exact ties. Save all four scores.
The small fit is a storage-equivalence replay, not independent evidence of
replication. The large fit supplies a supplementary model-family comparison
against the still-frozen neural fits. Report it as this named revision, including
the failed predecessor. The original architecture screen cannot be relabeled
PASS; a supplementary application of the same effect-size thresholds is
exploratory/adaptive and must be labeled. No cipher qualification or historical
language inference from source loss.

## Verification

Nineteen new tests compare every compact count and all four scores against
literal Python enumeration and the older suffix source, at five orders and
three cutoffs; include rare successors, leading-zero/high IDs, boundary resets,
absent contexts, archive equality, malformed counts and limit failures.
Full regression before freeze. Loaded arrays independently check strictly
sorted IDs, positive counts, totals and prefix/suffix closure.

For actual data: verify archive/input hashes, all root counts, four uniformly
spaced context strings at every represented depth using independent overlapping
substring searches (up to48contexts), and support cutoff for every retained row.
On small data, reconstruct and compare **every** count with the original frozen
archive and all four original source scores. For each selected mass, replay all
359,276Plinyletters using scalar Python bisect and a recursive literal-history
formula rather than the vectorized scorer; absolute score agreement≤1e-6bits.
Retain exact resources and any failures. These are separate algorithms by the
same research agent, not independent human replication.
