# KEY-BANK-NEURAL-002: explicit fixed-shape revision of neural candidate reranking

Registered 2026-09-30 after the artificial SEQUENCE-MEMORY-001 engineering pass,
before any trained-model scoring in this namespace. Exposed development test,
not fresh qualification. KEY-BANK-NEURAL-001's failed attempt remains recorded.

## Unchanged scientific question and inputs

Use exactly KEY-BANK-NEURAL-001's eight READ-002 positive cases, all104,290 frozen
complete candidate tuples, all compatible keys with their unchanged statistical
fit weights, both original large LSTM checkpoints(seeds31103/31109), and original
tuple ranking. Reconstruct the same statistical winners before neural scoring.
No retraining, new proposals, interpolation, ensemble selection, gold insertion,
partial-case selection, or source/length-law changes. The two trained sources
score37,032,100candidate letters before independent reference checks.

The source probability, whole-history BOS handling, shared-key support, original
source literature/Voynich limitations, numerical/literal audits, answer isolation,
failure accounting, and all four development criteria are identical to the
[original registration](KEY-BANK-NEURAL-001.md). Both seeds must independently
beat43/3,584edits, support all16records, and keep everykey≤5%CER, with all16
case/model outputs audited and complete. No statistical unseen bound is reused:
only a restricted candidate-set optimum is claimed, no full neural evidence/MAP
or null discrimination. Fresh CONFIRM-002 never changes based on this result.

## Engineering change and implementation isolation

Every MPS forward call now has8rows×270positions. Dummy rows/right-padding never
earn score; records longer than270 fail rather than truncate. The maximum270 was
already counted over the full original union before NEURAL-001. Complete history
and the geometric length law rho=1/225 are preserved. Float32 MPS forward and
hostfloat64 probability calculation remain. The independent CPU-float64 and
one-character-at-a-time checks use the original reference implementation.

The new runner imports the immutable001engine and temporarily substitutes only
its experiment/output paths, expanded freeze inventory, and MPS scoring adapter.
It restores these globals after the scoped call, even on exceptions. The engine
source is not edited. The adapter splits each original64-record outer group into
constant8-row inner calls and delegates all CPU reference calls unchanged.
Tests exercise namespace restoration, complete eight-case staging, no-gold guards,
partial failure/denominators, padding/length-law equality, and sampled guard failure.

Save a synchronized driver/tensor/host-memory trace after every inner batch,
including a sample that triggers the guard. Save total processed records/letters
and memory maxima on both success and caught failure. Memory is sampled after
batches, not continuously. Original per64-record guard also remains. No cache
clearing within a model; original between-model cache clearing remains.

Require the published artificial benchmark's successful fixed8result, matching
audit/campaign, unchanged traces/score arrays and source dependencies before
starting. That benchmark reached1,259,061,248driverbytes and agreed across all
saved scores within3.411e-13nats. It did not reproduce/explain the earlier real
candidate failure; the empirical attempt remains necessary.

## Resources, staging and failure policy

One Mac GPU process, twoCPU numericalthreads;0paidAPI/training. Same original
prediction2400wall/1800CPU seconds, evaluation300/240, validation300CPU allocation:
2340CPU seconds under40CPU-minute ceiling. Same sampled4GiBhost/2GiBdriver caps.
Engineering throughput suggests~456scoringseconds; plan8–12wallminutes including
preparation/audits, with40wallminute maximumprediction. No automatic retry,
fallback, budget extension, batch tuning, or subset continuation.

Publish source/protocol/benchmark before one prediction attempt. Preserve
exclusive attempt markers, every completed case and all failure/trace artifacts.
Publish predictions or terminal failure before reopening the old answers for one
fixed evaluation. Every missing record still contributes224deletion penalties;
distinguish unavailable accuracy from actual errors. Report both sources whether
they improve or worsen. Candidate histories and source versions stay checksummed;
no CONFIRM-002 answers or protected manuscript material is accessed.

```text
.venv/bin/python -u scripts/run_key_bank_neural002.py predict --freeze SOURCE_COMMIT
# Publish all terminal predictions/checks or failure first.
.venv/bin/python -u scripts/run_key_bank_neural002.py evaluate --freeze PREDICTION_COMMIT
```
