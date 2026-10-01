# CONTEXTUAL-GIBBS-001 — exact row conditionals versus matched-table random changes

2026-10-01. Exploratory contextual mapping-inference experiment on four EXPOSED artificial fixtures. [Method/source review and exact barrier control](../research/contextual-row-gibbs-2026-10-01.md). Goal actual decipherment; this experiment alone cannot qualify it.

## Fixed data and model

Use ONLY the four root queries inherited from CENSORED-CONTEXT-001, in case order0..3. Preserve both original glyph strings, zero offsets/initial source contexts and23unknown key rows. No answer-derived prefix bindings, prior fitted banks or oracle initializations. Existing fixtures contain two64-source-letter texts for cases0/1 and two224-letter texts for cases2/3; their earlier keys/readings/results are fully exposed. Model remains original order12Latin source/gotos, uniform42-code dictionary prior,6glyphs/one-or-two-glyph emissions, rho1/225and exact unclosed censoring/final EOS. Forced generation lengths differ from the inference model's geometric length prior, unchanged from earlier tests. No source training or candidate language selection.

Native qualification/source/count/array/library and all old186files remain frozen. Runner adds9explicit code/test/registration/review/result/audit paths:195total. Parent native and CENSORED result/full audit must bind exactly before fitting. Original fixture archive remains hash-bound; no change to charter, corpus, source, old benchmarks or samplers.

## Fixed panel and metrics

Two seeds81401/81402,32particles, balanced observation stride16, TWO arms per case/seed,16calls total:

- uniform42: the unchanged dictionary SMC implementation with42uniform-code random-row MH mutations per stage;
- gibbs: weight at old key, resample, then ONErandom-row update per particle enumerating all42contextual code likelihoods.

Initialization is identical for paired arms before different mutation RNG draws. Every complete stage costs32×43target tables per arm. The four schedules have17/16/56/51stages, so the maximum complete panel is770,560target-table attempts. No refill, prior refresh, joint move, annealing, adaptive stride, smoothing, score floor, guide or new decoder is introduced. Graph caps produce a distinct failed-call result with completed-stage trace and counters; never a zero likelihood. Each distinct registered call still runs once; no failed call is retried.

After fitting, choose the highest exact contextual-likelihood particle, breaking ties by original particle index. Using closed original ground truth ONLY for evaluation, measure its matches on actually used source-letter rows, whether all used rows match, and whether any final particle matches all used rows. Unused-row differences do not count as failure or success. This is mapping recovery, not a new Viterbi reading or gold-conditioned target. Extinct/capped calls have unavailable selected matches; their0contribution to the descriptive whole-panel rate is explicit. They fail all-complete criteria.

Prospective exploratory mapping-improvement gate: ALL16calls completepositive; Gibbs selected-used-row fraction at least10percentage points above uniform42; at leastONEof8Gibbs calls selects a complete used key. Separate stable-calibration gate: ALL16completepositive, allfour per-arm two-seed log-evidence spreads≤2nats, allcall maximum incremental weights≤.5. Passing only one clause does not pass its gate. Two seeds and exposed cases are weak uncertainty controls, not confidence intervals or confirmation. Equal target tables are not equal CPU/edge work; report both. No historical/recovery-language/neural qualification is claimed.

## Bounds, checks and publication

ONErun and ONEfullaudit, each1800wall/1600absoluteCPU seconds, sampled2GiBprocess peakRSS,128MiBignoredcompressed banks/traces, CPU1/BLAS1,0GPU/training/paid/cloud/manuscript holdout. Native perrecord300knodes/2Medges; percall2Bbridgeedges; panel8Bbridgeedges. The first rejected node/edge can be included in native counters. Native128MiBowned graph envelope excludes399571824bytes pinned source; separate particle/candidate32MiBguard. Source scanning/construction and IO costs counted in stage resources where applicable. Expected5–25minutes per stage, based on770560tables and the bank-specific benchmark's1–4msfull-score costs; arbitrary mappings can be more expensive. Hard limits hold regardless of estimate. No retries/resume/extensions/retuning or open-ended spending.

Fullaudit regenerates EVERYcall's RNG/native trajectory, proposals/resampling/gate/count/resource/hash/source identities, including recorded graph-cap failures. Every final key of a complete call is independently scored using the contextual Python DP:512maximum, ≤1e−7nats/zeroagreement. Python owned-memory768MiB/max300kcumulative states perrecord/1Browoperations percall bound explicit; a reference cap fails the audit, never counts as verified likelihood. Intermediate replay shares the native backend; this is same-author replay, not independent expert review. Exact tiny Fraction conditionals/weighted transport and the earlier5184native-prefix values provide additional references.

Before actual panel calls: complete finite conditional/barrier/SMC tests, FULL16-call tiny run+audit with successful and forced-cap controls, read-only actual input admission/tamper tests, full-tree tests/scopedlint, frozen195paths, links/JSON/diff; then commit/push/exactremote verify.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/run_contextual_gibbs001.py --freeze <registration-sha>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/audit_contextual_gibbs001.py
```

Publish compact per-call metrics/manifests/result/audit and negative results, retaining ignored full compressed banks/traces via immutable local hashes. No actual job is queued until frozen admission completes. Voynich UNSOLVED; goal ACTIVE.
