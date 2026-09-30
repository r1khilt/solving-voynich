# GLOBAL-SEARCH-SYSTEMS-001

2026-09-30. Prospective engineering benchmark, not a recovery qualification.
Do not open CONFIRM-002 fitting, transfer, labels or generating dictionaries.
Question: can exact strong-source replica search and partial-key bounds be
computed within a concrete local budget before another all-case development test?

## Inputs and method rationale

Use the already qualified order12/min-count4/tau64 Latin source and unchanged
native marginal ABI/build admitted by NATIVE-SUFFIX-001 and its audit. Its
selected-source manifest binds the compact count archive; no source retraining.
All source/runner/test/protocol dependencies in the runner's PATHS must exactly
match the published source freeze before the one measured invocation.

The [method review and derivations](../research/global-dictionary-search-after-confirm002.md)
cover previous Voynich limitations and Corlett/Penn2010, Nuhn2013,
Berg-Kirkpatrick/Klein2013 and neural decipherment precedents. Here the target is
the **sum** over unknown unit segmentation, with literal model cost. The paper's
Viterbi/bijection guarantees do not apply. Each symmetric Metropolis and replica
exchange component preserves its explicit target mathematically; finite output
has no convergence, posterior-sampling or global-optimum claim. Enumeration
tests check actual proposal probabilities, detailed balance, unsupported states
and a strict greedy trap. Rational full-history enumeration checks partial
marginals, every completion's bound, refinement monotonicity and duplicate units.

Workloads are generated directly in memory with seed94731: full42-unit pool,
six singleton and17distinct digram units permuted among23letters; three fixed
reshuffles of that dictionary; four iid-uniform224-letter artificial strings.
Short observations encode the first32letters per record. These strings are
neither Latin corpus passages nor draws from the source's geometric length law.
Known artificial keys exist solely to define reproducible cost workloads.
Their recovery is not measured. Raw inputs, lengths and checksums are retained.

Score every four-key/four-record short and full combination under exact native
inference. Independently replay all16short-record scores with Python; graph
node/edge/support counts must agree, likelihood tolerance1e-7. Measure native
and Python relaxation at0/11/20/23assigned rows on the first full observation,
with20,000nodes/80,000edges. Expected graph-cap failures are recorded, compared
and never replaced by partial bounds or probability zero.
Both backends must agree on completion versus cap; different state traversal
orders may hit different cap types first, so that order is not an equivalence
claim. Existing engineering reports are read for source/binary qualification;
the dedicated freeze allowlist excludes old learned dictionaries/cipher inputs.
Fully assigned relaxations must equal ordinary marginals when complete. A relaxation can exceed
probability1; it is not a normalized channel. Floating results are not interval
certificates. This phase estimates feasibility, not an A* implementation.

Then one replica search from fixed reshuffle1, on all four full artificial
observations; exact fit likelihood minus literal code bits*ln2. Seed94739,
temperatures1/2/4/8/16/32/64/128, replacement/swap/block weights2/1/1,
block sizes2/3/6, adjacent odd/even exchange every4iterations,
256iterations, at most2,048distinct fully scored keys and120cooperative seconds.
No tuning to artificial answers or old empirical cases. Keep every scored key,
proposal, exchange and work row in a checksum-bound ignored archive. Full-key
graph cap500,000nodes/2,000,000edges; scorer failures are terminal benchmark
failures, never unsupported probability. All observations reset to BOS.

## Budget, criteria and command

One CPU process/one BLAS thread, hard360wall/340CPU seconds; sampled4GiB host
RSS ceiling, no GPU, training, API or paid spend. Expected2–4minutes including
source construction; the deadline is a cap, not a performance forecast.
Run once after source checkpoint is pushed and its remote hash verified:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/benchmark_global_search_systems001.py --freeze SOURCE_COMMIT
```

Engineering PASS requires exact-equivalence tolerance, all eight complete-score
groups, bounded RSS, exact distinct-score accounting and at least one completed
proposal. A recorded partial-bound graph cap does not fail equivalence but does
limit practicality. Failure preserves started/progress/results and prohibits
silent retry/extension. Publish result and accounting before using measurements
to set a separate all32exposed-case development registration. No performance
or correctness gate here can establish decipherment or replace fresh recovery.
